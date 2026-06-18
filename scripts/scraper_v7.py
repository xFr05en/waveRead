#!/usr/bin/env python3
"""
Phase 1 v7 — Lyrics Scraper (Spotify discovery + Genius lyrics)
===============================================================
Decouples discovery from lyrics fetching:
  1. Spotify Web API  → reliable year-based track discovery
  2. lyricsgenius    → lyrics scraping (web scrape, not API)

Target: 2,000 songs each for 2000s, 2010s, 2020s
        (1980s + 1990s are already covered in lyrics_dataset_v5.csv)

Discovery strategy:
  - Query Spotify by individual year (e.g. year:2005)
  - Use alphabetical prefix trick (a*, b*, ...) to bypass 1,000-result offset cap
  - Filter by popularity >= MIN_POPULARITY to avoid obscure tracks with no lyrics

Usage:
    python3 scraper_v7.py --test        # ~10 songs from 2010, sanity check
    python3 scraper_v7.py               # full run (2000s + 2010s + 2020s)
    python3 scraper_v7.py --decade 2010s  # single decade

Resume support:
    Saves after every song. Restart anytime — already-saved songs are skipped.

Output:
    lyrics_dataset_v7.csv
"""

from __future__ import annotations
import os, re, sys, time, logging, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Lock

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = "3c0ecbdc0f2d4e55980d739378409ed6"
SPOTIFY_CLIENT_SECRET = "a21f8ae55d774c9fb6db64c4ced0fdaf"
GENIUS_TOKEN          = "msUmoUoLkKqxgFX0mXdvv9LClUZnob1jpYwUq3BbtleNxlSiZCPZEQEtCi6aepc9"

OUTPUT_FILE       = "lyrics_dataset_v7.csv"
TARGET_PER_DECADE = 2000
MAX_WORDS         = 512
SLEEP_LYRICS      = 1.0      # seconds between Genius requests per thread
GENIUS_WORKERS    = 3
MIN_POPULARITY    = 30       # 0–100; filters out very obscure tracks

# Decades to collect (skip 1980s/1990s — already in v5)
TARGET_DECADES: dict[str, list[int]] = {
    "2000s": list(range(2000, 2010)),
    "2010s": list(range(2010, 2020)),
    "2020s": list(range(2020, 2025)),
}

DECADE_LABEL_MAP = {"1980s": 0, "1990s": 1, "2000s": 2, "2010s": 3, "2020s": 4}

# Alphabetical prefixes to bypass Spotify's 1,000-result offset cap per query
PREFIXES = list("abcdefghijklmnopqrstuvwxyz") + [""]

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
_write_lock   = Lock()
_thread_local = threading.local()


# ── SPOTIFY DISCOVERY ─────────────────────────────────────────────────────────

def get_spotify() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    ))


def discover_year(sp: spotipy.Spotify, year: int, limit: int = 400) -> list[dict]:
    """
    Fetch up to `limit` tracks from Spotify released in `year`.
    Uses alphabetical prefix queries to access different result pools and
    bypass the 1,000-item offset cap per query.
    """
    decade_str = f"{(year // 10) * 10}s"
    results: list[dict] = []
    seen:    set[str]   = set()

    for prefix in PREFIXES:
        if len(results) >= limit:
            break

        query  = f"year:{year} track:{prefix}*" if prefix else f"year:{year}"
        offset = 0

        while offset < 1000 and len(results) < limit:
            try:
                resp  = sp.search(q=query, type="track", limit=10, offset=offset, market="US")
                items = resp.get("tracks", {}).get("items", [])
                if not items:
                    break

                for item in items:
                    if len(results) >= limit:
                        break
                    if item.get("popularity", 0) < MIN_POPULARITY:
                        continue
                    if not item.get("artists"):
                        continue
                    artist = item["artists"][0]["name"].strip()
                    title  = item["name"].strip()
                    if not artist or not title:
                        continue
                    key = f"{artist.lower()}|{title.lower()}"
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "artist":       artist,
                        "title":        title,
                        "release_year": year,
                        "decade":       decade_str,
                    })

                offset += 10
                time.sleep(0.1)

            except Exception as e:
                log.debug(f"Spotify error (year={year} prefix={prefix!r} offset={offset}): {e}")
                time.sleep(2)
                break

    return results


def discover_all(done_keys: set[str], decade_counts: dict[str, int],
                 only_decade: str | None = None) -> list[dict]:
    """
    Run Spotify discovery for all target decades (or one specific decade).
    Collects ~3× the target per decade to absorb Genius failure rate.
    """
    sp = get_spotify()
    all_candidates: list[dict] = []

    decades = {only_decade: TARGET_DECADES[only_decade]} if only_decade else TARGET_DECADES

    for decade, years in decades.items():
        already = decade_counts.get(decade, 0)
        needed  = TARGET_PER_DECADE - already
        if needed <= 0:
            log.info(f"{decade}: already at target ({already}), skipping.")
            continue

        # Collect 3× needed to cover ~30–50% Genius miss rate
        target_candidates = min(needed * 3, 6000)
        per_year = max(200, target_candidates // len(years) + 50)

        log.info(f"{decade}: need {needed} more songs → targeting {target_candidates} candidates")
        decade_candidates: list[dict] = []

        for year in tqdm(years, desc=f"{decade} Spotify", unit="yr"):
            if len(decade_candidates) >= target_candidates:
                break
            tracks = discover_year(sp, year, limit=per_year)
            for t in tracks:
                key = f"{t['artist'].lower()}|{t['title'].lower()}"
                if key not in done_keys:
                    decade_candidates.append(t)

        log.info(f"  → {len(decade_candidates)} candidates for {decade}")
        all_candidates.extend(decade_candidates)

    log.info(f"Total candidates across all decades: {len(all_candidates)}")
    return all_candidates


# ── GENIUS LYRICS ─────────────────────────────────────────────────────────────

def _get_genius() -> lyricsgenius.Genius:
    """Per-thread Genius client (lyricsgenius is not thread-safe)."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = lyricsgenius.Genius(
            GENIUS_TOKEN,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Instrumental)",
                            "(Karaoke)", "(Cover)", "(Acoustic)"],
            remove_section_headers=False,
            timeout=15,
            retries=3,
            sleep_time=0.5,
            verbose=False,
        )
    return _thread_local.client


def clean_lyrics(lyrics: str) -> str | None:
    if not lyrics or len(lyrics.strip()) < 20:
        return None
    clean = re.sub(r"\[.*?\]", "", lyrics)
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    if len(lines) < 3:
        return None
    words = " ".join(lines).split()
    if len(words) < 10:
        return None
    return " ".join(words[:MAX_WORDS])


def fetch_lyrics(artist: str, title: str) -> str | None:
    """Fetch and clean lyrics. Always sleeps SLEEP_LYRICS regardless of outcome."""
    genius = _get_genius()
    result = None
    try:
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            result = clean_lyrics(song.lyrics)
    except Exception as e:
        log.debug(f"Genius error ({artist} — {title}): {e}")
    finally:
        time.sleep(SLEEP_LYRICS)
    return result


# ── RESUME / SAVE ─────────────────────────────────────────────────────────────

def load_state(path: str) -> tuple[set[str], dict[str, int]]:
    if not os.path.exists(path):
        return set(), {}
    df = pd.read_csv(path, usecols=["artist", "title", "decade"])
    done_keys     = set((df["artist"] + "|" + df["title"]).str.lower())
    decade_counts = df["decade"].value_counts().to_dict()
    log.info(f"Resuming — {len(done_keys)} songs already saved.")
    log.info(f"Decade counts: {decade_counts}")
    return done_keys, decade_counts


def append_row(row: dict, path: str):
    with _write_lock:
        df = pd.DataFrame([row])
        write_header = not os.path.exists(path)
        df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def build_dataset(output_path: str, test_mode: bool, only_decade: str | None):
    done_keys, decade_counts = load_state(output_path)

    if test_mode:
        log.info("TEST MODE: fetching 10 Spotify tracks from 2010")
        sp = get_spotify()
        raw = discover_year(sp, 2010, limit=10)
        candidates = [c for c in raw
                      if f"{c['artist'].lower()}|{c['title'].lower()}" not in done_keys]
        log.info(f"Test candidates: {len(candidates)}")
    else:
        candidates = discover_all(done_keys, decade_counts, only_decade=only_decade)

    if not candidates:
        log.info("No new candidates found.")
        return

    log.info(f"Fetching lyrics for {len(candidates)} candidates...")
    saved = failed = skipped = 0

    with tqdm(total=len(candidates), desc="Fetching lyrics", unit="song") as pbar:
        with ThreadPoolExecutor(max_workers=GENIUS_WORKERS) as executor:
            future_to_song = {
                executor.submit(fetch_lyrics, c["artist"], c["title"]): c
                for c in candidates
            }
            for future in as_completed(future_to_song):
                song   = future_to_song[future]
                lyrics = future.result()
                decade = song["decade"]

                if not lyrics:
                    failed += 1
                    pbar.update(1)
                    continue

                key = f"{song['artist'].lower()}|{song['title'].lower()}"
                if key in done_keys:
                    skipped += 1
                    pbar.update(1)
                    continue

                if decade_counts.get(decade, 0) >= TARGET_PER_DECADE:
                    skipped += 1
                    pbar.update(1)
                    continue

                row = {
                    "artist":       song["artist"],
                    "title":        song["title"],
                    "release_year": song["release_year"],
                    "decade":       decade,
                    "decade_label": DECADE_LABEL_MAP.get(decade, -1),
                    "lyrics":       lyrics,
                    "valence":      None,
                    "energy":       None,
                    "danceability": None,
                    "tempo":        None,
                    "audio_path":   None,
                }
                append_row(row, output_path)
                done_keys.add(key)
                decade_counts[decade] = decade_counts.get(decade, 0) + 1
                saved += 1
                pbar.update(1)
                pbar.set_postfix(saved=saved, failed=failed, skipped=skipped)

    log.info(f"\nDone. Saved: {saved} | Failed (no lyrics): {failed} | Skipped: {skipped}")


def print_summary(path: str):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    log.info(f"\n{'='*55}")
    log.info(f"  DATASET SUMMARY — {len(df)} songs")
    log.info(f"{'='*55}")
    print(df.groupby("decade").size().sort_index().to_string())
    wc = df["lyrics"].dropna().str.split().str.len()
    log.info(f"\nLyrics — mean: {wc.mean():.0f} words  min: {wc.min()}  max: {wc.max()}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lyrical Emotion Analyzer v7 — Spotify discovery + Genius lyrics"
    )
    parser.add_argument("--test",   action="store_true",
                        help="Fetch ~10 songs from 2010 as a sanity check")
    parser.add_argument("--decade", choices=["2000s", "2010s", "2020s"], default=None,
                        help="Run discovery for one decade only")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help="Output CSV path (default: lyrics_dataset_v7.csv)")
    args = parser.parse_args()

    build_dataset(output_path=args.output, test_mode=args.test, only_decade=args.decade)
    print_summary(args.output)

    if not args.test:
        log.info("\nNext: combine v5 + v7, then run audio_scraper_v2.py")
