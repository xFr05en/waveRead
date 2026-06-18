#!/usr/bin/env python3
"""
build_dataset.py — Final Dataset Builder
=========================================
Abandons live API scraping in favor of a pre-compiled open-source lyrics database.

Strategy:
  - Source: theelderemo/genius-lyrics-cleaned (HuggingFace, ~2.56 GB)
  - Filter for 2000s, 2010s, 2020s → sample 2,000 per decade
  - Merge with existing lyrics_dataset_v5.csv (1980s + 1990s, trimmed to 2,000 each)
  - Output: lyrics_dataset_final.csv — 10,000 songs, balanced across 5 decades

Usage:
    pip install datasets pandas tqdm
    python3 build_dataset.py

    # If HuggingFace dataset is unavailable, use Kaggle fallback:
    python3 build_dataset.py --source kaggle --kaggle-path /path/to/song_lyrics.csv

Output:
    lyrics_dataset_final.csv
"""

from __future__ import annotations
import os, re, sys, argparse, logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
V5_PATH       = "lyrics_dataset_v5.csv"
OUTPUT_PATH   = "lyrics_dataset_final.csv"
TARGET        = 2000          # songs per decade
MAX_WORDS     = 512
RANDOM_SEED   = 42

DECADE_LABEL_MAP = {
    "1960s": 0, "1970s": 1, "1980s": 2, "1990s": 3,
    "2000s": 4, "2010s": 5, "2020s": 6,
}

NEW_DECADES = {
    "1960s": (1960, 1969),
    "1970s": (1970, 1979),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def year_to_decade(year) -> str | None:
    try:
        y = int(year)
        start = (y // 10) * 10
        label = f"{start}s"
        return label if label in DECADE_LABEL_MAP else None
    except (ValueError, TypeError):
        return None


def clean_lyrics(text) -> str | None:
    """Strip section headers, normalize whitespace, enforce word cap."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        return None
    text = re.sub(r"\[.*?\]", "", text)           # remove [Verse], [Chorus], etc.
    text = re.sub(r"\d+Embed$", "", text)          # strip Genius "123Embed" suffix
    text = re.sub(r"See .+ LiveGet tickets.*", "", text, flags=re.IGNORECASE)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return None
    words = " ".join(lines).split()
    if len(words) < 10:
        return None
    return " ".join(words[:MAX_WORDS])


def standardize_columns(df: pd.DataFrame, year_col: str, artist_col: str,
                         title_col: str, lyrics_col: str) -> pd.DataFrame:
    """Rename and standardize columns to the shared schema."""
    df = df.rename(columns={
        year_col:   "release_year",
        artist_col: "artist",
        title_col:  "title",
        lyrics_col: "lyrics",
    })
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year", "artist", "title", "lyrics"])
    df["release_year"] = df["release_year"].astype(int)
    df["decade"]       = df["release_year"].apply(year_to_decade)
    df["decade_label"] = df["decade"].map(DECADE_LABEL_MAP)
    df["lyrics"]       = df["lyrics"].apply(clean_lyrics)
    df = df.dropna(subset=["decade", "lyrics"])
    # Add empty audio columns for compatibility with downstream pipeline
    for col in ["valence", "energy", "danceability", "tempo", "audio_path"]:
        if col not in df.columns:
            df[col] = None
    return df[["artist", "title", "release_year", "decade", "decade_label",
               "lyrics", "valence", "energy", "danceability", "tempo", "audio_path"]]


# ── LOAD V5 (1980s + 1990s) ───────────────────────────────────────────────────

def load_v5(path: str) -> pd.DataFrame:
    log.info(f"Loading existing dataset: {path}")
    df = pd.read_csv(path)
    log.info(f"  Raw rows: {len(df)}")

    # Keep only 1980s and 1990s — the new dataset covers the rest
    df = df[df["decade"].isin(["1980s", "1990s"])].copy()
    log.info(f"  After filtering to 1980s/1990s: {len(df)} rows")

    # Trim each decade to TARGET to keep the final dataset balanced
    parts = []
    for decade in ["1980s", "1990s"]:
        chunk = df[df["decade"] == decade]
        if len(chunk) > TARGET:
            chunk = chunk.sample(n=TARGET, random_state=RANDOM_SEED)
            log.info(f"  {decade}: trimmed to {TARGET}")
        else:
            log.info(f"  {decade}: {len(chunk)} songs (under target)")
        parts.append(chunk)

    result = pd.concat(parts, ignore_index=True)
    log.info(f"  V5 contribution: {len(result)} songs")
    return result


# ── LOAD HUGGINGFACE DATASET ───────────────────────────────────────────────────

def load_huggingface() -> pd.DataFrame:
    log.info("Downloading theelderemo/genius-lyrics-cleaned from HuggingFace...")
    log.info("(First run downloads ~2.56 GB — subsequent runs use cache)")
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset("theelderemo/genius-lyrics-cleaned", split="train")
    df = ds.to_pandas()
    log.info(f"  Downloaded: {len(df):,} rows | Columns: {list(df.columns)}")
    return df


# ── LOAD KAGGLE FALLBACK ───────────────────────────────────────────────────────

def load_kaggle(path: str) -> pd.DataFrame:
    log.info(f"Loading Kaggle dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Loaded: {len(df):,} rows | Columns: {list(df.columns)}")
    return df


# ── DETECT COLUMNS ────────────────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Auto-detect which columns contain year, artist, title, and lyrics.
    Handles variations across HuggingFace and Kaggle schemas.
    """
    cols = {c.lower(): c for c in df.columns}

    year_candidates   = ["year", "release_year", "release_date", "date"]
    artist_candidates = ["artist", "artist_name", "singer", "performer"]
    title_candidates  = ["title", "song", "song_name", "track", "track_name", "name"]
    lyrics_candidates = ["lyrics", "lyric", "text", "cleaned_lyrics", "song_lyrics"]

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    mapping = {
        "year":   pick(year_candidates),
        "artist": pick(artist_candidates),
        "title":  pick(title_candidates),
        "lyrics": pick(lyrics_candidates),
    }

    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        log.error(f"Could not detect columns for: {missing}")
        log.error(f"Available columns: {list(df.columns)}")
        log.error("Use --year-col, --artist-col, --title-col, --lyrics-col to specify manually.")
        sys.exit(1)

    log.info(f"  Detected columns: {mapping}")
    return mapping


# ── SAMPLE NEW DECADES ────────────────────────────────────────────────────────

def sample_new_decades(df: pd.DataFrame, done_keys: set[str]) -> pd.DataFrame:
    """Filter and sample 2,000 songs per new decade, excluding already-collected songs."""
    parts = []

    for decade, (start, end) in NEW_DECADES.items():
        chunk = df[(df["release_year"] >= start) & (df["release_year"] <= end)].copy()
        log.info(f"  {decade}: {len(chunk):,} candidates after year filter")

        # Exclude duplicates already in v5
        mask  = ~((chunk["artist"].str.lower() + "|" + chunk["title"].str.lower()).isin(done_keys))
        chunk = chunk[mask]
        log.info(f"  {decade}: {len(chunk):,} after dedup")

        if len(chunk) == 0:
            log.warning(f"  {decade}: NO candidates found — check dataset columns/year range")
            continue

        n = min(TARGET, len(chunk))
        if n < TARGET:
            log.warning(f"  {decade}: only {n} songs available (target {TARGET})")
        sampled = chunk.sample(n=n, random_state=RANDOM_SEED)
        log.info(f"  {decade}: sampled {len(sampled)} songs")
        parts.append(sampled)

    if not parts:
        log.error("No songs sampled for any new decade. Check dataset and column detection.")
        sys.exit(1)

    return pd.concat(parts, ignore_index=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build balanced 10,000-song lyrics dataset from pre-compiled source"
    )
    parser.add_argument("--source", choices=["huggingface", "kaggle"], default="huggingface")
    parser.add_argument("--kaggle-path", default=None,
                        help="Path to Kaggle CSV (required if --source kaggle)")
    parser.add_argument("--year-col",   default=None, help="Override year column name")
    parser.add_argument("--artist-col", default=None, help="Override artist column name")
    parser.add_argument("--title-col",  default=None, help="Override title column name")
    parser.add_argument("--lyrics-col", default=None, help="Override lyrics column name")
    parser.add_argument("--output",     default=OUTPUT_PATH)
    parser.add_argument("--v5",         default=V5_PATH)
    args = parser.parse_args()

    # ── Step 1: Load v5 (1980s + 1990s) ──────────────────────────────────────
    v5 = load_v5(args.v5)
    done_keys = set((v5["artist"].str.lower() + "|" + v5["title"].str.lower()))

    # ── Step 2: Load source dataset ───────────────────────────────────────────
    if args.source == "huggingface":
        raw = load_huggingface()
    else:
        if not args.kaggle_path:
            log.error("--kaggle-path required when --source kaggle")
            sys.exit(1)
        raw = load_kaggle(args.kaggle_path)

    # ── Step 3: Detect and standardize columns ────────────────────────────────
    col_overrides = {
        "year":   args.year_col,
        "artist": args.artist_col,
        "title":  args.title_col,
        "lyrics": args.lyrics_col,
    }
    detected = detect_columns(raw)
    # Apply any manual overrides
    for k, v in col_overrides.items():
        if v:
            detected[k] = v

    log.info("Standardizing columns and cleaning lyrics...")
    clean = standardize_columns(
        raw,
        year_col=detected["year"],
        artist_col=detected["artist"],
        title_col=detected["title"],
        lyrics_col=detected["lyrics"],
    )
    log.info(f"  After cleaning: {len(clean):,} rows")

    # ── Step 4: Sample 2,000 per new decade ──────────────────────────────────
    log.info("Sampling new decades (2000s, 2010s, 2020s)...")
    new_songs = sample_new_decades(clean, done_keys)

    # ── Step 5: Merge and save ────────────────────────────────────────────────
    final = pd.concat([v5, new_songs], ignore_index=True)
    final = final.drop_duplicates(subset=["artist", "title"])
    final.to_csv(args.output, index=False, encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info(f"  FINAL DATASET — {len(final):,} songs → {args.output}")
    log.info(f"{'='*55}")
    print(final.groupby("decade").size().sort_index().to_string())
    wc = final["lyrics"].dropna().str.split().str.len()
    log.info(f"\nLyrics — mean: {wc.mean():.0f} words  min: {wc.min()}  max: {wc.max()}")
    log.info("\nNext: python3 audio_scraper_v2.py --csv lyrics_dataset_final.csv")


if __name__ == "__main__":
    main()
