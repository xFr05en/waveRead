#!/usr/bin/env python3
"""
Phase 1.5 v2 — Audio Scraper (parallel workers)
=================================================
Downloads 30s audio clips for every song in lyrics_dataset_v2.csv,
extracts Librosa features (valence, energy, danceability, tempo),
and merges results into lyrics_dataset_v2_with_audio.csv.

Key improvements over audio_scraper.py:
  - 5 parallel download workers (5x faster)
  - Per-song resume: crash at song 12,000 → restart from 12,001
  - Temp file safety: partial downloads never corrupt the output
  - Smarter sleep: randomized per worker, not blocking the whole script
  - Progress bar shows ETA across all workers

Usage:
    python3 audio_scraper_v2.py                            # default input
    python3 audio_scraper_v2.py --csv lyrics_dataset_v2.csv
    python3 audio_scraper_v2.py --workers 8               # more workers if connection is fast

Output:
    audio_clips_v2/                      — downloaded .wav clips
    audio_progress_v2.csv                — per-song checkpoint
    lyrics_dataset_v2_with_audio.csv     — final merged dataset
"""

from __future__ import annotations
import os
import sys
import time
import random
import shutil
import logging
import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH      = "lyrics_dataset_final.csv"
AUDIO_DIR     = "audio_clips_final"
PROGRESS_FILE = "audio_progress_final.csv"
OUTPUT_FILE   = "lyrics_dataset_final_with_audio.csv"

WORKERS       = 5        # parallel download threads
CLIP_START    = 60       # skip first N seconds (avoid intros)
CLIP_END      = 90       # end of clip (30s window)
SLEEP_MIN     = 3        # min sleep between downloads per worker
SLEEP_MAX     = 8        # max sleep between downloads per worker
RMS_THRESHOLD = 0.01     # silence filter threshold
SR_LIBROSA    = 22050    # Librosa sample rate

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Thread-safe lock for writing to the progress CSV
_write_lock = Lock()


# ── PREFLIGHT ─────────────────────────────────────────────────────────────────

def check_dependencies():
    import shutil as sh
    errors = []
    if sh.which("yt-dlp") is None:
        errors.append("yt-dlp not found. Run: pip install yt-dlp --break-system-packages")
    if sh.which("ffmpeg") is None:
        errors.append("ffmpeg not found. Run: brew install ffmpeg")
    try:
        import librosa  # noqa
    except ImportError:
        errors.append("librosa not found. Run: pip install librosa --break-system-packages")
    if errors:
        for e in errors:
            log.error(e)
        sys.exit(1)
    log.info("All dependencies OK.")


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def download_clip(artist: str, title: str, out_path: str) -> bool:
    """
    Download a 30s clip (seconds 60–90) from YouTube.
    Saves to a temp file first — only moves to out_path on success.
    Returns True on success, False on failure.
    """
    query = f"{artist} {title} official audio"

    with tempfile.TemporaryDirectory() as tmpdir:
        template = os.path.join(tmpdir, "audio.%(ext)s")
        cmd = [
            "yt-dlp",
            f"ytsearch1:{query}",
            "--extract-audio",
            "--audio-format", "wav",
            "--download-sections", f"*{CLIP_START}-{CLIP_END}",
            "--force-keyframes-at-cuts",
            "--output", template,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
        ]

        for attempt in range(4):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=180
                )
                wav_files = [f for f in os.listdir(tmpdir) if f.endswith(".wav")]
                if wav_files:
                    # Only move to final location after successful download
                    shutil.copy(os.path.join(tmpdir, wav_files[0]), out_path)
                    return True

                stderr = result.stderr.lower()
                if "429" in stderr or "too many requests" in stderr:
                    wait = min(2 ** attempt * 30, 300)
                    log.warning(f"Rate limited — sleeping {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue

                return False

            except subprocess.TimeoutExpired:
                log.debug(f"Timeout for '{title}' (attempt {attempt+1})")
                if attempt < 3:
                    time.sleep(10)
                continue

    return False


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_audio_labels(audio_path: str) -> dict | None:
    """
    Extract valence, energy, danceability, tempo from a 30s WAV clip.
    Returns None if the clip is silent or extraction fails.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SR_LIBROSA, duration=30.0)
    except Exception as e:
        log.debug(f"Load failed: {e}")
        return None

    rms_raw = float(np.sqrt(np.mean(y ** 2)))
    if rms_raw < RMS_THRESHOLD:
        return None

    try:
        # Energy: RMS normalized
        rms_frames = librosa.feature.rms(y=y)[0]
        energy = float(np.clip(np.mean(rms_frames) / 0.15, 0.0, 1.0))

        # Tempo: beat tracker
        tempo_bpm, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(tempo_bpm)

        # Valence: spectral brightness + chroma mode
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_norm = float(np.clip(np.mean(centroid) / 4000.0, 0.0, 1.0))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        major_score = float(chroma_mean[[0, 4, 7]].mean())
        minor_score = float(chroma_mean[[0, 3, 7]].mean())
        mode_bias = float(np.clip((major_score - minor_score + 0.1) / 0.2, 0.0, 1.0))
        valence = float(np.clip(0.55 * centroid_norm + 0.45 * mode_bias, 0.0, 1.0))

        # Danceability: beat regularity
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(beats) >= 2:
            beat_period = int(np.median(np.diff(beats)))
            ac = float(np.correlate(onset_env, onset_env, mode="full")[len(onset_env) - 1])
            if beat_period < len(onset_env):
                ac_beat = float(np.correlate(
                    onset_env, onset_env, mode="full"
                )[len(onset_env) - 1 + beat_period])
            else:
                ac_beat = ac * 0.5
            danceability = float(np.clip(ac_beat / (ac + 1e-6), 0.0, 1.0))
        else:
            danceability = 0.5

    except Exception as e:
        log.debug(f"Feature extraction failed: {e}")
        return None

    return {
        "valence":      round(valence, 4),
        "energy":       round(energy, 4),
        "danceability": round(danceability, 4),
        "tempo":        round(tempo_bpm, 2),
    }


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def load_done_keys(progress_file: str) -> set[str]:
    """Return 'artist|title' keys already saved in progress CSV."""
    if os.path.exists(progress_file):
        df = pd.read_csv(progress_file, usecols=["artist", "title"])
        keys = set(df["artist"] + "|" + df["title"])
        log.info(f"Resuming — {len(keys)} songs already processed.")
        return keys
    return set()


def append_progress(row: dict, progress_file: str):
    """Thread-safe append of one row to the progress CSV."""
    with _write_lock:
        df = pd.DataFrame([row])
        write_header = not os.path.exists(progress_file)
        df.to_csv(progress_file, mode="a", index=False,
                  header=write_header, encoding="utf-8")


# ── WORKER ────────────────────────────────────────────────────────────────────

def process_song(row: pd.Series, audio_dir: str, progress_file: str) -> dict:
    """
    Full pipeline for one song: download → extract features → save checkpoint.
    Runs in a thread worker. Returns a result dict for logging.
    """
    artist = str(row["artist"])
    title  = str(row["title"])

    # Build safe filename
    safe = f"{artist}_{title}".replace("/", "_").replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in "_-")[:60]
    audio_path = os.path.join(audio_dir, f"{safe}.wav")

    result = {**row.to_dict(), "audio_path": None}

    # Download
    dl_ok = download_clip(artist, title, audio_path)
    if not dl_ok:
        append_progress(result, progress_file)
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        return {"status": "download_failed", "title": title}

    # Extract features
    labels = extract_audio_labels(audio_path)
    if labels is None:
        result["audio_path"] = audio_path
        append_progress(result, progress_file)
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        return {"status": "features_failed", "title": title}

    # Success
    result.update({
        "valence":      labels["valence"],
        "energy":       labels["energy"],
        "danceability": labels["danceability"],
        "tempo":        labels["tempo"],
        "audio_path":   audio_path,
    })
    append_progress(result, progress_file)
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
    return {"status": "ok", "title": title, **labels}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 v2 — Parallel Audio Scraper")
    parser.add_argument("--csv",     default=CSV_PATH,      help="Input lyrics CSV")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Parallel download workers")
    args = parser.parse_args()

    check_dependencies()
    os.makedirs(AUDIO_DIR, exist_ok=True)

    df   = pd.read_csv(args.csv)
    done = load_done_keys(PROGRESS_FILE)

    todo = df[~(df["artist"] + "|" + df["title"]).isin(done)]
    log.info(f"Dataset: {len(df)} songs | Done: {len(done)} | Remaining: {len(todo)}")

    if todo.empty:
        log.info("All songs already processed. Skipping to merge step.")
    else:
        log.info(f"Starting {args.workers} parallel workers...")

        # Stats counters
        ok = failed_dl = failed_feat = 0

        with tqdm(total=len(todo), desc="Downloading audio", unit="song") as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_song, row, AUDIO_DIR, PROGRESS_FILE): row
                    for _, row in todo.iterrows()
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "ok":
                        ok += 1
                        pbar.set_postfix(ok=ok, fail=failed_dl + failed_feat)
                    elif result["status"] == "download_failed":
                        failed_dl += 1
                    else:
                        failed_feat += 1
                    pbar.update(1)

        log.info(f"\nResults: {ok} success | {failed_dl} download fails | {failed_feat} feature fails")

    # ── Merge into final CSV ──
    if not os.path.exists(PROGRESS_FILE):
        log.error("No progress file found — nothing to merge.")
        sys.exit(1)

    prog  = pd.read_csv(PROGRESS_FILE)
    final = prog.drop_duplicates(subset=["artist", "title"], keep="last").reset_index(drop=True)

    filled = final["valence"].notna().sum()
    total  = len(final)
    log.info(f"\nAudio labels filled: {filled}/{total} songs ({filled/total*100:.1f}%)")

    final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    log.info(f"Saved → {OUTPUT_FILE}")
    log.info("Phase 1.5 complete. Next: python3 preprocess.py --csv lyrics_dataset_final_with_audio.csv")


if __name__ == "__main__":
    main()
