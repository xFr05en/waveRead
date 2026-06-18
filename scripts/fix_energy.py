#!/usr/bin/env python3
"""
Post-processing fix: recalculate energy values using dataset-relative
normalization instead of the fixed 0.15 constant.

Run AFTER audio_scraper.py finishes:
    python3 fix_energy.py --csv lyrics_dataset_with_audio.csv
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def get_rms(audio_path: str) -> float | None:
    """Load clip and return mean RMS amplitude."""
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=30.0)
        rms = float(librosa.feature.rms(y=y).mean())
        return rms
    except Exception as e:
        log.warning(f"Could not load {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="lyrics_dataset_with_audio.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    has_audio = df["audio_path"].notna()
    log.info(f"Songs with audio: {has_audio.sum()} / {len(df)}")

    # Step 1: collect raw RMS for every song that has audio
    rms_values = []
    indices = []
    for i, row in df[has_audio].iterrows():
        rms = get_rms(str(row["audio_path"]))
        if rms is not None:
            rms_values.append(rms)
            indices.append(i)
            log.info(f"  [{len(rms_values)}/{has_audio.sum()}] {row['artist']} — {row['title']}: rms={rms:.4f}")

    if not rms_values:
        log.error("No valid audio files found.")
        return

    rms_arr = np.array(rms_values)
    log.info(f"\nRMS stats: min={rms_arr.min():.4f}  median={np.median(rms_arr):.4f}  max={rms_arr.max():.4f}")

    # Step 2: normalize to 0-1 using 5th–95th percentile range (robust to outliers)
    p5  = np.percentile(rms_arr, 5)
    p95 = np.percentile(rms_arr, 95)
    log.info(f"Normalizing energy with p5={p5:.4f}  p95={p95:.4f}")

    energy_norm = np.clip((rms_arr - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)

    # Step 3: write back into dataframe
    for idx, energy_val in zip(indices, energy_norm):
        df.at[idx, "energy"] = round(float(energy_val), 4)

    # Step 4: save
    out_path = args.csv.replace(".csv", "_fixed.csv")
    df.to_csv(out_path, index=False)
    log.info(f"\nSaved fixed dataset → {out_path}")
    log.info(f"Energy range after fix: {df.loc[indices, 'energy'].min():.3f} – {df.loc[indices, 'energy'].max():.3f}")


if __name__ == "__main__":
    main()
