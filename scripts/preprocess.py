#!/usr/bin/env python3
"""
Phase 2 — Preprocessing & DataLoader
Tokenizes lyrics with RoBERTa, extracts 20 Librosa audio features,
normalizes all labels, does a stratified 80/10/10 split by decade,
and saves PyTorch-ready datasets.

Usage:
    python3 preprocess.py --csv lyrics_dataset_final_with_audio.csv

Output:
    processed/train.pt
    processed/val.pt
    processed/test.pt
    processed/scaler.pkl      ← StandardScaler for tempo (needed at inference)
    processed/split_stats.txt ← sanity check on split distribution
"""

import os
import pickle
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAX_LENGTH   = 512      # RoBERTa token limit
N_MFCC       = 13       # MFCC coefficients
AUDIO_DIM    = 20       # total Librosa feature dimensions
SR           = 22050    # sample rate for feature extraction
OUT_DIR      = "processed"

DECADE_MAP = {
    "1960s": 0, "1970s": 1, "1980s": 2, "1990s": 3,
    "2000s": 4, "2010s": 5, "2020s": 6,
}

# ── LIBROSA FEATURES ─────────────────────────────────────────────────────────

def extract_librosa_features(audio_path: str) -> np.ndarray | None:
    """
    Extract 20-dim feature vector from a 30s audio clip.

    Features (20 total):
      MFCCs × 13  (mean of each coefficient across time)
      chroma_stft mean
      spectral_centroid mean
      spectral_bandwidth mean
      spectral_rolloff mean
      ZCR mean
      RMS mean
      tempo (BPM)
    """
    try:
        y, sr = librosa.load(audio_path, sr=SR, duration=30.0)

        mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).mean(axis=1)  # (13,)
        chroma   = float(librosa.feature.chroma_stft(y=y, sr=sr).mean())
        cent     = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        bw       = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
        rolloff  = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
        zcr      = float(librosa.feature.zero_crossing_rate(y).mean())
        rms      = float(librosa.feature.rms(y=y).mean())
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo    = float(np.atleast_1d(tempo)[0])

        features = np.concatenate([mfccs, [chroma, cent, bw, rolloff, zcr, rms, tempo]])
        return features.astype(np.float32)  # (20,)

    except Exception as e:
        log.warning(f"Librosa failed for {audio_path}: {e}")
        return None


# ── DATASET CLASS ─────────────────────────────────────────────────────────────

class LyricsAudioDataset(Dataset):
    """
    Each sample contains:
      input_ids       (512,)   — RoBERTa token ids
      attention_mask  (512,)   — RoBERTa attention mask
      audio_features  (20,)    — Librosa features (zeros if audio missing)
      audio_missing   bool     — True when audio was unavailable
      valence         float    — target label (0–1)
      energy          float    — target label (0–1)
      danceability    float    — target label (0–1)
      tempo_norm      float    — StandardScaler-normalized tempo
      decade          int      — class index (0=1960s … 6=2020s)
    """

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids":      torch.tensor(s["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "audio_features": torch.tensor(s["audio_features"], dtype=torch.float32),
            "audio_missing":  torch.tensor(s["audio_missing"],  dtype=torch.bool),
            "valence":        torch.tensor(s["valence"],        dtype=torch.float32),
            "energy":         torch.tensor(s["energy"],         dtype=torch.float32),
            "danceability":   torch.tensor(s["danceability"],   dtype=torch.float32),
            "tempo_norm":     torch.tensor(s["tempo_norm"],     dtype=torch.float32),
            "decade":         torch.tensor(s["decade"],         dtype=torch.long),
        }


def make_dataloader(samples: list[dict], batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        LyricsAudioDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="lyrics_dataset_final_with_audio.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load dataset ──
    df = pd.read_csv(args.csv)
    log.info(f"Loaded {len(df)} rows from {args.csv}")

    # Drop rows with no lyrics
    df = df[df["lyrics"].notna() & (df["lyrics"].str.strip() != "")].reset_index(drop=True)
    log.info(f"After dropping empty lyrics: {len(df)} rows")

    # ── 2. Tokenize lyrics ──
    log.info("Loading RoBERTa tokenizer ...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    log.info("Tokenizing lyrics (this takes ~1 min) ...")
    encodings = tokenizer(
        df["lyrics"].tolist(),
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    input_ids      = encodings["input_ids"]       # (N, 512)
    attention_mask = encodings["attention_mask"]  # (N, 512)
    log.info(f"Tokenization done. Shape: {input_ids.shape}")

    # ── 3. Extract Librosa features ──
    log.info("Extracting Librosa audio features ...")
    audio_features_list = []
    audio_missing_list  = []

    for i, row in df.iterrows():
        path = str(row.get("audio_path", ""))
        if row["audio_path"] != row["audio_path"] or path == "" or not os.path.exists(path):
            # Missing audio — use zero vector + flag
            audio_features_list.append(np.zeros(AUDIO_DIM, dtype=np.float32))
            audio_missing_list.append(True)
            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(df)}] missing audio")
        else:
            feats = extract_librosa_features(path)
            if feats is None:
                audio_features_list.append(np.zeros(AUDIO_DIM, dtype=np.float32))
                audio_missing_list.append(True)
            else:
                audio_features_list.append(feats)
                audio_missing_list.append(False)
            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(df)}] audio features extracted")

    audio_features = np.stack(audio_features_list)  # (N, 20)
    audio_missing  = np.array(audio_missing_list)    # (N,)
    log.info(f"Audio features shape: {audio_features.shape} | Missing: {audio_missing.sum()}")

    # ── 4. Encode decade labels ──
    mapped = df["decade"].map(DECADE_MAP)
    if mapped.isna().any():
        bad = df.loc[mapped.isna(), "decade"].unique()
        raise ValueError(f"Unknown decade values not in DECADE_MAP: {bad}")
    decade_labels = mapped.values.astype(np.int64)

    # ── 5. Stratified 80/10/10 split by decade ──
    # IMPORTANT: the split is computed BEFORE fitting any scaler, so that all
    # normalization statistics are derived from the TRAIN split only (no val/test
    # leakage into the saved scalers or the normalized features).
    strata = df["decade"]

    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, temp_idx = next(sss_outer.split(range(len(df)), strata))

    strata_temp = strata.iloc[temp_idx].reset_index(drop=True)
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(sss_inner.split(range(len(temp_idx)), strata_temp))

    val_idx  = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    # Boolean mask of training rows — used to fit scalers on the train split only.
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True

    # ── 6. Normalize audio features (fit StandardScaler on TRAIN split only) ──
    audio_scaler = StandardScaler()
    # Fit only on training rows that actually have audio.
    audio_fit_rows = train_mask & ~audio_missing
    audio_scaler.fit(audio_features[audio_fit_rows])
    audio_features_norm = audio_scaler.transform(audio_features)
    # Zero out missing rows after scaling (keeps null embedding learnable in model)
    audio_features_norm[audio_missing] = 0.0

    # ── 7. Normalize tempo (fit on TRAIN split only; other labels already 0–1) ──
    tempo_raw = df["tempo"].values.astype(np.float32)
    tempo_scaler = StandardScaler()
    tempo_fit_rows = train_mask & ~np.isnan(tempo_raw)
    tempo_scaler.fit(tempo_raw[tempo_fit_rows].reshape(-1, 1))
    tempo_norm = tempo_scaler.transform(tempo_raw.reshape(-1, 1)).flatten()
    tempo_norm = np.nan_to_num(tempo_norm, nan=0.0)

    # Save scalers for inference time (fit on train split only)
    scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"audio": audio_scaler, "tempo": tempo_scaler}, f)
    log.info(f"Saved scalers → {scaler_path}")

    # ── 8. Build sample list ──
    samples = []
    for i in range(len(df)):
        row = df.iloc[i]
        samples.append({
            "input_ids":      input_ids[i].tolist(),
            "attention_mask": attention_mask[i].tolist(),
            "audio_features": audio_features_norm[i].tolist(),
            "audio_missing":  bool(audio_missing[i]),
            "valence":        float(row["valence"]) if not pd.isna(row["valence"]) else 0.5,
            "energy":         float(row["energy"])  if not pd.isna(row["energy"])  else 0.5,
            "danceability":   float(row["danceability"]) if not pd.isna(row["danceability"]) else 0.5,
            "tempo_norm":     float(tempo_norm[i]),
            "decade":         int(decade_labels[i]),
            # Keep metadata for analysis
            "artist":         str(row["artist"]),
            "title":          str(row["title"]),
            "decade_str":     str(row["decade"]),
        })

    # ── 9. Slice samples into splits ──
    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]
    test_samples  = [samples[i] for i in test_idx]

    log.info(f"Split → train: {len(train_samples)} | val: {len(val_samples)} | test: {len(test_samples)}")

    # ── 10. Save splits ──
    for name, split in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = os.path.join(OUT_DIR, f"{name}.pt")
        torch.save(split, path)
        log.info(f"Saved {path}")

    # ── 11. Sanity check on split distribution ──
    stats_lines = ["Split distribution (decade)\n", "=" * 50]
    for name, idx_arr in [("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)]:
        stats_lines.append(f"\n{name}:")
        dist = strata.iloc[idx_arr].value_counts().sort_index()
        for label, count in dist.items():
            stats_lines.append(f"  {label}: {count}")

    stats_text = "\n".join(stats_lines)
    print("\n" + stats_text)

    stats_path = os.path.join(OUT_DIR, "split_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    log.info(f"Saved split stats → {stats_path}")

    # ── 12. Quick label stats ──
    train_df = pd.DataFrame([
        {"valence": s["valence"], "energy": s["energy"],
         "danceability": s["danceability"], "decade": s["decade"]}
        for s in train_samples
    ])
    log.info("\nTraining set label stats:")
    log.info(f"\n{train_df.describe().round(3)}")

    log.info("\nPhase 2 complete. Next: python3 model.py")


if __name__ == "__main__":
    main()
