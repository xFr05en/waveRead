#!/usr/bin/env python3
"""
Phase 5 — Evaluation
Loads best_model.pt and runs three analyses on the held-out test set:

  1. Regression metrics  — MAE and RMSE for valence, energy, danceability, tempo
  2. Decade classification — accuracy + 7×7 confusion matrix
  3. Ablation study      — text-only vs multimodal (force audio_missing=True for
                           the text-only pass to isolate the text branch's contribution)

Output:
    evaluation_report.txt   — human-readable summary
    evaluation_results.json — machine-readable results (for website / slides)

Usage:
    python3 evaluate.py
    python3 evaluate.py --ckpt checkpoints/best_model.pt
"""

import os
import json
import argparse
import logging

import numpy as np
import torch

from model import LyricalEmotionModel
from preprocess import make_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = "processed"
DECADE_NAMES  = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]


# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str) -> LyricalEmotionModel:
    model = LyricalEmotionModel(freeze_layers=12)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info(f"Loaded checkpoint: {ckpt_path}")
    return model


def move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def collect_predictions(model, loader, device, force_audio_missing: bool = False):
    """
    Run inference over loader and collect predictions + ground truth.

    force_audio_missing=True → forces all audio_missing flags to True,
    simulating text-only input for the ablation study.
    """
    preds = {k: [] for k in ["valence", "energy", "danceability", "tempo", "decade"]}
    gts   = {k: [] for k in ["valence", "energy", "danceability", "tempo_norm", "decade"]}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)

            audio_missing = batch["audio_missing"]
            if force_audio_missing:
                audio_missing = torch.ones_like(audio_missing)  # treat all as text-only

            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["audio_features"],
                audio_missing,
            )

            preds["valence"].extend(out["valence"].cpu().tolist())
            preds["energy"].extend(out["energy"].cpu().tolist())
            preds["danceability"].extend(out["danceability"].cpu().tolist())
            preds["tempo"].extend(out["tempo"].cpu().tolist())
            preds["decade"].extend(out["decade_logits"].argmax(dim=-1).cpu().tolist())

            gts["valence"].extend(batch["valence"].cpu().tolist())
            gts["energy"].extend(batch["energy"].cpu().tolist())
            gts["danceability"].extend(batch["danceability"].cpu().tolist())
            gts["tempo_norm"].extend(batch["tempo_norm"].cpu().tolist())
            gts["decade"].extend(batch["decade"].cpu().tolist())

    return (
        {k: np.array(v) for k, v in preds.items()},
        {k: np.array(v) for k, v in gts.items()},
    )


def regression_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    # Pearson correlation (catch edge case of zero std)
    if gt.std() < 1e-8 or pred.std() < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(pred, gt)[0, 1])
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "pearson_r": round(corr, 4)}


def confusion_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for p, g in zip(pred_labels, gt_labels):
        cm[int(g)][int(p)] += 1
    return cm


def format_cm(cm: np.ndarray, class_names: list) -> str:
    """Pretty-print a confusion matrix."""
    w = max(len(n) for n in class_names) + 2
    header = " " * (w + 2) + "  ".join(f"{n:>{w}}" for n in class_names)
    lines  = [header]
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>{w}}" for v in row)
        lines.append(f"{class_names[i]:>{w}}  {cells}")
    return "\n".join(lines)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    log.info(f"Device: {device}")

    # ── Load data ──
    test_data   = torch.load(
        os.path.join(PROCESSED_DIR, "test.pt"),
        map_location="cpu",
        weights_only=False,
    )
    log.info(f"Test samples: {len(test_data)}")
    test_loader = make_dataloader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── Load model ──
    model = load_model(args.ckpt, device)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. MULTIMODAL EVALUATION (text + audio)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n── Multimodal evaluation (text + audio) ──")
    mm_preds, gts = collect_predictions(model, test_loader, device, force_audio_missing=False)

    mm_reg = {}
    for task in ["valence", "energy", "danceability"]:
        mm_reg[task] = regression_metrics(mm_preds[task], gts[task])

    # Tempo: predictions are z-scored; compare in z-score space
    mm_reg["tempo"] = regression_metrics(mm_preds["tempo"], gts["tempo_norm"])

    mm_decade_acc = float((mm_preds["decade"] == gts["decade"]).mean())
    mm_cm = confusion_matrix(mm_preds["decade"], gts["decade"], n=len(DECADE_NAMES))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. TEXT-ONLY ABLATION (force audio_missing=True for all samples)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("── Text-only ablation ──")
    to_preds, _ = collect_predictions(model, test_loader, device, force_audio_missing=True)

    to_reg = {}
    for task in ["valence", "energy", "danceability"]:
        to_reg[task] = regression_metrics(to_preds[task], gts[task])
    to_reg["tempo"] = regression_metrics(to_preds["tempo"], gts["tempo_norm"])

    to_decade_acc = float((to_preds["decade"] == gts["decade"]).mean())
    to_cm = confusion_matrix(to_preds["decade"], gts["decade"], n=len(DECADE_NAMES))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. FORMAT REPORT
    # ══════════════════════════════════════════════════════════════════════════
    lines = []
    def h(title):
        lines.append("\n" + "=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)

    h("LYRICAL EMOTION ANALYZER — EVALUATION REPORT")
    lines.append(f"Checkpoint : {args.ckpt}")
    lines.append(f"Test set   : {len(test_data)} samples")

    # ── Regression results ──
    h("1. REGRESSION METRICS  (Multimodal vs Text-Only)")
    lines.append(f"\n{'Task':<16}  {'':>6}  {'MM MAE':>8}  {'MM RMSE':>9}  {'MM r':>7}  │  {'TO MAE':>8}  {'TO RMSE':>9}  {'TO r':>7}")
    lines.append("─" * 78)
    for task in ["valence", "energy", "danceability", "tempo"]:
        mm = mm_reg[task]
        to = to_reg[task]
        label = f"{task} (z-score)" if task == "tempo" else task
        lines.append(
            f"{label:<16}  {'':>6}  {mm['mae']:>8.4f}  {mm['rmse']:>9.4f}  {mm['pearson_r']:>7.4f}"
            f"  │  {to['mae']:>8.4f}  {to['rmse']:>9.4f}  {to['pearson_r']:>7.4f}"
        )

    # ── Decade classification ──
    h("2. DECADE CLASSIFICATION")
    lines.append(f"\n  Multimodal accuracy : {mm_decade_acc:.2%}  ({int(mm_decade_acc*len(test_data))}/{len(test_data)})")
    lines.append(f"  Text-only accuracy  : {to_decade_acc:.2%}  ({int(to_decade_acc*len(test_data))}/{len(test_data)})")
    lines.append(f"  Audio gain          : {(mm_decade_acc - to_decade_acc)*100:+.1f} pp")

    lines.append("\n  Multimodal confusion matrix (rows=actual, cols=predicted):")
    lines.append(format_cm(mm_cm, DECADE_NAMES))

    lines.append("\n  Text-only confusion matrix:")
    lines.append(format_cm(to_cm, DECADE_NAMES))

    # ── Ablation summary ──
    h("3. ABLATION SUMMARY  (Multimodal gain over Text-Only)")
    lines.append(f"\n{'Task':<16}  {'MAE Δ':>10}  {'RMSE Δ':>10}  {'r Δ':>10}")
    lines.append("─" * 50)
    for task in ["valence", "energy", "danceability", "tempo"]:
        mm = mm_reg[task]
        to = to_reg[task]
        d_mae  = to["mae"]  - mm["mae"]    # positive = multimodal better
        d_rmse = to["rmse"] - mm["rmse"]
        d_r    = mm["pearson_r"] - to["pearson_r"]
        lines.append(f"{task:<16}  {d_mae:>+10.4f}  {d_rmse:>+10.4f}  {d_r:>+10.4f}")
    lines.append(f"\n  Decade accuracy gain from audio: {(mm_decade_acc - to_decade_acc)*100:+.1f} percentage points")

    # ── Interpretation guide ──
    h("4. INTERPRETATION NOTES")
    lines.append("""
  MAE  = Mean Absolute Error (lower = better, same units as target)
  RMSE = Root Mean Squared Error (penalizes large errors more)
  r    = Pearson correlation (higher = better, max 1.0)

  Valence/energy/danceability targets are in [0, 1].
  Tempo target is z-scored (mean≈0, std≈1) — MAE of 1.0 ≈ 1 std dev off.

  Decade accuracy baseline (random): 14.3% (7 classes)
  Decade accuracy baseline (majority class): depends on split distribution.

  Audio gain columns: positive = multimodal outperforms text-only.
  If audio gains are small for valence/energy/danceability, this reflects
  the known limitation of Librosa-proxy labels rather than a model flaw.
""")

    report = "\n".join(lines)
    print(report)

    # Save text report
    report_path = "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"\nSaved → {report_path}")

    # Save JSON for website/slides
    results = {
        "multimodal": {
            "regression": mm_reg,
            "decade_accuracy": round(mm_decade_acc, 4),
            "confusion_matrix": mm_cm.tolist(),
        },
        "text_only": {
            "regression": to_reg,
            "decade_accuracy": round(to_decade_acc, 4),
            "confusion_matrix": to_cm.tolist(),
        },
        "ablation": {
            task: {
                "mae_gain":  round(to_reg[task]["mae"]  - mm_reg[task]["mae"],  4),
                "rmse_gain": round(to_reg[task]["rmse"] - mm_reg[task]["rmse"], 4),
                "r_gain":    round(mm_reg[task]["pearson_r"] - to_reg[task]["pearson_r"], 4),
            }
            for task in ["valence", "energy", "danceability", "tempo"]
        },
        "decade_accuracy_gain_pp": round((mm_decade_acc - to_decade_acc) * 100, 2),
        "test_n": len(test_data),
        "decade_names": DECADE_NAMES,
    }

    json_path = "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved → {json_path}")

    log.info("\nPhase 5 complete. Next: python3 inference.py")


if __name__ == "__main__":
    main()
