#!/usr/bin/env python3
"""
Phase 4 — Training Loop
Trains LyricalEmotionModel with AdamW (differential learning rates), a single
per-step warmup→cosine LR schedule, gradient clipping, early stopping, and
per-epoch checkpointing.

Fine-tuning strategy (dataset: ~11,200 training songs, 7 balanced decades):
    • New heads / projection / fusion train from scratch at --lr (default 5e-4).
    • The top --unfreeze-layers RoBERTa encoder layers are fine-tuned at the
      much lower --encoder-lr (default 2e-5); deeper layers + embeddings stay
      frozen. The default fine-tunes the top 6 layers — the strongest setting
      that still trains stably at this dataset size, guarded by early stopping,
      dropout, and weight decay.
    • Weight decay is applied only to weight matrices; biases, LayerNorm gains,
      the learned null-audio embedding, and the loss log-σ are excluded
      (decaying log-σ would pull σ→1 and undermine the uncertainty weighting).

    Drop --unfreeze-layers to 2–4 if you see train/val divergence, or raise to
    12 (full fine-tune) if you have the compute and val keeps improving. Each
    unfrozen layer adds ~7M trainable params and more compute per epoch.

Usage:
    python3 train.py                        # recommended defaults (top 6 layers)
    python3 train.py --unfreeze-layers 12   # full RoBERTa fine-tune (slowest)
    python3 train.py --unfreeze-layers 0    # freeze all RoBERTa (fastest)
    python3 train.py --epochs 40 --lr 3e-4

Output:
    checkpoints/best_model.pt    ← best val-loss checkpoint
    checkpoints/last_model.pt    ← end-of-training snapshot
    checkpoints/training_log.csv ← per-epoch metrics for plotting
"""

import os
import csv
import shutil
import argparse
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from model import LyricalEmotionModel, build_model, MIN_LOG_SIGMA
from preprocess import LyricsAudioDataset, make_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── DEFAULTS ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = "processed"
CKPT_DIR      = "checkpoints"
LOG_FILE      = "training_log.csv"


# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_split(name: str) -> list[dict]:
    path = os.path.join(PROCESSED_DIR, f"{name}.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    log.info(f"Loaded {name}: {len(data)} samples from {path}")
    return data


def move_batch(batch: dict, device: str) -> dict:
    """Move all tensor values in a batch dict to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def build_optimizer(model, lr: float, encoder_lr: float, weight_decay: float):
    """
    AdamW with differential learning rates (low for pretrained RoBERTa, higher
    for from-scratch heads) and decoupled weight decay.

    Weight decay is applied ONLY to weight matrices. Biases, LayerNorm gains,
    the learned null-audio embedding, and the loss log-σ parameters are excluded
    — standard practice, and decaying log-σ would pull σ→1 and fight the
    homoscedastic uncertainty weighting.

    Returns (optimizer, n_head_params, n_encoder_params).
    """
    norm_param_ids = {
        id(p) for m in model.modules() if isinstance(m, nn.LayerNorm)
        for p in m.parameters(recurse=False)
    }

    def is_no_decay(name: str, p) -> bool:
        return (id(p) in norm_param_ids
                or name.endswith(".bias")
                or name in ("log_sigma", "null_audio_embedding"))

    buckets = {"enc_d": [], "enc_n": [], "head_d": [], "head_n": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "enc" if name.startswith("roberta.") else "head"
        kind = "n" if is_no_decay(name, p) else "d"
        buckets[f"{side}_{kind}"].append(p)

    groups = []
    for key, glr in [("head_d", lr), ("head_n", lr),
                     ("enc_d", encoder_lr), ("enc_n", encoder_lr)]:
        if buckets[key]:
            groups.append({
                "params":       buckets[key],
                "lr":           glr,
                "weight_decay": 0.0 if key.endswith("_n") else weight_decay,
            })

    optimizer = AdamW(groups, betas=(0.9, 0.999))
    n_head = sum(p.numel() for k in ("head_d", "head_n") for p in buckets[k])
    n_enc  = sum(p.numel() for k in ("enc_d", "enc_n")  for p in buckets[k])
    return optimizer, n_head, n_enc


def run_epoch(
    model: LyricalEmotionModel,
    loader,
    device: str,
    optimizer=None,
    scheduler=None,
) -> dict:
    """
    One pass over the loader. Pass optimizer=None for eval mode.
    When training, the per-step LR scheduler is advanced after every batch.
    Returns dict of average losses for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    totals = {k: 0.0 for k in ["total", "valence", "energy", "danceability", "tempo", "decade"]}
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = move_batch(batch, device)

            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["audio_features"],
                batch["audio_missing"],
            )
            loss, components = model.compute_loss(outputs, batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            totals["total"]        += loss.item()
            totals["valence"]      += components["valence"]
            totals["energy"]       += components["energy"]
            totals["danceability"] += components["danceability"]
            totals["tempo"]        += components["tempo"]
            totals["decade"]       += components["decade"]
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def decade_accuracy(model: LyricalEmotionModel, loader, device: str) -> float:
    """Compute decade classification accuracy over a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["audio_features"],
                batch["audio_missing"],
            )
            preds = out["decade_logits"].argmax(dim=-1)
            correct += (preds == batch["decade"]).sum().item()
            total   += batch["decade"].size(0)
    return correct / max(total, 1)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch-size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=5e-4,
                        help="LR for new heads / projection / fusion (trained from scratch)")
    parser.add_argument("--encoder-lr",      type=float, default=2e-5,
                        help="LR for unfrozen RoBERTa layers (low — pretrained weights)")
    parser.add_argument("--weight-decay",    type=float, default=0.01)
    parser.add_argument("--warmup-frac",     type=float, default=0.1,
                        help="Fraction of total steps used for linear warmup")
    parser.add_argument("--patience",        type=int,   default=7,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--unfreeze-layers", type=int,   default=6,
                        help="Top RoBERTa encoder layers to fine-tune "
                             "(0 = all frozen; 6 = recommended for ~11k samples; 12 = full)")
    parser.add_argument("--processed-dir",   default=PROCESSED_DIR)
    parser.add_argument("--ckpt-dir",        default=CKPT_DIR)
    parser.add_argument("--resume",          default=None,
                        help="Resume model weights from this checkpoint. Uses a fresh "
                             "optimizer + scheduler and appends to the existing "
                             "training_log.csv (continuing the epoch numbering).")
    args = parser.parse_args()

    # ── Device ──
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    log.info(f"Device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Data ──
    train_data = load_split("train")
    val_data   = load_split("val")

    train_loader = make_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = make_dataloader(val_data,   batch_size=args.batch_size, shuffle=False)

    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ──
    # Default: freeze ALL 12 RoBERTa layers (freeze_layers=12)
    # If --unfreeze-layers N: freeze layers 0..(12-N-1), unfreeze (12-N)..11
    freeze_n = 12 - args.unfreeze_layers
    freeze_n = max(0, min(freeze_n, 12))

    log.info(f"Freezing first {freeze_n} RoBERTa encoder layers "
             f"(unfreeze_layers={args.unfreeze_layers})")

    model = LyricalEmotionModel(freeze_layers=freeze_n)

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        # Back up the source checkpoint first: the resumed run overwrites
        # best_model.pt as new bests are found, and the loss scale changed (σ
        # clamp), so a worse epoch could otherwise clobber your good weights.
        backup = args.resume + ".bak"
        shutil.copyfile(args.resume, backup)
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not load {args.resume} into the current model. If this "
                f"checkpoint predates recent model.py changes (LayerNorm audio branch "
                f"/ no pooler), it is incompatible — train fresh instead.\n{e}"
            ) from e
        log.info(f"Resumed weights from {args.resume} (backed up → {backup}). "
                 f"Fresh optimizer + scheduler; training {args.epochs} more epoch(s).")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    ratio     = trainable / max(len(train_data), 1)
    log.info(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%) "
             f"| {ratio:.0f} params/training-sample")
    if ratio > 8000:
        log.warning(
            f"⚠️  {ratio:.0f} trainable params/sample is high for {len(train_data)} samples. "
            f"If val loss diverges from train loss, lower --unfreeze-layers."
        )

    # Warm-start tempo sigma (fresh runs only): tempo loss is ~5x larger at init
    # due to z-scoring. On --resume we keep the checkpoint's learned sigmas.
    if not args.resume:
        with torch.no_grad():
            model.log_sigma[3] = 0.5   # sigma_tempo starts at exp(0.5) ≈ 1.65

    model = model.to(device)

    # ── Optimizer: differential LR × decoupled weight decay (see build_optimizer) ──
    optimizer, n_head, n_enc = build_optimizer(model, args.lr, args.encoder_lr, args.weight_decay)
    log.info(
        f"Optimizer: head={n_head:,} params @ lr={args.lr:g}"
        + (f" | encoder={n_enc:,} params @ lr={args.encoder_lr:g}" if n_enc else " | encoder fully frozen")
        + f" | weight_decay={args.weight_decay:g} (excl. norms/biases/log_sigma/null)"
    )

    # ── Scheduler: one per-step warmup→cosine schedule (scales every group equally) ──
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    # ── Training loop ──
    best_val_loss = float("inf")
    no_improve    = 0

    csv_fields = ["epoch", "train_total", "train_valence", "train_energy",
                  "train_danceability", "train_tempo", "train_decade",
                  "val_total", "val_valence", "val_energy",
                  "val_danceability", "val_tempo", "val_decade",
                  "val_decade_acc", "lr",
                  "sigma_valence", "sigma_energy", "sigma_danceability",
                  "sigma_tempo", "sigma_decade"]

    log_path = os.path.join(args.ckpt_dir, LOG_FILE)
    start_epoch_offset = 0
    if args.resume and os.path.exists(log_path):
        # Preserve history: migrate existing rows to the current schema (old logs
        # may lack the sigma_* columns), then append. Epoch numbers continue from
        # where the log left off so the dashboard's epoch axis stays monotonic;
        # the optimizer / scheduler / loop are still fresh (internal epochs from 1).
        with open(log_path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        start_epoch_offset = max((int(float(r.get("epoch") or 0)) for r in old_rows), default=0)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            for r in old_rows:
                w.writerow({k: r.get(k, "") for k in csv_fields})
        log.info(f"Appending to {log_path}: kept {len(old_rows)} prior epoch(s); "
                 f"logging new epochs as {start_epoch_offset+1}..{start_epoch_offset+args.epochs}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        train_metrics = run_epoch(
            model, train_loader, device,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # ── Validate ──
        val_metrics  = run_epoch(model, val_loader, device)
        val_dec_acc  = decade_accuracy(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:02d}/{args.epochs}"
            f"  train={train_metrics['total']:.4f}"
            f"  val={val_metrics['total']:.4f}"
            f"  dec_acc={val_dec_acc:.2%}"
            f"  lr={current_lr:.2e}"
        )
        log.info(
            f"          val  valence={val_metrics['valence']:.4f}"
            f"  energy={val_metrics['energy']:.4f}"
            f"  dance={val_metrics['danceability']:.4f}"
            f"  tempo={val_metrics['tempo']:.4f}"
            f"  decade={val_metrics['decade']:.4f}"
        )

        # Sigma tracking (clamped → effective σ used in the loss)
        sigmas = model.log_sigma.clamp(min=MIN_LOG_SIGMA).exp().detach().cpu().tolist()
        log.info(f"          σ  val={sigmas[0]:.3f}  eng={sigmas[1]:.3f}"
                 f"  dance={sigmas[2]:.3f}  tempo={sigmas[3]:.3f}  decade={sigmas[4]:.3f}")

        # ── Logging ──
        row = {
            "epoch":             start_epoch_offset + epoch,
            "train_total":       round(train_metrics["total"], 5),
            "train_valence":     round(train_metrics["valence"], 5),
            "train_energy":      round(train_metrics["energy"], 5),
            "train_danceability":round(train_metrics["danceability"], 5),
            "train_tempo":       round(train_metrics["tempo"], 5),
            "train_decade":      round(train_metrics["decade"], 5),
            "val_total":         round(val_metrics["total"], 5),
            "val_valence":       round(val_metrics["valence"], 5),
            "val_energy":        round(val_metrics["energy"], 5),
            "val_danceability":  round(val_metrics["danceability"], 5),
            "val_tempo":         round(val_metrics["tempo"], 5),
            "val_decade":        round(val_metrics["decade"], 5),
            "val_decade_acc":    round(val_dec_acc, 4),
            "lr":                current_lr,
            "sigma_valence":      round(sigmas[0], 5),
            "sigma_energy":       round(sigmas[1], 5),
            "sigma_danceability": round(sigmas[2], 5),
            "sigma_tempo":        round(sigmas[3], 5),
            "sigma_decade":       round(sigmas[4], 5),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        # ── Checkpoint ──
        last_path = os.path.join(args.ckpt_dir, "last_model.pt")
        torch.save(model.state_dict(), last_path)

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            no_improve    = 0
            best_path = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            log.info(f"  ✓ New best val loss: {best_val_loss:.4f} → {best_path}")
        else:
            no_improve += 1
            log.info(f"  No improvement {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    log.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints: {args.ckpt_dir}/")
    log.info(f"Log: {log_path}")
    log.info("Next: python3 evaluate.py")


if __name__ == "__main__":
    main()
