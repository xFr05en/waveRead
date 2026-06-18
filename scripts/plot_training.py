#!/usr/bin/env python3
"""
Phase 4b — Live training dashboard.

Run this in a SECOND terminal while train.py is training:

    python3 plot_training.py

It reads checkpoints/training_log.csv every 30 seconds and redraws a 2×3 grid
of live plots. Press Ctrl+C (or close the window) to exit.

    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    │ 1. train vs val      │ 2. per-task val      │ 3. decade accuracy   │
    │    total loss        │    losses (log y)    │    (val)             │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ 4. per-task σ        │ 5. learning rate     │ 6. status / sanity   │
    │    (uncertainty wts) │                      │    panel             │
    └─────────────────────┴─────────────────────┴─────────────────────┘

The σ panel is the important diagnostic: if all five σ lines stay pinned near
1.0, the homoscedastic uncertainty weighting is NOT adapting (all tasks weighted
equally) — you want to see them spread apart as training progresses.

Usage:
    python3 plot_training.py                       # live, refresh every 30s
    python3 plot_training.py --interval 15         # refresh every 15s
    python3 plot_training.py --log path/to.csv     # custom log path
    python3 plot_training.py --once                # render one frame to PNG and exit
"""

import argparse
import os
import time
from datetime import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ── Per-task identity (order matches train.py's log_sigma) ──────────────────────
TASKS  = ["valence", "energy", "danceability", "tempo", "decade"]
COLORS = {
    "valence":      "#1f77b4",
    "energy":       "#ff7f0e",
    "danceability": "#2ca02c",
    "tempo":        "#d62728",
    "decade":       "#9467bd",
}
NUM_DECADES   = 7
CHANCE_ACC    = 100.0 / NUM_DECADES   # 7-class random-guess accuracy (%)
DEFAULT_LOG   = os.path.join("checkpoints", "training_log.csv")


def apply_style():
    """Clean, readable matplotlib defaults (no reliance on version-specific style names)."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#f7f7f7",
        "axes.grid":         True,
        "grid.color":        "white",
        "grid.linewidth":    1.1,
        "axes.edgecolor":    "#cccccc",
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "lines.linewidth":   2.0,
    })


def read_log(path: str):
    """Read the CSV defensively; return a DataFrame with ≥1 row, else None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        # File may be mid-write; skip this refresh.
        return None
    if df is None or len(df) == 0 or "epoch" not in df.columns:
        return None
    return df


def _has_sigmas(df) -> bool:
    return all(f"sigma_{t}" in df.columns for t in TASKS)


def draw(fig, axes, df, log_path):
    for ax in axes.flat:
        ax.clear()

    ep = df["epoch"]
    latest = df.iloc[-1]

    # ── (1) Train vs val total loss ──
    ax = axes[0, 0]
    if "train_total" in df:
        ax.plot(ep, df["train_total"], label="train", color="#1f77b4")
    if "val_total" in df:
        ax.plot(ep, df["val_total"], label="val", color="#d62728")
        best_i = df["val_total"].idxmin()
        ax.scatter([df["epoch"][best_i]], [df["val_total"][best_i]],
                   color="#d62728", zorder=5, s=45, edgecolor="white",
                   label=f"best val ({df['val_total'][best_i]:.3f})")
    ax.set_title("1 · Total loss (train vs val)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(loc="upper right")

    # ── (2) Per-task validation losses (log y — scales differ by ~100×) ──
    ax = axes[0, 1]
    plotted = False
    for t in TASKS:
        col = f"val_{t}"
        if col in df:
            ax.plot(ep, df[col], label=t, color=COLORS[t])
            plotted = True
    ax.set_title("2 · Per-task val loss  (valence/energy/dance/tempo MSE, decade CE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    if plotted:
        ax.set_yscale("log")
        ax.legend(loc="upper right", ncol=2)

    # ── (3) Decade classification accuracy (val) ──
    ax = axes[0, 2]
    if "val_decade_acc" in df:
        ax.plot(ep, df["val_decade_acc"] * 100.0, color="#9467bd", marker="o", ms=3)
        ax.axhline(CHANCE_ACC, ls="--", lw=1.4, color="gray",
                   label=f"chance ({CHANCE_ACC:.1f}%)")
        ax.set_ylim(0, max(100.0, df["val_decade_acc"].max() * 100 + 10))
        ax.legend(loc="lower right")
    ax.set_title("3 · Decade accuracy (val)")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")

    # ── (4) Per-task sigma (uncertainty weights) ──
    ax = axes[1, 0]
    if _has_sigmas(df):
        for t in TASKS:
            ax.plot(ep, df[f"sigma_{t}"], label=t, color=COLORS[t])
        ax.axhline(1.0, ls=":", lw=1.4, color="gray", label="σ=1 (init / inactive)")
        ax.set_title("4 · Per-task σ  (uncertainty weighting)")
        ax.set_xlabel("epoch"); ax.set_ylabel("σ = exp(log σ)")
        ax.legend(loc="best", ncol=2)
    else:
        ax.set_title("4 · Per-task σ")
        ax.text(0.5, 0.5,
                "No sigma_* columns in log.\n"
                "Re-run training with the updated\n"
                "train.py to record σ values.",
                ha="center", va="center", fontsize=9, color="#888888",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    # ── (5) Learning rate ──
    ax = axes[1, 1]
    if "lr" in df:
        ax.plot(ep, df["lr"], color="#17becf")
        ax.set_yscale("log")
    ax.set_title("5 · Learning rate (head group)")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr (log)")

    # ── (6) Status / sanity panel ──
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("6 · Status")
    lines = []
    lines.append(f"epoch:            {int(latest['epoch'])}")
    if "train_total" in df and "val_total" in df:
        lines.append(f"train / val loss: {latest['train_total']:.4f} / {latest['val_total']:.4f}")
        best_i = df["val_total"].idxmin()
        lines.append(f"best val loss:    {df['val_total'][best_i]:.4f}  (epoch {int(df['epoch'][best_i])})")
    if "val_decade_acc" in df:
        lines.append(f"val decade acc:   {latest['val_decade_acc']*100:.2f}%   (chance {CHANCE_ACC:.1f}%)")
    if "lr" in df:
        lines.append(f"learning rate:    {latest['lr']:.2e}")

    verdict = ""
    if _has_sigmas(df):
        sig = {t: latest[f"sigma_{t}"] for t in TASKS}
        lines.append("")
        lines.append("σ (uncertainty weights):")
        for t in TASKS:
            lines.append(f"   {t:<13} {sig[t]:.3f}")
        spread = max(sig.values()) - min(sig.values())
        if int(latest["epoch"]) >= 3 and spread < 0.05:
            verdict = "⚠  σ ≈ all equal — uncertainty weighting NOT adapting"
        else:
            verdict = "✓  σ spreading — uncertainty weighting active"

    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=9.5, transform=ax.transAxes)
    if verdict:
        ax.text(0.0, 0.04, verdict, ha="left", va="bottom",
                fontsize=9, fontweight="bold",
                color=("#c0392b" if verdict.startswith("⚠") else "#27ae60"),
                transform=ax.transAxes)

    stamp = datetime.now().strftime("%H:%M:%S")
    fig.suptitle(f"Live training — {os.path.basename(log_path)}  ·  "
                 f"{len(df)} epochs logged  ·  updated {stamp}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def draw_waiting(fig, axes, log_path):
    for ax in axes.flat:
        ax.clear(); ax.axis("off")
    axes[0, 0].text(
        1.6, 0.5,
        f"Waiting for data in:\n{log_path}\n\n"
        "Start training in another terminal:\n    python3 train.py",
        ha="center", va="center", fontsize=12, color="#888888",
        transform=axes[0, 0].transAxes,
    )
    fig.suptitle("Live training — waiting for training_log.csv …",
                 fontsize=13, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Live training dashboard.")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to training_log.csv")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Refresh interval in seconds (default 30)")
    parser.add_argument("--once", action="store_true",
                        help="Render a single frame to a PNG and exit (no live loop)")
    parser.add_argument("--snapshot", default="training_dashboard.png",
                        help="Output PNG path when --once is used")
    args = parser.parse_args()

    if args.once:
        plt.switch_backend("Agg")   # headless: just render and save

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # ── One-shot snapshot mode (handy for testing / saving a final frame) ──
    if args.once:
        df = read_log(args.log)
        if df is None:
            draw_waiting(fig, axes, args.log)
        else:
            draw(fig, axes, df, args.log)
        fig.savefig(args.snapshot, dpi=150, bbox_inches="tight")
        print(f"Saved snapshot → {os.path.abspath(args.snapshot)}")
        return

    # ── Live auto-refresh loop ──
    plt.ion()
    plt.show(block=False)
    print(f"Live dashboard watching {args.log}  (refresh every {args.interval:g}s). Ctrl+C to exit.")
    try:
        while True:
            df = read_log(args.log)
            if df is None:
                draw_waiting(fig, axes, args.log)
            else:
                draw(fig, axes, df, args.log)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # Responsive sleep: poll in short slices so the window stays
            # interactive and we notice if it gets closed.
            waited = 0.0
            while waited < args.interval:
                if not plt.fignum_exists(fig.number):
                    print("\nWindow closed — exiting.")
                    return
                plt.pause(0.5)
                waited += 0.5
    except KeyboardInterrupt:
        print("\nExiting live dashboard.")


if __name__ == "__main__":
    main()
