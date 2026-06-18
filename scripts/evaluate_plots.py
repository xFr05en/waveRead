#!/usr/bin/env python3
"""
Phase 5b — Post-hoc evaluation plots.

Run ONCE after training completes. Loads checkpoints/best_model.pt, runs the
held-out test set, and saves every analysis figure as a 300-dpi PNG in plots/.

Figures produced:
    plots/confusion_matrix.png            — decade classification (counts + %)
    plots/valence_pred_vs_actual.png      — scatter + diagonal, Pearson r & MSE
    plots/valence_by_decade.png           — box plots per decade (emotional drift)
    plots/audio_ablation_accuracy.png     — accuracy: real audio vs null embedding
    plots/gradient_flow.png               — mean |grad| per param group (one step)
    plots/tsne_by_decade.png              — PCA→TSNE of CLS embeddings, by decade
    plots/tsne_by_valence_quartile.png    — same embedding, by valence quartile
    plots/attention_words_by_decade.png   — top-15 attended words per decade

Usage:
    python3 evaluate_plots.py
    python3 evaluate_plots.py --ckpt checkpoints/best_model.pt
    python3 evaluate_plots.py --unfreeze-layers 6   # match training (for grad-flow)
    python3 evaluate_plots.py --limit 128           # quick smoke test on a subset
"""

import argparse
import logging
import os
from collections import OrderedDict, defaultdict

import matplotlib
matplotlib.use("Agg")  # save-only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import RobertaTokenizer

from model import LyricalEmotionModel
from preprocess import make_dataloader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROCESSED_DIR = "processed"
PLOTS_DIR     = "plots"
DECADE_NAMES  = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
NUM_DECADES   = 7

# Function words / fillers filtered from the "top attended words" plot so the
# bars surface content words. Set STOPWORDS = set() to disable.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "am", "i", "you",
    "he", "she", "it", "we", "they", "me", "my", "your", "his", "her", "its",
    "our", "their", "this", "that", "these", "those", "as", "so", "no", "not",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "s", "t", "m", "re", "ve", "ll", "d", "up", "out", "all", "just", "got",
    "get", "go", "got", "now", "im", "dont", "yeah", "oh", "la", "na",
}

sns.set_theme(style="whitegrid", font_scale=0.95)


# ── SETUP HELPERS ────────────────────────────────────────────────────────────

def device_auto() -> str:
    return ("mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()         else "cpu")


def load_model(ckpt_path: str, device: str, freeze_layers: int) -> LyricalEmotionModel:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first (python3 train.py).")
    model = LyricalEmotionModel(freeze_layers=freeze_layers)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    log.info(f"Loaded checkpoint: {ckpt_path}  (freeze_layers={freeze_layers})")
    return model


def move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def savefig(fig, name: str) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    abspath = os.path.abspath(path)
    print(f"  saved → {abspath}")
    return abspath


# ── DATA COLLECTION (one pass over the test set) ───────────────────────────────

def collect(model, loader, device) -> dict:
    """
    Single eval pass collecting everything the plots need:
      cls            (N, 768)  final-layer CLS embedding
      attn           (N, seq)  final-layer CLS→token attention (mean over heads)
      ids            (N, seq)  token ids
      val_pred/val_gt, dec_pred/dec_gt, audio_missing

    The head path below mirrors model.forward() in eval mode (dropout and
    modality-dropout are inactive), so predictions match a normal forward —
    we just reuse the single RoBERTa pass that also yields attentions.
    """
    model.eval()
    cls_all, attn_all, ids_all = [], [], []
    vp, vg, dp, dg, am = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            b = move_batch(batch, device)
            r = model.roberta(input_ids=b["input_ids"],
                              attention_mask=b["attention_mask"],
                              output_attentions=True)
            cls = r.last_hidden_state[:, 0, :]                 # (B, 768)
            cls_attn = r.attentions[-1][:, :, 0, :].mean(1)    # (B, seq)

            text_emb  = model.text_proj(cls)
            audio_emb = model.audio_branch(b["audio_features"])
            null = model.null_audio_embedding.unsqueeze(0).expand(cls.size(0), -1)
            audio_emb = torch.where(b["audio_missing"].unsqueeze(1), null, audio_emb)
            fused = model.fusion(torch.cat([text_emb, audio_emb], dim=1))
            valence = torch.sigmoid(model.head_valence(fused)).squeeze(1)
            dec_pred = model.head_decade(fused).argmax(dim=-1)

            cls_all.append(cls.cpu()); attn_all.append(cls_attn.cpu())
            ids_all.append(b["input_ids"].cpu())
            vp.append(valence.cpu());  vg.append(b["valence"].cpu())
            dp.append(dec_pred.cpu());  dg.append(b["decade"].cpu())
            am.append(b["audio_missing"].cpu())

    return {
        "cls":           torch.cat(cls_all).float().numpy(),
        "attn":          torch.cat(attn_all).float().numpy(),
        "ids":           torch.cat(ids_all).long().numpy(),
        "val_pred":      torch.cat(vp).float().numpy(),
        "val_gt":        torch.cat(vg).float().numpy(),
        "dec_pred":      torch.cat(dp).long().numpy(),
        "dec_gt":        torch.cat(dg).long().numpy(),
        "audio_missing": torch.cat(am).bool().numpy(),
    }


# ── PLOTS ──────────────────────────────────────────────────────────────────────

def plot_confusion(dec_pred, dec_gt) -> str:
    cm = np.zeros((NUM_DECADES, NUM_DECADES), dtype=int)
    for g, p in zip(dec_gt, dec_pred):
        cm[g, p] += 1
    row_tot = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, np.maximum(row_tot, 1)) * 100.0
    annot = np.array([[f"{cm[i, j]}\n{pct[i, j]:.0f}%"
                       for j in range(NUM_DECADES)] for i in range(NUM_DECADES)])
    acc = np.trace(cm) / max(cm.sum(), 1)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar_kws={"label": "count"},
                xticklabels=DECADE_NAMES, yticklabels=DECADE_NAMES,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted decade"); ax.set_ylabel("Actual decade")
    ax.set_title(f"Decade Confusion Matrix  ·  test accuracy {acc:.1%}  (n={cm.sum()})")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return savefig(fig, "confusion_matrix.png")


def plot_valence_scatter(val_pred, val_gt) -> str:
    r = np.corrcoef(val_gt, val_pred)[0, 1] if val_gt.std() > 1e-8 else float("nan")
    mse = float(np.mean((val_pred - val_gt) ** 2))

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(val_gt, val_pred, s=12, alpha=0.35, color="#1f77b4", edgecolor="none")
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], ls="--", color="#d62728", lw=1.6, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Actual valence"); ax.set_ylabel("Predicted valence")
    ax.set_title(f"Predicted vs Actual Valence  ·  Pearson r = {r:.3f}  ·  MSE = {mse:.4f}")
    ax.legend(loc="upper left")
    return savefig(fig, "valence_pred_vs_actual.png")


def plot_valence_by_decade(val_gt, dec_gt) -> str:
    # Box per decade, ordered 1960s (top) → 2020s (bottom).
    data = [val_gt[dec_gt == d] for d in range(NUM_DECADES)]
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("viridis", NUM_DECADES)
    positions = list(range(NUM_DECADES, 0, -1))  # so decade 0 sits at top
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=5))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_yticks(positions)
    ax.set_yticklabels(DECADE_NAMES)
    ax.set_xlabel("Valence (0 = sad, 1 = happy)")
    ax.set_ylabel("Decade")
    ax.set_title("Valence Distribution by Decade  (temporal emotional drift)")
    ax.set_xlim(0, 1)
    return savefig(fig, "valence_by_decade.png")


def plot_audio_ablation(dec_pred, dec_gt, audio_missing) -> str:
    present = ~audio_missing
    groups, accs, ns = [], [], []
    for label, mask in [("Real audio", present), ("Missing audio\n(null embedding)", audio_missing)]:
        n = int(mask.sum())
        acc = float((dec_pred[mask] == dec_gt[mask]).mean()) if n else 0.0
        groups.append(label); accs.append(acc * 100); ns.append(n)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    bars = ax.bar(groups, accs, color=["#2ca02c", "#d62728"], width=0.55, alpha=0.9)
    ax.axhline(100 / NUM_DECADES, ls="--", color="gray",
               label=f"chance ({100/NUM_DECADES:.1f}%)")
    for bar, acc, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.0,
                f"{acc:.1f}%\n(n={n})", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Decade classification accuracy (%)")
    ax.set_ylim(0, max(accs) * 1.25 + 5)
    ax.set_title("Decade Accuracy: songs WITH vs WITHOUT audio features")
    ax.legend(loc="upper right")
    return savefig(fig, "audio_ablation_accuracy.png")


def plot_gradient_flow(model, batch, device) -> str:
    """Mean |grad| per parameter group after one training step."""
    model.train()
    model.zero_grad(set_to_none=True)
    b = move_batch(batch, device)
    out = model(b["input_ids"], b["attention_mask"], b["audio_features"], b["audio_missing"])
    loss, _ = model.compute_loss(out, b)
    loss.backward()

    def group_of(name: str) -> str:
        if name.startswith("roberta.embeddings"):
            return "roberta.embeddings"
        if name.startswith("roberta.encoder.layer."):
            return f"roberta.L{int(name.split('.')[3]):02d}"
        return name.split(".")[0]

    grads = defaultdict(list)
    frozen = defaultdict(lambda: True)
    for name, p in model.named_parameters():
        g = group_of(name)
        if p.requires_grad:
            frozen[g] = False
        grads[g].append(0.0 if p.grad is None else p.grad.abs().mean().item())

    model.zero_grad(set_to_none=True)
    model.eval()

    # Order: embeddings, L00..L11, then the rest in a sensible sequence.
    ordered = ["roberta.embeddings"] + [f"roberta.L{i:02d}" for i in range(12)]
    tail = ["text_proj", "audio_branch", "fusion",
            "head_valence", "head_energy", "head_danceability",
            "head_tempo", "head_decade", "null_audio_embedding", "log_sigma"]
    ordered += [g for g in tail if g in grads]
    ordered += [g for g in grads if g not in ordered]
    ordered = [g for g in ordered if g in grads]

    means = np.array([float(np.mean(grads[g])) for g in ordered])
    is_frozen = np.array([frozen[g] for g in ordered])
    nonzero = means[means > 0]
    floor = (nonzero.min() / 10.0) if len(nonzero) else 1e-12
    disp = np.where(means > 0, means, floor)

    colors = []
    for g, fr in zip(ordered, is_frozen):
        if fr:
            colors.append("#bbbbbb")               # frozen
        elif g.startswith("roberta"):
            colors.append("#4c72b0")               # trainable RoBERTa
        else:
            colors.append("#dd8452")               # heads / branches

    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(ordered))[::-1]              # first group at top
    ax.barh(y, disp, color=colors, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(ordered, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(left=floor / 2)
    ax.set_xlabel("mean |gradient|  (log scale)")
    ax.set_title("Gradient Flow — one training step\n"
                 "grey = frozen (≈0) · blue = trainable RoBERTa · orange = heads/branches")
    for yi, fr in zip(y, is_frozen):
        if fr:
            ax.text(floor, yi, " frozen", va="center", ha="left", fontsize=7, color="#666666")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#bbbbbb", label="frozen"),
                       Patch(color="#4c72b0", label="RoBERTa (trainable)"),
                       Patch(color="#dd8452", label="heads / branches")],
              loc="lower right")
    return savefig(fig, "gradient_flow.png")


def _tsne_embed(cls: np.ndarray, seed: int = 42) -> np.ndarray:
    n = cls.shape[0]
    n_pca = min(50, n, cls.shape[1])
    reduced = PCA(n_components=n_pca, random_state=seed).fit_transform(cls)
    perplexity = max(5, min(30, (n - 1) // 3))
    log.info(f"  TSNE: {n} points, PCA→{n_pca}, perplexity={perplexity}")
    return TSNE(n_components=2, random_state=seed, init="pca",
                perplexity=perplexity).fit_transform(reduced)


def plot_tsne_by_decade(emb2, dec_gt) -> str:
    fig, ax = plt.subplots(figsize=(9, 8))
    palette = sns.color_palette("viridis", NUM_DECADES)
    for d in range(NUM_DECADES):
        m = dec_gt == d
        if m.any():
            ax.scatter(emb2[m, 0], emb2[m, 1], s=14, alpha=0.6,
                       color=palette[d], label=DECADE_NAMES[d], edgecolor="none")
    ax.set_xlabel("TSNE-1"); ax.set_ylabel("TSNE-2")
    ax.set_title("CLS Embeddings (PCA→TSNE) — colored by decade")
    ax.legend(title="decade", loc="best", framealpha=0.9)
    return savefig(fig, "tsne_by_decade.png")


def plot_tsne_by_valence(emb2, val_gt) -> str:
    qs = np.quantile(val_gt, [0.25, 0.5, 0.75])
    bins = np.digitize(val_gt, qs)  # 0..3
    labels = ["low", "mid-low", "mid-high", "high"]
    palette = sns.color_palette("rocket", 4)
    fig, ax = plt.subplots(figsize=(9, 8))
    for q in range(4):
        m = bins == q
        if m.any():
            ax.scatter(emb2[m, 0], emb2[m, 1], s=14, alpha=0.6,
                       color=palette[q], label=labels[q], edgecolor="none")
    ax.set_xlabel("TSNE-1"); ax.set_ylabel("TSNE-2")
    ax.set_title("CLS Embeddings (PCA→TSNE) — colored by valence quartile")
    ax.legend(title="valence", loc="best", framealpha=0.9)
    return savefig(fig, "tsne_by_valence_quartile.png")


def plot_attention_words(ids, attn, dec_gt, tokenizer, top_k=15, min_count=5) -> str:
    special = set(tokenizer.all_special_ids)
    pad_id = tokenizer.pad_token_id

    def clean(tok: str) -> str:
        return tok.replace("Ġ", "").replace("Ċ", "").strip().lower()

    per_decade = {d: defaultdict(lambda: [0.0, 0]) for d in range(NUM_DECADES)}  # word -> [sum, count]
    for i in range(ids.shape[0]):
        d = int(dec_gt[i])
        toks = tokenizer.convert_ids_to_tokens(ids[i].tolist())
        row = attn[i]
        for pos, (tid, tok) in enumerate(zip(ids[i], toks)):
            if tid in special or tid == pad_id:
                continue
            w = clean(tok)
            if len(w) < 2 or not w.isalpha() or w in STOPWORDS:
                continue
            acc = per_decade[d][w]
            acc[0] += float(row[pos]); acc[1] += 1

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()
    for d in range(NUM_DECADES):
        ax = axes[d]
        items = [(w, s / c) for w, (s, c) in per_decade[d].items() if c >= min_count]
        items.sort(key=lambda x: x[1])           # ascending → biggest at top of barh
        items = items[-top_k:]
        ax.set_title(DECADE_NAMES[d], fontweight="bold")
        if not items:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes, color="#999999")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        words = [w for w, _ in items]
        vals  = [v for _, v in items]
        ax.barh(range(len(words)), vals, color=sns.color_palette("viridis", NUM_DECADES)[d],
                alpha=0.9)
        ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=8)
        ax.set_xlabel("mean CLS attention", fontsize=8)
    axes[7].axis("off")  # 8th cell unused (7 decades)
    fig.suptitle("Top attended words per decade (final-layer CLS attention, "
                 f"stopwords removed, ≥{min_count} occurrences)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return savefig(fig, "attention_words_by_decade.png")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate post-hoc evaluation plots.")
    parser.add_argument("--ckpt", default="checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--unfreeze-layers", type=int, default=6,
                        help="Match training, so the gradient-flow plot is faithful")
    parser.add_argument("--limit", type=int, default=0,
                        help="Use only the first N test samples (0 = all; for quick tests)")
    args = parser.parse_args()

    device = device_auto()
    log.info(f"Device: {device}")

    test_data = torch.load(os.path.join(PROCESSED_DIR, "test.pt"),
                           map_location="cpu", weights_only=False)
    if args.limit and args.limit < len(test_data):
        test_data = test_data[:args.limit]
    log.info(f"Test samples: {len(test_data)}")
    loader = make_dataloader(test_data, batch_size=args.batch_size, shuffle=False)

    freeze_layers = max(0, min(12, 12 - args.unfreeze_layers))
    model = load_model(args.ckpt, device, freeze_layers)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    log.info("Collecting predictions, CLS embeddings, and attention …")
    data = collect(model, loader, device)
    acc = float((data["dec_pred"] == data["dec_gt"]).mean())
    log.info(f"Test decade accuracy: {acc:.2%}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\nSaving plots to {os.path.abspath(PLOTS_DIR)}/\n")

    # Each plot is isolated so one failure doesn't abort the rest.
    jobs = [
        ("confusion matrix",   lambda: plot_confusion(data["dec_pred"], data["dec_gt"])),
        ("valence scatter",    lambda: plot_valence_scatter(data["val_pred"], data["val_gt"])),
        ("valence by decade",  lambda: plot_valence_by_decade(data["val_gt"], data["dec_gt"])),
        ("audio ablation",     lambda: plot_audio_ablation(data["dec_pred"], data["dec_gt"],
                                                           data["audio_missing"])),
        ("gradient flow",      lambda: plot_gradient_flow(model, next(iter(loader)), device)),
        ("attention words",    lambda: plot_attention_words(data["ids"], data["attn"],
                                                           data["dec_gt"], tokenizer)),
    ]

    saved = []
    for label, fn in jobs:
        try:
            log.info(f"Plot: {label}")
            saved.append(fn())
        except Exception as e:
            log.error(f"  FAILED ({label}): {e}")

    # TSNE shares one embedding for both colorings.
    try:
        log.info("Plot: TSNE (PCA→TSNE of CLS embeddings)")
        emb2 = _tsne_embed(data["cls"])
        saved.append(plot_tsne_by_decade(emb2, data["dec_gt"]))
        saved.append(plot_tsne_by_valence(emb2, data["val_gt"]))
    except Exception as e:
        log.error(f"  FAILED (tsne): {e}")

    print(f"\nDone — {len(saved)} plot(s) written to {os.path.abspath(PLOTS_DIR)}/")


if __name__ == "__main__":
    main()
