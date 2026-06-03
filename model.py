#!/usr/bin/env python3
"""
Phase 3 — Model Architecture
Multimodal model that fuses RoBERTa lyrics embeddings with Librosa audio
features to simultaneously predict valence, energy, danceability, tempo,
and decade (1960s–2020s, 7 classes).

Architecture overview:
    Text branch  : RoBERTa-base CLS → Linear(768 → 256)
    Audio branch : Linear(20 → 128) → LayerNorm → ReLU → Dropout → Linear(128 → 256)
    Null audio   : learned nn.Parameter(256,) used when audio is missing
    Modality drop: p=0.20 — randomly zeros audio during training (forces
                   text branch to work solo, mirrors inference when no audio)
    Fusion       : concat(text_emb, audio_emb) → Linear(512 → 512) → GELU → Dropout
    Heads        :
        valence, energy, danceability → Sigmoid  (0–1 continuous)
        tempo_norm                    → linear   (z-scored at preprocess time)
        decade                        → Linear(512 → 7) (7-way classification)

    Loss         : Homoscedastic uncertainty weighting (Kendall et al. 2018)
                   total = Σ_i [ (1 / 2σ_i²) * L_i + log σ_i ]
                   where σ_i are learned per-task scalars.

Usage:
    from model import LyricalEmotionModel
    model = LyricalEmotionModel()

    # forward
    out  = model(input_ids, attention_mask, audio_features, audio_missing)
    loss = model.compute_loss(out, batch)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TEXT_DIM   = 768   # RoBERTa-base hidden size
PROJ_DIM   = 256   # shared projection dim for both branches
FUSION_DIM = 512   # concat(text_proj, audio_proj)
AUDIO_DIM  = 20    # Librosa feature vector size
NUM_DECADES = 7    # 1960s–2020s
AUDIO_DROP  = 0.20 # modality dropout probability

# Floor on the per-task uncertainty σ in the homoscedastic loss. Without it,
# easily-fit regression tasks (valence/energy/danceability) push σ→0, which
# blows up their precision (0.5/σ²) and starves the decade head of gradient.
MIN_SIGMA     = 0.3                  # σ clamp floor (precision capped at 0.5/σ² ≈ 5.6); raise toward 0.5 if decade still starves
MIN_LOG_SIGMA = math.log(MIN_SIGMA)  # = log(0.3) ≈ -1.204; a float, so clamp() stays device-safe (CPU/MPS/CUDA)


# ── AUDIO BRANCH ───────────────────────────────────────────────────────────────

class AudioBranch(nn.Module):
    """
    Encodes the 20-dim Librosa feature vector to a 256-dim embedding.

    Architecture:
        Linear(20 → 128) → LayerNorm(128) → ReLU → Dropout(0.3)
        → Linear(128 → 256) → ReLU

    LayerNorm (not BatchNorm) is used deliberately: ~6% of songs have no audio
    and are fed as zero-vectors, and audio is replaced with the null embedding
    *after* this branch. BatchNorm would fold those zero-rows into its batch /
    running statistics and would crash on a batch of size 1; LayerNorm
    normalizes each sample independently, so it is robust to both.
    """
    def __init__(self, in_dim: int = AUDIO_DIM, out_dim: int = PROJ_DIM, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 20) → (B, 256)"""
        return self.net(x)


# ── MAIN MODEL ─────────────────────────────────────────────────────────────────

class LyricalEmotionModel(nn.Module):
    """
    Multimodal model for lyric-based emotion and decade prediction.

    Inputs:
        input_ids       (B, 512)  — RoBERTa token ids
        attention_mask  (B, 512)  — RoBERTa attention mask
        audio_features  (B, 20)   — Librosa features (zeros when missing)
        audio_missing   (B,)      — bool tensor; True = no audio available

    Outputs dict:
        valence      (B,)  — sigmoid output, 0–1
        energy       (B,)  — sigmoid output, 0–1
        danceability (B,)  — sigmoid output, 0–1
        tempo        (B,)  — linear output, z-scored
        decade_logits (B, 7) — raw logits for CrossEntropy
    """

    def __init__(
        self,
        roberta_name: str = "roberta-base",
        freeze_layers: int = 6,   # freeze first N encoder layers
        fusion_dropout: float = 0.3,
        audio_modality_drop: float = AUDIO_DROP,
    ):
        super().__init__()

        # ── Text branch ──
        self.roberta = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        self._freeze_roberta(freeze_layers)
        self.text_proj = nn.Linear(TEXT_DIM, PROJ_DIM)  # 768 → 256

        # ── Audio branch ──
        self.audio_branch = AudioBranch(in_dim=AUDIO_DIM, out_dim=PROJ_DIM)

        # Learned null embedding — used when audio is absent
        # Initialized to zeros; becomes meaningful through training
        self.null_audio_embedding = nn.Parameter(torch.zeros(PROJ_DIM))

        self.audio_modality_drop = audio_modality_drop

        # ── Fusion ──
        self.fusion = nn.Sequential(
            nn.Linear(FUSION_DIM, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        # ── Output heads ──
        self.head_valence      = nn.Linear(FUSION_DIM, 1)
        self.head_energy       = nn.Linear(FUSION_DIM, 1)
        self.head_danceability = nn.Linear(FUSION_DIM, 1)
        self.head_tempo        = nn.Linear(FUSION_DIM, 1)
        self.head_decade       = nn.Linear(FUSION_DIM, NUM_DECADES)

        # ── Homoscedastic uncertainty weights (one per task) ──
        # log(σ) initialized to 0 → σ=1 → equal initial weighting
        # Order: valence, energy, danceability, tempo, decade
        self.log_sigma = nn.Parameter(torch.zeros(5))

    # ── HELPERS ────────────────────────────────────────────────────────────────

    def _freeze_roberta(self, n_layers: int):
        """Freeze the embedding layer + first n_layers encoder layers."""
        # Always freeze embeddings
        for p in self.roberta.embeddings.parameters():
            p.requires_grad = False
        # Freeze encoder layers 0 … n_layers-1
        for i in range(n_layers):
            for p in self.roberta.encoder.layer[i].parameters():
                p.requires_grad = False

    # ── FORWARD ────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, 512)
        attention_mask: torch.Tensor,  # (B, 512)
        audio_features: torch.Tensor,  # (B, 20)
        audio_missing: torch.Tensor,   # (B,)  bool
    ) -> dict:

        B = input_ids.size(0)

        # ── 1. Text embedding ──
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = roberta_out.last_hidden_state[:, 0, :]   # (B, 768) CLS token
        text_emb = self.text_proj(cls_emb)                 # (B, 256)

        # ── 2. Audio embedding ──
        audio_emb = self.audio_branch(audio_features)      # (B, 256)

        # Replace missing audio with learned null embedding
        null = self.null_audio_embedding.unsqueeze(0).expand(B, -1)  # (B, 256)
        audio_emb = torch.where(
            audio_missing.unsqueeze(1),   # (B, 1) broadcast
            null,
            audio_emb,
        )

        # Modality dropout during training: randomly treat audio as missing
        if self.training and self.audio_modality_drop > 0.0:
            drop_mask = torch.rand(B, device=audio_emb.device) < self.audio_modality_drop
            audio_emb = torch.where(
                drop_mask.unsqueeze(1),
                null,
                audio_emb,
            )

        # ── 3. Fusion ──
        fused = self.fusion(torch.cat([text_emb, audio_emb], dim=1))  # (B, 512)

        # ── 4. Heads ──
        valence      = torch.sigmoid(self.head_valence(fused)).squeeze(1)       # (B,)
        energy       = torch.sigmoid(self.head_energy(fused)).squeeze(1)        # (B,)
        danceability = torch.sigmoid(self.head_danceability(fused)).squeeze(1)  # (B,)
        tempo        = self.head_tempo(fused).squeeze(1)                        # (B,) linear
        decade_logits = self.head_decade(fused)                                 # (B, 7)

        return {
            "valence":       valence,
            "energy":        energy,
            "danceability":  danceability,
            "tempo":         tempo,
            "decade_logits": decade_logits,
        }

    # ── LOSS ───────────────────────────────────────────────────────────────────

    def compute_loss(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Homoscedastic uncertainty-weighted multi-task loss.

        For each task i:
            L_total_i = (1 / 2σ_i²) * L_i + log(σ_i)

        where σ_i = exp(log_sigma_i) for numerical stability, so:
            L_total_i = exp(-log_sigma_i*2) * L_i * 0.5 + log_sigma_i

        Regression tasks (valence, energy, danceability, tempo): MSE
        Classification task (decade): CrossEntropy

        Returns:
            total_loss: scalar tensor
            component_losses: dict of individual unweighted losses for logging
        """
        # ── Per-task unweighted losses ──
        l_valence      = F.mse_loss(outputs["valence"],      batch["valence"])
        l_energy       = F.mse_loss(outputs["energy"],       batch["energy"])
        l_danceability = F.mse_loss(outputs["danceability"], batch["danceability"])
        l_tempo        = F.mse_loss(outputs["tempo"],        batch["tempo_norm"])
        l_decade       = F.cross_entropy(outputs["decade_logits"], batch["decade"], label_smoothing=0.1)

        raw_losses = torch.stack([l_valence, l_energy, l_danceability, l_tempo, l_decade])

        # ── Homoscedastic weighting (with a σ floor) ──
        # log_sigma shape: (5,). Clamp σ ≥ MIN_SIGMA so no single task's
        # precision (0.5/σ²) can run away and starve the others of gradient.
        # clamp() passes gradients through where σ > floor and blocks them at
        # the floor, so a clamped task simply stops driving σ lower.
        # weighted = 0.5 * exp(-2*log_sigma_clamped) * L + log_sigma_clamped
        log_sigma_clamped = self.log_sigma.clamp(min=MIN_LOG_SIGMA)
        precision = torch.exp(-2.0 * log_sigma_clamped)
        weighted  = 0.5 * precision * raw_losses + log_sigma_clamped
        total_loss = weighted.sum()

        component_losses = {
            "valence":      l_valence.item(),
            "energy":       l_energy.item(),
            "danceability": l_danceability.item(),
            "tempo":        l_tempo.item(),
            "decade":       l_decade.item(),
            # Log the *clamped* (effective) σ used in the loss, so the dashboard
            # reflects what is actually weighting each task.
            "sigma_valence":      log_sigma_clamped[0].exp().item(),
            "sigma_energy":       log_sigma_clamped[1].exp().item(),
            "sigma_danceability": log_sigma_clamped[2].exp().item(),
            "sigma_tempo":        log_sigma_clamped[3].exp().item(),
            "sigma_decade":       log_sigma_clamped[4].exp().item(),
        }

        return total_loss, component_losses


# ── FACTORY ────────────────────────────────────────────────────────────────────

def build_model(device: str = "cpu") -> LyricalEmotionModel:
    """Convenience builder — loads RoBERTa weights and moves model to device."""
    model = LyricalEmotionModel()
    model = model.to(device)
    return model


# ── QUICK SANITY CHECK ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Device: {device}")

    model = build_model(device)

    # Count trainable params
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.1f}%)")

    # Dummy batch
    B = 4
    dummy_batch = {
        "input_ids":      torch.randint(0, 50265, (B, 512)).to(device),
        "attention_mask": torch.ones(B, 512, dtype=torch.long).to(device),
        "audio_features": torch.randn(B, 20).to(device),
        "audio_missing":  torch.tensor([False, True, False, False]).to(device),
        "valence":        torch.rand(B).to(device),
        "energy":         torch.rand(B).to(device),
        "danceability":   torch.rand(B).to(device),
        "tempo_norm":     torch.randn(B).to(device),
        "decade":         torch.randint(0, 7, (B,)).to(device),
    }

    model.train()
    out = model(
        dummy_batch["input_ids"],
        dummy_batch["attention_mask"],
        dummy_batch["audio_features"],
        dummy_batch["audio_missing"],
    )
    loss, components = model.compute_loss(out, dummy_batch)

    print(f"\nOutput shapes:")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")

    print(f"\nTotal loss: {loss.item():.4f}")
    print("Component losses:")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ model.py sanity check passed. Next: python3 train.py")
