#!/usr/bin/env python3
"""
Phase 6a — Inference
Clean prediction pipeline that takes raw lyrics + an optional audio file
and returns all 5 model outputs in a human-readable dict.

The FastAPI app (app.py) wraps this module. You can also run it directly
from the terminal for quick spot-checks.

Usage (CLI):
    python3 inference.py --lyrics "I used to love her but I had to kill her"
    python3 inference.py --lyrics "..." --audio audio_clips/song.wav
    python3 inference.py --lyrics-file my_song.txt --audio song.wav

Output dict keys:
    valence        float  0–1       (higher = more positive / happy)
    energy         float  0–1       (higher = more energetic)
    danceability   float  0–1       (higher = more danceable)
    tempo_bpm      float  BPM       (back-converted from z-score using scaler)
    decade         str              "1960s" … "2020s"
    decade_probs   dict             {"1960s": 0.xx, …, "2020s": 0.xx}
    audio_used     bool             True if audio was successfully extracted
    word_scores    list[dict]       [{word, score}, ...] for attention highlight
                                   (approximated via gradient × embedding norm)
"""

import os
import re
import pickle
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer

from model import LyricalEmotionModel
from preprocess import extract_librosa_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
CKPT_PATH    = "checkpoints/best_model.pt"
SCALER_PATH  = "processed/scaler.pkl"
MAX_LENGTH   = 512
AUDIO_DIM    = 20
DECADE_NAMES = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]

# Minimum word count before we warn the user
MIN_WORD_COUNT = 10


# ── PREDICTOR CLASS ────────────────────────────────────────────────────────────

class LyricsPredictor:
    """
    Stateful predictor — loads model + tokenizer + scalers once,
    then serves predictions with zero reload overhead.

    Instantiate once at app startup:
        predictor = LyricsPredictor()
    Then call:
        result = predictor.predict(lyrics="...", audio_path=None)
    """

    def __init__(
        self,
        ckpt_path:   str = CKPT_PATH,
        scaler_path: str = SCALER_PATH,
    ):
        self.device = (
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else
            "cpu"
        )
        log.info(f"Inference device: {self.device}")

        # ── Tokenizer ──
        log.info("Loading RoBERTa tokenizer ...")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # ── Model ──
        log.info(f"Loading model from {ckpt_path} ...")
        self.model = LyricalEmotionModel(freeze_layers=12)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # ── Scalers ──
        log.info(f"Loading scalers from {scaler_path} ...")
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        self.audio_scaler = scalers["audio"]
        self.tempo_scaler = scalers["tempo"]

        log.info("Predictor ready.")

    # ── TOKENIZE ───────────────────────────────────────────────────────────────

    def _tokenize(self, lyrics: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Tokenize lyrics. Returns input_ids, attention_mask, and word list.
        Also returns the decoded token strings for word_scores alignment.
        """
        words = lyrics.split()
        if len(words) < MIN_WORD_COUNT:
            log.warning(
                f"Only {len(words)} words — predictions may be unreliable. "
                f"Aim for at least {MIN_WORD_COUNT} words."
            )

        enc = self.tokenizer(
            lyrics,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)       # (1, 512)
        attention_mask = enc["attention_mask"].to(self.device)  # (1, 512)
        return input_ids, attention_mask, words

    # ── AUDIO ──────────────────────────────────────────────────────────────────

    def _extract_audio(self, audio_path: str | None) -> tuple[np.ndarray, bool]:
        """
        Returns (feature_vector, audio_missing_flag).
        Normalizes features with the fitted audio scaler.
        """
        if audio_path is None or not os.path.exists(audio_path):
            if audio_path is not None:
                log.warning(f"Audio file not found: {audio_path}")
            return np.zeros(AUDIO_DIM, dtype=np.float32), True

        feats = extract_librosa_features(audio_path)
        if feats is None:
            log.warning(f"Feature extraction failed for: {audio_path}")
            return np.zeros(AUDIO_DIM, dtype=np.float32), True

        feats_norm = self.audio_scaler.transform(feats.reshape(1, -1)).flatten()
        return feats_norm.astype(np.float32), False

    # ── WORD SCORES ────────────────────────────────────────────────────────────

    def _word_scores(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        audio_missing:  torch.Tensor,
        words:          list[str],
    ) -> list[dict]:
        """
        Approximate word-level importance using gradient × embedding norm.

        All RoBERTa layers are frozen (requires_grad=False), so we can't use
        retain_grad() on the embedding output directly. Instead we:
          1. Manually fetch word embeddings and detach + re-enable grad
          2. Pass inputs_embeds through RoBERTa (bypassing frozen word_embeddings)
          3. Reconstruct the forward pass through projection → fusion → heads
          4. Backprop to inputs_embeds → gradient magnitude ≈ token importance
          5. Group subword tokens back to whitespace-split words
        """
        self.model.eval()

        try:
            with torch.enable_grad():
                # Get word embeddings with grad enabled (detach from frozen param graph)
                inputs_embeds = (
                    self.model.roberta.embeddings.word_embeddings(input_ids)
                    .detach()
                    .requires_grad_(True)
                )  # (1, 512, 768)

                # Full RoBERTa forward using inputs_embeds
                roberta_out = self.model.roberta(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                cls_emb  = roberta_out.last_hidden_state[:, 0, :]   # (1, 768)
                text_emb = self.model.text_proj(cls_emb)             # (1, 256)

                # Audio branch (detached — we only want text grad)
                with torch.no_grad():
                    audio_emb_raw = self.model.audio_branch(audio_features)
                    null = self.model.null_audio_embedding.unsqueeze(0).expand(
                        audio_features.size(0), -1
                    )
                    audio_emb = torch.where(
                        audio_missing.unsqueeze(1), null, audio_emb_raw
                    ).detach()

                fused         = self.model.fusion(torch.cat([text_emb, audio_emb], dim=1))
                decade_logits = self.model.head_decade(fused)
                valence       = torch.sigmoid(self.model.head_valence(fused))

                score = decade_logits.abs().sum() + valence.abs().sum()
                score.backward()

            if inputs_embeds.grad is not None:
                # Gradient × embedding norm per token position
                grad_norm = (inputs_embeds.grad * inputs_embeds.detach()).norm(dim=-1).squeeze(0)  # (512,)
                mask      = attention_mask.squeeze(0).float()
                grad_norm = grad_norm * mask

                # Align tokens to words
                token_ids  = input_ids.squeeze(0).tolist()
                token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
                grad_vals  = grad_norm.detach().cpu().numpy()

                # Group subword tokens back to words
                word_grads = []
                current_word_grads = []
                word_i = 0
                for tok, g in zip(token_strs, grad_vals):
                    if tok in ["<s>", "</s>", "<pad>"]:
                        continue
                    # RoBERTa uses Ġ prefix for word-start tokens
                    if tok.startswith("Ġ") and current_word_grads:
                        word_grads.append(float(np.mean(current_word_grads)))
                        word_i += 1
                        current_word_grads = [g]
                    else:
                        current_word_grads.append(g)
                if current_word_grads:
                    word_grads.append(float(np.mean(current_word_grads)))

                # Normalize to 0–1
                arr = np.array(word_grads[:len(words)])
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                else:
                    arr = np.zeros_like(arr)

                return [{"word": w, "score": round(float(s), 4)}
                        for w, s in zip(words, arr)]

        except Exception as e:
            log.warning(f"Word score computation failed: {e}")

        # Fallback: uniform scores
        return [{"word": w, "score": 0.0} for w in words]

    # ── PREDICT ────────────────────────────────────────────────────────────────

    def predict(
        self,
        lyrics:     str,
        audio_path: str | None = None,
    ) -> dict:
        """
        Main prediction entry point.

        Args:
            lyrics:     Raw song lyrics (any length; truncated to 512 tokens)
            audio_path: Path to a .wav/.mp3 audio file, or None for text-only

        Returns:
            dict with keys: valence, energy, danceability, tempo_bpm,
                            decade, decade_probs, audio_used, word_scores,
                            warnings (list of strings)
        """
        warnings = []
        lyrics = lyrics.strip()

        if not lyrics:
            raise ValueError("Lyrics cannot be empty.")

        # 1. Tokenize
        input_ids, attention_mask, words = self._tokenize(lyrics)
        if len(words) < MIN_WORD_COUNT:
            warnings.append(
                f"Short input ({len(words)} words) — acoustic feature predictions "
                "are especially uncertain. Add more lyrics for better results."
            )

        # 2. Audio features
        audio_np, audio_missing_flag = self._extract_audio(audio_path)
        audio_tensor   = torch.tensor(audio_np).unsqueeze(0).to(self.device)   # (1, 20)
        missing_tensor = torch.tensor([audio_missing_flag]).to(self.device)     # (1,)

        if audio_missing_flag and audio_path is not None:
            warnings.append("Audio file could not be processed — using text-only predictions.")
        if audio_missing_flag:
            warnings.append(
                "No audio provided. Tempo, energy, and danceability predictions "
                "are estimated from lyrics alone and may be less accurate."
            )

        # 3. Forward pass (no grad for speed)
        with torch.no_grad():
            out = self.model(input_ids, attention_mask, audio_tensor, missing_tensor)

        # 4. Decade probabilities
        decade_probs_tensor = F.softmax(out["decade_logits"], dim=-1).squeeze(0)
        decade_idx          = int(decade_probs_tensor.argmax().item())
        decade_probs        = {
            name: round(float(p), 4)
            for name, p in zip(DECADE_NAMES, decade_probs_tensor.tolist())
        }

        # 5. Back-convert tempo from z-score → BPM
        tempo_z   = float(out["tempo"].item())
        tempo_bpm = float(
            self.tempo_scaler.inverse_transform([[tempo_z]])[0][0]
        )
        tempo_bpm = round(max(40.0, min(220.0, tempo_bpm)), 1)  # clamp to sane BPM range

        # 6. Word importance scores (lightweight — uses gradient hook)
        word_scores = self._word_scores(
            input_ids, attention_mask, audio_tensor, missing_tensor, words
        )

        return {
            "valence":      round(float(out["valence"].item()), 4),
            "energy":       round(float(out["energy"].item()), 4),
            "danceability": round(float(out["danceability"].item()), 4),
            "tempo_bpm":    tempo_bpm,
            "decade":       DECADE_NAMES[decade_idx],
            "decade_probs": decade_probs,
            "audio_used":   not audio_missing_flag,
            "word_scores":  word_scores,
            "warnings":     warnings,
        }


# ── CLI ENTRY POINT ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lyrical Emotion Analyzer — inference")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lyrics",      type=str, help="Lyrics as a string")
    group.add_argument("--lyrics-file", type=str, help="Path to a .txt file with lyrics")
    parser.add_argument("--audio",      type=str, default=None, help="Optional audio file path")
    parser.add_argument("--ckpt",       default=CKPT_PATH)
    parser.add_argument("--scaler",     default=SCALER_PATH)
    args = parser.parse_args()

    if args.lyrics_file:
        with open(args.lyrics_file, "r") as f:
            lyrics = f.read()
    else:
        lyrics = args.lyrics

    predictor = LyricsPredictor(ckpt_path=args.ckpt, scaler_path=args.scaler)
    result    = predictor.predict(lyrics=lyrics, audio_path=args.audio)

    print("\n" + "=" * 50)
    print("  PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Decade       : {result['decade']}")
    print(f"  Decade probs : {result['decade_probs']}")
    print(f"  Valence      : {result['valence']:.4f}  (0=negative, 1=positive)")
    print(f"  Energy       : {result['energy']:.4f}  (0=calm, 1=energetic)")
    print(f"  Danceability : {result['danceability']:.4f}  (0=not danceable, 1=very)")
    print(f"  Tempo        : {result['tempo_bpm']} BPM")
    print(f"  Audio used   : {result['audio_used']}")
    if result["warnings"]:
        print("\n  Warnings:")
        for w in result["warnings"]:
            print(f"    ⚠  {w}")
    print("\n  Top-10 important words:")
    top = sorted(result["word_scores"], key=lambda x: x["score"], reverse=True)[:10]
    for item in top:
        bar = "█" * int(item["score"] * 20)
        print(f"    {item['word']:<20} {bar}  ({item['score']:.3f})")
    print("=" * 50)


if __name__ == "__main__":
    main()
