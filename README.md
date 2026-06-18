---
title: WaveRead
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# WaveRead

**Multimodal music emotion and era prediction from lyrics + audio.**

WaveRead takes a song's lyrics and an optional audio clip and returns five predictions: emotional **valence**, **energy**, **danceability**, **tempo**, and the **decade** the song most likely comes from. Word-level attention highlights show exactly which lyrics drove the prediction.

🔗 **Live demo:** [fr05enzz-waveread.hf.space](https://fr05enzz-waveread.hf.space)

---

## How it works

WaveRead is a dual-branch model trained on 14,000 songs:

- **Text branch** — fine-tuned RoBERTa (`roberta-base`) encodes lyrics and extracts attention weights for word highlighting
- **Audio branch** — a 3-layer MLP processes 20 Librosa features (MFCCs, spectral centroid, zero-crossing rate, RMS energy, tempo)
- **Fusion** — both branches are combined via homoscedastic uncertainty weighting (Kendall et al., 2018), letting the model automatically down-weight the noisier modality at inference time

The decade classifier outputs a probability distribution across seven decades (1960s–2020s). Continuous outputs (valence, energy, danceability, tempo) are regression heads trained jointly with the decade classifier.

---

## Features

- Paste any lyrics or use one of three built-in samples (Folk Pop, 90s Hip-Hop, 2010s Pop)
- Record a 30-second audio clip directly in the browser or upload a file
- See per-word attention highlighting on the lyrics
- Mood quadrant visualization (valence × energy)
- Dark/light mode toggle

---

## Tech stack

| Component | Details |
|---|---|
| Text model | `roberta-base` (HuggingFace Transformers) |
| Audio features | Librosa |
| Training | PyTorch, homoscedastic uncertainty weighting |
| Backend | FastAPI + Python |
| Frontend | Vanilla JS, Chart.js |
| Deployment | HuggingFace Spaces (Docker) |

---

## Dataset

Lyrics scraped from Genius across seven decades (1960s–2020s), audio clips sourced from YouTube. 14,000 songs total (2,000 per decade) after filtering. Audio features extracted with Librosa at 22,050 Hz.

---

## Project structure

```
waveRead/
├── app.py              # FastAPI server
├── model.py            # Dual-branch model architecture
├── inference.py        # Inference pipeline
├── preprocess.py       # Audio feature extraction
├── index.html          # Frontend
├── checkpoints/        # Trained model weights
├── notebooks/          # Training notebooks
└── scripts/            # Data collection scripts
```
