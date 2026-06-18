#!/usr/bin/env python3
"""
Phase 6b — FastAPI Backend
Wraps LyricsPredictor and exposes three endpoints:

    GET  /health                  — readiness check
    POST /predict                 — lyrics-only prediction (JSON body)
    POST /predict-with-audio      — lyrics + audio file upload (multipart)

CORS is open (*) so the static website can call it from any origin.

Local dev:
    pip install fastapi uvicorn python-multipart --break-system-packages
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

HuggingFace Spaces:
    The Space's CMD in Dockerfile runs:
        uvicorn app:app --host 0.0.0.0 --port 7860
"""

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from inference import LyricsPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── GLOBAL PREDICTOR (loaded once at startup) ──────────────────────────────────
predictor: LyricsPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the server starts, clean up on shutdown."""
    global predictor
    log.info("Loading LyricsPredictor at startup ...")
    predictor = LyricsPredictor()
    log.info("Model ready. Server accepting requests.")
    yield
    log.info("Shutting down.")


# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="waveRead API",
    description="Predicts emotional valence, energy, danceability, tempo, and decade from song lyrics and optional audio.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── SCHEMAS ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    lyrics: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "lyrics": "I used to roll the dice, feel the fear in my enemy's eyes"
            }]
        }
    }


class WordScore(BaseModel):
    word:  str
    score: float


class PredictResponse(BaseModel):
    valence:      float
    energy:       float
    danceability: float
    tempo_bpm:    float
    decade:       str
    decade_probs: dict[str, float]
    audio_used:   bool
    word_scores:  list[WordScore]
    warnings:     list[str]


# ── ENDPOINTS ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse("index.html")


@app.get("/health")
def health():
    """
    Readiness probe. Returns 200 when the model is loaded and ready.
    The website pings this on page load so it can show a 'Model ready' badge.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok", "model": "lyrical-emotion-analyzer-v1"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Lyrics-only prediction.

    Body (JSON):
        { "lyrics": "full song lyrics here..." }

    Returns all 5 predictions. Tempo/energy/danceability will include a
    warning that audio was not provided.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    lyrics = req.lyrics.strip()
    if not lyrics:
        raise HTTPException(status_code=422, detail="lyrics field cannot be empty.")

    try:
        result = predictor.predict(lyrics=lyrics, audio_path=None)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return PredictResponse(
        valence      = result["valence"],
        energy       = result["energy"],
        danceability = result["danceability"],
        tempo_bpm    = result["tempo_bpm"],
        decade       = result["decade"],
        decade_probs = result["decade_probs"],
        audio_used   = result["audio_used"],
        word_scores  = [WordScore(**ws) for ws in result["word_scores"]],
        warnings     = result["warnings"],
    )


@app.post("/predict-with-audio", response_model=PredictResponse)
async def predict_with_audio(
    lyrics: str      = Form(..., description="Full song lyrics"),
    audio:  UploadFile = File(..., description="Audio clip (.wav or .mp3, ideally 30s)"),
):
    """
    Multimodal prediction — lyrics + audio file.

    Body (multipart/form-data):
        lyrics: str   — song lyrics
        audio:  file  — .wav or .mp3 audio clip

    The audio file is saved to a temp directory, processed by Librosa,
    then immediately deleted.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    lyrics = lyrics.strip()
    if not lyrics:
        raise HTTPException(status_code=422, detail="lyrics field cannot be empty.")

    # Validate file extension — fall back to content-type if filename has no extension
    filename  = audio.filename or "upload"
    extension = os.path.splitext(filename)[-1].lower()
    if not extension and audio.content_type:
        ct_map = {
            "audio/webm": ".webm", "audio/ogg": ".ogg",
            "audio/mp4": ".mp4",   "audio/mpeg": ".mp3",
            "audio/wav": ".wav",   "audio/x-wav": ".wav",
        }
        ct = audio.content_type.split(";")[0].strip()
        extension = ct_map.get(ct, ".webm")
    if extension not in {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".mp4"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported audio format '{extension}'. Use .wav, .mp3, or .webm.",
        )

    # Save upload to temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False, dir="/tmp"
        ) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        log.info(f"Audio upload saved to {tmp_path} ({os.path.getsize(tmp_path):,} bytes)")

        result = predictor.predict(lyrics=lyrics, audio_path=tmp_path)

    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            log.info("Temp audio file deleted.")

    return PredictResponse(
        valence      = result["valence"],
        energy       = result["energy"],
        danceability = result["danceability"],
        tempo_bpm    = result["tempo_bpm"],
        decade       = result["decade"],
        decade_probs = result["decade_probs"],
        audio_used   = result["audio_used"],
        word_scores  = [WordScore(**ws) for ws in result["word_scores"]],
        warnings     = result["warnings"],
    )


# ── LOCAL DEV ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
