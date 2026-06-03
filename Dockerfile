FROM python:3.11-slim

# ffmpeg is required by librosa for audio loading
RUN apt-get update && \
    apt-get install -y ffmpeg git-lfs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY app.py .
COPY model.py .
COPY inference.py .
COPY preprocess.py .
COPY index.html .

# Copy model checkpoint and scaler
COPY checkpoints/best_model.pt checkpoints/best_model.pt
COPY processed/scaler.pkl processed/scaler.pkl

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
