# RunPod serverless image for Qwen3-TTS voice cloning (Base variant)
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

# System deps: ffmpeg (mp3/opus), sox (qwen-tts), git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsox-dev git && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# App Python deps (transformers/accelerate pinned by qwen-tts)
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# flash-attn is optional; build best-effort, code falls back to sdpa if absent
RUN MAX_JOBS=4 python3 -m pip install flash-attn --no-build-isolation || \
    echo "flash-attn unavailable; will use sdpa attention"

RUN python3 -m pip cache purge || true

# App code (flat modules)
COPY config.py chunking.py srt.py audio.py registry.py inference.py actions.py handler.py /app/

# Model + voice registry live on the network volume, shared across workers
ENV HF_HOME=/runpod-volume \
    HF_HUB_CACHE=/runpod-volume \
    VOICE_DIR=/runpod-volume/voices \
    PYTHONUNBUFFERED=1

CMD ["python3", "-u", "handler.py"]
