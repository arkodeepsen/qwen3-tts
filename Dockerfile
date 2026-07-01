# RunPod serverless image for Qwen3-TTS voice cloning (Base variant)
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

# System deps: ffmpeg (mp3/opus), sox (qwen-tts), git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsox-dev git && rm -rf /var/lib/apt/lists/*

# Install app deps into a venv that INHERITS the base image's system packages
# (--system-site-packages), so torch/torchaudio (cu128) stay put with no
# re-download. New/upgraded deps (e.g. runpod needs cryptography>=48, but the
# base ships an apt/Debian cryptography with no pip RECORD that pip cannot
# uninstall) install INTO the venv, shadowing the system copies instead of
# hard-failing on `uninstall-no-record-file`.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN python3 -m pip install --upgrade pip

# App Python deps (transformers/accelerate pinned by qwen-tts; torch pinned to
# the base image version so pip treats it as already-satisfied — no re-download)
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# flash-attn is intentionally NOT built — it's heavy/fragile on constrained
# builders (RunPod Hub) and the code uses PyTorch sdpa attention, which is
# plenty for this 1.7B model. Re-add a `pip install flash-attn` here if you
# want it and your build environment can compile it.
RUN python3 -m pip cache purge || true

# App code (flat modules)
COPY config.py chunking.py srt.py audio.py registry.py inference.py actions.py handler.py /app/

# Model + voice registry live on the network volume, shared across workers
ENV HF_HOME=/runpod-volume \
    HF_HUB_CACHE=/runpod-volume \
    TRANSFORMERS_CACHE=/runpod-volume \
    VOICE_DIR=/runpod-volume/voices \
    PYTHONUNBUFFERED=1

CMD ["python3", "-u", "handler.py"]
