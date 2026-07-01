"""Shared, env-overridable configuration constants."""
import os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICE_DIR = os.getenv("VOICE_DIR", "/runpod-volume/voices")

MAX_CHARS = int(os.getenv("MAX_CHARS", "200"))
INTER_CHUNK_GAP_SEC = float(os.getenv("INTER_CHUNK_GAP_SEC", "0.15"))

SUPPORTED_FORMATS = ("wav", "mp3", "flac", "opus")
DEFAULT_FORMAT = "wav"
