"""Shared, env-overridable configuration constants."""
import os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICE_DIR = os.getenv("VOICE_DIR", "/runpod-volume/voices")

MAX_CHARS = int(os.getenv("MAX_CHARS", "200"))
INTER_CHUNK_GAP_SEC = float(os.getenv("INTER_CHUNK_GAP_SEC", "0.15"))

SUPPORTED_FORMATS = ("wav", "mp3", "flac", "opus")
DEFAULT_FORMAT = "wav"

# Generation stability. The 1.7B-Base model can fail to emit an end-of-sequence
# token and run away (minutes of audio for one sentence), especially with loose
# sampling. These follow Qwen/community guidance for stable voice cloning; all
# are env-overridable. See QwenLM/Qwen3-TTS issue #239.
TOP_K = int(os.getenv("TOP_K", "50"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))

# Reference audio is trimmed to this many seconds at registration — overly long
# reference clips are a known trigger for the model never emitting EOS.
REF_AUDIO_MAX_SEC = float(os.getenv("REF_AUDIO_MAX_SEC", "30"))
