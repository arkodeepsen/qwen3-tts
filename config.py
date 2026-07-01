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

# Optional S3-compatible object storage — lets generate return a URL instead of
# base64 for large results (a single response can't carry hours of base64 audio).
# Works with RunPod network-volume S3, Cloudflare R2, AWS S3, Backblaze B2, MinIO,
# etc. If unset, URL output is disabled (output=auto falls back to base64).
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")          # e.g. https://<acct>.r2.cloudflarestorage.com (blank for AWS)
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_REGION = os.getenv("S3_REGION", "auto")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", "")    # set if the bucket serves public URLs; else presigned URLs are used
S3_URL_EXPIRY = int(os.getenv("S3_URL_EXPIRY", "604800"))  # presigned URL lifetime in seconds (default 7 days)
S3_PREFIX = os.getenv("S3_PREFIX", "qwen3-tts")            # key prefix for all objects

# `generate` with output="auto" returns base64 when the encoded audio is at or
# below this size, else uploads to S3 and returns a URL (base64 over ~this size
# risks exceeding the serverless response limit).
MAX_INLINE_BYTES = int(os.getenv("MAX_INLINE_BYTES", str(5 * 1024 * 1024)))  # 5 MB

# Outputs older than this are pruned from the bucket on each upload — the RunPod
# network-volume bucket is shared with the model cache + voices, so stale output
# files must not accumulate. Set to 0 to disable auto-pruning.
OUTPUT_TTL_SEC = int(os.getenv("OUTPUT_TTL_SEC", "86400"))  # 24 hours
