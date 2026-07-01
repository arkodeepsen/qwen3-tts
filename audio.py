"""Concatenate waveforms and encode to the requested container format."""
import base64
import io
import subprocess
import numpy as np
import soundfile as sf

_SF_FORMATS = {"wav": "WAV", "flac": "FLAC"}
_FFMPEG_FORMATS = {"mp3": ("mp3", "libmp3lame"), "opus": ("ogg", "libopus")}


def concat_waveforms(wavs: list[np.ndarray], sr: int, gap: float = 0.0) -> np.ndarray:
    if not wavs:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(round(gap * sr)), dtype=np.float32)
    parts = []
    for i, w in enumerate(wavs):
        if i > 0 and len(silence):
            parts.append(silence)
        parts.append(np.asarray(w, dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


def _encode_ffmpeg(wav: np.ndarray, sr: int, fmt: str) -> bytes:
    container, codec = _FFMPEG_FORMATS[fmt]
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, wav, sr, format="WAV")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-c:a", codec, "-f", container, "pipe:1"],
        input=wav_bytes.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {fmt}: {proc.stderr.decode(errors='ignore')[:500]}")
    return proc.stdout


def encode_audio(wav: np.ndarray, sr: int, fmt: str) -> bytes:
    fmt = fmt.lower()
    if fmt in _SF_FORMATS:
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format=_SF_FORMATS[fmt])
        return buf.getvalue()
    if fmt in _FFMPEG_FORMATS:
        return _encode_ffmpeg(wav, sr, fmt)
    raise ValueError(f"Unsupported format: {fmt}")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")
