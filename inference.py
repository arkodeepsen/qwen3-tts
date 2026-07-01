"""Model singleton and the synthesize orchestration (chunk -> generate -> concat -> encode)."""
import numpy as np
import torch

import config
from audio import b64, concat_waveforms, encode_audio
from chunking import pack_sentences, split_sentences
from srt import build_segments, segments_to_srt

_MODEL = None


def get_model():
    """Load the Base model once per worker; fall back to sdpa if flash-attn is unavailable."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from qwen_tts import Qwen3TTSModel
    common = dict(device_map="cuda:0", dtype=torch.bfloat16)
    try:
        _MODEL = Qwen3TTSModel.from_pretrained(config.MODEL_ID, attn_implementation="flash_attention_2", **common)
    except Exception:
        _MODEL = Qwen3TTSModel.from_pretrained(config.MODEL_ID, attn_implementation="sdpa", **common)
    return _MODEL


def _units(text: str, return_srt: bool) -> list[str]:
    sentences = split_sentences(text, max_chars=config.MAX_CHARS)
    if not sentences:
        return []
    return sentences if return_srt else pack_sentences(sentences, max_chars=config.MAX_CHARS)


def synthesize(prompt_items, text, language, seed: int = 42, return_srt: bool = False,
               response_format: str = None, model=None) -> dict:
    model = model or get_model()
    fmt = (response_format or config.DEFAULT_FORMAT).lower()
    if fmt not in config.SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")

    units = _units(text, return_srt)
    if not units:
        raise ValueError("No synthesizable text provided.")

    wavs, durations, sr = [], [], None
    for unit in units:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        out_wavs, out_sr = model.generate_voice_clone(
            text=unit, language=language, voice_clone_prompt=prompt_items)
        wav = np.asarray(out_wavs[0], dtype=np.float32).reshape(-1)
        sr = int(out_sr)
        wavs.append(wav)
        durations.append(len(wav) / sr)

    gap = config.INTER_CHUNK_GAP_SEC
    full = concat_waveforms(wavs, sr, gap=gap)
    data = encode_audio(full, sr, fmt)

    segments = srt_str = None
    if return_srt:
        segments = build_segments(units, durations, gap=gap)
        srt_str = segments_to_srt(segments)

    return {
        "audio_base64": b64(data),
        "format": fmt,
        "sample_rate": sr,
        "duration_sec": round(len(full) / sr, 3),
        "chunks": len(units),
        "size_bytes": len(data),
        "srt": srt_str,
        "segments": segments,
    }
