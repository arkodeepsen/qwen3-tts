"""Model singleton and the synthesize orchestration (chunk -> generate -> concat -> encode)."""
import numpy as np
import torch

import config
from audio import b64, concat_waveforms, encode_audio
from chunking import pack_sentences, split_sentences
from srt import build_segments, segments_to_srt

_MODEL = None


def get_model():
    """Load the Base model once per worker.

    Selects flash_attention_2 only if flash-attn is importable, else sdpa. The
    choice is made up front (cheap import probe) so a cold start never loads the
    ~5 GB model twice; a final sdpa retry guards any other attn init failure.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from qwen_tts import Qwen3TTSModel
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except Exception:
        attn = "sdpa"
    common = dict(device_map="cuda:0", dtype=torch.bfloat16)
    try:
        _MODEL = Qwen3TTSModel.from_pretrained(config.MODEL_ID, attn_implementation=attn, **common)
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

    # Explicit, stability-focused generation params. Without these the Base model
    # relies on loose defaults (top_p=1.0, no repetition penalty) and can fail to
    # emit EOS -> runaway audio. max_new_tokens also hard-caps a single chunk.
    gen_kwargs = dict(
        do_sample=True,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        temperature=config.TEMPERATURE,
        repetition_penalty=config.REPETITION_PENALTY,
        subtalker_dosample=True,
        subtalker_top_k=config.TOP_K,
        subtalker_top_p=config.TOP_P,
        subtalker_temperature=config.TEMPERATURE,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    wavs, durations, sr = [], [], None
    for unit in units:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        out_wavs, out_sr = model.generate_voice_clone(
            text=unit, language=language, voice_clone_prompt=prompt_items, **gen_kwargs)
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
