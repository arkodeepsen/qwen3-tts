"""Model singleton and the synthesize orchestration (chunk -> generate -> concat -> encode)."""
import numpy as np
import torch

import config
from audio import b64, concat_waveforms, content_type, encode_audio
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
               response_format: str = None, temperature: float = None, top_p: float = None,
               top_k: int = None, repetition_penalty: float = None, max_new_tokens: int = None,
               to_url: bool = False, s3_key: str = None, model=None) -> dict:
    model = model or get_model()
    fmt = (response_format or config.DEFAULT_FORMAT).lower()
    if fmt not in config.SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")

    units = _units(text, return_srt)
    if not units:
        raise ValueError("No synthesizable text provided.")

    # Generation params: per-request overrides win, else the env-tunable config
    # defaults. These are stability-focused so the Base model reliably emits EOS
    # (loose defaults -> runaway audio; see QwenLM/Qwen3-TTS#239). max_new_tokens
    # also hard-caps a single chunk.
    top_k_v = int(top_k) if top_k is not None else config.TOP_K
    top_p_v = float(top_p) if top_p is not None else config.TOP_P
    temp_v = float(temperature) if temperature is not None else config.TEMPERATURE
    rep_v = float(repetition_penalty) if repetition_penalty is not None else config.REPETITION_PENALTY
    max_tok_v = int(max_new_tokens) if max_new_tokens is not None else config.MAX_NEW_TOKENS
    gen_kwargs = dict(
        do_sample=True,
        top_k=top_k_v, top_p=top_p_v, temperature=temp_v, repetition_penalty=rep_v,
        subtalker_dosample=True, subtalker_top_k=top_k_v, subtalker_top_p=top_p_v,
        subtalker_temperature=temp_v, max_new_tokens=max_tok_v,
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

    result = {
        "format": fmt,
        "sample_rate": sr,
        "duration_sec": round(len(full) / sr, 3),
        "chunks": len(units),
        "size_bytes": len(data),
        "srt": srt_str,
        "segments": segments,
    }
    if to_url:  # upload to S3 and return a URL instead of base64 (large outputs)
        import uuid
        import storage
        key = storage.object_key(s3_key or f"outputs/{uuid.uuid4().hex}.{fmt}")
        result["url"] = storage.upload(key, data, content_type(fmt))
        result["key"] = key
    else:
        result["audio_base64"] = b64(data)
    return result


def merge_audio(keys, response_format: str = None, gap_sec: float = None,
                output_key: str = None) -> dict:
    """Concatenate already-generated audio parts (by S3 key) into one file and
    upload it, returning a URL. This is the long-form assembly step: pure I/O, no
    GPU generation, so it finishes fast even for hours of audio. Parts are decoded
    and re-encoded to `response_format`."""
    import io
    import uuid
    import soundfile as sf
    import storage

    if not keys:
        raise ValueError("merge requires a non-empty 'keys' list.")
    fmt = (response_format or config.DEFAULT_FORMAT).lower()
    if fmt not in config.SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    gap = float(gap_sec) if gap_sec is not None else config.INTER_CHUNK_GAP_SEC

    wavs, sr = [], None
    for k in keys:
        wav, s = sf.read(io.BytesIO(storage.download(k)), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        wavs.append(np.asarray(wav, dtype=np.float32).reshape(-1))
        sr = int(s) if sr is None else sr

    full = concat_waveforms(wavs, sr, gap=gap)
    data = encode_audio(full, sr, fmt)
    key = storage.object_key(output_key or f"outputs/{uuid.uuid4().hex}.{fmt}")
    url = storage.upload(key, data, content_type(fmt))
    return {
        "url": url,
        "key": key,
        "format": fmt,
        "sample_rate": sr,
        "duration_sec": round(len(full) / sr, 3),
        "parts": len(keys),
        "size_bytes": len(data),
    }
