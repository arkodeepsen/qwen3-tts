"""Client-side long-form / audiobook generator for the Qwen3-TTS RunPod endpoint.

A single serverless job cannot return hours of audio (bounded by execution
timeout ~300s and response payload size). This script splits a script into
text blocks, generates each block via the API (concurrently, order preserved),
and concatenates the results client-side into one long output file.
"""
import argparse
import base64
import concurrent.futures
import io
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import requests
import soundfile as sf

# Ensure this directory (client/) is importable as a top-level path so
# `from cli import poll_result` resolves both when running this file
# directly (python client/longform.py ...) and when imported as
# client.longform (e.g. from tests).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from cli import poll_result

# Load ENDPOINT_ID / RUNPOD_API_KEY from the repo-root .env if python-dotenv is
# installed; otherwise fall back to real environment variables.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass


_SENTENCE_SPLIT_RE = re.compile(r"[^.!?。！？\n]*[.!?。！？\n]+|[^.!?。！？\n]+$")


def _split_sentences(text: str) -> list[str]:
    pieces = _SENTENCE_SPLIT_RE.findall(text or "")
    out = []
    for p in pieces:
        s = p.strip()
        if s:
            out.append(s)
    return out


def _hard_split_word_boundary(sentence: str, block_chars: int) -> list[str]:
    """Split a single overlong sentence at word boundaries, never mid-word
    unless a single word itself exceeds block_chars."""
    if len(sentence) <= block_chars:
        return [sentence]
    words = sentence.split(" ")
    out, cur = [], ""
    for w in words:
        candidate = w if not cur else cur + " " + w
        if len(candidate) <= block_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            # a single word longer than block_chars: hard-slice it
            while len(w) > block_chars:
                out.append(w[:block_chars])
                w = w[block_chars:]
            cur = w
    if cur:
        out.append(cur)
    return out


def _pack(units: list[str], block_chars: int) -> list[str]:
    """Greedily pack consecutive units (paragraphs or sentences) into blocks
    <= block_chars, joining with a single space/newline-normalized separator."""
    out, cur = [], ""
    for u in units:
        candidate = u if not cur else cur + "\n\n" + u
        if len(candidate) <= block_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            cur = u
    if cur:
        out.append(cur)
    return out


def split_blocks(text: str, block_chars: int = 1200) -> list[str]:
    """Split a script into blocks each <= block_chars characters.

    Splits on blank lines (paragraphs) first, then packs consecutive
    paragraphs into blocks <= block_chars. A paragraph longer than
    block_chars is sentence-split (on . ! ? CJK 。！？ and newlines) and
    packed; a sentence still too long is hard-split at word boundaries
    (never mid-word unless a single word exceeds block_chars). Empty /
    whitespace-only blocks are dropped. Content is preserved aside from
    whitespace normalization.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs on blank lines.
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        # normalize internal whitespace runs (keep single newlines as spaces)
        norm = re.sub(r"\s+", " ", p).strip()
        if norm:
            paragraphs.append(norm)

    if not paragraphs:
        return []

    # Expand any paragraph that alone exceeds block_chars into
    # sentence-packed sub-units before the final pack pass.
    units = []
    for p in paragraphs:
        if len(p) <= block_chars:
            units.append(p)
            continue
        sentences = _split_sentences(p)
        if not sentences:
            continue
        # hard-split any sentence that's still too long
        expanded = []
        for s in sentences:
            expanded.extend(_hard_split_word_boundary(s, block_chars))
        # pack sentences into sub-blocks <= block_chars, joined with a space
        sub_out, cur = [], ""
        for s in expanded:
            candidate = s if not cur else cur + " " + s
            if len(candidate) <= block_chars:
                cur = candidate
            else:
                if cur:
                    sub_out.append(cur)
                cur = s
        if cur:
            sub_out.append(cur)
        units.extend(sub_out)

    blocks = _pack(units, block_chars)
    return [b for b in (blk.strip() for blk in blocks) if b]


def _build_payload(voice_id, text, language, tuning):
    inp = {
        "action": "generate",
        "voice_id": voice_id,
        "text": text,
        "language": language,
        "response_format": "wav",  # always request lossless wav per block for concatenation
    }
    inp.update(tuning)
    return {"input": inp}


def _post(url, api_key, payload):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(url, json=payload, headers=headers, timeout=300).json()


def _generate_block(url, api_key, voice_id, text, language, tuning):
    payload = _build_payload(voice_id, text, language, tuning)
    job = _post(url, api_key, payload)
    out = poll_result(url, api_key, job)
    if not out.get("success") or not out.get("audio_base64"):
        raise RuntimeError(out.get("error", "generation failed"))
    audio_bytes = base64.b64decode(out["audio_base64"])
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return wav, int(sr)


def _generate_block_with_retry(url, api_key, index, total, voice_id, text, language, tuning):
    start = time.monotonic()
    try:
        wav, sr = _generate_block(url, api_key, voice_id, text, language, tuning)
    except Exception:
        try:
            wav, sr = _generate_block(url, api_key, voice_id, text, language, tuning)
        except Exception as e:
            raise RuntimeError(f"Block {index} failed after retry: {e}") from e
    elapsed = time.monotonic() - start
    print(f"[{index}/{total}] {elapsed:.1f}s")
    return index, wav, sr


def generate_blocks(url, api_key, voice_id, blocks, language, tuning, concurrency=2):
    total = len(blocks)
    max_workers = max(1, min(concurrency, 3))
    results = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_generate_block_with_retry, url, api_key, i + 1, total,
                      voice_id, block, language, tuning)
            for i, block in enumerate(blocks)
        ]
        for fut in concurrent.futures.as_completed(futures):
            index, wav, sr = fut.result()
            results[index - 1] = (wav, sr)
    return results


def concat_with_gap(results, gap: float):
    sample_rates = {sr for _, sr in results}
    assert len(sample_rates) == 1, f"Inconsistent sample rates across blocks: {sample_rates}"
    sr = sample_rates.pop()
    silence = np.zeros(int(round(gap * sr)), dtype=np.float32)
    parts = []
    for i, (wav, _) in enumerate(results):
        if i > 0 and len(silence):
            parts.append(silence)
        parts.append(wav)
    full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return full, sr


def write_output(full: np.ndarray, sr: int, output: str, fmt: str):
    fmt = fmt.lower()
    if fmt in ("wav", "flac"):
        sf.write(output, full, sr, format="WAV" if fmt == "wav" else "FLAC")
        return
    if fmt in ("mp3", "opus"):
        codec = "libmp3lame" if fmt == "mp3" else "libopus"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmpwav = tmp.name
        try:
            sf.write(tmpwav, full, sr, format="WAV")
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", tmpwav, "-c:a", codec, output],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {fmt}: {proc.stderr.decode(errors='ignore')[:500]}")
        finally:
            try:
                os.remove(tmpwav)
            except OSError:
                pass
        return
    raise ValueError(f"Unsupported output format: {fmt}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Qwen3-TTS long-form / audiobook client")
    p.add_argument("--url", default=(f"https://api.runpod.ai/v2/{os.getenv('ENDPOINT_ID')}/run"
                                     if os.getenv("ENDPOINT_ID") else None))
    p.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    p.add_argument("--voice-id", required=True)
    p.add_argument("--input", required=True, help="Path to a UTF-8 .txt script")
    p.add_argument("--output", default="audiobook.wav")
    p.add_argument("--language", default="English")
    p.add_argument("--format", default="wav", choices=["wav", "mp3", "flac", "opus"])
    p.add_argument("--block-chars", type=int, default=1200)
    p.add_argument("--gap", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=2)
    # passthrough tuning: only included in the request if explicitly set
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args(argv)
    if not args.url:
        print("Set ENDPOINT_ID/--url", file=sys.stderr)
        return 2

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = split_blocks(text, block_chars=args.block_chars)
    if not blocks:
        print("No synthesizable text found in input.", file=sys.stderr)
        return 2

    tuning = {}
    for key in ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "seed"):
        val = getattr(args, key)
        if val is not None:
            tuning[key] = val

    print(f"Split into {len(blocks)} block(s); generating with concurrency={min(args.concurrency, 3)}...")
    results = generate_blocks(args.url, args.api_key, args.voice_id, blocks,
                               args.language, tuning, concurrency=args.concurrency)

    full, sr = concat_with_gap(results, args.gap)
    write_output(full, sr, args.output, args.format)

    duration_sec = len(full) / sr if sr else 0.0
    print(f"Wrote {args.output}: {duration_sec:.1f}s total, {len(blocks)} block(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
