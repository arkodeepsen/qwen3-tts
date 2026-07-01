# Qwen3-TTS Voice-Cloning API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RunPod serverless endpoint that clones a voice from a reference clip once, then synthesizes arbitrary-length speech (with optional sentence-level SRT) in that voice by `voice_id`.

**Architecture:** A thin `handler.py` (RunPod entry) delegates to an `actions.py` router. Business logic lives in small, independently testable modules: `chunking` (sentence split/pack), `srt` (segments + SRT string), `audio` (concat + encode + base64), `registry` (voice profiles on the network volume), and `inference` (model singleton + synthesize orchestration). The `Qwen/Qwen3-TTS-12Hz-1.7B-Base` model is loaded once per worker; voice profiles are cached on `/runpod-volume` and reused by id.

**Tech Stack:** Python ≥3.9, `qwen-tts` (`transformers==4.57.3`, `accelerate==1.12.0`, `torch` bf16), `runpod`, `soundfile`, `librosa`, `ffmpeg` (mp3/opus), Docker on a `runpod/pytorch` base, pytest for tests, Streamlit + argparse clients.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec (`docs/superpowers/specs/2026-07-01-qwen3-tts-voice-cloning-api-design.md`) and the verified `qwen_tts` source.

- **Model (only):** `Qwen/Qwen3-TTS-12Hz-1.7B-Base` — voice cloning via `generate_voice_clone` / `create_voice_clone_prompt`.
- **Python:** `>=3.9` (qwen-tts supports 3.9–3.13; the base image Python is fine).
- **Pinned deps (from qwen-tts pyproject):** `transformers==4.57.3`, `accelerate==1.12.0`. Also `qwen-tts`, `runpod`, `soundfile`, `librosa`, `torchaudio`, `sox`, `onnxruntime`, `einops`.
- **Model load kwargs:** `device_map="cuda:0"`, `dtype=torch.bfloat16` (note: `dtype`, not `torch_dtype`), `attn_implementation="flash_attention_2"` with fallback to `"sdpa"` if flash-attn is unavailable.
- **Network volume env:** `HF_HOME=/runpod-volume`, `HF_HUB_CACHE=/runpod-volume`, `VOICE_DIR=/runpod-volume/voices`.
- **Response envelope:** JSON, `{"success": true|false, ...}`; binary is base64. Errors: `{"success": false, "error": "<message>"}`.
- **Audio formats:** `wav` (default), `mp3`, `flac`, `opus`.
- **Chunking:** split to sentences; each generation unit ≤ 200 characters; **same fixed `seed` applied before every chunk** (`torch.manual_seed`) for consistent timbre. SRT requested ⇒ one sentence per generation unit (so each SRT segment has a measured duration).
- **Voice profile storage (`/runpod-volume/voices/<voice_id>/`):** `meta.json`, `ref.wav` (source of truth), `prompt.pt` (best-effort cache). `voice_id` is `slug(name)-<shortuuid>`, regex-validated, never interpolated raw into a path. Writes are atomic (temp dir → `os.rename`).
- **`prompt.pt` serialization:** store a **plain dict** of tensors+primitives (NOT the `VoiceClonePromptItem` dataclass) so `torch.load(..., weights_only=True)` (torch ≥2.6 default) can read it. Reconstruct the dataclass on load and move tensors to the model device.
- **Scale-to-zero / billing:** `min_workers=0`, `idle_timeout=5`, `flashboot=true`, `max_workers=3`, `execution_timeout=300`. Only `/run` and `/runsync` bill workers; `/status` polling is free and never wakes a worker.
- **Commit footer (every commit):**
  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

## File Structure

Flat modules at repo root (matches the author's `qwen-image`/`kokoro` flat style) but split by responsibility so each is independently testable. This refines the spec's single-`inference.py` sketch into focused files.

| File | Responsibility |
|---|---|
| `config.py` | Env-driven constants: `MODEL_ID`, `VOICE_DIR`, `MAX_CHARS`, `INTER_CHUNK_GAP_SEC`, supported formats. |
| `chunking.py` | `split_sentences(text)`, `pack_sentences(sentences, max_chars)` — pure. |
| `srt.py` | `build_segments(texts, durations, gap)`, `segments_to_srt(segments)` — pure. |
| `audio.py` | `concat_waveforms(wavs, sr, gap)`, `encode_audio(wav, sr, fmt)`, `b64(bytes)` — pure/IO-light. |
| `registry.py` | `VoiceRegistry`: `register`, `list_voices`, `delete`, `load_prompt` (+ id sanitization, atomic write, prompt (de)serialization, in-process LRU). |
| `inference.py` | `get_model()` singleton; `synthesize(prompt_items, text, language, seed, return_srt, response_format)` orchestrating chunk→generate→concat→durations→(srt)→encode. |
| `actions.py` | `handle(job_input)` router: `register_voice`, `generate`, `list_voices`, `delete_voice` + validation + error envelopes. |
| `handler.py` | Thin RunPod entry: `runpod.serverless.start({"handler": handler})`, wraps `actions.handle`. |
| `tests/test_*.py` | One test module per logic file. |
| `Dockerfile`, `requirements.txt`, `build.sh` | Image build + push to `ghcr.io/arkodeepsen`. |
| `runpod.toml`, `.runpod/hub.json`, `.runpod/tests.json` | RunPod deploy config + Hub listing + smoke test. |
| `client/cli.py`, `client/app.py`, `client/.env.example` | CLI + Streamlit clients. |
| `.github/workflows/build.yml` | CI image build/push. |
| `README.md`, `LICENSE`, `.gitignore` | Docs + Apache-2.0 + ignores. |
| `scripts/smoke_test.py` | Manual GPU integration smoke test (register→generate→srt). |

**Test strategy:** No test loads the 5 GB model or needs a GPU. `chunking`/`srt`/`audio` are pure. `registry`/`inference`/`actions` take collaborators (the model, the registry) by injection so tests pass fakes. Real model inference is covered by the manual `scripts/smoke_test.py` on RunPod (Task 14).

---

### Task 1: Repo scaffold, tooling, and shared config

**Files:**
- Create: `LICENSE`, `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.MODEL_ID: str`, `config.VOICE_DIR: str`, `config.MAX_CHARS: int`, `config.INTER_CHUNK_GAP_SEC: float`, `config.SUPPORTED_FORMATS: tuple[str,...]`, `config.DEFAULT_FORMAT: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import importlib, os

def test_defaults(monkeypatch):
    monkeypatch.delenv("VOICE_DIR", raising=False)
    import config; importlib.reload(config)
    assert config.MODEL_ID == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert config.VOICE_DIR == "/runpod-volume/voices"
    assert config.MAX_CHARS == 200
    assert config.DEFAULT_FORMAT == "wav"
    assert set(config.SUPPORTED_FORMATS) == {"wav", "mp3", "flac", "opus"}

def test_voice_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_DIR", str(tmp_path))
    import config; importlib.reload(config)
    assert config.VOICE_DIR == str(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Create the supporting files**

`config.py`:
```python
"""Shared, env-overridable configuration constants."""
import os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICE_DIR = os.getenv("VOICE_DIR", "/runpod-volume/voices")

MAX_CHARS = int(os.getenv("MAX_CHARS", "200"))
INTER_CHUNK_GAP_SEC = float(os.getenv("INTER_CHUNK_GAP_SEC", "0.15"))

SUPPORTED_FORMATS = ("wav", "mp3", "flac", "opus")
DEFAULT_FORMAT = "wav"
```

`requirements.txt`:
```
qwen-tts
transformers==4.57.3
accelerate==1.12.0
torch
torchaudio
librosa
soundfile
sox
onnxruntime
einops
runpod>=1.0.0
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest
numpy
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

`.gitignore`:
```
__pycache__/
*.pyc
.env
*.wav
*.mp3
*.srt
.pytest_cache/
outputs/
```

`LICENSE`: the Apache-2.0 license text (copy from `https://www.apache.org/licenses/LICENSE-2.0.txt`, standard boilerplate).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add config.py requirements.txt requirements-dev.txt pytest.ini .gitignore LICENSE tests/test_config.py
git commit -m "chore: scaffold repo, tooling, and shared config

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Sentence splitting and packing (`chunking.py`)

**Files:**
- Create: `chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces:
  - `split_sentences(text: str, max_chars: int = 200) -> list[str]` — splits on terminal punctuation (`. ! ?` and CJK `。！？`) and newlines; hard-splits any sentence longer than `max_chars` at word/char boundaries; drops empties/whitespace.
  - `pack_sentences(sentences: list[str], max_chars: int = 200) -> list[str]` — greedily concatenates consecutive sentences while staying ≤ `max_chars`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
from chunking import split_sentences, pack_sentences

def test_split_basic_punctuation():
    assert split_sentences("Hello world. How are you? Fine!") == \
        ["Hello world.", "How are you?", "Fine!"]

def test_split_cjk_and_newlines():
    assert split_sentences("你好。今天天气不错！\nBye.") == \
        ["你好。", "今天天气不错！", "Bye."]

def test_split_hard_splits_overlong_sentence():
    long = "word " * 100  # 500 chars, no terminal punctuation
    out = split_sentences(long, max_chars=200)
    assert all(len(s) <= 200 for s in out)
    assert "".join(out).replace(" ", "") == long.replace(" ", "")

def test_split_drops_empty():
    assert split_sentences("  . .. \n\n") == []

def test_pack_greedy():
    sents = ["A.", "B.", "C.", "D."]  # 2 chars each
    assert pack_sentences(sents, max_chars=5) == ["A. B.", "C. D."]

def test_pack_single_overlong_passes_through():
    assert pack_sentences(["x" * 250], max_chars=200) == ["x" * 250]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chunking'`

- [ ] **Step 3: Write minimal implementation**

`chunking.py`:
```python
"""Split text into short generation units for stable long-form synthesis."""
import re

_TERMINATORS = ".!?。！？\n"
_SPLIT_RE = re.compile(r"[^.!?。！？\n]*[.!?。！？\n]+|[^.!?。！？\n]+$")


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    words, out, cur = sentence.split(" "), [], ""
    for w in words:
        candidate = w if not cur else cur + " " + w
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            # a single word longer than max_chars: slice it
            while len(w) > max_chars:
                out.append(w[:max_chars]); w = w[max_chars:]
            cur = w
    if cur:
        out.append(cur)
    return out


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    pieces = _SPLIT_RE.findall(text or "")
    out: list[str] = []
    for p in pieces:
        s = p.strip().strip("\n").strip()
        if not s or all(ch in _TERMINATORS + " " for ch in s):
            continue
        out.extend(_hard_split(s, max_chars))
    return out


def pack_sentences(sentences: list[str], max_chars: int = 200) -> list[str]:
    out: list[str] = []
    cur = ""
    for s in sentences:
        candidate = s if not cur else cur + " " + s
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add chunking.py tests/test_chunking.py
git commit -m "feat: sentence splitting and packing for long-form chunking

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Sentence-level SRT (`srt.py`)

**Files:**
- Create: `srt.py`
- Test: `tests/test_srt.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces:
  - `build_segments(texts: list[str], durations: list[float], gap: float = 0.0) -> list[dict]` — returns `[{"index": int, "start": float, "end": float, "text": str}]`; each segment starts after the previous end plus `gap`.
  - `segments_to_srt(segments: list[dict]) -> str` — standard SRT (`HH:MM:SS,mmm`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_srt.py
from srt import build_segments, segments_to_srt

def test_build_segments_no_gap():
    segs = build_segments(["A", "B"], [1.0, 2.0], gap=0.0)
    assert segs == [
        {"index": 1, "start": 0.0, "end": 1.0, "text": "A"},
        {"index": 2, "start": 1.0, "end": 3.0, "text": "B"},
    ]

def test_build_segments_with_gap():
    segs = build_segments(["A", "B"], [1.0, 1.0], gap=0.5)
    assert segs[1]["start"] == 1.5
    assert segs[1]["end"] == 2.5

def test_build_segments_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        build_segments(["A"], [1.0, 2.0])

def test_segments_to_srt_format():
    segs = [{"index": 1, "start": 0.0, "end": 1.25, "text": "Hi"}]
    out = segments_to_srt(segs)
    assert out == "1\n00:00:00,000 --> 00:00:01,250\nHi\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_srt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'srt'`

- [ ] **Step 3: Write minimal implementation**

`srt.py`:
```python
"""Build sentence-level subtitle segments and an SRT string from chunk durations."""


def build_segments(texts, durations, gap: float = 0.0) -> list[dict]:
    if len(texts) != len(durations):
        raise ValueError(f"texts ({len(texts)}) and durations ({len(durations)}) must match")
    segments, cursor = [], 0.0
    for i, (text, dur) in enumerate(zip(texts, durations), start=1):
        start = cursor
        end = start + float(dur)
        segments.append({"index": i, "start": round(start, 3), "end": round(end, 3), "text": text})
        cursor = end + gap
    return segments


def _fmt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    blocks = []
    for seg in segments:
        blocks.append(f"{seg['index']}\n{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n{seg['text']}\n")
    return "\n".join(blocks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_srt.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add srt.py tests/test_srt.py
git commit -m "feat: sentence-level SRT segments and formatting

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Audio concat and encoding (`audio.py`)

**Files:**
- Create: `audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `concat_waveforms(wavs: list[np.ndarray], sr: int, gap: float = 0.0) -> np.ndarray` — concatenates 1-D float32 waveforms, inserting `gap` seconds of silence between them.
  - `encode_audio(wav: np.ndarray, sr: int, fmt: str) -> bytes` — encodes to `wav`/`flac` via soundfile, `mp3`/`opus` via ffmpeg; raises `ValueError` on unsupported fmt.
  - `b64(data: bytes) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio.py
import io, shutil, base64
import numpy as np
import soundfile as sf
import pytest
from audio import concat_waveforms, encode_audio, b64

def _tone(n=8000):
    return (0.1 * np.sin(np.linspace(0, 20, n))).astype(np.float32)

def test_concat_inserts_silence():
    a, bwav = _tone(1000), _tone(1000)
    out = concat_waveforms([a, bwav], sr=1000, gap=0.5)  # 0.5s @1000hz = 500 samples
    assert len(out) == 1000 + 500 + 1000
    assert np.allclose(out[1000:1500], 0.0)

def test_concat_single():
    a = _tone(1000)
    assert len(concat_waveforms([a], sr=1000, gap=0.5)) == 1000

def test_encode_wav_roundtrip():
    wav = _tone()
    data = encode_audio(wav, 8000, "wav")
    back, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == 8000 and len(back) == len(wav)

def test_encode_flac_roundtrip():
    wav = _tone()
    data = encode_audio(wav, 8000, "flac")
    back, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == 8000

def test_encode_unsupported():
    with pytest.raises(ValueError):
        encode_audio(_tone(), 8000, "aiff")

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_mp3_nonempty():
    data = encode_audio(_tone(), 8000, "mp3")
    assert isinstance(data, bytes) and len(data) > 0

def test_b64_roundtrip():
    assert base64.b64decode(b64(b"abc")) == b"abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio'`

- [ ] **Step 3: Write minimal implementation**

`audio.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio.py -v`
Expected: PASS (7 passed, or 6 passed + 1 skipped if ffmpeg absent locally)

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_audio.py
git commit -m "feat: waveform concat and multi-format audio encoding

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Voice registry on the network volume (`registry.py`)

**Files:**
- Create: `registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: a `model` object exposing `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False) -> list[VoiceClonePromptItem]` and a `.device`. A `VoiceClonePromptItem` has fields `ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text`.
- Produces: `VoiceRegistry(root: str, model_getter=callable)` with:
  - `register(name: str, ref_audio, ref_text: str, language: str) -> dict` → `{"voice_id", "sample_rate", "name", "language"}`.
  - `list_voices() -> list[dict]`.
  - `delete(voice_id: str) -> bool`.
  - `load_prompt(voice_id: str) -> list[VoiceClonePromptItem]` (device-correct, LRU-cached).
  - `sanitize_voice_id(name: str) -> str` (module function).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import types
import numpy as np
import pytest
import torch
from registry import VoiceRegistry, sanitize_voice_id

class FakeItem:
    def __init__(self):
        self.ref_code = torch.zeros(4, 2)
        self.ref_spk_embedding = torch.ones(8)
        self.x_vector_only_mode = False
        self.icl_mode = True
        self.ref_text = "ref transcript"

class FakeModel:
    device = torch.device("cpu")
    def __init__(self): self.calls = 0
    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode=False):
        self.calls += 1
        return [FakeItem()]

@pytest.fixture
def reg(tmp_path):
    model = FakeModel()
    return VoiceRegistry(root=str(tmp_path), model_getter=lambda: model), model

def _wav_b64():
    import io, base64, soundfile as sf
    buf = io.BytesIO(); sf.write(buf, np.zeros(2400, dtype=np.float32), 24000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

def test_sanitize_blocks_traversal():
    vid = sanitize_voice_id("../../etc/passwd")
    assert "/" not in vid and ".." not in vid

def test_register_creates_profile(reg):
    r, model = reg
    out = r.register("Client Narrator", _wav_b64(), "ref transcript", "English")
    assert out["voice_id"].startswith("client-narrator-")
    voices = r.list_voices()
    assert len(voices) == 1 and voices[0]["language"] == "English"

def test_load_prompt_uses_cache(reg):
    r, model = reg
    out = r.register("v", _wav_b64(), "ref transcript", "English")
    p1 = r.load_prompt(out["voice_id"])
    p2 = r.load_prompt(out["voice_id"])
    assert p1 is p2                      # LRU returns same object
    assert p1[0].ref_text == "ref transcript"

def test_load_prompt_rebuilds_when_cache_file_missing(reg, tmp_path):
    r, model = reg
    out = r.register("v", _wav_b64(), "ref transcript", "English")
    # delete prompt.pt, clear in-process cache -> must rebuild from ref.wav
    (tmp_path / "voices" if False else None)
    import os
    os.remove(os.path.join(r.root, out["voice_id"], "prompt.pt"))
    r._cache.clear()
    calls_before = model.calls
    prompt = r.load_prompt(out["voice_id"])
    assert model.calls == calls_before + 1
    assert prompt[0].ref_spk_embedding.shape[0] == 8

def test_delete(reg):
    r, _ = reg
    out = r.register("v", _wav_b64(), "ref transcript", "English")
    assert r.delete(out["voice_id"]) is True
    assert r.list_voices() == []
    assert r.delete("nonexistent") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'registry'`

- [ ] **Step 3: Write minimal implementation**

`registry.py`:
```python
"""Persistent voice-profile registry on the RunPod network volume.

Layout: <root>/<voice_id>/{meta.json, ref.wav, prompt.pt}
ref.wav is the source of truth; prompt.pt is a best-effort cache serialized as a
plain dict (tensors + primitives) so torch.load(weights_only=True) can read it.
"""
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import OrderedDict

from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch


@dataclass
class PromptItem:
    """Structural mirror of qwen_tts.VoiceClonePromptItem.

    generate_voice_clone only reads these attributes (duck-typed), so a local
    dataclass keeps the registry decoupled from qwen_tts and unit-testable
    without installing the model package. Fields must match VoiceClonePromptItem.
    """
    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: object = None


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")


def sanitize_voice_id(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-") or "voice"
    return f"{slug[:40]}-{uuid.uuid4().hex[:6]}"


def _load_audio_np(ref_audio):
    """Decode base64/URL/path/tuple into (np.float32 mono, sr) using the model's own loader semantics."""
    import base64, io, urllib.request, librosa
    if isinstance(ref_audio, tuple):
        wav, sr = ref_audio
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)
    s = ref_audio
    if s.startswith("http://") or s.startswith("https://"):
        with urllib.request.urlopen(s) as resp:
            wav, sr = sf.read(io.BytesIO(resp.read()), dtype="float32", always_2d=False)
    elif s.startswith("data:audio") or ("/" not in s and "\\" not in s and len(s) > 256):
        if "," in s and s.startswith("data:"):
            s = s.split(",", 1)[1]
        wav, sr = sf.read(io.BytesIO(base64.b64decode(s)), dtype="float32", always_2d=False)
    else:
        wav, sr = librosa.load(s, sr=None, mono=True)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    return wav.astype(np.float32), int(sr)


class VoiceRegistry:
    def __init__(self, root: str, model_getter, cache_size: int = 16):
        self.root = root
        self._model_getter = model_getter
        self._cache: "OrderedDict[str, list]" = OrderedDict()
        self._cache_size = cache_size
        os.makedirs(self.root, exist_ok=True)

    # ---- helpers ----
    def _dir(self, voice_id: str) -> str:
        if not _ID_RE.match(voice_id or ""):
            raise ValueError(f"Invalid voice_id: {voice_id!r}")
        return os.path.join(self.root, voice_id)

    @staticmethod
    def _item_to_dict(it) -> dict:
        return {
            "ref_code": it.ref_code,
            "ref_spk_embedding": it.ref_spk_embedding,
            "x_vector_only_mode": bool(it.x_vector_only_mode),
            "icl_mode": bool(it.icl_mode),
            "ref_text": it.ref_text,
        }

    @staticmethod
    def _dict_to_item(d: dict, device) -> PromptItem:
        rc = d["ref_code"]
        emb = d["ref_spk_embedding"]
        return PromptItem(
            ref_code=(rc.to(device) if isinstance(rc, torch.Tensor) else rc),
            ref_spk_embedding=emb.to(device),
            x_vector_only_mode=bool(d["x_vector_only_mode"]),
            icl_mode=bool(d["icl_mode"]),
            ref_text=d.get("ref_text"),
        )

    # ---- public API ----
    def register(self, name: str, ref_audio, ref_text: str, language: str) -> dict:
        model = self._model_getter()
        voice_id = sanitize_voice_id(name)
        wav, sr = _load_audio_np(ref_audio)
        items = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False)

        tmp = tempfile.mkdtemp(dir=self.root)
        try:
            sf.write(os.path.join(tmp, "ref.wav"), wav, sr, format="WAV")
            torch.save([self._item_to_dict(it) for it in items], os.path.join(tmp, "prompt.pt"))
            meta = {"voice_id": voice_id, "name": name, "language": language,
                    "ref_text": ref_text, "sample_rate": int(sr)}
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.rename(tmp, os.path.join(self.root, voice_id))
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return {"voice_id": voice_id, "sample_rate": int(sr), "name": name, "language": language}

    def list_voices(self) -> list[dict]:
        out = []
        for vid in sorted(os.listdir(self.root)):
            meta_path = os.path.join(self.root, vid, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    m = json.load(f)
                out.append({"voice_id": m["voice_id"], "name": m.get("name"),
                            "language": m.get("language"), "sample_rate": m.get("sample_rate")})
        return out

    def delete(self, voice_id: str) -> bool:
        try:
            d = self._dir(voice_id)
        except ValueError:
            return False
        if not os.path.isdir(d):
            return False
        shutil.rmtree(d, ignore_errors=True)
        self._cache.pop(voice_id, None)
        return True

    def load_prompt(self, voice_id: str) -> list:
        if voice_id in self._cache:
            self._cache.move_to_end(voice_id)
            return self._cache[voice_id]
        model = self._model_getter()
        device = model.device
        d = self._dir(voice_id)
        if not os.path.isdir(d):
            raise KeyError(f"Unknown voice_id: {voice_id}")
        prompt_path = os.path.join(d, "prompt.pt")
        items = None
        if os.path.isfile(prompt_path):
            try:
                dicts = torch.load(prompt_path, map_location=device, weights_only=True)
                items = [self._dict_to_item(x, device) for x in dicts]
            except Exception:
                items = None
        if items is None:  # rebuild from source of truth
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            built = model.create_voice_clone_prompt(
                ref_audio=os.path.join(d, "ref.wav"), ref_text=meta.get("ref_text"), x_vector_only_mode=False)
            torch.save([self._item_to_dict(it) for it in built], prompt_path)
            items = [self._dict_to_item(self._item_to_dict(it), device) for it in built]
        self._cache[voice_id] = items
        self._cache.move_to_end(voice_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return items
```

> Note: `weights_only=True` requires the saved payload to be tensors + primitives only (dicts/lists/str/bool/None are allowed). That is exactly what `_item_to_dict` produces — do not save the dataclass directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add registry.py tests/test_registry.py
git commit -m "feat: network-volume voice registry with prompt caching

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Model loader and synthesize orchestration (`inference.py`)

**Files:**
- Create: `inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `chunking.split_sentences`, `chunking.pack_sentences`, `audio.concat_waveforms`, `audio.encode_audio`, `audio.b64`, `srt.build_segments`, `srt.segments_to_srt`, `config.*`; a model with `generate_voice_clone(text, language, voice_clone_prompt, **kw) -> (list[np.ndarray], int)`.
- Produces:
  - `get_model() -> Qwen3TTSModel` — cached singleton (loads once; flash-attn→sdpa fallback).
  - `synthesize(prompt_items, text, language, seed=42, return_srt=False, response_format="wav", model=None) -> dict` returning `{"audio_base64","format","sample_rate","duration_sec","chunks","size_bytes","srt","segments"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
import base64, io
import numpy as np
import soundfile as sf
from inference import synthesize

class FakeModel:
    """Returns 1 sample-second of audio per call so durations are deterministic."""
    def __init__(self): self.calls = []; self.seeds = []
    def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
        import torch
        self.calls.append(text)
        self.seeds.append(torch.initial_seed())
        n = 100  # 100 samples @ sr=100 => 1.0s
        return [np.full(n, 0.2, dtype=np.float32)], 100

def _prompt():  # opaque to synthesize; just passed through
    return [object()]

def test_single_chunk_wav():
    m = FakeModel()
    out = synthesize(_prompt(), "Hello world.", "English", response_format="wav", model=m)
    assert out["format"] == "wav" and out["sample_rate"] == 100
    assert out["chunks"] == 1
    wav, sr = sf.read(io.BytesIO(base64.b64decode(out["audio_base64"])), dtype="float32")
    assert sr == 100 and len(wav) == 100
    assert out["srt"] is None and out["segments"] is None

def test_multi_sentence_packs_without_srt():
    m = FakeModel()
    out = synthesize(_prompt(), "A. B. C.", "English", return_srt=False, model=m)
    # short sentences pack into a single <=200 char unit -> 1 generate call
    assert len(m.calls) == 1 and out["chunks"] == 1

def test_srt_is_sentence_level():
    m = FakeModel()
    out = synthesize(_prompt(), "One. Two. Three.", "English", return_srt=True, model=m)
    assert len(m.calls) == 3            # one generate per sentence
    assert out["segments"][0]["text"] == "One."
    assert out["segments"][1]["start"] > 0
    assert out["srt"].startswith("1\n00:00:00,000 -->")

def test_seed_is_applied_each_chunk():
    m = FakeModel()
    synthesize(_prompt(), "One. Two.", "English", seed=123, return_srt=True, model=m)
    assert set(m.seeds) == {123}        # same fixed seed reset before each chunk
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference'`

- [ ] **Step 3: Write minimal implementation**

`inference.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_inference.py
git commit -m "feat: model singleton and synthesize orchestration

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Action router (`actions.py`)

**Files:**
- Create: `actions.py`
- Test: `tests/test_actions.py`

**Interfaces:**
- Consumes: `registry.VoiceRegistry`, `inference.synthesize`, `inference.get_model`, `config.VOICE_DIR`.
- Produces:
  - `handle(job_input: dict, registry=None) -> dict` — routes on `job_input["action"]` (default `"generate"`); returns a `{"success": ...}` envelope; never raises (converts exceptions to error envelopes).
  - `get_registry() -> VoiceRegistry` — process singleton bound to `config.VOICE_DIR` and `inference.get_model`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_actions.py
import numpy as np
from actions import handle

class FakeRegistry:
    def __init__(self): self.voices = {}
    def register(self, name, ref_audio, ref_text, language):
        vid = f"{name}-x"; self.voices[vid] = {"voice_id": vid, "name": name, "language": language, "sample_rate": 24000}
        return {"voice_id": vid, "sample_rate": 24000, "name": name, "language": language}
    def list_voices(self): return list(self.voices.values())
    def delete(self, voice_id): return self.voices.pop(voice_id, None) is not None
    def load_prompt(self, voice_id):
        if voice_id not in self.voices: raise KeyError(voice_id)
        return [object()]

def _reg(): return FakeRegistry()

def test_register_voice_happy(monkeypatch):
    out = handle({"action": "register_voice", "name": "narr", "ref_audio": "b64",
                  "ref_text": "hi", "language": "English"}, registry=_reg())
    assert out["success"] and out["voice_id"] == "narr-x"

def test_register_missing_fields():
    out = handle({"action": "register_voice", "name": "narr"}, registry=_reg())
    assert out["success"] is False and "ref_audio" in out["error"]

def test_generate_unknown_voice():
    out = handle({"action": "generate", "voice_id": "nope", "text": "hi"}, registry=_reg())
    assert out["success"] is False and "nope" in out["error"]

def test_generate_happy(monkeypatch):
    reg = _reg(); reg.register("narr", "b64", "hi", "English")
    import actions
    monkeypatch.setattr(actions, "synthesize",
        lambda prompt, text, language, seed=42, return_srt=False, response_format="wav":
            {"audio_base64": "AAA", "format": response_format, "chunks": 1})
    out = handle({"action": "generate", "voice_id": "narr-x", "text": "Hello.",
                  "response_format": "mp3"}, registry=reg)
    assert out["success"] and out["format"] == "mp3"

def test_generate_requires_text():
    reg = _reg(); reg.register("narr", "b64", "hi", "English")
    out = handle({"action": "generate", "voice_id": "narr-x"}, registry=reg)
    assert out["success"] is False and "text" in out["error"]

def test_list_and_delete():
    reg = _reg(); reg.register("narr", "b64", "hi", "English")
    assert handle({"action": "list_voices"}, registry=reg)["voices"][0]["voice_id"] == "narr-x"
    d = handle({"action": "delete_voice", "voice_id": "narr-x"}, registry=reg)
    assert d["success"] and d["deleted"] == "narr-x"

def test_unknown_action():
    out = handle({"action": "frobnicate"}, registry=_reg())
    assert out["success"] is False and "frobnicate" in out["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_actions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'actions'`

- [ ] **Step 3: Write minimal implementation**

`actions.py`:
```python
"""Route a RunPod job input to a voice-cloning operation and return a JSON envelope."""
import config
from inference import get_model, synthesize
from registry import VoiceRegistry

_REGISTRY = None


def get_registry() -> VoiceRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = VoiceRegistry(root=config.VOICE_DIR, model_getter=get_model)
    return _REGISTRY


def _require(d: dict, *keys):
    missing = [k for k in keys if not d.get(k)]
    if missing:
        raise ValueError(f"Missing required parameter(s): {', '.join(missing)}")


def handle(job_input: dict, registry: VoiceRegistry = None) -> dict:
    registry = registry or get_registry()
    action = (job_input or {}).get("action", "generate")
    try:
        if action == "register_voice":
            _require(job_input, "name", "ref_audio", "ref_text")
            res = registry.register(
                name=job_input["name"], ref_audio=job_input["ref_audio"],
                ref_text=job_input["ref_text"], language=job_input.get("language", "Auto"))
            return {"success": True, **res}

        if action == "generate":
            _require(job_input, "voice_id", "text")
            try:
                prompt = registry.load_prompt(job_input["voice_id"])
            except KeyError:
                return {"success": False, "error": f"Unknown voice_id: {job_input['voice_id']}"}
            res = synthesize(
                prompt, text=job_input["text"], language=job_input.get("language", "Auto"),
                seed=int(job_input.get("seed", 42)), return_srt=bool(job_input.get("return_srt", False)),
                response_format=job_input.get("response_format", config.DEFAULT_FORMAT))
            return {"success": True, **res}

        if action == "list_voices":
            return {"success": True, "voices": registry.list_voices()}

        if action == "delete_voice":
            _require(job_input, "voice_id")
            ok = registry.delete(job_input["voice_id"])
            return {"success": ok, "deleted": job_input["voice_id"] if ok else None,
                    **({} if ok else {"error": f"Unknown voice_id: {job_input['voice_id']}"})}

        return {"success": False, "error": f"Unknown action: {action}"}
    except Exception as e:  # never let the handler crash — return an envelope
        return {"success": False, "error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_actions.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add actions.py tests/test_actions.py
git commit -m "feat: action router for register/generate/list/delete

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: RunPod handler (`handler.py`)

**Files:**
- Create: `handler.py`
- Test: `tests/test_handler.py`

**Interfaces:**
- Consumes: `actions.handle`.
- Produces: `handler(job: dict) -> dict` — extracts `job["input"]`, delegates to `actions.handle`, wraps unexpected errors.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_handler.py
import handler as h

def test_handler_delegates(monkeypatch):
    seen = {}
    monkeypatch.setattr(h, "handle", lambda job_input: seen.setdefault("in", job_input) or {"success": True, "ok": 1})
    out = h.handler({"input": {"action": "list_voices"}})
    assert out == {"success": True, "ok": 1}
    assert seen["in"] == {"action": "list_voices"}

def test_handler_missing_input():
    out = h.handler({})
    assert out["success"] is False and "input" in out["error"]

def test_handler_wraps_exceptions(monkeypatch):
    def boom(_): raise RuntimeError("kaboom")
    monkeypatch.setattr(h, "handle", boom)
    out = h.handler({"input": {"action": "generate"}})
    assert out["success"] is False and "kaboom" in out["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_handler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'handler'`

- [ ] **Step 3: Write minimal implementation**

`handler.py`:
```python
"""RunPod serverless entry point for the Qwen3-TTS voice-cloning API."""
from actions import handle


def handler(job: dict) -> dict:
    job_input = (job or {}).get("input")
    if job_input is None:
        return {"success": False, "error": "Missing 'input' in job payload."}
    try:
        return handle(job_input)
    except Exception as e:  # defensive: handle() already envelopes, but never crash the worker
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_handler.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest -q`
Expected: all tests pass.

```bash
git add handler.py tests/test_handler.py
git commit -m "feat: RunPod serverless handler entry point

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Dockerfile, and image build script

**Files:**
- Create: `Dockerfile`, `build.sh`
- (Uses `requirements.txt` from Task 1.)

**Interfaces:** produces a container whose `CMD` runs `python -u handler.py`.

- [ ] **Step 1: Write the Dockerfile**

`Dockerfile`:
```dockerfile
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
```

- [ ] **Step 2: Write the build script**

`build.sh`:
```bash
#!/bin/bash
set -e
GITHUB_USERNAME="${GITHUB_USERNAME:-arkodeepsen}"
IMAGE_NAME="qwen3-tts"
REGISTRY="ghcr.io/${GITHUB_USERNAME}"
TAG="${1:-latest}"

echo "Building ${REGISTRY}/${IMAGE_NAME}:${TAG}..."
docker build -f Dockerfile -t "${IMAGE_NAME}:${TAG}" -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" .

echo "✅ Built. To push:"
echo "  docker login ghcr.io -u ${GITHUB_USERNAME}"
echo "  docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}"
```

- [ ] **Step 3: Verify the Dockerfile builds (manual / CI — requires Docker + network)**

Run (where Docker is available): `bash build.sh test`
Expected: image builds; the flash-attn line either succeeds or prints the fallback message without failing the build.
(If Docker is not available in this environment, this verification is performed by the GitHub Actions workflow in Task 13 or on the deploy host. Note this in the commit message.)

- [ ] **Step 4: Commit**

```bash
chmod +x build.sh
git add Dockerfile build.sh
git commit -m "build: Dockerfile and image build script (ffmpeg, optional flash-attn)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: RunPod deploy config (`runpod.toml`, `.runpod/hub.json`, `.runpod/tests.json`)

**Files:**
- Create: `runpod.toml`, `.runpod/hub.json`, `.runpod/tests.json`
- Test: `tests/test_deploy_config.py`

**Interfaces:** consumed by RunPod at deploy time. The test only asserts the files parse and contain the scale-to-zero + volume settings.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_config.py
import json, tomllib, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_runpod_toml_scale_to_zero():
    cfg = tomllib.loads((ROOT / "runpod.toml").read_text())["runpod"]
    assert cfg["min_workers"] == 0
    assert cfg["idle_timeout"] == 5
    assert cfg["flashboot"] is True
    assert cfg["volume_gb"] >= 20
    assert cfg["execution_timeout"] == 300

def test_hub_json_serverless_audio():
    hub = json.loads((ROOT / ".runpod" / "hub.json").read_text())
    assert hub["type"] == "serverless"
    assert hub["config"]["env"] and any(e["key"] == "VOICE_DIR" for e in hub["config"]["env"])

def test_tests_json_has_register_and_generate():
    t = json.loads((ROOT / ".runpod" / "tests.json").read_text())
    actions = [c["input"]["action"] for c in t["tests"]]
    assert "register_voice" in actions and "generate" in actions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deploy_config.py -v`
Expected: FAIL — `FileNotFoundError` for `runpod.toml`.

- [ ] **Step 3: Create the config files**

`runpod.toml`:
```toml
[runpod]
name = "qwen3-tts-voice-clone"
dockerfile = "Dockerfile"

# 16-24GB class GPUs (1.7B model, ~5GB weights)
gpu_types = ["NVIDIA GeForce RTX 4090", "NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA L40S"]
min_vram_gb = 16

container_disk_gb = 10
volume_gb = 20              # ~5GB model + voice registry

execution_timeout = 300     # caps runaway/hung jobs
min_workers = 0             # NO always-on workers -> $0 at rest
max_workers = 3
idle_timeout = 5            # billed warm tail after each job; keep low to save $
flashboot = true            # fast cold starts so a low idle_timeout is cheap
```

`.runpod/hub.json`:
```json
{
  "title": "Qwen3-TTS Voice Clone",
  "description": "Serverless voice cloning with Qwen3-TTS-12Hz-1.7B-Base. Register a voice once, then synthesize arbitrary-length speech (with optional SRT) by voice_id.",
  "type": "serverless",
  "category": "audio",
  "iconUrl": "https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/685b40411e38197aff6351df_poddy-3.webp",
  "config": {
    "runsOn": "GPU",
    "containerDiskInGb": 10,
    "env": [
      { "key": "HF_HOME", "value": "/runpod-volume" },
      { "key": "HF_HUB_CACHE", "value": "/runpod-volume" },
      { "key": "VOICE_DIR", "value": "/runpod-volume/voices" }
    ]
  }
}
```

`.runpod/tests.json`:
```json
{
  "tests": [
    {
      "name": "register_voice",
      "input": {
        "action": "register_voice",
        "name": "smoke_voice",
        "ref_audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        "ref_text": "Okay. Yeah. I resent you. I love you. I respect you. But you blew it!",
        "language": "English"
      }
    },
    {
      "name": "generate",
      "input": {
        "action": "generate",
        "voice_id": "REPLACE_WITH_REGISTERED_ID",
        "text": "This is a smoke test of the cloned voice.",
        "language": "English",
        "response_format": "wav",
        "return_srt": true
      }
    }
  ],
  "config": { "gpuTypeId": "NVIDIA L4", "gpuCount": 1, "containerDiskInGb": 10, "volumeInGb": 20 }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_deploy_config.py -v`
Expected: PASS (3 passed). (Requires Python ≥3.11 for `tomllib`; if the dev env is 3.9/3.10, `pip install tomli` and swap the import — note in the test file.)

- [ ] **Step 5: Commit**

```bash
git add runpod.toml .runpod/hub.json .runpod/tests.json tests/test_deploy_config.py
git commit -m "deploy: RunPod serverless config with scale-to-zero and volume

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: CLI client (`client/cli.py`)

**Files:**
- Create: `client/cli.py`, `client/.env.example`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `build_register_payload(name, ref_audio_path, ref_text, language) -> dict` — reads the audio file and base64-encodes it into a `register_voice` payload.
  - `build_generate_payload(voice_id, text, language, response_format, return_srt, seed) -> dict`.
  - `poll_result(base_url, api_key, job) -> dict` — returns immediately on `COMPLETED`/direct output, else polls `/status` (free, per billing rules).
  - `main(argv=None)` — argparse: `register` / `generate` subcommands.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import base64
from client.cli import build_register_payload, build_generate_payload

def test_register_payload_encodes_audio(tmp_path):
    p = tmp_path / "ref.wav"; p.write_bytes(b"RIFFfake")
    payload = build_register_payload("Narr", str(p), "hello", "English")
    assert payload["input"]["action"] == "register_voice"
    assert base64.b64decode(payload["input"]["ref_audio"]) == b"RIFFfake"
    assert payload["input"]["name"] == "Narr"

def test_generate_payload_fields():
    payload = build_generate_payload("v-1", "Hi there.", "English", "mp3", True, 7)
    inp = payload["input"]
    assert inp["action"] == "generate" and inp["voice_id"] == "v-1"
    assert inp["response_format"] == "mp3" and inp["return_srt"] is True and inp["seed"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'client'` (add empty `client/__init__.py`).

- [ ] **Step 3: Write minimal implementation**

Create empty `client/__init__.py`, then `client/cli.py`:
```python
"""CLI client for the Qwen3-TTS RunPod endpoint (register + generate)."""
import argparse, base64, os, sys, time
import requests


def build_register_payload(name, ref_audio_path, ref_text, language):
    with open(ref_audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"input": {"action": "register_voice", "name": name, "ref_audio": b64,
                      "ref_text": ref_text, "language": language}}


def build_generate_payload(voice_id, text, language, response_format, return_srt, seed):
    return {"input": {"action": "generate", "voice_id": voice_id, "text": text,
                      "language": language, "response_format": response_format,
                      "return_srt": return_srt, "seed": seed}}


def poll_result(base_url, api_key, job):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if job.get("status") in ("IN_QUEUE", "IN_PROGRESS") and job.get("id"):
        status_url = f"{base_url.rsplit('/', 1)[0]}/status/{job['id']}"
        while True:
            job = requests.get(status_url, headers=headers, timeout=30).json()
            if job.get("status") in ("COMPLETED", "FAILED"):
                break
            time.sleep(2)  # /status polling is free — never re-submit to check progress
    return job.get("output", job)


def _post(url, api_key, payload):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(url, json=payload, headers=headers, timeout=300).json()


def main(argv=None):
    p = argparse.ArgumentParser(description="Qwen3-TTS RunPod client")
    p.add_argument("--url", default=(f"https://api.runpod.ai/v2/{os.getenv('ENDPOINT_ID')}/runsync"
                                     if os.getenv("ENDPOINT_ID") else None))
    p.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY", ""))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--name", required=True); r.add_argument("--ref-audio", required=True)
    r.add_argument("--ref-text", required=True); r.add_argument("--language", default="English")

    g = sub.add_parser("generate")
    g.add_argument("--voice-id", required=True); g.add_argument("--text", required=True)
    g.add_argument("--language", default="English"); g.add_argument("--format", default="wav")
    g.add_argument("--srt", action="store_true"); g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output", default="output")

    args = p.parse_args(argv)
    if not args.url:
        print("Set ENDPOINT_ID/--url", file=sys.stderr); return 2

    if args.cmd == "register":
        out = poll_result(args.url, args.api_key, _post(args.url, args.api_key,
                build_register_payload(args.name, args.ref_audio, args.ref_text, args.language)))
        print(out); return 0

    out = poll_result(args.url, args.api_key, _post(args.url, args.api_key,
            build_generate_payload(args.voice_id, args.text, args.language, args.format, args.srt, args.seed)))
    if out.get("success") and out.get("audio_base64"):
        with open(f"{args.output}.{out['format']}", "wb") as f:
            f.write(base64.b64decode(out["audio_base64"]))
        if out.get("srt"):
            open(f"{args.output}.srt", "w", encoding="utf-8").write(out["srt"])
        print(f"Saved {args.output}.{out['format']}")
    else:
        print("Error:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`client/.env.example`:
```
ENDPOINT_ID=your_runpod_endpoint_id
RUNPOD_API_KEY=your_runpod_api_key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add client/__init__.py client/cli.py client/.env.example tests/test_cli.py
git commit -m "feat: CLI client for register and generate

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Streamlit GUI client (`client/app.py`)

**Files:**
- Create: `client/app.py`
- Test: `tests/test_app_import.py`

**Interfaces:** consumes `client/cli.py` payload builders. GUI is verified by a smoke import + manual run (Streamlit UIs are not unit-tested).

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_app_import.py
import importlib.util, pathlib

def test_app_module_parses():
    path = pathlib.Path(__file__).resolve().parents[1] / "client" / "app.py"
    spec = importlib.util.spec_from_file_location("app_check", path)
    assert spec is not None
    compile(path.read_text(encoding="utf-8"), str(path), "exec")  # parses without syntax error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_import.py -v`
Expected: FAIL — `client/app.py` does not exist.

- [ ] **Step 3: Write the Streamlit app**

`client/app.py`:
```python
"""Streamlit GUI for the Qwen3-TTS RunPod endpoint: register a voice, then generate."""
import base64, os
import requests
import streamlit as st
from cli import build_generate_payload, build_register_payload, poll_result  # noqa: E402

st.set_page_config(page_title="Qwen3-TTS Voice Clone", layout="wide")

default_id = os.getenv("ENDPOINT_ID", "")
url = st.sidebar.text_input("RunPod runsync URL",
        value=(f"https://api.runpod.ai/v2/{default_id}/runsync" if default_id else ""))
api_key = st.sidebar.text_input("RunPod API Key", value=os.getenv("RUNPOD_API_KEY", ""), type="password")


def _post(payload):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(url, json=payload, headers=headers, timeout=300).json()


st.title("Qwen3-TTS Voice Clone")
tab_reg, tab_gen = st.tabs(["1 · Register voice", "2 · Generate"])

with tab_reg:
    name = st.text_input("Voice name", "client_narrator")
    ref_text = st.text_area("Reference transcript (exact words in the clip)")
    ref_file = st.file_uploader("Reference audio (wav/mp3)", type=["wav", "mp3", "flac"])
    language = st.selectbox("Language", ["English", "Chinese", "Japanese", "Korean", "German",
                                          "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"])
    if st.button("Register", type="primary") and ref_file and ref_text:
        payload = {"input": {"action": "register_voice", "name": name,
                             "ref_audio": base64.b64encode(ref_file.getvalue()).decode(),
                             "ref_text": ref_text, "language": language}}
        out = poll_result(url, api_key, _post(payload))
        if out.get("success"):
            st.success(f"Registered! voice_id = {out['voice_id']}")
            st.code(out["voice_id"])
        else:
            st.error(out)

with tab_gen:
    voice_id = st.text_input("voice_id (from step 1)")
    text = st.text_area("Text to synthesize", height=180)
    g_lang = st.selectbox("Language ", ["English", "Chinese", "Japanese", "Korean", "German",
                                         "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"], key="glang")
    fmt = st.selectbox("Format", ["wav", "mp3", "flac", "opus"])
    want_srt = st.checkbox("Return SRT subtitles")
    seed = st.number_input("Seed", value=42, step=1)
    if st.button("Generate", type="primary") and voice_id and text:
        payload = build_generate_payload(voice_id, text, g_lang, fmt, want_srt, int(seed))
        out = poll_result(url, api_key, _post(payload))
        if out.get("success") and out.get("audio_base64"):
            audio = base64.b64decode(out["audio_base64"])
            st.audio(audio, format=f"audio/{fmt}")
            st.download_button("Download audio", audio, f"output.{fmt}", mime=f"audio/{fmt}")
            if out.get("srt"):
                st.text_area("SRT", out["srt"], height=200)
                st.download_button("Download SRT", out["srt"], "output.srt", "text/plain")
        else:
            st.error(out)
```

> Streamlit runs from inside `client/`, so `from cli import ...` resolves. To run: `cd client && streamlit run app.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app_import.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add client/app.py tests/test_app_import.py
git commit -m "feat: Streamlit GUI client (register + generate)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: README, GitHub Actions build, and repo docs

**Files:**
- Create: `README.md`, `.github/workflows/build.yml`
- Test: `tests/test_ci_workflow.py`

**Interfaces:** documentation + CI. The test asserts the workflow YAML parses and targets ghcr.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ci_workflow.py
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_workflow_parses_and_targets_ghcr():
    try:
        import yaml
    except ImportError:
        import pytest; pytest.skip("pyyaml not installed")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "build.yml").read_text())
    assert "jobs" in wf
    assert "ghcr.io" in (ROOT / ".github" / "workflows" / "build.yml").read_text()

def test_readme_has_api_and_billing_sections():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ["register_voice", "generate", "voice_id", "scale to zero", "/status"]:
        assert needle in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ci_workflow.py -v`
Expected: FAIL — `README.md` / workflow missing.

- [ ] **Step 3: Create the workflow and README**

`.github/workflows/build.yml`:
```yaml
name: build-and-push
on:
  push:
    branches: [main, master]
    tags: ["v*"]
  workflow_dispatch:
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/qwen3-tts:latest
```

`README.md`: document the project. Must include (checked by the test) the actions `register_voice`, `generate`, the `voice_id` flow, the phrase "scale to zero", and the `/status` billing note. Structure it like the `kokoro` README: Features, Model Specs, API Usage (JSON for each action), Parameters tables, Output Format, Python + cURL examples, Deployment (network volume + scale-to-zero), a **Cost & Billing** section (only `/run`/`/runsync` bill; `/status` polling is free; never health-check via `/run`), Performance (~1.56× RTF), License (Apache-2.0), Credits (QwenLM/Qwen3-TTS).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ci_workflow.py -v`
Expected: PASS (2 passed, or 1 passed + 1 skipped if pyyaml absent)

- [ ] **Step 5: Commit**

```bash
git add README.md .github/workflows/build.yml tests/test_ci_workflow.py
git commit -m "docs: README and CI build/push workflow

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: GPU integration smoke test + final verification

**Files:**
- Create: `scripts/smoke_test.py`

**Interfaces:** an end-to-end manual test run against a deployed endpoint (or a GPU box). Not part of the CI unit suite (needs the real model + GPU).

- [ ] **Step 1: Write the smoke script**

`scripts/smoke_test.py`:
```python
"""End-to-end smoke test against a deployed RunPod endpoint.

Usage:
  ENDPOINT_ID=... RUNPOD_API_KEY=... python scripts/smoke_test.py
Registers a voice from the public sample clip, then generates speech + SRT.
"""
import base64, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))
import requests
from cli import poll_result

URL = f"https://api.runpod.ai/v2/{os.environ['ENDPOINT_ID']}/runsync"
KEY = os.environ["RUNPOD_API_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
REF = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you blew it!"

reg = poll_result(URL, KEY, requests.post(URL, headers=H, json={"input": {
    "action": "register_voice", "name": "smoke", "ref_audio": REF,
    "ref_text": REF_TEXT, "language": "English"}}).json())
print("register ->", reg)
vid = reg["voice_id"]

gen = poll_result(URL, KEY, requests.post(URL, headers=H, json={"input": {
    "action": "generate", "voice_id": vid,
    "text": "This is a cloned voice speaking a long-form sentence. And a second one, for SRT.",
    "language": "English", "response_format": "mp3", "return_srt": True}}).json())
assert gen["success"] and gen["audio_base64"], gen
open("smoke.mp3", "wb").write(base64.b64decode(gen["audio_base64"]))
open("smoke.srt", "w", encoding="utf-8").write(gen["srt"])
print(f"OK: {gen['duration_sec']}s, {gen['chunks']} chunks, sr={gen['sample_rate']} -> smoke.mp3 + smoke.srt")
```

- [ ] **Step 2: Run the full unit suite one last time**

Run: `pytest -q`
Expected: all tests pass (config, chunking, srt, audio, registry, inference, actions, handler, deploy_config, cli, app_import, ci_workflow).

- [ ] **Step 3: Manual deploy verification (documented, run by the operator)**

1. Build & push: `bash build.sh latest && docker push ghcr.io/arkodeepsen/qwen3-tts:latest`
2. Create a RunPod serverless endpoint from the image; attach a ≥20 GB network volume; import `runpod.toml` settings (min_workers=0, idle_timeout=5, flashboot).
3. Run `python scripts/smoke_test.py` with `ENDPOINT_ID`/`RUNPOD_API_KEY` set → expect `smoke.mp3` + `smoke.srt`.
4. In the RunPod dashboard, confirm: after `scale_down_delay`, workers drop to **0** ($0 idle); polling `/status` during/after the job does **not** change the running-worker count.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "test: end-to-end GPU smoke test + deploy verification

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Serverless handler + scale-to-zero → Tasks 8, 10. ✓
- Register-once/reuse-by-id on network volume (atomic, sanitized, ref.wav truth, prompt.pt cache, LRU) → Task 5. ✓
- Base variant voice cloning (`create_voice_clone_prompt` / `generate_voice_clone`) → Tasks 5, 6. ✓
- Long-form sentence chunking + same-seed timbre → Tasks 2, 6. ✓
- Sentence-level SRT from chunk durations → Tasks 3, 6. ✓
- Audio formats wav/mp3/flac/opus + base64 → Task 4. ✓
- API contract (register/generate/list/delete + error envelope) → Task 7. ✓
- Docker + deps (transformers/accelerate pins, flash-attn fallback, ffmpeg) → Task 9. ✓
- runpod.toml/hub.json/tests.json → Task 10. ✓
- Client CLI + Streamlit → Tasks 11, 12. ✓
- README (with billing safety) + CI → Task 13. ✓
- Billing safety (status polling free) → documented in Tasks 11 (poll, not resubmit), 13 (README), 14 (dashboard check). ✓
- Success criteria (idle→0, status doesn't scale, register→generate) → Task 14. ✓

**2. Placeholder scan:** No "TBD"/"implement later"/"add error handling" left. The only intentional literal placeholder is `"REPLACE_WITH_REGISTERED_ID"` in `.runpod/tests.json`, which is inherent to a two-step register→generate hub test and is documented as such.

**3. Type consistency:**
- `VoiceRegistry(root, model_getter, cache_size)` — same signature in Task 5 impl, Task 7 (`VoiceRegistry(root=config.VOICE_DIR, model_getter=get_model)`), and tests. ✓
- `load_prompt(voice_id) -> list[VoiceClonePromptItem]` — produced in Task 5, consumed in Task 7 → passed as `prompt` to `synthesize` in Task 6. ✓
- `synthesize(prompt_items, text, language, seed, return_srt, response_format, model)` — defined Task 6, called with matching kwargs in Task 7. ✓
- `handle(job_input, registry=None)` — defined Task 7, called in Task 8 as `handle(job_input)`. ✓
- Response keys (`audio_base64`, `format`, `sample_rate`, `duration_sec`, `chunks`, `size_bytes`, `srt`, `segments`) — produced in Task 6, surfaced unchanged through Tasks 7–8 and consumed by clients in Tasks 11–12. ✓
- `build_generate_payload` / `build_register_payload` / `poll_result` — defined Task 11, imported by Task 12 and Task 14. ✓

No inconsistencies found.
