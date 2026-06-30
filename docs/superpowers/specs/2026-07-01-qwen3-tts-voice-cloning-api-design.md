# Qwen3-TTS Voice-Cloning API — Design Spec

- **Date:** 2026-07-01
- **Status:** Approved (pending written-spec review)
- **Repo name:** `qwen3-tts`
- **Author:** Arkodeep Sen (with Claude)

## 1. Goal

Ship a production-ready **RunPod serverless** endpoint wrapping
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` for **voice cloning**, to be consumed as an
API by the client's existing internal website.

User experience is two-phase:

1. **Register a voice once** — upload reference audio + its transcript; the
   service computes and caches a reusable voice profile on the network volume
   and returns a `voice_id`.
2. **Generate speech any number of times** — send `voice_id` + text; get back
   base64 audio (and optionally an SRT subtitle track) in that cloned voice.

The repo mirrors the conventions of the author's existing RunPod repos
(`arkodeepsen/qwen-image` for the serverless pattern, `arkodeepsen/kokoro` for
client tooling and the JSON response contract).

## 2. Scope

### In scope
- RunPod serverless handler with **scale-to-zero** cost behavior.
- `Qwen3-TTS-12Hz-1.7B-Base` only (voice cloning via `generate_voice_clone`).
- Register-once / reuse-by-`voice_id` voice registry on the network volume.
- Arbitrary-length text via internal sentence chunking + concatenation.
- **Sentence-level** SRT export, derived for free from chunk durations.
- Audio output formats: `wav` (default), `mp3`, `flac`, `opus`.
- Client tooling: a CLI client and a Streamlit GUI client.
- Standard repo furniture: Dockerfile, `runpod.toml`, `.runpod/` hub config,
  `build.sh`, GitHub Actions build/push, README, LICENSE (Apache-2.0).

### Out of scope (explicitly not building now)
- CustomVoice (preset speakers) and VoiceDesign (text→voice) variants.
- Pod / always-on FastAPI deployment.
- Word-level SRT / forced alignment.
- Real-time streaming (WebSocket / first-packet streaming).
- Fine-tuning.
- Inline per-request cloning (every `generate` requires a registered `voice_id`).

## 3. Architecture overview

Single serverless worker process:

- `handler.py` — RunPod entry point. Routes on an `action` field in the job
  input (single-endpoint pattern, like the kokoro handler's `endpoint` field).
- `inference.py` — model loading (global singleton), the voice registry, the
  long-form chunking + generation pipeline, SRT building, and audio encoding.
- Model is loaded **once per worker** on first use (cold start) and reused for
  the worker's lifetime.
- The **network volume** (`/runpod-volume`) holds both the Hugging Face model
  cache and the persistent voice registry, shared across all workers.

## 4. API contract

All requests follow RunPod's `{ "input": { ... } }` envelope. The `action`
field selects the operation (default `generate`). All binary output is base64
in JSON. Errors return `{ "success": false, "error": "<message>" }`, matching
the kokoro contract.

### 4.1 `register_voice`
Compute and cache a reusable voice profile.

Request:
```json
{ "input": {
    "action": "register_voice",
    "name": "client_narrator",
    "ref_audio": "<base64 audio | https URL>",
    "ref_text": "exact transcript of the reference clip",
    "language": "English"
}}
```
Response:
```json
{ "success": true, "voice_id": "client_narrator-a1b2c3", "sample_rate": 24000 }
```
Notes:
- `ref_audio` accepts base64-encoded audio or an `https` URL (the `qwen_tts`
  API also accepts local paths / numpy tuples; the public API exposes base64
  and URL).
- `ref_text` is the verbatim transcript of the reference clip (required for
  best-quality cloning).
- `language` is one of the 10 supported languages or `"Auto"`.
- `sample_rate` in all responses is the model's native output rate (the `sr`
  returned by `qwen_tts`); `24000` in these examples is illustrative, not a
  hardcoded assumption — the implementation echoes whatever the model returns.

### 4.2 `generate` (default action)
Synthesize speech in a registered voice. Supports arbitrary-length `text`.

Request:
```json
{ "input": {
    "action": "generate",
    "voice_id": "client_narrator-a1b2c3",
    "text": "Any length of text. Chunked and concatenated internally.",
    "language": "English",
    "response_format": "mp3",
    "seed": 42,
    "return_srt": false
}}
```
Response:
```json
{ "success": true,
  "audio_base64": "...",
  "format": "mp3",
  "sample_rate": 24000,
  "duration_sec": 12.4,
  "chunks": 3,
  "size_bytes": 198321,
  "srt": null,
  "segments": null }
```
When `return_srt: true`, `srt` is a ready-to-save SRT string and `segments` is
`[{ "index": 1, "start": 0.0, "end": 4.2, "text": "..." }, ...]` at
**sentence** granularity.

### 4.3 `list_voices`
```json
{ "input": { "action": "list_voices" } }
```
```json
{ "success": true, "voices": [
    { "voice_id": "client_narrator-a1b2c3", "name": "client_narrator",
      "language": "English", "created_at": "2026-07-01T12:00:00Z" }
]}
```

### 4.4 `delete_voice`
```json
{ "input": { "action": "delete_voice", "voice_id": "client_narrator-a1b2c3" } }
```
```json
{ "success": true, "deleted": "client_narrator-a1b2c3" }
```

## 5. Voice registry (the "register once" core)

Stored on the network volume, shared across all workers:

```
/runpod-volume/voices/<voice_id>/
    meta.json     # {name, language, ref_text, sample_rate, created_at}
    ref.wav       # the decoded reference audio — source of truth
    prompt.pt     # cached precomputed voice_clone_prompt (best-effort optimization)
```

- **`voice_id`** = `slug(name)` + `-` + short uuid. It is regex-validated and
  sanitized; it is **never** interpolated raw into a filesystem path (prevents
  `../` traversal).
- **`register_voice`** decodes `ref_audio`, runs
  `model.create_voice_clone_prompt(ref_audio, ref_text)`, and writes the folder
  **atomically** (write to a temp dir, then `os.rename`) so concurrent workers
  never read a half-written profile.
- **`generate`** loads the profile with a robust fallback chain:
  1. In-process LRU cache keyed by `voice_id` (prompt already in VRAM) → use it.
  2. Else load `prompt.pt`, move tensors to the worker's device → cache it.
  3. Else (missing/incompatible `prompt.pt`) rebuild the prompt from `ref.wav` +
     `ref_text`, persist `prompt.pt`, cache it.
- `ref.wav` being the source of truth means the registry is always recoverable
  even if a cached `prompt.pt` is unusable on a different worker/version.

## 6. Generation pipeline

1. **Model load (once per worker):**
   ```python
   Qwen3TTSModel.from_pretrained(
       "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
       device_map="cuda:0", dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",  # fallback to "sdpa" if unavailable
   )
   ```
   Global singleton, mirroring the qwen-image `load_model()` pattern.
2. **Chunking:** split `text` into sentences on terminal punctuation
   (`. ! ?` and CJK `。！？` plus newlines), then pack sentences into chunks of
   ≤ ~200 characters. Rationale: the model is autoregressive; keeping chunks
   short preserves stability and quality on long-form input.
3. **Per-chunk synthesis:** call `generate_voice_clone(text=chunk, language,
   voice_clone_prompt=<cached prompt>)` for each chunk, using the **same fixed
   `seed`** across all chunks so timbre stays consistent. Concatenate the
   resulting waveforms with a short inter-chunk silence.
4. **SRT (sentence-level):** while concatenating, accumulate each chunk's exact
   audio duration to compute `start`/`end` timecodes per sentence. Build both
   the `segments` array and the formatted `srt` string. No extra model, no extra
   cost — the timings come directly from the audio we already generated.
5. **Encode:** write the final waveform to `response_format` (`wav` via
   `soundfile`; `mp3` / `flac` / `opus` via `ffmpeg`) and base64-encode.

## 7. Repo structure

```
qwen3-tts/
├── handler.py            # RunPod serverless entry (action router)
├── inference.py          # model load, voice registry, chunking, SRT, encode
├── Dockerfile
├── build.sh              # docker build + push to ghcr.io/arkodeepsen
├── requirements.txt
├── runpod.toml           # GPU / volume / scaling config
├── .runpod/
│   ├── hub.json          # RunPod Hub listing (type=serverless, category=audio)
│   └── tests.json        # register -> generate smoke test
├── client/
│   ├── cli.py            # CLI client (register + generate), polls /status
│   ├── app.py            # Streamlit GUI (upload ref -> register -> generate)
│   └── .env.example      # ENDPOINT_ID, RUNPOD_API_KEY
├── .github/workflows/
│   └── build.yml         # automated image build/push
├── README.md
└── LICENSE               # Apache-2.0
```

## 8. Docker + RunPod configuration

### Image
- **Base:** `runpod/pytorch` (cu128 / torch2.8 family, as in qwen-image). Must
  ship **Python 3.12** (a `qwen-tts` requirement) — verified at build time;
  fall back to a uv/conda 3.12 env if the chosen tag is not 3.12.
- **Python deps:** `qwen-tts`, `runpod`, `soundfile`,
  `huggingface_hub[hf_transfer]`, and `flash-attn` (optional, built with
  `MAX_JOBS=4`; the model falls back to `sdpa` attention if the wheel or GPU is
  incompatible, so the image always builds).
- **System deps:** `ffmpeg` (for mp3/opus/flac encoding), `git`.
- **Env:** `HF_HOME=/runpod-volume`, `HF_HUB_CACHE=/runpod-volume`,
  `VOICE_DIR=/runpod-volume/voices`.
- **CMD:** `python -u handler.py`.

### `runpod.toml`
```
gpu_types        = RTX 4090 / L4 / A5000 / RTX A4000 / L40S   # 16-24GB class
min_vram_gb      = 16
container_disk_gb = 10
volume_gb        = 20        # ~5GB model + voice registry
execution_timeout = 300      # caps runaway/hung jobs -> no infinite billing
min_workers      = 0         # NO always-on workers -> $0 at rest
max_workers      = 3         # burst fan-out
idle_timeout     = 5         # billed "warm" tail after each job; keep low to save $
flashboot        = true      # fast cold starts so a low idle_timeout is cheap
```
- `idle_timeout = 5` (down from 60) minimizes the billed warm tail after each
  job. FlashBoot keeps the resulting cold starts fast/cheap. If the site shows
  excessive cold starts under bursty load, this is the one knob to raise
  (trade a little idle billing for fewer cold starts).

### Cost controls & billing safety (explicit)

RunPod bills **only GPU worker execution time**, "from when a worker starts
until it stops," rounded up to the second. A worker only **starts** when a job
is **enqueued** — i.e. via **`/run`** or **`/runsync`**.

- **`/status`, `/health`, `/stream`, `/cancel` are control-plane / queue
  operations. They do NOT start a worker and incur NO GPU billing.** Polling
  `/status` for a job is free and safe at any frequency.
- **Only `/run` and `/runsync` spin up (and bill) workers.**

Client contract (enforced in `client/` and documented in the README) to prevent
accidental spend:
1. Submit each generation **once** via `/run`, then **poll `/status`** until
   `COMPLETED` / `FAILED`. Never re-submit a job to "check progress."
2. Any website uptime/health check hits **`/health`**, never `/run`/`/runsync`.
3. `min_workers = 0` — no always-on (active) workers are ever provisioned.
4. `execution_timeout = 300` bounds a hung job's maximum billable time.

### Cost / latency behavior (scale-to-zero)
- Idle → all workers terminated → **$0/hr while no requests**.
- New request after idle → cold start (model loads from the 5 GB network-volume
  cache; FlashBoot accelerates subsequent starts), then warm.
- Generation is asynchronous; clients poll `/status` (free) until done.

## 9. Client tooling

- **`client/cli.py`** — `register` and `generate` subcommands; reads
  `ENDPOINT_ID` / `RUNPOD_API_KEY` from `.env`; submits to `/runsync` and polls
  `/status`; saves audio (and `.srt` when requested). Modeled on the kokoro
  `inference.py`.
- **`client/app.py`** — Streamlit GUI: one tab to upload/record reference audio
  + transcript and register a voice; one tab to pick a `voice_id`, type text,
  generate, play inline, and download audio + SRT. Modeled on the kokoro
  `app.py`.

## 10. Risks & verification items

1. **`voice_clone_prompt` serializability** — confirm `torch.save`/`torch.load`
   round-trips the prompt object across workers/versions. *Mitigation already in
   design:* `ref.wav` is the source of truth, so `prompt.pt` is best-effort and
   always rebuildable.
2. **Python 3.12 base image** — confirm the chosen `runpod/pytorch` tag ships
   3.12; otherwise provision a 3.12 env via uv/conda.
3. **flash-attn build** — heavy and environment-sensitive; kept optional with an
   `sdpa` fallback so the image always builds.
4. **Sentence segmentation across 10 languages** — a punctuation + length
   heuristic, not full NLP segmentation. Documented as a known simplification.
5. **Cold-start RTF** — generation runs at ~1.56× real-time on a 4080-class GPU
   (a short sentence ≈ a few seconds warm). Acceptable for an internal site that
   treats generation as async.

## 11. Success criteria

- A request with `action: register_voice` returns a `voice_id`, and the profile
  persists on the network volume across worker restarts.
- A subsequent `action: generate` with only `voice_id` + `text` returns audio in
  the registered voice — no reference audio resent.
- Long text (multiple paragraphs) returns a single concatenated audio file with
  consistent timbre across chunks.
- `return_srt: true` returns a valid, correctly-timed sentence-level SRT string.
- With no traffic, the endpoint scales to zero workers (verified $0 idle in the
  RunPod dashboard); a new request spins a worker up automatically.
- Polling `/status` while a job runs (and while idle) does **not** spin up extra
  workers or change the billed worker count (verified in the dashboard); only
  `/run` / `/runsync` start workers.
- The image builds and deploys via `build.sh` / GitHub Actions to
  `ghcr.io/arkodeepsen`.

## 12. Future / possible extensions (not now)

- Optional pod / FastAPI deployment for steady-traffic or streaming use.
- CustomVoice (preset speakers + emotion) and VoiceDesign (text→voice).
- Word-level SRT via forced alignment.
- Inline per-request cloning (ref audio in the `generate` call).
