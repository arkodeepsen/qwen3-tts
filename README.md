# Qwen3-TTS Voice Cloning — Serverless RunPod Worker

[![RunPod](https://api.runpod.io/badge/arkodeepsen/qwen3-tts)](https://console.runpod.io/hub/arkodeepsen/qwen3-tts)

Deploy in one click from the [RunPod Hub](https://console.runpod.io/hub/arkodeepsen/qwen3-tts), or build the image yourself (below).

A [RunPod Serverless](https://www.runpod.io/serverless-gpu) worker that wraps
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) voice cloning: register a
speaker from a short reference clip, then generate speech in that voice from
arbitrary text — with automatic sentence-level chunking, consistent timbre
across chunks, multi-format audio encoding, and optional SRT subtitle output.

The worker scales to zero when idle. You only pay while a job is actually
running.

## Features

- **Voice cloning from a short reference clip** — register a speaker once
  (`register_voice`), then reuse the `voice_id` for unlimited `generate`
  calls without re-uploading reference audio.
- **Persistent voice registry** on a RunPod network volume
  (`/runpod-volume/voices/<voice_id>/`) — voices survive worker restarts and
  cold starts.
- **Long-text chunking** — input text is split into sentences (each
  generation unit ≤ 200 characters) and re-stitched with a fixed inter-chunk
  gap, so arbitrarily long scripts synthesize with consistent timbre (a
  single fixed seed is re-applied before every chunk).
- **Multi-format output** — `wav`, `mp3`, `flac`, `opus`, returned as
  base64 in the JSON response by default.
- **Optional SRT subtitles** — request `return_srt: true` to get
  per-sentence timed subtitle segments alongside the audio.
- **Optional object storage (S3)** — set `output: "url"` on `generate` to get
  a URL back instead of base64, and use the new `merge` action to stitch
  already-generated parts into one file. Powers the long-form client's
  `--to-url` mode: **one URL for hours of audio**. See
  [Object storage (S3)](#object-storage-s3).
- **Scale-to-zero billing** — `min_workers=0`, aggressive `idle_timeout`,
  and `flashboot` for fast cold starts, so the endpoint costs nothing at
  rest.

## Model Specs

| | |
|---|---|
| Model | [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |
| Task | Zero-shot / few-shot voice cloning text-to-speech |
| Precision | `bfloat16` |
| Attention | `flash_attention_2` (falls back to `sdpa` if unavailable) |
| Device | Single CUDA GPU (`cuda:0`) |
| Min VRAM | 16 GB (weights ≈ 5 GB) |
| Recommended GPUs | RTX 4090, L4, RTX A5000, RTX A4000, L40S |

## API Usage

> **Plugging this into another project?** See **[API.md](./API.md)** — a
> language-agnostic HTTP integration guide with the production async
> `/run` → poll `/status` pattern and copy-paste **cURL / Python / Node·TypeScript**
> clients. The section below is a quick reference.

The worker exposes a single RunPod job endpoint that dispatches on an
`"action"` field in the job input: `register_voice`, `generate`, `merge`,
`list_voices`, `delete_voice`.

Every response is a JSON envelope: `{"success": true, ...}` on success or
`{"success": false, "error": "<message>"}` on failure.

### 1. `register_voice`

Registers a new voice from a short reference clip + transcript and returns a
`voice_id` you reuse for all future `generate` calls.

Request:

```json
{
  "input": {
    "action": "register_voice",
    "name": "narrator",
    "ref_audio": "<base64-encoded WAV/FLAC/MP3 bytes, data URI, or https:// URL>",
    "ref_text": "This is the exact transcript of the reference clip.",
    "language": "Auto"
  }
}
```

Response:

```json
{
  "success": true,
  "voice_id": "narrator-a1b2c3",
  "sample_rate": 24000,
  "name": "narrator",
  "language": "Auto"
}
```

### 2. `generate`

Synthesizes speech in a previously registered voice.

Request:

```json
{
  "input": {
    "action": "generate",
    "voice_id": "narrator-a1b2c3",
    "text": "Hello there! This is a longer script that will be split into sentence-level chunks and re-stitched into one continuous clip.",
    "language": "Auto",
    "seed": 42,
    "return_srt": false,
    "response_format": "wav"
  }
}
```

Optional per-request tuning params (override the worker's env defaults — see
[Generation tuning & stability](#generation-tuning--stability)):

```json
{
  "input": {
    "action": "generate",
    "voice_id": "narrator-a1b2c3",
    "text": "Hello there!",
    "temperature": 0.7,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1024
  }
}
```

Response:

```json
{
  "success": true,
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "format": "wav",
  "sample_rate": 24000,
  "duration_sec": 4.816,
  "chunks": 2,
  "size_bytes": 231044,
  "srt": null,
  "segments": null
}
```

With `"return_srt": true`, `srt` and `segments` are populated (one segment
per sentence, so each has a precisely measured duration):

```json
{
  "success": true,
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "format": "wav",
  "sample_rate": 24000,
  "duration_sec": 4.816,
  "chunks": 2,
  "size_bytes": 231044,
  "srt": "1\n00:00:00,000 --> 00:00:02,340\nHello there!\n\n2\n00:00:02,490 --> 00:00:04,816\nThis is a longer script...\n",
  "segments": [
    {"index": 1, "start": 0.0, "end": 2.34, "text": "Hello there!"},
    {"index": 2, "start": 2.49, "end": 4.816, "text": "This is a longer script..."}
  ]
}
```

**Getting a URL instead of base64:** pass `"output": "url"` (requires
[object storage](#object-storage-s3) configured on the endpoint). The
response drops `audio_base64` and adds `url` + `key` — every other field
(`format`, `sample_rate`, `duration_sec`, `chunks`, `size_bytes`, `srt`,
`segments`) is unchanged:

```json
{
  "input": {
    "action": "generate",
    "voice_id": "narrator-a1b2c3",
    "text": "Hello there!",
    "output": "url",
    "key": "narrations/intro.wav"
  }
}
```

```json
{
  "success": true,
  "url": "https://<bucket>.<endpoint>/qwen3-tts/narrations/intro.wav?X-Amz-...",
  "key": "qwen3-tts/narrations/intro.wav",
  "format": "wav",
  "sample_rate": 24000,
  "duration_sec": 4.816,
  "chunks": 2,
  "size_bytes": 231044,
  "srt": null,
  "segments": null
}
```

`key` is optional — omit it and the worker auto-generates one
(`outputs/<uuid>.<format>`), namespaced under `S3_PREFIX` either way.

### 3. `merge`

Concatenates already-generated audio parts (identified by their S3 keys,
e.g. from `generate` calls with `output: "url"`) into **one** file and
returns a URL. This is a pure I/O job — no GPU involved — so it finishes fast
even when the parts add up to hours of audio. This is what powers the
long-form client's [`--to-url` mode](#long-form-audio-audiobooks--hours).

Request:

```json
{
  "input": {
    "action": "merge",
    "keys": ["qwen3-tts/sessions/abc123/part-00000.wav", "qwen3-tts/sessions/abc123/part-00001.wav"],
    "response_format": "mp3",
    "gap_sec": 0.15,
    "output_key": "sessions/abc123/audiobook.mp3"
  }
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `keys` | yes | — | S3 object keys of the parts, in the order to concatenate them. |
| `response_format` | no | `"wav"` | `wav`, `mp3`, `flac`, or `opus`. Parts are decoded and re-encoded to this format. |
| `gap_sec` | no | `0.15` | Silence inserted between parts, in seconds. |
| `output_key` | no | auto-generated | Object name for the merged file. |

Response:

```json
{
  "success": true,
  "url": "https://<bucket>.<endpoint>/qwen3-tts/sessions/abc123/audiobook.mp3?X-Amz-...",
  "key": "qwen3-tts/sessions/abc123/audiobook.mp3",
  "format": "mp3",
  "sample_rate": 24000,
  "duration_sec": 7230.4,
  "parts": 2,
  "size_bytes": 57874112
}
```

`merge` requires [object storage](#object-storage-s3) to be configured —
it needs to both download the input parts and upload the merged result.

### 4. `list_voices`

Lists every voice currently registered on the network volume.

Request:

```json
{
  "input": {
    "action": "list_voices"
  }
}
```

Response:

```json
{
  "success": true,
  "voices": [
    {"voice_id": "narrator-a1b2c3", "name": "narrator", "language": "Auto", "sample_rate": 24000, "created_at": "2026-07-01T12:00:00+00:00"},
    {"voice_id": "assistant-9f8e7d", "name": "assistant", "language": "en", "sample_rate": 22050, "created_at": "2026-07-01T12:05:00+00:00"}
  ]
}
```

### 5. `delete_voice`

Permanently removes a voice profile from the registry.

Request:

```json
{
  "input": {
    "action": "delete_voice",
    "voice_id": "narrator-a1b2c3"
  }
}
```

Response:

```json
{
  "success": true,
  "deleted": "narrator-a1b2c3"
}
```

If the `voice_id` is unknown, `success` is `false` and `deleted` is `null`:

```json
{
  "success": false,
  "deleted": null,
  "error": "Unknown voice_id: narrator-a1b2c3"
}
```

## Parameters

### `register_voice`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Human-readable voice name. Slugified into part of the generated `voice_id`. |
| `ref_audio` | string | yes | — | Reference clip: base64 audio, a `data:` URI, or a public `https://` URL. Local filesystem paths are rejected. |
| `ref_text` | string | yes | — | Exact transcript of `ref_audio`. |
| `language` | string | no | `"Auto"` | Language hint for prompt construction. |

### `generate`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `voice_id` | string | yes | — | `voice_id` returned by `register_voice`. |
| `text` | string | yes | — | Text to synthesize. Split into ≤ 200-character sentence chunks internally. |
| `language` | string | no | `"Auto"` | Language hint for synthesis. |
| `seed` | integer | no | `42` | Applied before every chunk (`torch.manual_seed`) so timbre stays consistent across a long script. |
| `return_srt` | boolean | no | `false` | If `true`, forces one generation unit per sentence and returns timed `srt`/`segments`. |
| `response_format` | string | no | `"wav"` | One of `wav`, `mp3`, `flac`, `opus`. |
| `temperature` | float | no | `0.8` | Sampling temperature. Higher = more varied/expressive, lower = more stable. |
| `top_p` | float | no | `0.9` | Nucleus sampling threshold. Lower = more stable. |
| `top_k` | integer | no | `50` | Top-k sampling cutoff. |
| `repetition_penalty` | float | no | `1.1` | Raise toward `1.3` if you hear looping/rambling. |
| `max_new_tokens` | integer | no | `1024` | Hard cap on tokens per sentence-chunk (safety against runaway generation). |
| `output` | string | no | `"base64"` | `"base64"` (inline) or `"url"` (upload to S3; response has `url`+`key` instead of `audio_base64`). Requires [object storage](#object-storage-s3). |
| `key` | string | no | auto-generated | Object name for the upload when `output: "url"`, namespaced under `S3_PREFIX`. |

All five tuning params are optional per-request overrides of the worker's
env-tunable defaults — omit them to use the operator defaults (see
[Generation tuning & stability](#generation-tuning--stability)).

### `merge`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `keys` | array of strings | yes | — | S3 object keys of the parts, in the order to concatenate. |
| `response_format` | string | no | `"wav"` | One of `wav`, `mp3`, `flac`, `opus`. |
| `gap_sec` | float | no | `0.15` | Silence inserted between parts, in seconds. |
| `output_key` | string | no | auto-generated | Object name for the merged file. |

Requires [object storage](#object-storage-s3) to be configured.

### `delete_voice`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `voice_id` | string | yes | — | Voice profile to delete. |

## Generation tuning & stability

The 1.7B-Base model can occasionally **ramble or fail to emit an
end-of-sequence token**, producing minutes of audio for what should be a
single short sentence. This is a known model quirk, not a bug in this worker
— see [QwenLM/Qwen3-TTS#239](https://github.com/QwenLM/Qwen3-TTS/issues/239).

Two known triggers:

- **Mismatched `ref_text`.** If the transcript passed to `register_voice`
  does not *exactly* match the words spoken in `ref_audio`, the ICL
  voice-clone conditioning is corrupted and the model is more likely to run
  away. **Always register with the precise transcript of the clip.**
- **Loose sampling.** High temperature / high top_p / low repetition penalty
  make runaway generation more likely. This worker ships **stability-focused
  defaults** (`repetition_penalty=1.1`, `top_p=0.9`, `temperature=0.8`)
  specifically to minimize this.

Reference audio is also trimmed to `REF_AUDIO_MAX_SEC` (30s by default) at
registration — long reference clips are another EOS-failure trigger.

**If you still hear rambling/looping on a particular voice or line:**

- Raise `repetition_penalty` (`1.1` → `1.3`).
- Lower `top_p` (`0.9` → `0.8`) and/or `temperature` (`0.8` → `0.7`).
- For more expressive/varied delivery (at some stability cost), raise
  `temperature` a little and watch for looping.

Tune these either **per-request** (the `generate` params above) or as
**operator defaults** via env vars, which apply to every request that
doesn't override them:

| Env var | Default | Purpose |
|---|---|---|
| `TOP_K` | `50` | Default `top_k` for all `generate` calls. |
| `TOP_P` | `0.9` | Default `top_p`. |
| `TEMPERATURE` | `0.8` | Default `temperature`. |
| `REPETITION_PENALTY` | `1.1` | Default `repetition_penalty`. |
| `MAX_NEW_TOKENS` | `1024` | Default `max_new_tokens` (per-chunk hard cap). |
| `REF_AUDIO_MAX_SEC` | `30` | Max reference-clip length kept at registration. |

Per-request `generate` params always override these env defaults.

## Object storage (S3)

**Optional.** The default `generate` behavior (base64 audio inline in the
response) works with **no storage configured at all**. Object storage is
only needed for two things:

- `generate` with `"output": "url"` — get a `url` back instead of base64.
- the `merge` action — stitch pre-generated parts into one file server-side.

Any S3-compatible provider works. **Cloudflare R2** is recommended
(S3-compatible API, 10 GB free tier, zero egress fees), but AWS S3, Backblaze
B2, MinIO, and RunPod S3 all work too. Configure it via environment variables
on the RunPod endpoint (Environment variables):

| Env var | Required? | Default | Purpose |
|---|---|---|---|
| `S3_BUCKET` | required (for URL/merge) | — | Bucket name. |
| `S3_ACCESS_KEY_ID` | required (for URL/merge) | — | Access key. |
| `S3_SECRET_ACCESS_KEY` | required (for URL/merge) | — | Secret key. |
| `S3_ENDPOINT_URL` | required for non-AWS | — | e.g. R2's `https://<acct>.r2.cloudflarestorage.com`. Blank for AWS S3. |
| `S3_REGION` | no | `auto` | Region passed to the S3 client. |
| `S3_PUBLIC_BASE_URL` | no | — | If set, returned URLs are public (`S3_PUBLIC_BASE_URL/<key>`). If unset, time-limited presigned URLs are returned instead. |
| `S3_URL_EXPIRY` | no | `604800` (7 days) | Presigned URL lifetime, in seconds. |
| `S3_PREFIX` | no | `qwen3-tts` | Key prefix all auto-generated/namespaced objects are stored under. |

Storage is considered configured once `S3_BUCKET`, `S3_ACCESS_KEY_ID`, and
`S3_SECRET_ACCESS_KEY` are all set. If `output: "url"` or `merge` is
requested without storage configured, the job returns the standard error
envelope (`{"success": false, "error": "S3 storage is not configured. ..."}`)
instead of failing silently.

## Long-form audio (audiobooks / hours)

A single serverless job **cannot** return hours of audio: it's bounded by the
per-job `execution_timeout` (~300s, see [Deployment](#deployment)) and by the
response payload size, since audio is returned as base64 inside the JSON
response. Long-form synthesis is therefore always **split into many blocks**,
generated as separate short jobs. `client/longform.py` supports two ways to
assemble the blocks into one output: **local merge** (default) and
**`--to-url`** (server-side merge, one URL — best for the longest jobs).

Register the voice **once** first with `register_voice` (or the `client/cli.py
register` subcommand), then pass its `voice_id` to `longform.py`. Total
compute time is roughly `audio length × real-time factor` (see
[Performance](#performance)) — e.g. a two-hour audiobook takes on the order
of two hours of billed GPU time, spread across many short parallel jobs
instead of one long-running one (the per-job `execution_timeout` wouldn't
allow the latter anyway).

### Default mode — local merge

```bash
python client/longform.py --voice-id <id> --input script.txt --output audiobook.wav \
  [--block-chars 1200] [--concurrency 2] [--format mp3]
```

It splits the script paragraph/sentence-aware, generates blocks in parallel
(bounded by the endpoint's `max_workers=3`), downloads each block's base64
audio, and concatenates them **on your machine** into one file. Fine for
shorter pieces; no S3 needed. It also accepts the tuning flags from the
section above (`--temperature`, `--top-p`, `--top-k`,
`--repetition-penalty`, `--max-new-tokens`).

### `--to-url` — one URL for hours of audio

```bash
python client/longform.py --voice-id <id> --input book.txt --format mp3 --to-url --concurrency 3
```

Requires [object storage](#object-storage-s3) configured on the endpoint.
This is the right way to turn a multi-hour script into a single downloadable
link: instead of downloading every block locally, each block is generated
with `output: "url"` so the worker uploads it to S3 and returns only a small
key (not audio) over the wire; then one `merge` call stitches all the parts
server-side and returns a single final URL.

**Why it works:** generation is split across many short GPU jobs, each
returning a tiny response — so no individual job risks the
`execution_timeout` or a large response payload. The `merge` step is a single
fast, non-GPU I/O job, so it isn't bound by generation time at all — it
finishes in seconds no matter how many hours of audio it's concatenating.
Neither the execution-timeout nor the response-size limit is ever hit,
regardless of how long the source text is.

Contrast with the default local-merge mode above: that mode downloads every
block's audio to your machine and concatenates locally — simple and needs no
S3 setup, and fine for shorter pieces, but not the right choice once you're
talking about hours of audio.

## Output Format

By default, all binary audio is returned as a base64 string in
`audio_base64`, alongside:

| Field | Type | Description |
|---|---|---|
| `audio_base64` | string | Base64-encoded audio bytes in the requested `format`. Present when `output` is `"base64"` (the default) or omitted. |
| `format` | string | Echo of the resolved output format (`wav`, `mp3`, `flac`, `opus`). |
| `sample_rate` | integer | Sample rate of the generated audio, in Hz. This is model-native — it is whatever the model emits, not a caller-selectable parameter. |
| `duration_sec` | float | Total duration of the stitched clip, in seconds. |
| `chunks` | integer | Number of generation units (sentences/packed groups) synthesized and concatenated. |
| `size_bytes` | integer | Size of the decoded (pre-base64) audio bytes. |
| `srt` | string \| null | SRT-formatted subtitle text when `return_srt: true`, else `null`. |
| `segments` | array \| null | Per-sentence `{index, start, end, text}` timing objects when `return_srt: true`, else `null`. |

When `"output": "url"` is set instead, `audio_base64` is replaced by `url`
(a public or time-limited presigned link, see
[Object storage](#object-storage-s3)) and `key` (the S3 object key) — every
other field above is unchanged.

## Python Example

```python
import base64
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# 1. Register a voice from a local reference clip
with open("reference.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode("utf-8")

reg = endpoint.run_sync({
    "action": "register_voice",
    "name": "narrator",
    "ref_audio": ref_b64,
    "ref_text": "This is the exact transcript of the reference clip.",
    "language": "Auto",
})
voice_id = reg["voice_id"]

# 2. Generate speech in that voice
result = endpoint.run_sync({
    "action": "generate",
    "voice_id": voice_id,
    "text": "Hello there! This is Qwen3-TTS speaking in a cloned voice.",
    "return_srt": True,
    "response_format": "mp3",
})

with open("output.mp3", "wb") as f:
    f.write(base64.b64decode(result["audio_base64"]))

print(result["duration_sec"], "seconds,", result["chunks"], "chunks")
print(result["srt"])
```

## cURL Example

```bash
# register_voice
curl -s -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "action": "register_voice",
          "name": "narrator",
          "ref_audio": "'"$(base64 -w0 reference.wav)"'",
          "ref_text": "This is the exact transcript of the reference clip.",
          "language": "Auto"
        }
      }'

# generate (using the voice_id returned above)
curl -s -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "action": "generate",
          "voice_id": "narrator-a1b2c3",
          "text": "Hello there! This is Qwen3-TTS speaking in a cloned voice.",
          "response_format": "wav"
        }
      }'
```

## Deployment

### Network volume

Voice profiles must persist across cold starts and across workers, so this
worker requires a RunPod **network volume** attached at `/runpod-volume`:

| Env var | Value | Purpose |
|---|---|---|
| `HF_HOME` | `/runpod-volume` | Hugging Face cache root, persisted on the volume. |
| `HF_HUB_CACHE` | `/runpod-volume` | Model weights cache, persisted on the volume (avoids re-downloading ~5 GB on every cold start). |
| `VOICE_DIR` | `/runpod-volume/voices` | Voice registry root — `meta.json`, `ref.wav`, `prompt.pt` per `voice_id`. |

Recommended volume size: **20 GB** (≈5 GB model weights + voice registry
headroom).

### Scale to zero

The endpoint is configured (see `runpod.toml`) to **scale to zero** when
idle, so it costs nothing between requests:

| Setting | Value | Effect |
|---|---|---|
| `min_workers` | `0` | No always-on workers — $0 at rest. |
| `idle_timeout` | `5` | Worker stays warm only 5s after the last job before spinning down. |
| `flashboot` | `true` | Fast cold starts, making a short `idle_timeout` practical. |
| `max_workers` | `3` | Caps concurrent scale-out. |
| `execution_timeout` | `300` | Caps runaway/hung jobs. |

## Cost & Billing

- Only **`/run`** (async) and **`/runsync`** (sync) invocations start a
  worker and bill GPU time. Every `generate`, `register_voice`,
  `list_voices`, and `delete_voice` call above goes through one of these
  two endpoints and is billable while the worker is active.
- **`/status`** polling (used to check the result of an async `/run` job)
  is **free** — it queries RunPod's control plane and **never wakes a
  worker**. Poll `/status` as often as you like while waiting on a job.
- **Never health-check via `/run` or `/runsync`.** Doing so bills a worker
  and defeats scale-to-zero. For liveness, use RunPod's endpoint-level
  `GET https://api.runpod.ai/v2/<endpoint_id>/health` — it reports queue and
  worker counts from the control plane and does **not** wake or bill a worker.
  This service's own handler exposes no HTTP route (it is a RunPod serverless
  worker), so all interaction is through the `/run`, `/runsync`, and `/status`
  job API.
- Because `min_workers=0`, the endpoint bills **$0 while idle** — cost is
  incurred only for the wall-clock duration of active `/run`/`/runsync`
  jobs, plus the short `idle_timeout` warm tail after each job.

## Performance

Benchmarked on a single 24 GB-class GPU (RTX 4090 / L40S) with
`flash_attention_2` and `bfloat16`: **~1.56× real-time factor (RTF)** —
i.e. generating 1 second of audio takes roughly 0.64 seconds of compute
(excluding cold start / model load time).

## License

[Apache License 2.0](./LICENSE).

## Credits

Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the
[QwenLM](https://github.com/QwenLM) team. Model weights:
[`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).
