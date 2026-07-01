# Qwen3-TTS Voice Cloning — HTTP API

Integration guide for calling the deployed endpoint from **any** language or
framework. It talks plain HTTP + JSON over RunPod's Serverless job API — no SDK
required.

- **Base URL:** `https://api.runpod.ai/v2/<ENDPOINT_ID>`
- **Auth:** header `Authorization: Bearer <RUNPOD_API_KEY>`
- **Content type:** `application/json`
- **Body shape (always):** `{ "input": { "action": "...", ... } }`

`<ENDPOINT_ID>` and `<RUNPOD_API_KEY>` come from your RunPod console (Serverless
endpoint + Settings → API Keys).

---

## 1. How a call works (the RunPod job model)

Every request is a **job**. There are two ways to submit one:

| Method | URL | Behavior | Use when |
|---|---|---|---|
| **Synchronous** | `POST /runsync` | Blocks and returns the result if the job finishes within the sync window; otherwise returns a job `id` to poll. | Quick calls, scripts, testing. |
| **Asynchronous** | `POST /run` | Returns a job `id` immediately. You then poll `GET /status/<id>` until it's done. | **Production** / long text / anything that might take more than a few seconds. |

**Polling `/status` is free** — it reads RunPod's queue and never wakes or bills
a GPU worker. Only `/run` and `/runsync` start (and bill) a worker. Never poll by
re-submitting the job.

### Job lifecycle

`/run` → `{ "id": "...", "status": "IN_QUEUE" }`, then `GET /status/<id>` returns
one of:

| `status` | Meaning |
|---|---|
| `IN_QUEUE` | Waiting for a worker (cold start may be spinning up). |
| `IN_PROGRESS` | Running. |
| `COMPLETED` | Done — read `output`. |
| `FAILED` | Worker error — read `error`. |

The worker's own JSON envelope is nested under **`output`**:

```json
{
  "id": "abc-123",
  "status": "COMPLETED",
  "output": { "success": true, "audio_base64": "...", "format": "mp3", "...": "..." }
}
```

So the integration pattern is always: submit → poll until `status == "COMPLETED"`
→ read `output` → check `output.success`.

> **First call is slow.** With scale-to-zero, the first request after idle spins
> up a worker and (once ever) downloads the ~5 GB model to the network volume.
> Budget a generous client timeout (e.g. 5 min) for the first `register_voice`
> or `generate`; subsequent warm calls are fast.

---

## 2. Actions

All actions are selected by `input.action`: `register_voice`, `generate`,
`merge`, `list_voices`, `delete_voice`. Every response is
`{"success": true, ...}` or `{"success": false, "error": "<message>"}`.

### `register_voice` → returns a `voice_id`

Clone a voice once from a reference clip; reuse the `voice_id` forever.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `action` | string | yes | — | `"register_voice"` |
| `name` | string | yes | — | Human name; slugified into the `voice_id`. |
| `ref_audio` | string | yes | — | Base64 audio bytes, a `data:` URI, **or** a public `https://` URL. Local file paths are rejected; URLs are SSRF-filtered (no private/loopback/metadata hosts). |
| `ref_text` | string | yes | — | Exact transcript of `ref_audio`. |
| `language` | string | no | `"Auto"` | Language hint. One of: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, or `Auto`. |

Response: `{ success, voice_id, sample_rate, name, language }`

### `generate` → returns audio (+ optional SRT)

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `action` | string | yes | — | `"generate"` |
| `voice_id` | string | yes | — | From `register_voice`. |
| `text` | string | yes | — | Any length — split into ≤200-char sentence chunks and re-stitched. |
| `language` | string | no | `"Auto"` | Language hint (see list above). |
| `response_format` | string | no | `"wav"` | `wav` \| `mp3` \| `flac` \| `opus`. |
| `return_srt` | boolean | no | `false` | If `true`, returns timed `srt` + `segments` (one per sentence). |
| `seed` | integer | no | `42` | Re-applied before every chunk for consistent timbre. |
| `temperature` | float | no | `0.8` | Sampling temperature; higher = more varied/expressive, lower = more stable. |
| `top_p` | float | no | `0.9` | Nucleus sampling; lower = more stable. |
| `top_k` | integer | no | `50` | Top-k sampling cutoff. |
| `repetition_penalty` | float | no | `1.1` | Raise toward `1.3` if you hear looping/rambling. |
| `max_new_tokens` | integer | no | `1024` | Hard cap on tokens per sentence-chunk (safety against runaway generation). |
| `output` | string | no | `"base64"` | `"base64"` (inline in the response) or `"url"` (upload to S3, response contains `url`+`key` instead). See [§3.5 Object storage (S3)](#35-object-storage-s3). |
| `key` | string | no | auto-generated | Only used when `output: "url"`. Object name for the upload, namespaced under `S3_PREFIX`. Omit to let the worker generate one (`outputs/<uuid>.<format>`). |

All five tuning params are optional per-request overrides of the worker's
env-tunable defaults — see [§3 Generation tuning & stability](#3-generation-tuning--stability).

Response fields (`output: "base64"`, the default):

| Field | Type | Description |
|---|---|---|
| `success` | bool | |
| `audio_base64` | string | Base64 of the audio bytes in `format`. Decode to get the file. |
| `format` | string | Echo of `response_format`. |
| `sample_rate` | int | Model-native output rate (Hz). |
| `duration_sec` | float | Total length of the stitched clip. |
| `chunks` | int | Number of sentence chunks synthesized. |
| `size_bytes` | int | Decoded (pre-base64) byte size. |
| `srt` | string \| null | SRT subtitle text when `return_srt: true`. |
| `segments` | array \| null | `[{index, start, end, text}]` when `return_srt: true`. |

Response fields (`output: "url"`) — identical except `audio_base64` is
replaced by `url` + `key`; `format`, `sample_rate`, `duration_sec`, `chunks`,
`size_bytes`, `srt`, `segments` are unchanged:

| Field | Type | Description |
|---|---|---|
| `success` | bool | |
| `url` | string | Public URL (if `S3_PUBLIC_BASE_URL` is set) or a time-limited presigned URL, good for `S3_URL_EXPIRY` seconds. |
| `key` | string | The S3 object key the audio was uploaded to (echoes `key` if you passed one). |
| `format` / `sample_rate` / `duration_sec` / `chunks` / `size_bytes` / `srt` / `segments` | — | Same as base64 mode. |

`output: "url"` requires S3 storage to be configured on the endpoint (§3.5).
If it isn't, the job returns the standard error envelope, e.g.
`{"success": false, "error": "S3 storage is not configured. Set S3_BUCKET, ..."}`.

### `merge` → concatenate uploaded parts into ONE file, returns a URL

Stitches audio parts that were previously generated with `output: "url"`
(identified by their S3 keys) into a single file and uploads the result. This
is a pure I/O job — no GPU, no model load — so it completes in seconds even
for parts totaling hours of audio. It's the mechanism behind the long-form
client's `--to-url` mode (§5).

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `action` | string | yes | — | `"merge"` |
| `keys` | array of strings | yes | — | S3 object keys of the parts, **in the order they should be concatenated**. |
| `response_format` | string | no | `"wav"` | `wav` \| `mp3` \| `flac` \| `opus`. Parts are decoded and re-encoded to this format. |
| `gap_sec` | float | no | `0.15` | Silence inserted between consecutive parts, in seconds. |
| `output_key` | string | no | auto-generated | Object name for the merged file, namespaced under `S3_PREFIX`. Omit to let the worker generate one. |

Response: `{ success, url, key, format, sample_rate, duration_sec, parts, size_bytes }`

- `url` / `key` — location of the merged file (same semantics as `generate`'s
  `output: "url"` response).
- `parts` — number of keys merged.
- `sample_rate` / `duration_sec` / `size_bytes` — describe the merged output.

`merge` requires S3 storage to be configured (§3.5) — both to download the
input parts and to upload the merged result.

### `list_voices` → all registered voices

Input: `{ "action": "list_voices" }` →
`{ success, voices: [{ voice_id, name, language, sample_rate, created_at }] }`

### `delete_voice`

Input: `{ "action": "delete_voice", "voice_id": "..." }` →
`{ success, deleted }` (`deleted` is `null` + `error` set if the id is unknown).

---

## 3. Generation tuning & stability

The 1.7B-Base model can occasionally **ramble or fail to emit an
end-of-sequence token**, producing minutes of audio for what should be a
single short sentence. This is a documented model quirk — see
[QwenLM/Qwen3-TTS#239](https://github.com/QwenLM/Qwen3-TTS/issues/239).

Two known triggers:

- **Mismatched `ref_text`.** A transcript passed to `register_voice` that
  does not *exactly* match the words spoken in `ref_audio` corrupts the ICL
  voice-clone conditioning and makes runaway generation more likely. **Always
  register with the precise transcript of the clip.**
- **Loose sampling.** High `temperature` / high `top_p` / low
  `repetition_penalty` make runaway generation more likely, which is why this
  worker ships stability-focused defaults (`repetition_penalty=1.1`,
  `top_p=0.9`, `temperature=0.8`).

Reference audio is also trimmed to `REF_AUDIO_MAX_SEC` (30s by default) at
registration — long reference clips are another EOS-failure trigger.

**If a particular voice or line still rambles/loops:**

- Raise `repetition_penalty` (`1.1` → `1.3`).
- Lower `top_p` (`0.9` → `0.8`) and/or `temperature` (`0.8` → `0.7`).
- For more expressive variety (at some stability cost), raise `temperature`
  a little and watch for looping.

Set these either **per-request** via the `generate` params in §2, or as
**operator defaults** via env vars (apply to every request that doesn't pass
an override):

| Env var | Default | Purpose |
|---|---|---|
| `TOP_K` | `50` | Default `top_k`. |
| `TOP_P` | `0.9` | Default `top_p`. |
| `TEMPERATURE` | `0.8` | Default `temperature`. |
| `REPETITION_PENALTY` | `1.1` | Default `repetition_penalty`. |
| `MAX_NEW_TOKENS` | `1024` | Default `max_new_tokens` (per-chunk hard cap). |
| `REF_AUDIO_MAX_SEC` | `30` | Max reference-clip length kept at registration. |

Per-request `generate` params always override these env defaults.

### 3.5 Object storage (S3)

**Optional.** Base64 output (the default) needs no storage at all. Object
storage is only required for `generate`'s `output: "url"` mode and for the
`merge` action (§2) — i.e. for getting a URL back instead of inline base64,
and for the long-form `--to-url` flow (§5).

Any S3-compatible provider works: **Cloudflare R2** (S3-compatible, 10 GB
free tier, zero egress fees — recommended), AWS S3, Backblaze B2, MinIO, or
RunPod S3. Configure it entirely via environment variables on the RunPod
endpoint (Settings → Environment Variables):

| Env var | Required? | Default | Purpose |
|---|---|---|---|
| `S3_BUCKET` | **yes** (for URL/merge) | — | Bucket name. |
| `S3_ACCESS_KEY_ID` | **yes** (for URL/merge) | — | Access key. |
| `S3_SECRET_ACCESS_KEY` | **yes** (for URL/merge) | — | Secret key. |
| `S3_ENDPOINT_URL` | required for non-AWS providers | `""` | e.g. `https://<account_id>.r2.cloudflarestorage.com` for R2. Leave blank for AWS S3. |
| `S3_REGION` | no | `auto` | Region passed to the S3 client. |
| `S3_PUBLIC_BASE_URL` | no | `""` | If set, returned URLs are `S3_PUBLIC_BASE_URL/<key>` (public bucket/CDN). If unset, the worker returns a presigned URL instead. |
| `S3_URL_EXPIRY` | no | `604800` (7 days) | Lifetime in seconds of presigned URLs (ignored when `S3_PUBLIC_BASE_URL` is set). |
| `S3_PREFIX` | no | `qwen3-tts` | Key prefix every auto-generated/namespaced object is stored under. |

Storage is considered "configured"/enabled once `S3_BUCKET`,
`S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY` are all set. If `output: "url"`
or `merge` is requested while storage isn't configured, the job returns
`{"success": false, "error": "S3 storage is not configured. ..."}` rather than
failing silently.

---

## 4. Clients (copy-paste)

Replace `ENDPOINT_ID` / `RUNPOD_API_KEY`. All examples do the production
**async submit → poll** flow except the first cURL (which uses `/runsync` for a
quick test).

### cURL — quick sync test

```bash
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"action":"list_voices"}}'
```

### cURL — async submit + poll

```bash
# submit
JOB=$(curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"action":"generate","voice_id":"narrator-a1b2c3","text":"Hello!","response_format":"mp3"}}' \
  | jq -r '.id')

# poll
while :; do
  RES=$(curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB" \
        -H "Authorization: Bearer $RUNPOD_API_KEY")
  STATUS=$(echo "$RES" | jq -r '.status')
  [ "$STATUS" = "COMPLETED" ] && { echo "$RES" | jq -r '.output.audio_base64' | base64 -d > out.mp3; break; }
  [ "$STATUS" = "FAILED" ] && { echo "$RES"; break; }
  sleep 2
done
```

### Python — no SDK (`requests`), async submit + poll

```python
import base64, time, requests

ENDPOINT_ID = "YOUR_ENDPOINT_ID"
API_KEY = "YOUR_RUNPOD_API_KEY"
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def call(action_input: dict, poll_timeout: int = 300) -> dict:
    """Submit a job and poll /status until COMPLETED. Returns the worker output."""
    job = requests.post(f"{BASE}/run", json={"input": action_input}, headers=H, timeout=30).json()
    job_id = job["id"]
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/status/{job_id}", headers=H, timeout=30).json()
        if r["status"] == "COMPLETED":
            return r["output"]
        if r["status"] == "FAILED":
            raise RuntimeError(r.get("error") or r)
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish in {poll_timeout}s")


# register once (first call also downloads the model — allow a few minutes)
with open("reference.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()
reg = call({"action": "register_voice", "name": "narrator",
            "ref_audio": ref_b64, "ref_text": "Exact transcript here.", "language": "English"})
voice_id = reg["voice_id"]

# generate as many times as you like
out = call({"action": "generate", "voice_id": voice_id,
            "text": "Hello there, this is a cloned voice.",
            "response_format": "mp3", "return_srt": True})
open("out.mp3", "wb").write(base64.b64decode(out["audio_base64"]))
if out["srt"]:
    open("out.srt", "w", encoding="utf-8").write(out["srt"])
```

### Node.js / TypeScript — `fetch`, async submit + poll

```ts
const ENDPOINT_ID = process.env.ENDPOINT_ID!;
const API_KEY = process.env.RUNPOD_API_KEY!;
const BASE = `https://api.runpod.ai/v2/${ENDPOINT_ID}`;
const H = { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json" };

async function call(input: Record<string, unknown>, pollTimeoutMs = 300_000) {
  const submit = await fetch(`${BASE}/run`, {
    method: "POST", headers: H, body: JSON.stringify({ input }),
  }).then((r) => r.json());

  const id = submit.id as string;
  const deadline = Date.now() + pollTimeoutMs;
  while (Date.now() < deadline) {
    const r = await fetch(`${BASE}/status/${id}`, { headers: H }).then((x) => x.json());
    if (r.status === "COMPLETED") return r.output;
    if (r.status === "FAILED") throw new Error(r.error ?? JSON.stringify(r));
    await new Promise((res) => setTimeout(res, 2000));
  }
  throw new Error(`job ${id} timed out`);
}

// Register a voice (ref_audio can be base64 OR a public https URL)
const reg = await call({
  action: "register_voice",
  name: "narrator",
  ref_audio: "https://example.com/reference.wav",
  ref_text: "Exact transcript here.",
  language: "English",
});
const voiceId = reg.voice_id as string;

// Generate, then decode the base64 audio to a Buffer / Blob
const out = await call({
  action: "generate",
  voice_id: voiceId,
  text: "Hello there, this is a cloned voice.",
  response_format: "mp3",
  return_srt: true,
});
const audio = Buffer.from(out.audio_base64, "base64"); // Node
// Browser: const bytes = Uint8Array.from(atob(out.audio_base64), c => c.charCodeAt(0));
require("fs").writeFileSync("out.mp3", audio);
```

---

## 5. Long-form audio (audiobooks / hours)

A single serverless job **cannot** return hours of audio: it's bounded by the
per-job `execution_timeout` (~300s) and by the response payload size, since
audio is returned as base64 inside the JSON response. Long-form synthesis is
therefore always **split into many blocks**, generated as separate short jobs.
`client/longform.py` supports two ways to assemble those blocks back into one
file: **local merge** (default, no storage needed) and **`--to-url`** (server
merge, one URL, best for the longest jobs).

Register the voice **once** first via `register_voice` (§2), then pass its
`voice_id` to `longform.py`. Total compute time is roughly
`audio length × real-time factor` (see the README's Performance section) —
budget accordingly and prefer several parallel jobs over one enormous one.

### 5.1 Default mode — local merge

```bash
python client/longform.py --voice-id <id> --input script.txt --output audiobook.wav \
  [--block-chars 1200] [--concurrency 2] [--format mp3]
```

It splits the script paragraph/sentence-aware, generates blocks in parallel
(bounded by the endpoint's `max_workers=3`), downloads each block's base64
audio, and concatenates them **on your machine** into one file — suitable for
shorter pieces, no S3 needed. It also accepts the tuning flags from §3
(`--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`,
`--max-new-tokens`).

### 5.2 `--to-url` — one URL for hours of audio (recommended for long jobs)

```bash
python client/longform.py --voice-id <id> --input book.txt --format mp3 --to-url --concurrency 3
```

Requires S3 storage configured on the endpoint (§3.5). Instead of downloading
audio to your machine, this mode:

1. Generates every block with `output: "url"` (§2) — each block's audio is
   uploaded to S3 by the worker and only a small JSON response (a `key`, not
   the audio) comes back over the wire.
2. Calls the `merge` action (§2) once with all the block keys, in order —
   a pure I/O job (no GPU) that stitches them server-side and uploads the
   final file.
3. Prints the single resulting `url`.

**Why this is the right way to get a multi-hour audiobook as a download
link:** generation is split across many short GPU jobs, each returning a tiny
response (a key, not audio) — so no individual job risks hitting the
`execution_timeout` or a large response payload. The merge step that stitches
everything together is a single fast non-GPU job, so it isn't subject to
generation time at all — it finishes in seconds regardless of how many hours
of audio it's concatenating. Neither the per-job execution-timeout nor the
response-size limit is ever hit, no matter how long the source text is.

Contrast with §5.1 (no `--to-url`): that mode downloads every block's base64
audio to your machine and concatenates locally. It works fine for shorter
pieces and needs no S3 setup, but for very long scripts you're pulling all
the audio over HTTP and holding it in memory client-side — `--to-url` avoids
that entirely by keeping the audio server-side until the single final
download.

---

## 6. Errors

Two layers:

1. **Job level** (`status: "FAILED"`) — worker crash / timeout. Read the top-level
   `error`. Rare in normal operation.
2. **Application level** (`status: "COMPLETED"`, but `output.success === false`) —
   e.g. missing parameter, unknown `voice_id`, unsupported `response_format`,
   rejected `ref_audio`. Read `output.error`. Always check `output.success`.

Example application error:

```json
{ "status": "COMPLETED", "output": { "success": false, "error": "Unknown voice_id: narrator-a1b2c3" } }
```

---

## 7. Operational notes

- **Register once, generate many.** Registration extracts and caches a speaker
  profile on the network volume; `generate` only needs the `voice_id`. Don't
  re-send reference audio per generation.
- **Long text is fine.** Text is chunked by sentence internally; the response's
  `chunks` tells you how many units were synthesized. Keep individual sentences
  reasonable (< ~200 chars each) for best stability.
- **`ref_audio` sources:** base64 bytes or a **public** `https://` URL. Private /
  loopback / cloud-metadata hosts and local file paths are rejected by design.
- **Health checks:** use `GET /v2/<ENDPOINT_ID>/health` (free, control-plane).
  Never health-check by submitting a `/run` job.
- **Cost:** billed only for active worker time during `/run`/`/runsync`. `$0`
  while idle (scale-to-zero). `/status` polling is free.
- **URL output is optional.** `generate` defaults to inline `audio_base64`;
  pass `output: "url"` only if you've configured S3 (§3.5) and want a link
  instead of inline bytes.
- **Generation tuning and stability quirks:** see §3. **Object storage
  (S3):** see §3.5. **Multi-hour audio:** see §5.

See the [README](./README.md) for deployment, model specs, and performance.
