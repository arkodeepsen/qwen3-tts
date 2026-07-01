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
