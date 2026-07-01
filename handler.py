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
