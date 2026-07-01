"""Route a RunPod job input to a voice-cloning operation and return a JSON envelope."""
import traceback

import config
from inference import get_model, merge_audio, synthesize
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
                response_format=job_input.get("response_format", config.DEFAULT_FORMAT),
                temperature=job_input.get("temperature"), top_p=job_input.get("top_p"),
                top_k=job_input.get("top_k"), repetition_penalty=job_input.get("repetition_penalty"),
                max_new_tokens=job_input.get("max_new_tokens"),
                to_url=(job_input.get("output") == "url"), s3_key=job_input.get("key"))
            return {"success": True, **res}

        if action == "merge":
            _require(job_input, "keys")
            res = merge_audio(
                job_input["keys"], response_format=job_input.get("response_format", config.DEFAULT_FORMAT),
                gap_sec=job_input.get("gap_sec"), output_key=job_input.get("output_key"))
            return {"success": True, **res}

        if action == "list_voices":
            return {"success": True, "voices": registry.list_voices()}

        if action == "delete_voice":
            _require(job_input, "voice_id")
            ok = registry.delete(job_input["voice_id"])
            return {"success": ok, "deleted": job_input["voice_id"] if ok else None,
                    **({} if ok else {"error": f"Unknown voice_id: {job_input['voice_id']}"})}

        return {"success": False, "error": f"Unknown action: {action}"}
    except ValueError as e:  # expected client/validation error — no stack trace
        return {"success": False, "error": str(e)}
    except Exception as e:  # unexpected: surface a traceback in worker logs, still envelope
        traceback.print_exc()
        return {"success": False, "error": str(e)}
