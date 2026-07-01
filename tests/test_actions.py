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
