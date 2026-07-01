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


# --- security regression tests (SSRF + arbitrary file access) ---

def test_load_audio_rejects_local_paths():
    from registry import _load_audio_np
    with pytest.raises(ValueError):
        _load_audio_np("/etc/passwd")
    with pytest.raises(ValueError):
        _load_audio_np(r"C:\Windows\win.ini")

def test_load_audio_rejects_loopback_url():
    from registry import _load_audio_np
    with pytest.raises(ValueError):
        _load_audio_np("http://127.0.0.1/evil.wav")

def test_load_audio_rejects_cloud_metadata_url():
    from registry import _load_audio_np
    with pytest.raises(ValueError):  # 169.254.0.0/16 link-local (cloud metadata)
        _load_audio_np("http://169.254.169.254/latest/meta-data/")

def test_load_audio_rejects_non_http_scheme():
    from registry import _load_audio_np
    with pytest.raises(ValueError):
        _load_audio_np("file:///etc/passwd")

def test_register_rejects_local_path(reg):
    r, _ = reg
    with pytest.raises(ValueError):
        r.register("v", "/etc/passwd", "ref transcript", "English")
