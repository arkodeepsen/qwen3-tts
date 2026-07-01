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
