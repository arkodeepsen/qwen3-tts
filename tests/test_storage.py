import io
import numpy as np
import soundfile as sf
import pytest

import config
import storage
from inference import merge_audio, synthesize


class FakeS3:
    """In-memory stand-in for a boto3 S3 client (no boto3 / network needed)."""
    def __init__(self): self.store = {}
    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.store[(Bucket, Key)] = Body
    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}
    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://fake-s3/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(config, "S3_BUCKET", "bkt")
    monkeypatch.setattr(config, "S3_ACCESS_KEY_ID", "k")
    monkeypatch.setattr(config, "S3_SECRET_ACCESS_KEY", "s")
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "")
    monkeypatch.setattr(config, "S3_PREFIX", "pfx")
    monkeypatch.setattr(storage, "_get_client", lambda: fake)
    return fake


def _wav_bytes(seconds, sr=100):
    buf = io.BytesIO()
    sf.write(buf, np.full(int(seconds * sr), 0.1, dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def test_enabled_reflects_config(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    assert storage.enabled() is False
    monkeypatch.setattr(config, "S3_BUCKET", "b")
    monkeypatch.setattr(config, "S3_ACCESS_KEY_ID", "k")
    monkeypatch.setattr(config, "S3_SECRET_ACCESS_KEY", "s")
    assert storage.enabled() is True

def test_object_key_prefix(s3):
    assert storage.object_key("a/b.wav") == "pfx/a/b.wav"

def test_upload_download_roundtrip(s3):
    key = storage.object_key("a/b.wav")
    url = storage.upload(key, b"hello", "audio/wav")
    assert "pfx/a/b.wav" in url
    assert storage.download(key) == b"hello"

def test_upload_requires_config(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    with pytest.raises(ValueError):
        storage.upload("k", b"x")

def test_public_base_url(monkeypatch, s3):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    assert storage.url_for("pfx/x.mp3") == "https://cdn.example.com/pfx/x.mp3"

def test_merge_audio_concatenates_and_uploads(s3):
    k1, k2 = storage.object_key("s/p0.wav"), storage.object_key("s/p1.wav")
    s3.store[(config.S3_BUCKET, k1)] = _wav_bytes(1.0)
    s3.store[(config.S3_BUCKET, k2)] = _wav_bytes(1.0)
    res = merge_audio([k1, k2], response_format="wav", gap_sec=0.0)
    assert res["parts"] == 2 and abs(res["duration_sec"] - 2.0) < 0.05
    assert res["url"] and res["key"]
    w, sr = sf.read(io.BytesIO(storage.download(res["key"])), dtype="float32")
    assert abs(len(w) / sr - 2.0) < 0.05

def test_merge_requires_keys():
    with pytest.raises(ValueError):
        merge_audio([])

def test_merge_requires_s3(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    with pytest.raises(ValueError):
        merge_audio(["k"])

def test_synthesize_to_url(s3):
    class FakeModel:
        def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
            return [np.full(100, 0.2, dtype=np.float32)], 100
    out = synthesize([object()], "Hello.", "English", to_url=True, model=FakeModel())
    assert "audio_base64" not in out
    assert out["url"] and out["key"].startswith("pfx/outputs/")
