import datetime
import io
import numpy as np
import soundfile as sf
import pytest

import config
import storage
from inference import synthesize


class _FakePaginator:
    def __init__(self, s3): self.s3 = s3
    def paginate(self, Bucket, Prefix=""):
        contents = [{"Key": k, "LastModified": self.s3.mtimes[(b, k)]}
                    for (b, k) in list(self.s3.store) if b == Bucket and k.startswith(Prefix)]
        yield {"Contents": contents}


class FakeS3:
    """In-memory stand-in for a boto3 S3 client (no boto3 / network needed)."""
    def __init__(self): self.store = {}; self.mtimes = {}
    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.store[(Bucket, Key)] = Body
        self.mtimes[(Bucket, Key)] = datetime.datetime.now(datetime.timezone.utc)
    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}
    def delete_object(self, Bucket, Key):
        self.store.pop((Bucket, Key), None); self.mtimes.pop((Bucket, Key), None)
    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://fake-s3/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"
    def get_paginator(self, name):
        return _FakePaginator(self)


class _FakeModel:
    def __init__(self, n=100): self.n = n
    def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
        return [np.full(self.n, 0.2, dtype=np.float32)], 100


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


# --- storage primitives ---

def test_enabled_reflects_config(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    assert storage.enabled() is False
    monkeypatch.setattr(config, "S3_BUCKET", "b")
    monkeypatch.setattr(config, "S3_ACCESS_KEY_ID", "k")
    monkeypatch.setattr(config, "S3_SECRET_ACCESS_KEY", "s")
    assert storage.enabled() is True

def test_object_key_prefix(s3):
    assert storage.object_key("a/b.wav") == "pfx/a/b.wav"

def test_upload_download_delete_roundtrip(s3):
    key = storage.object_key("a/b.wav")
    url = storage.upload(key, b"hello", "audio/wav")
    assert "pfx/a/b.wav" in url
    assert storage.download(key) == b"hello"
    storage.delete(key)
    assert (config.S3_BUCKET, key) not in s3.store

def test_upload_requires_config(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    with pytest.raises(ValueError):
        storage.upload("k", b"x")

def test_public_base_url(monkeypatch, s3):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    assert storage.url_for("pfx/x.mp3") == "https://cdn.example.com/pfx/x.mp3"

def test_prune_prefix_deletes_only_old(s3):
    k_new, k_old = storage.object_key("outputs/new.wav"), storage.object_key("outputs/old.wav")
    storage.upload(k_new, b"n", "audio/wav")
    storage.upload(k_old, b"o", "audio/wav")
    s3.mtimes[(config.S3_BUCKET, k_old)] = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2))
    deleted = storage.prune_prefix(storage.object_key("outputs/"), older_than_sec=86400)
    assert deleted == 1
    assert (config.S3_BUCKET, k_old) not in s3.store
    assert (config.S3_BUCKET, k_new) in s3.store


# --- synthesize output selection (base64 | url | auto) ---

def test_output_url_uploads(s3):
    out = synthesize([object()], "Hello.", "English", output="url", model=_FakeModel())
    assert "audio_base64" not in out
    assert out["url"] and out["key"].startswith("pfx/outputs/")

def test_output_auto_small_is_base64(s3):
    out = synthesize([object()], "Hello.", "English", output="auto", model=_FakeModel())
    assert "audio_base64" in out and "url" not in out

def test_output_auto_large_is_url(s3, monkeypatch):
    monkeypatch.setattr(config, "MAX_INLINE_BYTES", 10)  # force "large"
    out = synthesize([object()], "Hello.", "English", output="auto", model=_FakeModel())
    assert "url" in out and "audio_base64" not in out

def test_output_url_without_s3_errors(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    with pytest.raises(ValueError):
        synthesize([object()], "Hello.", "English", output="url", model=_FakeModel())

def test_output_auto_no_s3_falls_back_to_base64(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "")
    monkeypatch.setattr(config, "MAX_INLINE_BYTES", 10)  # would be "large"
    out = synthesize([object()], "Hello.", "English", output="auto", model=_FakeModel())
    assert "audio_base64" in out and "url" not in out
