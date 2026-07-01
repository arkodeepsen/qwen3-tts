"""Optional S3-compatible object storage: upload audio and hand back a URL.

This powers two things a single serverless job cannot do:
  - return audio as a URL instead of base64 (large outputs),
  - the long-form `merge` flow (hours of audio assembled from pre-generated parts).

Any S3-compatible provider works — Cloudflare R2, AWS S3, Backblaze B2, MinIO,
RunPod S3 — configured entirely via the S3_* env vars in config.py. If storage
is not configured, `enabled()` is False and the URL/merge features are disabled
with a clear error rather than silently failing.

Keys passed to upload()/download() are used verbatim; build namespaced keys with
object_key() so everything lives under S3_PREFIX.
"""
import config

_client = None  # lazily created boto3 client (test seam: monkeypatch _get_client)


def enabled() -> bool:
    """True when the minimum S3 credentials are configured."""
    return bool(config.S3_BUCKET and config.S3_ACCESS_KEY_ID and config.S3_SECRET_ACCESS_KEY)


def _get_client():
    global _client
    if _client is None:
        import boto3  # lazy import: only needed when storage is actually used
        _client = boto3.client(
            "s3",
            endpoint_url=config.S3_ENDPOINT_URL or None,
            aws_access_key_id=config.S3_ACCESS_KEY_ID,
            aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
            region_name=config.S3_REGION or None,
        )
    return _client


def _require():
    if not enabled():
        raise ValueError(
            "S3 storage is not configured. Set S3_BUCKET, S3_ACCESS_KEY_ID and "
            "S3_SECRET_ACCESS_KEY (plus S3_ENDPOINT_URL for non-AWS providers) to "
            "enable URL output and long-form merge.")


def object_key(name: str) -> str:
    """Namespace an object name under S3_PREFIX, e.g. 'a.wav' -> 'qwen3-tts/a.wav'."""
    prefix = (config.S3_PREFIX or "").strip("/")
    name = name.lstrip("/")
    return f"{prefix}/{name}" if prefix else name


def url_for(key: str) -> str:
    """Public URL if S3_PUBLIC_BASE_URL is set, else a time-limited presigned URL."""
    if config.S3_PUBLIC_BASE_URL:
        return f"{config.S3_PUBLIC_BASE_URL.rstrip('/')}/{key.lstrip('/')}"
    return _get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": config.S3_BUCKET, "Key": key},
        ExpiresIn=config.S3_URL_EXPIRY,
    )


def upload(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes under `key` (used verbatim) and return a URL to fetch them."""
    _require()
    _get_client().put_object(Bucket=config.S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    return url_for(key)


def download(key: str) -> bytes:
    """Fetch object bytes for `key` (used verbatim)."""
    _require()
    return _get_client().get_object(Bucket=config.S3_BUCKET, Key=key)["Body"].read()
