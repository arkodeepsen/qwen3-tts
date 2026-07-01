"""Optional S3-compatible object storage: upload audio and hand back a URL.

Lets `generate` return a URL instead of base64 when the result is large (the
serverless response has a size limit). Any S3-compatible provider works — RunPod
network-volume S3, Cloudflare R2, AWS S3, Backblaze B2, MinIO — configured via the
S3_* env vars in config.py. If storage is not configured, `enabled()` is False and
URL output is disabled (output="auto" falls back to base64; output="url" errors).

Keys passed to upload()/download()/delete() are used verbatim; build namespaced
keys with object_key() so everything lives under S3_PREFIX. Old outputs are pruned
via prune_outputs() so the (possibly shared) volume never fills.
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
        from botocore.config import Config
        # Path-style + s3v4 is required by S3-compatible gateways (RunPod S3,
        # R2, MinIO) — the default virtual-host addressing needs per-bucket DNS
        # and triggers SignatureDoesNotMatch on custom endpoints.
        _client = boto3.client(
            "s3",
            endpoint_url=config.S3_ENDPOINT_URL or None,
            aws_access_key_id=config.S3_ACCESS_KEY_ID,
            aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
            region_name=config.S3_REGION or None,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
    return _client


def _require():
    if not enabled():
        raise ValueError(
            "S3 storage is not configured. Set S3_BUCKET, S3_ACCESS_KEY_ID and "
            "S3_SECRET_ACCESS_KEY (plus S3_ENDPOINT_URL for non-AWS providers) to "
            "enable URL output for large results.")


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


def delete(key: str) -> None:
    """Delete an object by key (used verbatim)."""
    _require()
    _get_client().delete_object(Bucket=config.S3_BUCKET, Key=key)


def prune_prefix(prefix: str, older_than_sec: int) -> int:
    """Delete objects under `prefix` older than `older_than_sec`. Returns the
    number deleted. Best-effort housekeeping so the (shared) volume never fills."""
    if older_than_sec <= 0:
        return 0
    import datetime
    client = _get_client()
    now = datetime.datetime.now(datetime.timezone.utc)
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if (now - obj["LastModified"]).total_seconds() > older_than_sec:
                client.delete_object(Bucket=config.S3_BUCKET, Key=obj["Key"])
                deleted += 1
    return deleted


def prune_outputs() -> int:
    """Prune old objects under the outputs/ prefix (best-effort; never raises)."""
    try:
        return prune_prefix(object_key("outputs/"), config.OUTPUT_TTL_SEC)
    except Exception:
        return 0
