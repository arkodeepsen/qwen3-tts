"""Persistent voice-profile registry on the RunPod network volume.

Layout: <root>/<voice_id>/{meta.json, ref.wav, prompt.pt}
ref.wav is the source of truth; prompt.pt is a best-effort cache serialized as a
plain dict (tensors + primitives) so torch.load(weights_only=True) can read it.
"""
import base64
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import tempfile
import urllib.request
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch


@dataclass
class PromptItem:
    """Structural mirror of qwen_tts.VoiceClonePromptItem.

    generate_voice_clone only reads these attributes (duck-typed), so a local
    dataclass keeps the registry decoupled from qwen_tts and unit-testable
    without installing the model package. Fields must match VoiceClonePromptItem.
    """
    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: object = None


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")


def sanitize_voice_id(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-") or "voice"
    return f"{slug[:40]}-{uuid.uuid4().hex[:6]}"


_ALLOWED_URL_SCHEMES = ("http", "https")
_URL_FETCH_TIMEOUT = 15
_MAX_REF_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB cap on decoded/fetched reference audio


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects so a validated URL cannot be bounced to an internal target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"ref_audio URL redirect blocked: {newurl}")


_URL_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _assert_public_url(url: str) -> None:
    """SSRF guard: allow only http(s) to hosts that resolve to public addresses.

    Note: the fetch re-resolves DNS, so a determined DNS-rebinding attacker is not
    fully blocked; for an internal reference-audio fetch, this range check plus
    redirect blocking is a proportionate mitigation.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"ref_audio URL scheme not allowed: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("ref_audio URL has no host.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve ref_audio URL host: {host}") from e
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"ref_audio URL resolves to a non-public address: {ip}")


def _read_audio_bytes(data: bytes) -> tuple:
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return wav, int(sr)


def _load_audio_np(ref_audio):
    """Decode CLIENT-supplied ref_audio into (float32 mono, sr).

    Accepts ONLY: a (np.ndarray, sr) tuple, a base64 / data-URI audio string, or
    an http(s) URL to public audio (SSRF-guarded, redirects blocked, size-capped).
    Local filesystem paths are rejected so a client cannot read worker files.
    """
    if isinstance(ref_audio, tuple):
        wav, sr = ref_audio
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:  # average channels BEFORE flattening (don't interleave)
            wav = np.mean(wav, axis=-1)
        return wav.reshape(-1).astype(np.float32), int(sr)
    if not isinstance(ref_audio, str):
        raise ValueError("ref_audio must be a base64 string, https URL, or (waveform, sr) tuple.")

    s = ref_audio
    if s.startswith("http://") or s.startswith("https://"):
        _assert_public_url(s)
        with _URL_OPENER.open(s, timeout=_URL_FETCH_TIMEOUT) as resp:
            data = resp.read(_MAX_REF_AUDIO_BYTES + 1)
        if len(data) > _MAX_REF_AUDIO_BYTES:
            raise ValueError("ref_audio exceeds maximum allowed size.")
        wav, sr = _read_audio_bytes(data)
    else:
        # Anything that is not an http(s) URL must be base64 / data-URI audio.
        # Filesystem paths fail to decode here and are rejected (no file access).
        if s.startswith("data:") and "," in s:
            s = s.split(",", 1)[1]
        try:
            raw = base64.b64decode(s, validate=True)
        except Exception as e:
            raise ValueError(
                "ref_audio must be base64/data-URI audio or an https URL; "
                "local paths are not accepted.") from e
        if len(raw) > _MAX_REF_AUDIO_BYTES:
            raise ValueError("ref_audio exceeds maximum allowed size.")
        wav, sr = _read_audio_bytes(raw)

    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    return wav.astype(np.float32), int(sr)


class VoiceRegistry:
    def __init__(self, root: str, model_getter, cache_size: int = 16):
        self.root = root
        self._model_getter = model_getter
        self._cache: "OrderedDict[str, list]" = OrderedDict()
        self._cache_size = cache_size
        os.makedirs(self.root, exist_ok=True)

    # ---- helpers ----
    def _dir(self, voice_id: str) -> str:
        if not _ID_RE.match(voice_id or ""):
            raise ValueError(f"Invalid voice_id: {voice_id!r}")
        return os.path.join(self.root, voice_id)

    @staticmethod
    def _item_to_dict(it) -> dict:
        return {
            "ref_code": it.ref_code,
            "ref_spk_embedding": it.ref_spk_embedding,
            "x_vector_only_mode": bool(it.x_vector_only_mode),
            "icl_mode": bool(it.icl_mode),
            "ref_text": it.ref_text,
        }

    @staticmethod
    def _dict_to_item(d: dict, device) -> PromptItem:
        rc = d["ref_code"]
        emb = d["ref_spk_embedding"]
        return PromptItem(
            ref_code=(rc.to(device) if isinstance(rc, torch.Tensor) else rc),
            ref_spk_embedding=emb.to(device),
            x_vector_only_mode=bool(d["x_vector_only_mode"]),
            icl_mode=bool(d["icl_mode"]),
            ref_text=d.get("ref_text"),
        )

    # ---- public API ----
    def register(self, name: str, ref_audio, ref_text: str, language: str) -> dict:
        model = self._model_getter()
        voice_id = sanitize_voice_id(name)
        wav, sr = _load_audio_np(ref_audio)
        # Pass the already-decoded waveform (never the raw string) so the model
        # does not re-fetch a URL or re-read a path.
        items = model.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=ref_text, x_vector_only_mode=False)

        tmp = tempfile.mkdtemp(dir=self.root)
        try:
            sf.write(os.path.join(tmp, "ref.wav"), wav, sr, format="WAV")
            torch.save([self._item_to_dict(it) for it in items], os.path.join(tmp, "prompt.pt"))
            meta = {"voice_id": voice_id, "name": name, "language": language,
                    "ref_text": ref_text, "sample_rate": int(sr),
                    "created_at": datetime.now(timezone.utc).isoformat()}
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.rename(tmp, os.path.join(self.root, voice_id))
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return {"voice_id": voice_id, "sample_rate": int(sr), "name": name, "language": language}

    def list_voices(self) -> list[dict]:
        out = []
        for vid in sorted(os.listdir(self.root)):
            meta_path = os.path.join(self.root, vid, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    m = json.load(f)
                out.append({"voice_id": m["voice_id"], "name": m.get("name"),
                            "language": m.get("language"), "sample_rate": m.get("sample_rate"),
                            "created_at": m.get("created_at")})
        return out

    def delete(self, voice_id: str) -> bool:
        try:
            d = self._dir(voice_id)
        except ValueError:
            return False
        if not os.path.isdir(d):
            return False
        shutil.rmtree(d, ignore_errors=True)
        self._cache.pop(voice_id, None)
        return True

    def load_prompt(self, voice_id: str) -> list:
        if voice_id in self._cache:
            self._cache.move_to_end(voice_id)
            return self._cache[voice_id]
        model = self._model_getter()
        device = model.device
        d = self._dir(voice_id)
        if not os.path.isdir(d):
            raise KeyError(f"Unknown voice_id: {voice_id}")
        prompt_path = os.path.join(d, "prompt.pt")
        items = None
        if os.path.isfile(prompt_path):
            try:
                dicts = torch.load(prompt_path, map_location=device, weights_only=True)
                items = [self._dict_to_item(x, device) for x in dicts]
            except Exception:
                items = None
        if items is None:  # rebuild from source of truth (server-controlled ref.wav)
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            ref_wav, ref_sr = sf.read(os.path.join(d, "ref.wav"), dtype="float32", always_2d=False)
            if ref_wav.ndim > 1:
                ref_wav = np.mean(ref_wav, axis=-1).astype(np.float32)
            built = model.create_voice_clone_prompt(
                ref_audio=(ref_wav, int(ref_sr)), ref_text=meta.get("ref_text"), x_vector_only_mode=False)
            torch.save([self._item_to_dict(it) for it in built], prompt_path)
            items = [self._dict_to_item(self._item_to_dict(it), device) for it in built]
        self._cache[voice_id] = items
        self._cache.move_to_end(voice_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return items
