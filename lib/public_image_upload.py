"""Upload local images to a public HTTP image host."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

DEFAULT_PUBLIC_IMAGE_UPLOAD_URL = "http://114.67.242.123:8887/upload.php"


def configured_public_image_upload_url() -> str:
    return (
        os.getenv("MANXUE_PUBLIC_IMAGE_UPLOAD_URL")
        or os.getenv("PUBLIC_IMAGE_UPLOAD_URL")
        or DEFAULT_PUBLIC_IMAGE_UPLOAD_URL
    ).strip()


def configured_public_image_upload_token() -> str | None:
    return (os.getenv("MANXUE_PUBLIC_IMAGE_UPLOAD_TOKEN") or os.getenv("PUBLIC_IMAGE_UPLOAD_TOKEN") or "").strip() or None


async def upload_public_image(path: Path, *, timeout: float = 60.0) -> str:
    upload_url = configured_public_image_upload_url()
    if not upload_url:
        raise RuntimeError("public image upload URL is not configured")
    if not path.exists():
        raise FileNotFoundError(f"public image upload file not found: {path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers: dict[str, str] = {}
    token = configured_public_image_upload_token()
    data: dict[str, str] = {}
    if token:
        headers["X-Upload-Token"] = token
        data["token"] = token

    with path.open("rb") as fh:
        files = {"file": (path.name, fh, mime_type)}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upload_url, headers=headers, data=data, files=files)
    resp.raise_for_status()
    try:
        body = resp.json()
    except ValueError as exc:
        preview = resp.text[:300].replace("\n", " ").strip()
        raise RuntimeError(f"public image upload returned non-JSON response: {preview}") from exc
    public_url = _extract_uploaded_url(body)
    if not public_url:
        raise RuntimeError(f"public image upload response missing url: {body}")
    if not _is_http_url(public_url):
        raise RuntimeError(f"public image upload returned non-http URL: {public_url}")
    return public_url


def _extract_uploaded_url(body: Any) -> str | None:
    if isinstance(body, dict):
        for key in ("url", "public_url", "file_url"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
        data = body.get("data")
        if isinstance(data, dict):
            return _extract_uploaded_url(data)
    return None


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
