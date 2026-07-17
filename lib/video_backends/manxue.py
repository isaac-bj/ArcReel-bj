"""Manxue API video backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from lib.aspect_size import parse_aspect_ratio
from lib.logging_utils import format_kwargs_for_log
from lib.providers import PROVIDER_MANXUE
from lib.retry import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DOWNLOAD_BACKOFF_SECONDS,
    DOWNLOAD_MAX_ATTEMPTS,
    with_retry_async,
)
from lib.video_backends.base import (
    ProviderJobIdPersistenceMixin,
    ResumeExpiredError,
    VideoCapabilities,
    VideoCapability,
    VideoGenerationRequest,
    VideoGenerationResult,
    download_video,
    poll_with_retry,
    should_retry_download,
    should_retry_poll,
    should_retry_submit,
    submit_post,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "1ren-dance-2-ka"
DEFAULT_SEEDANCE_MODEL = "guanfang-seedance-2"
DEFAULT_BASE_URL = "https://manxueapi.com/v1"

_POLL_INTERVAL_SECONDS = 5.0
_MIN_POLL_TIMEOUT_SECONDS = 1200
_POLL_TIMEOUT_PER_SECOND = 30
_MAX_REFERENCE_IMAGES = 9
_VALID_SIZES = {"1280x720", "720x1280", "1024x1024"}
_SEEDANCE_REFERENCE_IMAGE_LIMIT = 9
_SEEDANCE_DURATIONS = (10, 15)


class ManxueVideoBackend(ProviderJobIdPersistenceMixin):
    """OpenAI-compatible Manxue /videos backend for ``1ren-dance-2-ka``."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("ManxueVideoBackend requires api_key")
        self._api_key = api_key
        self._base_url = _normalize_base_url(base_url or DEFAULT_BASE_URL)
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
        self._capabilities: set[VideoCapability] = {
            VideoCapability.TEXT_TO_VIDEO,
            VideoCapability.IMAGE_TO_VIDEO,
        }

    @property
    def name(self) -> str:
        return PROVIDER_MANXUE

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[VideoCapability]:
        return self._capabilities

    @property
    def video_capabilities(self) -> VideoCapabilities:
        return self.video_capabilities_for_model(self._model)

    @staticmethod
    def video_capabilities_for_model(_model: str) -> VideoCapabilities:
        return VideoCapabilities(reference_images=True, max_reference_images=_MAX_REFERENCE_IMAGES)

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        payload = await self._prepare_payload(request)

        logger.info("Manxue video generation start model=%s duration=%s", self._model, _payload_duration(payload))
        logger.info("Calling %s video API payload=%s", self.name, format_kwargs_for_log(payload))

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            provider_task_id = await self._create_task(client, payload)
            await self._persist_provider_job_id(request, provider_task_id, provider=PROVIDER_MANXUE)
            return await self._poll_and_build(client, provider_task_id, request, is_resume=False)

    async def resume_video(self, job_id: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            return await self._poll_and_build(client, job_id, request, is_resume=True)

    async def _prepare_payload(self, request: VideoGenerationRequest) -> dict[str, Any]:
        return self._build_payload(request)

    def _build_payload(self, request: VideoGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": request.prompt,
            "seconds": str(request.duration_seconds or 8),
            "size": _resolve_size(request.resolution, request.aspect_ratio),
        }

        refs = _collect_reference_images(request)
        if refs:
            from lib.image_backends.base import image_to_base64_data_uri

            payload["input_reference"] = image_to_base64_data_uri(refs[0])
            if len(refs) > 1:
                payload["images"] = [image_to_base64_data_uri(path) for path in refs[1:]]
        return payload

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retry_if=should_retry_submit,
    )
    async def _create_task(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        try:
            resp = await submit_post(
                lambda: client.post(
                    f"{self._base_url}/videos",
                    json=payload,
                    headers=self._headers(),
                ),
                provider=PROVIDER_MANXUE,
            )
        except httpx.HTTPStatusError as exc:
            _raise_manxue_create_error(exc)
        body = resp.json()
        task_id = body.get("id") or body.get("task_id")
        if not task_id:
            raise RuntimeError(f"Manxue create task response missing id/task_id: {body}")
        return str(task_id)

    async def _poll_and_build(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        request: VideoGenerationRequest,
        *,
        is_resume: bool,
    ) -> VideoGenerationResult:
        async def _gated_poll() -> dict[str, Any]:
            try:
                return await self._poll_once(client, task_id)
            except httpx.HTTPStatusError as exc:
                if is_resume and exc.response.status_code == 404:
                    raise ResumeExpiredError(job_id=task_id, provider=PROVIDER_MANXUE) from exc
                raise

        final = await poll_with_retry(
            poll_fn=_gated_poll,
            is_done=lambda state: str(state.get("status", "")).lower() in _TERMINAL_STATUSES,
            is_failed=_extract_failure,
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_wait=self._max_wait(request.duration_seconds),
            retry_if=should_retry_poll,
            label="Manxue",
        )

        status = str(final.get("status", "")).lower()
        if status == "cancelled":
            raise RuntimeError(f"Manxue video task cancelled: {task_id}")

        video_url = _extract_video_url(final)
        if video_url:
            await self._download_with_retry(video_url, request.output_path)
        else:
            await self._download_content_with_retry(client, task_id, request.output_path)

        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_MANXUE,
            model=self._model,
            duration_seconds=request.duration_seconds,
            task_id=task_id,
            video_uri=video_url,
        )

    async def _poll_once(self, client: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
        resp = await client.get(f"{self._base_url}/videos/{task_id}", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retry_if=should_retry_download,
    )
    async def _download_with_retry(video_url: str, output_path: Path) -> None:
        await download_video(video_url, output_path)

    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retry_if=should_retry_download,
    )
    async def _download_content_with_retry(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        output_path: Path,
    ) -> None:
        resp = await client.get(f"{self._base_url}/videos/{task_id}/content", headers=self._headers())
        resp.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(resp.content)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _max_wait(duration_seconds: int) -> float:
        return max(_MIN_POLL_TIMEOUT_SECONDS, duration_seconds * _POLL_TIMEOUT_PER_SECOND)


_TERMINAL_STATUSES = {"completed", "success", "succeeded", "failed", "error", "cancelled"}


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return DEFAULT_BASE_URL
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _resolve_size(resolution: str | None, aspect_ratio: str) -> str:
    if resolution:
        explicit = resolution.strip().lower()
        if explicit in _VALID_SIZES:
            return explicit

    aw, ah = parse_aspect_ratio(aspect_ratio)
    if aw == ah:
        return "1024x1024"
    return "1280x720" if aw > ah else "720x1280"


def _collect_reference_images(request: VideoGenerationRequest) -> list[Path]:
    refs: list[Path] = []
    if request.start_image:
        refs.append(Path(request.start_image))
    if request.reference_images:
        refs.extend(Path(path) for path in request.reference_images)
    existing = [path for path in refs if path.exists()]
    if len(existing) > _MAX_REFERENCE_IMAGES:
        logger.warning("Manxue supports at most %d reference images; extra images ignored", _MAX_REFERENCE_IMAGES)
    return existing[:_MAX_REFERENCE_IMAGES]


def _collect_reference_urls(request: VideoGenerationRequest, *, limit: int, warn_local: bool = True) -> list[str]:
    refs: list[object] = []
    if request.start_image:
        refs.append(request.start_image)
    if request.reference_images:
        refs.extend(request.reference_images)

    urls: list[str] = []
    local_count = 0
    for ref in refs:
        value = str(ref)
        if _is_public_http_url(value):
            urls.append(value)
            continue
        if Path(value).exists():
            local_count += 1

    if warn_local and local_count and not urls:
        logger.warning(
            "Manxue 1ren dance models require public http/https media URLs; "
            "omitting %d local reference image(s)",
            local_count,
        )
    if len(urls) > limit:
        logger.warning("Manxue supports at most %d public reference URLs; extra URLs ignored", limit)
    return urls[:limit]


def _is_public_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _extract_failure(state: dict[str, Any]) -> str | None:
    if str(state.get("status", "")).lower() not in {"failed", "error"}:
        return None
    error = state.get("error")
    message = None
    if isinstance(error, dict):
        message = error.get("message")
    message = message or state.get("message") or state.get("fail_reason") or state.get("error_message") or "unknown"
    return f"Manxue video generation failed: {message}"


def _extract_video_url(state: dict[str, Any]) -> str | None:
    for key in ("video_url", "result_url", "url"):
        value = state.get(key)
        if isinstance(value, str) and value:
            return value
    output = state.get("output")
    if isinstance(output, dict):
        value = output.get("url")
        if isinstance(value, str) and value:
            return value
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                value = item.get("url")
                if isinstance(value, str) and value:
                    return value
    if isinstance(output, str) and output:
        return output
    for key in ("video", "result", "data", "metadata"):
        value = state.get(key)
        if isinstance(value, dict):
            url = value.get("url")
            if isinstance(url, str) and url:
                return url
    return None


def _raise_manxue_create_error(exc: httpx.HTTPStatusError) -> None:
    body: dict[str, Any] | None = None
    try:
        parsed = exc.response.json()
        if isinstance(parsed, dict):
            body = parsed
    except ValueError:
        body = None

    if body and body.get("code") == "invalid_resource_url":
        raise RuntimeError(
            "Manxue video create failed: reference media must be a public http/https URL. "
            "The selected model does not accept local uploads or base64 data URIs."
        ) from exc
    if body and exc.response.status_code == 429:
        message = body.get("message") or "Too many requests"
        raise RuntimeError(f"Manxue video create failed: {message}") from exc
    raise exc


def _payload_duration(payload: dict[str, Any]) -> Any:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("duration") is not None:
        return metadata["duration"]
    return payload.get("duration") or payload.get("seconds")


class ManxueSeedanceVideoBackend(ManxueVideoBackend):
    """Manxue official Seedance /v1/videos backend using metadata payloads."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model or DEFAULT_SEEDANCE_MODEL,
            http_timeout=http_timeout,
        )

    @staticmethod
    def video_capabilities_for_model(_model: str) -> VideoCapabilities:
        return VideoCapabilities(reference_images=True, max_reference_images=_SEEDANCE_REFERENCE_IMAGE_LIMIT)

    async def _prepare_payload(self, request: VideoGenerationRequest) -> dict[str, Any]:
        if _is_1ren_dance_model(self.model):
            return await _build_1ren_dance_payload_with_upload(self.model, request)
        return await _build_seedance_payload_with_upload(self.model, request)

    def _build_payload(self, request: VideoGenerationRequest) -> dict[str, Any]:
        if _is_1ren_dance_model(self.model):
            return _build_1ren_dance_payload(self.model, request)
        return _build_seedance_payload(self.model, request)


def _is_1ren_dance_model(model: str) -> bool:
    return model.startswith("1ren-dance-2")


def _build_seedance_payload(model: str, request: VideoGenerationRequest) -> dict[str, Any]:
    image_urls = _collect_reference_urls(
        request,
        limit=_SEEDANCE_REFERENCE_IMAGE_LIMIT,
        warn_local=False,
    )
    return _build_seedance_payload_from_urls(model, request, image_urls)


async def _build_seedance_payload_with_upload(model: str, request: VideoGenerationRequest) -> dict[str, Any]:
    image_urls = _collect_reference_urls(
        request,
        limit=_SEEDANCE_REFERENCE_IMAGE_LIMIT,
        warn_local=False,
    )
    remaining = _SEEDANCE_REFERENCE_IMAGE_LIMIT - len(image_urls)
    if remaining > 0:
        local_refs = _collect_reference_images(request)[:remaining]
        if local_refs:
            from lib.public_image_upload import upload_public_image

            uploaded: list[str] = []
            for path in local_refs:
                uploaded.append(await upload_public_image(path))
            logger.info("Uploaded %d local reference image(s) for Manxue Seedance", len(uploaded))
            image_urls.extend(uploaded)
    return _build_seedance_payload_from_urls(model, request, image_urls[:_SEEDANCE_REFERENCE_IMAGE_LIMIT])


def _build_seedance_payload_from_urls(
    model: str,
    request: VideoGenerationRequest,
    image_urls: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "ratio": request.aspect_ratio,
        "resolution": _resolve_seedance_resolution(request.resolution),
        "duration": _resolve_seedance_duration(request.duration_seconds),
        "generate_audio": bool(request.generate_audio),
    }

    if image_urls:
        metadata["images"] = [
            {"url": url, "role": "reference_image"}
            for url in image_urls[:_SEEDANCE_REFERENCE_IMAGE_LIMIT]
        ]

    if request.resolution and request.resolution.strip().lower() == "1080p":
        metadata["resolution"] = "720p"
        metadata["super_resolution_config"] = {
            "resolution": "1080p",
            "scene": "short_series",
            "tool_version": "standard",
            "fps": 24,
        }

    return {
        "model": model,
        "prompt": request.prompt,
        "metadata": metadata,
    }


def _build_1ren_dance_payload(model: str, request: VideoGenerationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "ratio": request.aspect_ratio,
        "duration": _resolve_1ren_dance_duration(model, request.duration_seconds),
        "resolution": _resolve_1ren_dance_resolution(model, request.resolution),
        "images": [],
        "videos": [],
        "audios": [],
    }

    max_images = 4 if model == "1ren-dance-2-4" else _MAX_REFERENCE_IMAGES
    payload["images"] = _collect_reference_urls(request, limit=max_images)

    return payload


async def _build_1ren_dance_payload_with_upload(model: str, request: VideoGenerationRequest) -> dict[str, Any]:
    payload = _build_1ren_dance_payload(model, request)
    max_images = 4 if model == "1ren-dance-2-4" else _MAX_REFERENCE_IMAGES
    urls = _collect_reference_urls(request, limit=max_images, warn_local=False)
    remaining = max_images - len(urls)
    if remaining > 0:
        local_refs = _collect_reference_images(request)[:remaining]
        if local_refs:
            from lib.public_image_upload import upload_public_image

            uploaded: list[str] = []
            for path in local_refs:
                uploaded_url = await upload_public_image(path)
                uploaded.append(uploaded_url)
            logger.info("Uploaded %d local reference image(s) for Manxue 1ren", len(uploaded))
            urls.extend(uploaded)
    payload["images"] = urls[:max_images]
    return payload


def _resolve_1ren_dance_duration(model: str, duration_seconds: int | None) -> int:
    duration = int(duration_seconds or 10)
    if model == "1ren-dance-2-4":
        return min(15, max(5, duration))
    if duration in _SEEDANCE_DURATIONS:
        return duration
    resolved = min(_SEEDANCE_DURATIONS, key=lambda allowed: abs(allowed - duration))
    logger.info(
        "Manxue 1ren dance duration %s is unsupported; using nearest supported duration %s",
        duration,
        resolved,
    )
    return resolved


def _resolve_1ren_dance_resolution(model: str, resolution: str | None) -> str:
    if model == "1ren-dance-2-3":
        return "480p"
    if resolution and resolution.strip().lower() == "480p" and model == "1ren-dance-2-3":
        return "480p"
    return "720p"


def _resolve_seedance_resolution(resolution: str | None) -> str:
    if not resolution:
        return "720p"
    normalized = resolution.strip().lower()
    if normalized in {"480p", "720p", "1080p"}:
        return normalized
    return "720p"


def _resolve_seedance_duration(duration_seconds: int | None) -> int:
    if duration_seconds is None:
        return _SEEDANCE_DURATIONS[0]
    duration = int(duration_seconds)
    if duration in _SEEDANCE_DURATIONS:
        return duration
    resolved = min(_SEEDANCE_DURATIONS, key=lambda allowed: abs(allowed - duration))
    logger.info(
        "Manxue Seedance duration %s is unsupported; using nearest supported duration %s",
        duration,
        resolved,
    )
    return resolved
