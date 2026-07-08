from pathlib import Path
from unittest.mock import AsyncMock, patch

from lib.video_backends.base import VideoGenerationRequest
from lib.video_backends.manxue import (
    ManxueSeedanceVideoBackend,
    ManxueVideoBackend,
    _extract_failure,
    _extract_video_url,
    _normalize_base_url,
    _resolve_seedance_duration,
    _resolve_size,
)


def test_infer_endpoint_routes_1ren_dance_to_manxue_video():
    from lib.custom_provider.endpoints import infer_endpoint

    assert infer_endpoint("1ren-dance-2-ka", "openai") == "manxue-video"


def test_infer_endpoint_routes_guanfang_seedance_to_dedicated_option():
    from lib.custom_provider.endpoints import infer_endpoint

    assert infer_endpoint("guanfang-seedance-2-fast", "openai") == "manxue-seedance-video"


def test_normalize_base_url_appends_v1_once():
    assert _normalize_base_url("https://manxueapi.com") == "https://manxueapi.com/v1"
    assert _normalize_base_url("https://manxueapi.com/v1") == "https://manxueapi.com/v1"


def test_resolve_size_uses_supported_manxue_sizes():
    assert _resolve_size(None, "16:9") == "1280x720"
    assert _resolve_size(None, "9:16") == "720x1280"
    assert _resolve_size(None, "1:1") == "1024x1024"
    assert _resolve_size("720x1280", "16:9") == "720x1280"


def test_resolve_seedance_duration_uses_supported_values():
    assert _resolve_seedance_duration(None) == 10
    assert _resolve_seedance_duration(10) == 10
    assert _resolve_seedance_duration(11) == 10
    assert _resolve_seedance_duration(15) == 15


def test_build_payload_splits_first_reference_from_images(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.webp"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    backend = ManxueVideoBackend(api_key="sk-test")
    req = VideoGenerationRequest(
        prompt="dance",
        output_path=tmp_path / "out.mp4",
        aspect_ratio="9:16",
        duration_seconds=10,
        start_image=first,
        reference_images=[second],
        generate_audio=True,
    )

    payload = backend._build_payload(req)

    assert payload == {
        "model": "1ren-dance-2-ka",
        "prompt": "dance",
        "seconds": "10",
        "size": "720x1280",
        "input_reference": "data:image/png;base64,Zmlyc3Q=",
        "images": ["data:image/webp;base64,c2Vjb25k"],
    }
    assert "generate_audio" not in payload
    assert "duration" not in payload
    assert "resolution" not in payload
    assert "aspect_ratio" not in payload
    assert "medias" not in payload


def test_seedance_build_payload_uses_metadata_media_objects(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    backend = ManxueSeedanceVideoBackend(api_key="sk-test", model="guanfang-seedance-2-fast")
    req = VideoGenerationRequest(
        prompt="cinematic city night",
        output_path=tmp_path / "out.mp4",
        aspect_ratio="16:9",
        duration_seconds=10,
        resolution="1080p",
        start_image=first,
        reference_images=[second],
        generate_audio=True,
    )

    payload = backend._build_payload(req)

    assert payload == {
        "model": "guanfang-seedance-2-fast",
        "prompt": "cinematic city night",
        "metadata": {
            "ratio": "16:9",
            "resolution": "720p",
            "duration": 10,
            "generate_audio": True,
            "images": [
                {"url": "data:image/png;base64,Zmlyc3Q=", "role": "reference_image"},
                {"url": "data:image/jpeg;base64,c2Vjb25k", "role": "reference_image"},
            ],
            "super_resolution_config": {
                "resolution": "1080p",
                "scene": "short_series",
                "tool_version": "standard",
                "fps": 24,
            },
        },
    }
    assert "images" not in payload
    assert "ratio" not in payload
    assert "duration" not in payload


def test_seedance_endpoint_omits_local_images_for_1ren_dance_models(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    backend = ManxueSeedanceVideoBackend(api_key="sk-test", model="1ren-dance-2-fast-1")
    req = VideoGenerationRequest(
        prompt="dance with reference",
        output_path=tmp_path / "out.mp4",
        aspect_ratio="9:16",
        duration_seconds=11,
        resolution="1080p",
        start_image=first,
        reference_images=[second],
        generate_audio=True,
    )

    payload = backend._build_payload(req)

    assert payload == {
        "model": "1ren-dance-2-fast-1",
        "prompt": "dance with reference",
        "ratio": "9:16",
        "duration": 10,
        "resolution": "720p",
        "images": [],
        "videos": [],
        "audios": [],
    }
    assert "metadata" not in payload
    assert "input_reference" not in payload
    assert "seconds" not in payload


def test_seedance_endpoint_uses_public_urls_for_1ren_dance_models(tmp_path: Path):
    backend = ManxueSeedanceVideoBackend(api_key="sk-test", model="1ren-dance-2-fast-1")
    req = VideoGenerationRequest(
        prompt="dance with public reference",
        output_path=tmp_path / "out.mp4",
        aspect_ratio="9:16",
        duration_seconds=10,
        resolution="720p",
        start_image="https://cdn.example.com/first.png",  # type: ignore[arg-type]
        reference_images=["https://cdn.example.com/second.jpg"],  # type: ignore[list-item]
        generate_audio=True,
    )

    payload = backend._build_payload(req)

    assert payload["images"] == [
        "https://cdn.example.com/first.png",
        "https://cdn.example.com/second.jpg",
    ]
    assert "metadata" not in payload


def test_video_capabilities_allow_nine_reference_images():
    caps = ManxueVideoBackend.video_capabilities_for_model("1ren-dance-2-ka")
    assert caps.reference_images is True
    assert caps.max_reference_images == 9


def test_extract_video_url_from_common_response_shapes():
    assert _extract_video_url({"url": "https://cdn/out.mp4"}) == "https://cdn/out.mp4"
    assert _extract_video_url({"video": {"url": "https://cdn/video.mp4"}}) == "https://cdn/video.mp4"
    assert _extract_video_url({"result": {"url": "https://cdn/result.mp4"}}) == "https://cdn/result.mp4"
    assert _extract_video_url({"data": {"url": "https://cdn/data.mp4"}}) == "https://cdn/data.mp4"
    assert _extract_video_url({"metadata": {"url": "https://cdn/meta.mp4"}}) == "https://cdn/meta.mp4"
    assert _extract_video_url({"output": "https://cdn/output.mp4"}) == "https://cdn/output.mp4"
    assert _extract_video_url({"video_url": "https://cdn/video-url.mp4"}) == "https://cdn/video-url.mp4"
    assert _extract_video_url({"result_url": "https://cdn/result-url.mp4"}) == "https://cdn/result-url.mp4"
    assert _extract_video_url({"output": [{"url": "https://cdn/output-list.mp4"}]}) == "https://cdn/output-list.mp4"


def test_extract_failure_uses_documented_error_fields():
    assert _extract_failure({"status": "failed", "error": {"message": "bad ref"}}).endswith("bad ref")
    assert _extract_failure({"status": "failed", "fail_reason": "blocked"}).endswith("blocked")
    assert _extract_failure({"status": "completed"}) is None


async def test_generate_creates_polls_and_downloads(tmp_path: Path):
    backend = ManxueVideoBackend(api_key="sk-test", base_url="https://manxueapi.com")
    req = VideoGenerationRequest(prompt="p", output_path=tmp_path / "out.mp4", duration_seconds=8)

    with (
        patch.object(backend, "_create_task", new=AsyncMock(return_value="task-1")) as create_task,
        patch.object(
            backend,
            "_poll_once",
            new=AsyncMock(return_value={"status": "completed", "url": "https://cdn/v.mp4"}),
        ),
        patch.object(backend, "_download_with_retry", new=AsyncMock()) as download,
        patch("lib.video_backends.base.persist_provider_job_id", new=AsyncMock()),
    ):
        result = await backend.generate(req)

    create_task.assert_awaited_once()
    download.assert_awaited_once_with("https://cdn/v.mp4", req.output_path)
    assert result.provider == "manxue"
    assert result.model == "1ren-dance-2-ka"
    assert result.task_id == "task-1"
