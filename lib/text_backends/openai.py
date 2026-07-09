"""OpenAITextBackend — OpenAI 文本生成后端。"""

from __future__ import annotations

import logging
import json

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError

from lib.config.url_utils import is_official_openai_base_url
from lib.logging_utils import format_kwargs_for_log
from lib.openai_shared import OPENAI_RETRYABLE_ERRORS, create_openai_client
from lib.providers import PROVIDER_OPENAI
from lib.retry import with_retry_async
from lib.text_backends.base import (
    TextCapability,
    TextGenerationRequest,
    TextGenerationResult,
    TokenParam,
    is_valid_json,
    resolve_schema,
    structured_fallback_reason,
    warn_if_truncated,
)
from lib.text_utils import strip_json_code_fences

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAITextBackend:
    """OpenAI 文本生成后端，支持 Chat Completions API。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider_name: str = PROVIDER_OPENAI,
    ):
        # 禁用 SDK 内置重试，由本层 generate() 统一管理重试策略
        self._client = create_openai_client(api_key=api_key, base_url=base_url, max_retries=0)
        self._model = model or DEFAULT_MODEL
        # 复用 OpenAI 兼容协议的 provider（如 dashscope）须用真实 provider 记账，
        # 否则计费查表会命中 OpenAI 的 USD 费率而非自身定价。
        self._provider_name = provider_name
        # 官方端点已弃用 max_tokens（推理模型直接拒绝），用 max_completion_tokens；
        # 第三方兼容端点（自定义供应商、dashscope 等）不保证支持新参数，保守沿用 max_tokens
        self._max_tokens_param: TokenParam = (
            "max_completion_tokens" if is_official_openai_base_url(base_url) else "max_tokens"
        )
        self._capabilities: set[TextCapability] = {
            TextCapability.TEXT_GENERATION,
            TextCapability.STRUCTURED_OUTPUT,
            TextCapability.VISION,
        }

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[TextCapability]:
        return self._capabilities

    @with_retry_async(max_attempts=4, backoff_seconds=(2, 4, 8), retryable_errors=OPENAI_RETRYABLE_ERRORS)
    async def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        """生成文本回复。

        单一重试循环包裹整个流程：
        1. 尝试原生 response_format 调用
        2. 若遇 schema 不兼容错误 → 本次 attempt 内降级到 Instructor
        3. 若遇瞬态错误（429/500/503/网络）→ 由装饰器自动重试整个流程

        这样无论是原生调用还是降级路径遇到瞬态错误，都统一由外层重试处理。
        """
        messages = _build_messages(request)
        kwargs: dict = {"model": self._model, "messages": messages}
        if request.max_output_tokens is not None:
            kwargs[self._max_tokens_param] = request.max_output_tokens

        if request.response_schema:
            if _uses_prompted_json_schema(self._provider_name, self._model):
                kwargs["messages"] = _messages_with_json_instruction(messages)
            else:
                schema = resolve_schema(request.response_schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": schema,
                    },
                }

        logger.info("调用 %s 文本 SDK kwargs=%s", self.name, format_kwargs_for_log(kwargs))
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if request.response_schema and _is_schema_error(exc):
                logger.warning(
                    "原生 response_format 失败 (%s)，降级到 Instructor 路径",
                    exc,
                )
                return await _instructor_fallback(
                    self._client,
                    self._model,
                    request,
                    messages,
                    provider=self._provider_name,
                    token_param=self._max_tokens_param,
                )
            raise

        usage = response.usage
        choice = response.choices[0]
        output_tokens = usage.completion_tokens if usage else None
        text = choice.message.content or ""

        prompted_json_schema = request.response_schema and _uses_prompted_json_schema(
            self._provider_name, self._model
        )
        if prompted_json_schema:
            text = _normalize_prompted_json_text(text, request.response_schema)
            if not is_valid_json(text):
                raise ValueError(f"{self._provider_name}/{self._model} returned non-JSON content")

        if request.response_schema:
            fallback_reason = structured_fallback_reason(text, request.response_schema)
            if fallback_reason:
                if prompted_json_schema and is_valid_json(text):
                    logger.warning(
                        "prompt JSON %s; returning parseable JSON and skipping Instructor fallback",
                        fallback_reason,
                    )
                else:
                    logger.warning(
                        "native response_format %s; falling back to Instructor",
                        fallback_reason,
                    )
                    result = await _instructor_fallback(
                        self._client,
                        self._model,
                        request,
                        messages,
                        provider=self._provider_name,
                        token_param=self._max_tokens_param,
                    )
                    # The native 200 response was billed; merge its tokens into fallback usage.
                    if usage:
                        result.input_tokens = (result.input_tokens or 0) + (usage.prompt_tokens or 0)
                        result.output_tokens = (result.output_tokens or 0) + (usage.completion_tokens or 0)
                    return result

        warn_if_truncated(
            getattr(choice, "finish_reason", None),
            provider=self._provider_name,
            model=self._model,
            output_tokens=output_tokens,
        )
        return TextGenerationResult(
            text=text,
            provider=self._provider_name,
            model=self._model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=output_tokens,
        )


def _build_messages(request: TextGenerationRequest) -> list[dict]:
    """将 TextGenerationRequest 转为 OpenAI messages 格式。"""
    messages: list[dict] = []

    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})

    # 构建 user message
    if request.images:
        from lib.image_backends.base import image_to_base64_data_uri

        content: list[dict] = []
        for img in request.images:
            if img.path:
                data_uri = image_to_base64_data_uri(img.path)
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            elif img.url:
                content.append({"type": "image_url", "image_url": {"url": img.url}})
        content.append({"type": "text", "text": request.prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": request.prompt})

    return messages


def _uses_prompted_json_schema(provider: str, model: str) -> bool:
    """Return True for OpenAI-compatible providers that reject json_schema response_format."""
    target = f"{provider} {model}".lower()
    return "minimax" in target


def _messages_with_json_instruction(messages: list[dict]) -> list[dict]:
    """Ask the provider for raw JSON without using response_format."""
    instruction = (
        "\n\nReturn only one valid JSON object that matches the requested schema. "
        "Do not include Markdown fences, explanations, or any text outside the JSON object."
    )
    patched = [dict(message) for message in messages]
    if not patched:
        return [{"role": "user", "content": instruction.strip()}]

    last = patched[-1]
    content = last.get("content")
    if isinstance(content, str):
        if "JSON" not in content.upper():
            last["content"] = content + instruction
        return patched

    if isinstance(content, list):
        last["content"] = [*content, {"type": "text", "text": instruction.strip()}]
        return patched

    last["content"] = instruction.strip()
    return patched


def _normalize_prompted_json_text(text: str, response_schema: dict | type | None) -> str:
    """Clean prompt-only JSON output and normalize it when a Pydantic schema is available."""
    candidate = strip_json_code_fences(text)
    if not is_valid_json(candidate):
        extracted = _extract_best_json_value(candidate)
        if extracted:
            candidate = extracted

    if not is_valid_json(candidate):
        return candidate

    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        data = json.loads(candidate)
        data = _normalize_script_like_payload(data)
        candidate = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        try:
            validated = response_schema.model_validate_json(candidate, strict=False)
            return json.dumps(validated.model_dump(), ensure_ascii=False, separators=(",", ":"))
        except ValidationError:
            if isinstance(data, dict):
                for key in ("response", "result", "data", "script", "episode"):
                    value = data.get(key)
                    if value is None:
                        continue
                    value = _normalize_script_like_payload(value)
                    try:
                        validated = response_schema.model_validate(value, strict=False)
                        return json.dumps(validated.model_dump(), ensure_ascii=False, separators=(",", ":"))
                    except ValidationError:
                        continue
    return candidate


_SHOT_TYPE_ALIASES = {
    "extreme close up": "Extreme Close-up",
    "extreme close-up": "Extreme Close-up",
    "特写": "Close-up",
    "大特写": "Extreme Close-up",
    "近景": "Close-up",
    "中近景": "Medium Close-up",
    "中景": "Medium Shot",
    "中远景": "Medium Long Shot",
    "远景": "Long Shot",
    "大远景": "Extreme Long Shot",
    "过肩镜头": "Over-the-shoulder",
    "肩越し": "Over-the-shoulder",
    "主观镜头": "Point-of-view",
    "第一人称": "Point-of-view",
    "pov": "Point-of-view",
}

_CAMERA_MOTION_ALIASES = {
    "fixed": "Static",
    "none": "Static",
    "still": "Static",
    "静止": "Static",
    "固定": "Static",
    "左摇": "Pan Left",
    "向左摇": "Pan Left",
    "右摇": "Pan Right",
    "向右摇": "Pan Right",
    "上摇": "Tilt Up",
    "向上摇": "Tilt Up",
    "下摇": "Tilt Down",
    "向下摇": "Tilt Down",
    "推近": "Zoom In",
    "拉近": "Zoom In",
    "放大": "Zoom In",
    "拉远": "Zoom Out",
    "缩小": "Zoom Out",
    "跟拍": "Tracking Shot",
    "跟随": "Tracking Shot",
    "跟踪镜头": "Tracking Shot",
}


def _normalize_script_like_payload(data, *, root: bool = True):
    """Normalize common LLM drift in ArcReel script-shaped JSON."""
    if isinstance(data, dict):
        normalized = {
            key: _normalize_script_like_payload(value, root=False)
            for key, value in data.items()
        }
        if root:
            for wrapper_key in ("response", "result", "data", "script", "episode"):
                wrapped = normalized.get(wrapper_key)
                if _is_script_root(wrapped) or _looks_like_script_item(wrapped):
                    merged = _normalize_script_like_payload(wrapped, root=True)
                    if isinstance(merged, dict):
                        normalized = {
                            **merged,
                            **{key: value for key, value in normalized.items() if key not in {wrapper_key}},
                        }
                    break

            if _looks_like_script_item(normalized):
                items_key = _items_key_for_script_item(normalized) or "segments"
                return {"title": "Untitled", items_key: [_normalize_script_item(normalized)]}

            for items_key in ("segments", "scenes", "shots"):
                items = normalized.get(items_key)
                if isinstance(items, list) and _looks_like_script_items(items):
                    normalized[items_key] = [_normalize_script_item(item) for item in items]
                    if not normalized.get("title"):
                        normalized["title"] = "Untitled"
        return normalized
    if isinstance(data, list):
        if root and _looks_like_script_items(data):
            first_key = _items_key_for_script_item(next(item for item in data if isinstance(item, dict))) or "segments"
            return {"title": "Untitled", first_key: [_normalize_script_item(item) for item in data]}
        return [_normalize_script_like_payload(item, root=False) for item in data]
    return data


def _is_script_root(value) -> bool:
    return isinstance(value, dict) and any(
        isinstance(value.get(key), list) and _looks_like_script_items(value.get(key))
        for key in ("segments", "scenes", "shots")
    )


def _looks_like_script_items(items) -> bool:
    return isinstance(items, list) and any(_looks_like_script_item(item) for item in items)


def _looks_like_script_item(item) -> bool:
    return isinstance(item, dict) and any(key in item for key in ("segment_id", "scene_id", "shot_id"))


def _items_key_for_script_item(item) -> str | None:
    if not isinstance(item, dict):
        return None
    if "segment_id" in item:
        return "segments"
    if "scene_id" in item:
        return "scenes"
    if "shot_id" in item:
        return "shots"
    return None


def _normalize_script_item(item):
    if not isinstance(item, dict):
        return item
    item = dict(item)

    video_prompt = item.get("video_prompt")
    if isinstance(video_prompt, dict):
        video_prompt = dict(video_prompt)
        if "ambiance_audio" not in video_prompt and "ambience_audio" in video_prompt:
            video_prompt["ambiance_audio"] = video_prompt.pop("ambience_audio")
        if "camera_motion" in video_prompt:
            video_prompt["camera_motion"] = _normalize_enum_value(
                video_prompt["camera_motion"],
                _CAMERA_MOTION_ALIASES,
                {
                    "Static",
                    "Pan Left",
                    "Pan Right",
                    "Tilt Up",
                    "Tilt Down",
                    "Zoom In",
                    "Zoom Out",
                    "Tracking Shot",
                },
                "Static",
            )
        if "dialogue" not in video_prompt and "dialogue" in item:
            dialogue = item.pop("dialogue")
            if isinstance(dialogue, list):
                video_prompt["dialogue"] = dialogue
        item["video_prompt"] = video_prompt
    else:
        item.pop("dialogue", None)

    image_prompt = item.get("image_prompt")
    if isinstance(image_prompt, dict):
        composition = image_prompt.get("composition")
        if isinstance(composition, dict) and "shot_type" in composition:
            composition = dict(composition)
            composition["shot_type"] = _normalize_enum_value(
                composition["shot_type"],
                _SHOT_TYPE_ALIASES,
                {
                    "Extreme Close-up",
                    "Close-up",
                    "Medium Close-up",
                    "Medium Shot",
                    "Medium Long Shot",
                    "Long Shot",
                    "Extreme Long Shot",
                    "Over-the-shoulder",
                    "Point-of-view",
                },
                "Medium Shot",
            )
            image_prompt = dict(image_prompt)
            image_prompt["composition"] = composition
            item["image_prompt"] = image_prompt

    return item


def _normalize_enum_value(value, aliases: dict[str, str], allowed: set[str], default: str):
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    if stripped in allowed:
        return stripped
    key = stripped.lower().replace("_", " ").replace("-", " ")
    if key in aliases:
        return aliases[key]
    compact = key.replace(" ", "")
    for allowed_value in allowed:
        if compact == allowed_value.lower().replace("-", "").replace(" ", ""):
            return allowed_value
    return aliases.get(stripped, default)


def _extract_best_json_value(text: str) -> str | None:
    """Extract the most script-like balanced JSON value from text."""
    candidates = []
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        candidate = _extract_balanced_json_value(text, idx)
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        candidates.append((_json_candidate_score(parsed), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _json_candidate_score(value) -> int:
    if _is_script_root(value):
        return 100
    if _looks_like_script_items(value):
        return 90
    if _looks_like_script_item(value):
        return 80
    if isinstance(value, dict):
        return 10
    if isinstance(value, list):
        return 1
    return 0


def _extract_balanced_json_value(text: str, start: int) -> str | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escaped = False

    for idx in range(start + 1, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : idx + 1].strip()

    return None


_SCHEMA_ERROR_KEYWORDS = (
    "response_schema",
    "json_schema",
    "Unknown name",
    "Cannot find field",
    "Invalid JSON payload",
)


def _is_schema_error(exc: BaseException) -> bool:
    """判断异常是否为 JSON Schema 不兼容导致的错误。

    除了标准的 400 BadRequestError，一些 OpenAI 兼容代理（如 Gemini
    兼容端点）会将上游 schema 错误包装成其他状态码（如 429），
    因此也检查错误信息中是否包含 schema 相关关键字。
    """
    if isinstance(exc, BadRequestError):
        return True
    # 代理可能把上游 schema 错误包装成非 400 状态码
    error_str = str(exc)
    return any(kw in error_str for kw in _SCHEMA_ERROR_KEYWORDS)


async def _instructor_fallback(
    client: AsyncOpenAI,
    model: str,
    request: TextGenerationRequest,
    messages: list[dict],
    *,
    provider: str = PROVIDER_OPENAI,
    token_param: TokenParam = "max_tokens",
) -> TextGenerationResult:
    """Instructor 降级：当原生 response_format 不可用时的备选路径。"""
    from lib.text_backends.instructor_support import instructor_fallback_async

    return await instructor_fallback_async(
        client=client,
        model=model,
        messages=messages,
        response_schema=request.response_schema,
        provider=provider,
        max_tokens=request.max_output_tokens,
        token_param=token_param,
    )
