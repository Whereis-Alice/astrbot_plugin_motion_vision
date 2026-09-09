"""语音转写（STT）。

两种后端，优先级依次是：

1. **AstrBot 已配置的语音识别服务商** —— 在 WebUI 里选一个即可，最省事；
2. **OpenAI 兼容接口** —— 自己填 API 地址、密钥和模型，走标准的
   /audio/transcriptions 多段表单上传。

两者都没配好时抛 SttError，调用方会把音轨原样附给模型（如果模型支持音频）。
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import httpx

from .settings import AudioSettings

TRANSCRIPTION_ROUTE = "/audio/transcriptions"

MAX_ATTEMPTS = 3
"""429 / 5xx 时的总尝试次数。"""

BACKOFF_BASE_SECONDS = 1.5
MAX_BACKOFF_SECONDS = 20.0

_GATE = asyncio.Semaphore(1)
"""转写接口按量计费且容易限流，全进程串行调用，稳比快重要。"""


class SttError(RuntimeError):
    """转写失败，message 可直接给用户看。"""


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if limit > 0 and len(cleaned) > limit:
        return cleaned[:limit].rstrip() + "…"
    return cleaned


def _endpoint(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if not base:
        raise SttError("未填写转写接口地址")
    if base.endswith(TRANSCRIPTION_ROUTE):
        return base
    return base + TRANSCRIPTION_ROUTE


async def _via_provider(path: Path, provider: Any) -> str:
    getter = getattr(provider, "get_text", None)
    if not callable(getter):
        raise SttError("所选服务商不支持语音转写")
    # AstrBot provider 和自定义 OpenAI 兼容接口共用同一条上游转写闸门，
    # 避免多个会话同时上传音频触发服务商 429。
    async with _GATE:
        try:
            return str(await getter(audio_url=str(path)) or "")
        except Exception as exc:
            raise SttError(f"服务商转写失败：{exc}") from exc


def _retry_delay(attempt: int, response: httpx.Response | None) -> float:
    """优先听服务端的 Retry-After，否则指数退避 + 抖动。"""
    if response is not None:
        try:
            hinted = float(response.headers.get("retry-after", ""))
        except ValueError:
            hinted = 0.0
        if hinted > 0:
            return min(hinted, MAX_BACKOFF_SECONDS)
    delay = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
    return delay * (0.7 + random.random() * 0.6)


def _parse_transcription(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()

    if isinstance(body, dict):
        for key in ("text", "transcript", "result"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise SttError("转写接口没有返回文本")


async def _via_openai_compatible(
    path: Path, settings: AudioSettings, client: httpx.AsyncClient
) -> str:
    """调用 OpenAI 兼容的转写接口，遇到限流会退避重试。"""
    url = _endpoint(settings.api_base)
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    payload = {"model": settings.model, "temperature": "0"}
    last_error = "未知错误"

    async with _GATE:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with path.open("rb") as handle:
                    files = {"file": (path.name, handle, "audio/wav")}
                    response = await client.post(
                        url, headers=headers, data=payload, files=files, timeout=120.0
                    )
            except OSError as exc:
                raise SttError(f"读不到音轨文件：{exc}") from exc
            except httpx.HTTPError as exc:
                last_error = f"请求失败：{exc}"
                if attempt >= MAX_ATTEMPTS:
                    break
                await asyncio.sleep(_retry_delay(attempt, None))
                continue

            if response.status_code < 400:
                return _parse_transcription(response)

            retryable = response.status_code == 429 or response.status_code >= 500
            last_error = (
                "接口限流（429）"
                if response.status_code == 429
                else f"接口返回 {response.status_code}"
            )
            if not retryable or attempt >= MAX_ATTEMPTS:
                break
            await asyncio.sleep(_retry_delay(attempt, response))

    raise SttError(f"语音转写失败：{last_error}")


def resolve_provider(context: Any, settings: AudioSettings) -> Any:
    """按配置找 STT 服务商；没指定就用 AstrBot 当前启用的那个。"""
    if settings.stt_provider_id:
        getter = getattr(context, "get_provider_by_id", None)
        if callable(getter):
            try:
                provider = getter(settings.stt_provider_id)
            except Exception:
                provider = None
            if provider is not None:
                return provider
    getter = getattr(context, "get_using_stt_provider", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


async def transcribe(
    path: Path,
    settings: AudioSettings,
    context: Any,
    client: httpx.AsyncClient,
) -> str:
    """把音轨转成文字。"""
    provider = resolve_provider(context, settings)
    if provider is not None:
        text = await _via_provider(path, provider)
        if text.strip():
            return _clip(text, settings.max_transcript_chars)

    if settings.use_custom_api:
        text = await _via_openai_compatible(path, settings, client)
        return _clip(text, settings.max_transcript_chars)

    raise SttError("没有可用的语音识别服务，请在插件配置里选一个或填写自定义接口")


def describe_backend(context: Any, settings: AudioSettings) -> str:
    """给诊断命令用的一句话说明。"""
    if not settings.transcribe:
        return "未启用"
    provider = resolve_provider(context, settings)
    if provider is not None:
        meta = getattr(provider, "meta", None)
        label = ""
        try:
            label = getattr(meta(), "id", "") if callable(meta) else getattr(meta, "id", "")
        except Exception:
            label = ""
        return f"AstrBot 服务商（{label or '当前启用'}）"
    if settings.use_custom_api:
        return f"自定义接口（{settings.model}）"
    return "未配置"
