"""整片视频理解后端。

AstrBot 的通用 ``ProviderRequest`` 目前没有统一的视频字段，所以整片视频不能
像图片一样直接塞给当前 Provider。本模块把这条能力做成一个可选、低并发的
外部后端，并把结果作为“不可信事实报告”交回主模型。没有配置或调用失败时，
主流程仍然继续使用本地抽帧、字幕和音频，不会因为某个视频 API 挂掉而丢消息。

支持四类常见接口：

* Gemini ``generateContent``，官方大文件走 Files API；
* OpenAI 兼容 ``video_url``；
* 百炼 Qwen 的内嵌 Base64 / 临时 OSS URL；
* Moonshot Kimi 的 ``/files`` + ``ms://`` 引用。

所有网络请求共用一个串行闸门，并且只对 429/5xx 做一次短退避，避免把上游
限流放大成并发风暴。
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import mimetypes
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .models import MediaItem, MediaKind
from .settings import NativeVideoSettings
from .tempstore import TempStore

MB = 1024 * 1024
_GEMINI_HOSTS = {"generativelanguage.googleapis.com", "aiplatform.googleapis.com"}
_QWEN_INLINE_BYTES = int(7.4 * MB)
_MAX_RESPONSE_BYTES = 2 * MB
_CHUNK_SIZE = 1024 * 1024
_QWEN_UPLOAD_HOST_SUFFIXES = ("aliyuncs.com", "aliyun.com")


class NativeVideoError(RuntimeError):
    """原生视频后端失败，文本可安全展示给模型。"""

    def __init__(self, detail: str, user_message: str | None = None) -> None:
        super().__init__(detail)
        self.user_message = user_message or detail


class NativeVideoAnalyzer:
    """可选的整片视频分析器。"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: NativeVideoSettings,
        *,
        runner: Any = None,
        store: Any = None,
        log: Any = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.runner = runner
        self.store = store
        self.log = log or (lambda _message: None)
        self._gate = asyncio.Semaphore(1)
        self._last_request = 0.0
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

    @property
    def only_mode(self) -> bool:
        """保留兼容接口；当前设计始终允许抽帧回退。"""
        return False

    def configure(self, settings: NativeVideoSettings) -> None:
        if settings != self.settings:
            self._cache.clear()
        self.settings = settings

    async def close(self) -> None:
        self._cache.clear()

    def should_attempt(self, item: MediaItem) -> bool:
        return bool(
            self.settings.automatic
            and item.kind is MediaKind.VIDEO
            and item.path is not None
            and item.path.is_file()
            and self.settings.api_key
            and self._model()
        )

    async def analyze(
        self,
        path: Path,
        name: str = "视频",
        question: str = "",
    ) -> str:
        """把本地视频交给配置的原生模型，返回受长度限制的事实报告。"""
        if not self.settings.enabled:
            raise NativeVideoError(
                "native video mode is disabled",
                "整片视频模型功能当前是关闭的。",
            )
        if not self.settings.api_key:
            raise NativeVideoError("missing native video API key", "没有配置整片视频模型 API Key。")
        provider = self._provider()
        if provider not in {"gemini", "openai", "qwen", "kimi"}:
            raise NativeVideoError(
                "native provider is not configured",
                "整片视频模型没有配置可识别的服务商。",
            )
        if not self._model():
            raise NativeVideoError(
                "native model is not configured",
                "整片视频模型没有配置模型名。",
            )
        base = self._base(provider)
        parsed_base = urlparse(base)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
            raise NativeVideoError(
                "native API base is not configured",
                "整片视频模型没有配置有效的 API 地址。",
            )
        if not path.is_file():
            raise NativeVideoError("video file is missing", "视频源文件已经不存在，无法整片上传。")

        size = path.stat().st_size
        limit = self.settings.max_upload_mb * MB
        if size <= 0:
            raise NativeVideoError("video file is empty", "视频文件为空，无法上传。")
        if size > limit:
            raise NativeVideoError(
                "native upload size limit exceeded",
                (
                    f"视频约 {size / MB:.1f} MB，超过整片视频模型的 "
                    f"{self.settings.max_upload_mb} MB 限制。"
                ),
            )

        key = self._cache_key(path, name, question)
        cached = self._cache.get(key)
        if cached is not None and cached[0] > time.monotonic():
            self._cache.move_to_end(key)
            return cached[1]

        async with self._gate:
            cached = self._cache.get(key)
            if cached is not None and cached[0] > time.monotonic():
                self._cache.move_to_end(key)
                return cached[1]
            report = await self._analyze_locked(path, name, question)
            self._cache[key] = (time.monotonic() + 900.0, report)
            self._cache.move_to_end(key)
            while len(self._cache) > 32:
                self._cache.popitem(last=False)
            return report

    def _cache_key(self, path: Path, name: str, question: str) -> str:
        try:
            stat = path.stat()
            fingerprint = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            fingerprint = str(path)
        return "|".join(
            (
                fingerprint,
                self._provider(),
                self._model(),
                name,
                question.strip(),
                self.settings.prompt,
            )
        )

    def user_error(self, exc: Exception) -> str:
        if isinstance(exc, NativeVideoError):
            return exc.user_message
        return "整片视频模型暂时不可用，已回退到关键帧、字幕和音频分析。"

    async def _analyze_locked(self, path: Path, name: str, question: str) -> str:
        working = path
        temporary: Path | None = None
        try:
            provider = self._provider()
            inline_limit = self.settings.inline_mb * MB
            if provider == "qwen":
                inline_limit = min(inline_limit, _QWEN_INLINE_BYTES)

            if path.stat().st_size > inline_limit and self.settings.auto_compress:
                if self.runner is None or self.store is None:
                    raise NativeVideoError(
                        "compression dependencies unavailable",
                        "整片视频超过内嵌上限，且当前没有可用的 ffmpeg 压缩路径。",
                    )
                temporary = self.store.download_path(".mp4")
                await self.runner.compress_video(
                    path,
                    temporary,
                    max_seconds=self.settings.compress_max_seconds,
                    height=self.settings.compress_height,
                    crf=self.settings.compress_crf,
                    timeout=float(self.settings.timeout_seconds),
                )
                working = temporary

            mime = mimetypes.guess_type(working.name)[0] or "video/mp4"
            if provider == "gemini":
                return await self._analyze_gemini(working, mime, name, question, inline_limit)
            if provider == "qwen":
                return await self._analyze_qwen(working, mime, name, question, inline_limit)
            if provider == "kimi":
                return await self._analyze_kimi(working, mime, name, question)
            return await self._analyze_openai(working, mime, name, question, inline_limit)
        finally:
            if temporary is not None:
                TempStore.discard(temporary)

    # ------------------------------------------------------------------
    # Provider selection and common request helpers
    # ------------------------------------------------------------------

    def _model(self) -> str:
        if self.settings.model:
            return self.settings.model.removeprefix("models/").strip()
        provider = self._provider()
        return {
            "gemini": "gemini-2.5-flash",
            "qwen": "qwen3.7-plus",
            "kimi": "kimi-k2.6",
        }.get(provider, "")

    def _base(self, provider: str | None = None) -> str:
        provider = provider or self._provider()
        if self.settings.api_base:
            return self.settings.api_base.rstrip("/")
        return {
            "gemini": "https://generativelanguage.googleapis.com",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "kimi": "https://api.moonshot.cn/v1",
        }.get(provider, "")

    def _provider(self) -> str:
        explicit = self.settings.provider
        if explicit != "auto":
            return explicit
        base = self._base_from_raw().lower()
        model = self.settings.model.lower()
        host = (urlparse(base).hostname or "").lower()
        if host in _GEMINI_HOSTS or "gemini" in model:
            return "gemini"
        if "moonshot" in host or "kimi" in model:
            return "kimi"
        if "dashscope" in host or "maas.aliyuncs.com" in host or "qwen" in model:
            return "qwen"
        return "openai"

    def _base_from_raw(self) -> str:
        return self.settings.api_base or ""

    def _is_gemini_official(self) -> bool:
        host = (urlparse(self._base("gemini")).hostname or "").lower()
        return host in _GEMINI_HOSTS

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        content: Any = None,
        files: Any = None,
        data: Any = None,
        params: Any = None,
        retry: bool = True,
        expect_json: bool = True,
    ) -> tuple[httpx.Response, dict[str, Any]]:
        attempts = 2 if retry else 1
        last_response: httpx.Response | None = None
        for attempt in range(attempts):
            wait = 0.65 - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = await self.client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    content=content,
                    files=files,
                    data=data,
                    params=params,
                    timeout=float(self.settings.timeout_seconds),
                )
            except httpx.HTTPError as exc:
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(4.0, 1.0 + attempt))
                    continue
                raise NativeVideoError(
                    f"native request failed: {type(exc).__name__}",
                    "整片视频模型网络请求失败，已回退到本地视觉分析。",
                ) from exc
            last_response = response
            if response.status_code == 429 and attempt + 1 < attempts:
                await asyncio.sleep(_retry_after(response.headers.get("retry-after")))
                continue
            if response.status_code >= 500 and attempt + 1 < attempts:
                await asyncio.sleep(min(4.0, 1.0 + attempt))
                continue
            break

        assert last_response is not None
        response = last_response
        if response.status_code < 200 or response.status_code >= 300:
            message = _error_text(response)
            if response.status_code == 429:
                message = "整片视频模型接口限流（429），本轮已回退到关键帧分析。"
            raise NativeVideoError(
                f"native API HTTP {response.status_code}: {message}",
                message,
            )
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise NativeVideoError(
                "native response too large",
                "整片视频模型返回内容过大，已忽略。",
            )
        if not expect_json and not response.content.strip():
            return response, {}
        try:
            payload = response.json()
        except ValueError as exc:
            if not expect_json:
                return response, {}
            raise NativeVideoError(
                "native response is not JSON",
                "整片视频模型返回了无法解析的结果。",
            ) from exc
        if not isinstance(payload, dict):
            raise NativeVideoError(
                "native response is not an object",
                "整片视频模型返回了无效结果。",
            )
        return response, payload

    def _headers(
        self,
        *,
        gemini: bool = False,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if gemini and self._is_gemini_official():
            headers = {"x-goog-api-key": self.settings.api_key}
        else:
            headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        if extra:
            headers.update(extra)
        return headers

    def _prompt(self, name: str, question: str) -> str:
        suffix = f"\n用户额外问题：{question.strip()}" if question.strip() else ""
        return f"{self.settings.prompt}\n视频名称：{name}{suffix}"[:30000]

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    async def _analyze_gemini(
        self,
        path: Path,
        mime: str,
        name: str,
        question: str,
        inline_limit: int,
    ) -> str:
        size = path.stat().st_size
        file_name = ""
        file_uri = ""
        try:
            if size <= inline_limit:
                data = await asyncio.to_thread(path.read_bytes)
                payload = {
                    "system_instruction": {"parts": [{"text": self._prompt(name, question)}]},
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": mime,
                                        "data": base64.b64encode(data).decode("ascii"),
                                    }
                                },
                                {"text": "请分析整段视频。"},
                            ],
                        }
                    ],
                }
                return await self._gemini_generate(payload)

            if not self.settings.use_files_api or not self._is_gemini_official():
                raise NativeVideoError(
                    "large Gemini upload is unavailable",
                    "视频超过内嵌上限；只有 Gemini 官方接口支持当前的 Files API，"
                    "请开启官方接口或自动压缩。",
                )
            file_name, file_uri = await self._gemini_upload(path, mime)
            payload = {
                "system_instruction": {"parts": [{"text": self._prompt(name, question)}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"file_data": {"mime_type": mime, "file_uri": file_uri}},
                            {"text": "请分析整段视频。"},
                        ],
                    }
                ],
            }
            return await self._gemini_generate(payload)
        finally:
            if file_name:
                await self._gemini_delete(file_name)

    async def _gemini_generate(self, payload: dict[str, Any]) -> str:
        base = self._base("gemini")
        for suffix in ("/v1beta/openai", "/v1beta", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        url = f"{base}/v1beta/models/{self._model()}:generateContent"
        _response, body = await self._request_json(
            "POST", url, headers=self._headers(gemini=True), json_body=payload
        )
        candidates = body.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "\n".join(
            str(part.get("text", "")).strip()
            for part in parts
            if isinstance(part, dict) and str(part.get("text", "")).strip()
        ).strip()
        if not text:
            raise NativeVideoError("Gemini returned no text", "整片视频模型没有返回可用报告。")
        return _clip(text, self.settings.max_report_chars)

    async def _gemini_upload(self, path: Path, mime: str) -> tuple[str, str]:
        base = self._base("gemini")
        for suffix in ("/v1beta/openai", "/v1beta", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        url = f"{base}/upload/v1beta/files"
        start_headers = self._headers(
            gemini=True,
            extra={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            },
        )
        response, _payload = await self._request_json(
            "POST",
            url,
            headers=start_headers,
            json_body={"file": {"display_name": path.name, "mime_type": mime}},
            expect_json=False,
        )
        upload_url = response.headers.get("x-goog-upload-url", "")
        if not upload_url:
            raise NativeVideoError("Gemini upload URL missing", "Gemini 文件上传没有返回上传地址。")

        async def chunks() -> AsyncIterator[bytes]:
            with path.open("rb") as handle:
                while True:
                    chunk = await asyncio.to_thread(handle.read, _CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk

        put_headers = self._headers(
            gemini=True,
            extra={
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
                "Content-Type": "application/octet-stream",
                "Content-Length": str(path.stat().st_size),
            },
        )
        _response, body = await self._request_json(
            "PUT", upload_url, headers=put_headers, content=chunks(), retry=False
        )
        file_info = body.get("file") or {}
        file_name = str(file_info.get("name") or "")
        file_uri = str(file_info.get("uri") or "")
        state = str(file_info.get("state") or "")
        if not file_name or not file_uri:
            raise NativeVideoError(
                "Gemini upload response invalid",
                "Gemini 上传响应缺少文件信息。",
            )
        if state == "FAILED":
            raise NativeVideoError("Gemini file processing failed", "Gemini 处理视频文件失败。")
        if state != "ACTIVE":
            await self._wait_gemini_file(file_name, file_uri)
        return file_name, file_uri

    async def _wait_gemini_file(self, name: str, uri: str) -> None:
        base = self._base("gemini")
        for suffix in ("/v1beta/openai", "/v1beta", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        deadline = time.monotonic() + self.settings.timeout_seconds
        while time.monotonic() < deadline:
            _response, body = await self._request_json(
                "GET", f"{base}/v1beta/{name}", headers=self._headers(gemini=True), retry=False
            )
            info = body.get("file") or {}
            state = str(info.get("state") or "")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise NativeVideoError("Gemini file processing failed", "Gemini 处理视频文件失败。")
            await asyncio.sleep(1.5)
        raise NativeVideoError("Gemini file processing timed out", "Gemini 处理视频文件超时。")

    async def _gemini_delete(self, name: str) -> None:
        base = self._base("gemini")
        for suffix in ("/v1beta/openai", "/v1beta", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        try:
            await self.client.delete(
                f"{base}/v1beta/{name}",
                headers=self._headers(gemini=True),
                timeout=float(self.settings.timeout_seconds),
            )
        except httpx.HTTPError:
            self.log("Gemini 临时文件清理失败，服务端会自行过期")

    # ------------------------------------------------------------------
    # OpenAI-compatible / Qwen / Kimi
    # ------------------------------------------------------------------

    async def _analyze_openai(
        self, path: Path, mime: str, name: str, question: str, inline_limit: int
    ) -> str:
        if path.stat().st_size > inline_limit:
            raise NativeVideoError(
                "OpenAI compatible inline limit exceeded",
                "当前 OpenAI 兼容视频接口只接受内嵌视频；请开启自动压缩或降低视频大小。",
            )
        data = await asyncio.to_thread(path.read_bytes)
        ref = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        return await self._openai_request(ref, name, question)

    async def _analyze_qwen(
        self, path: Path, mime: str, name: str, question: str, inline_limit: int
    ) -> str:
        if path.stat().st_size <= min(inline_limit, _QWEN_INLINE_BYTES):
            data = await asyncio.to_thread(path.read_bytes)
            ref = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            return await self._openai_request(ref, name, question)
        ref = await self._qwen_upload(path, mime)
        return await self._openai_request(ref, name, question, oss=True)

    async def _analyze_kimi(self, path: Path, mime: str, name: str, question: str) -> str:
        ref = await self._kimi_upload(path, mime)
        try:
            return await self._openai_request(ref, name, question)
        finally:
            # 删除失败不影响报告，临时文件会在服务端过期。
            with contextlib.suppress(Exception):
                await self.client.delete(
                    f"{self._kimi_files_base()}/{ref.removeprefix('ms://')}",
                    headers=self._headers(),
                    timeout=float(self.settings.timeout_seconds),
                )

    async def _openai_request(
        self, ref: str, name: str, question: str, *, oss: bool = False
    ) -> str:
        base = self._base(self._provider())
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        video_part: dict[str, Any] = {"type": "video_url", "video_url": {"url": ref}}
        if self.settings.fps > 0:
            video_part["fps"] = self.settings.fps
        headers = self._headers(extra={"Content-Type": "application/json"})
        if oss:
            headers["X-DashScope-OssResourceResolve"] = "enable"
        payload = {
            "model": self._model(),
            "messages": [
                {"role": "system", "content": self._prompt(name, question)},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "请分析整段视频。"}, video_part],
                },
            ],
        }
        _response, body = await self._request_json("POST", url, headers=headers, json_body=payload)
        choices = body.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else ""
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise NativeVideoError(
                "OpenAI-compatible API returned no text",
                "整片视频模型没有返回可用报告。",
            )
        return _clip(content.strip(), self.settings.max_report_chars)

    async def _qwen_upload(self, path: Path, mime: str) -> str:
        base = self._base("qwen")
        for suffix in ("/compatible-mode/v1", "/chat/completions", "/api/v1", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        policy_url = f"{base}/api/v1/uploads"
        _response, body = await self._request_json(
            "GET",
            policy_url,
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
            params={"action": "getPolicy", "model": self._model()},
        )
        policy = body.get("data") or {}
        if not isinstance(policy, dict) or not policy.get("upload_host"):
            raise NativeVideoError("Qwen upload policy invalid", "百炼没有返回有效的视频上传凭证。")
        upload_host = str(policy["upload_host"])
        upload_host = _secure_qwen_upload_url(upload_host)
        key = f"{str(policy.get('upload_dir') or '').strip('/')}/{path.name}".lstrip("/")
        form = {
            "OSSAccessKeyId": str(policy.get("oss_access_key_id") or ""),
            "signature": str(policy.get("signature") or ""),
            "policy": str(policy.get("policy") or ""),
            "key": key,
            "success_action_status": "200",
        }
        if policy.get("x_oss_object_acl"):
            form["x-oss-object-acl"] = str(policy["x_oss_object_acl"])
        with path.open("rb") as handle:
            files = {"file": (path.name, handle, mime)}
            _response, _upload_body = await self._request_json(
                "POST",
                upload_host,
                data=form,
                files=files,
                retry=False,
                expect_json=False,
            )
        return f"oss://{key}"

    async def _kimi_upload(self, path: Path, mime: str) -> str:
        base = self._base("kimi").rstrip("/")
        url = f"{base}/files" if not base.endswith("/files") else base
        with path.open("rb") as handle:
            files = {"file": (path.name, handle, mime)}
            _response, body = await self._request_json(
                "POST",
                url,
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                data={"purpose": "video"},
                files=files,
                retry=False,
            )
        file_id = str(body.get("id") or (body.get("file") or {}).get("id") or "")
        if not file_id:
            raise NativeVideoError(
                "Kimi upload response invalid",
                "Kimi 视频上传没有返回文件编号。",
            )
        return f"ms://{file_id}"

    def _kimi_files_base(self) -> str:
        """返回 Kimi 文件资源根地址，兼容 api_base 已带 ``/files``。"""
        base = self._base("kimi").rstrip("/")
        return base if base.endswith("/files") else f"{base}/files"


def _retry_after(value: str | None) -> float:
    try:
        return min(8.0, max(1.0, float(value or 2)))
    except (TypeError, ValueError):
        return 2.0


def _secure_qwen_upload_url(value: str) -> str:
    """只允许百炼返回的阿里云对象存储上传地址。"""
    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise NativeVideoError(
            "Qwen upload URL is invalid", "百炼返回了无效的视频上传地址。"
        ) from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not any(
        host == suffix or host.endswith("." + suffix) for suffix in _QWEN_UPLOAD_HOST_SUFFIXES
    ):
        raise NativeVideoError(
            "Qwen upload URL is outside the allowlist",
            "百炼返回的视频上传地址不在安全域名范围内。",
        )
    return parsed.geturl()


def _error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:240] if text else f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or "请求被拒绝")
        if error:
            return str(error)
        if payload.get("message"):
            return str(payload["message"])
    return f"HTTP {response.status_code}"


def _clip(value: str, limit: int) -> str:
    value = re.sub(r"\x00", "", value).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 40)] + "\n[报告已按长度限制截断]"


__all__ = ["NativeVideoAnalyzer", "NativeVideoError"]
