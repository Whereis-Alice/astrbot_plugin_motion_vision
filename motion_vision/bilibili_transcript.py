"""B 站字幕统一服务。

优先使用 B 站官方/AI 字幕；没有字幕且用户明确开启回退时，才把已下载视频的音轨
送到必剪语音转写。BCut 上传地址严格校验域名，Cookie 永远不会随音频上传。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .bilibili_parser import BilibiliInfo
from .ffmpeg import FfmpegError, FfmpegRunner
from .settings import BilibiliSettings
from .tempstore import TempStore

_BCUT_BASE = "https://member.bilibili.com/x/bcut/rubick-interface"
_BCUT_HEADERS = {
    "User-Agent": "Bilibili/1.0.0 (https://www.bilibili.com)",
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
}
_BCUT_UPLOAD_SUFFIXES = ("biliapi.net", "bilivideo.com", "hdslb.com")
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_UPLOAD_CHUNKS = 512


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    text: str
    source: str
    full_text: str = ""
    language: str = ""
    fallback: bool = False


class BilibiliTranscriptService:
    """官方字幕、必剪回退和有界缓存的统一入口。"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        bili: Any,
        runner: FfmpegRunner,
        store: TempStore,
        settings: BilibiliSettings,
        *,
        log: Any = None,
    ) -> None:
        self.client = client
        self.bili = bili
        self.runner = runner
        self.store = store
        self.settings = settings
        self.log = log or (lambda _message: None)
        self._gate = asyncio.Semaphore(1)
        self._bcut_work_gate = asyncio.Semaphore(1)
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache: OrderedDict[str, tuple[float, TranscriptResult]] = OrderedDict()
        self._official_source: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._negative: OrderedDict[str, float] = OrderedDict()
        self._last_request = 0.0
        self._bcut_client: httpx.AsyncClient | None = None

    def configure(self, settings: BilibiliSettings) -> None:
        old_cookie_fingerprint = self._cookie_fingerprint()
        if (
            settings.cookie != self.settings.cookie
            or settings.subtitle_language != self.settings.subtitle_language
            or settings.subtitle_fallback != self.settings.subtitle_fallback
            or settings.max_subtitle_chars != self.settings.max_subtitle_chars
            or settings.caption_full_max_chars != self.settings.caption_full_max_chars
        ):
            self._cache.clear()
            self._official_source.clear()
            self._negative.clear()
        self.settings = settings
        # 扫码登录得到的 Cookie 由外部 provider 提供，配置对象本身可能完全没变。
        # 把脱敏指纹纳入缓存生命周期，避免登录/退出后继续复用旧字幕。
        if old_cookie_fingerprint != self._cookie_fingerprint():
            self._cache.clear()
            self._official_source.clear()
            self._negative.clear()

    async def close(self) -> None:
        client = self._bcut_client
        self._bcut_client = None
        if client is not None:
            await client.aclose()
        self._cache.clear()
        self._official_source.clear()
        self._negative.clear()
        self._locks.clear()

    async def fetch(
        self,
        info: BilibiliInfo,
        *,
        full: bool = False,
        allow_fallback: bool = True,
        max_chars: int | None = None,
    ) -> TranscriptResult | None:
        """取一份字幕；``full`` 只改变文字上限，不会绕过安全截断。"""
        if max_chars is None:
            limit = (
                self.settings.caption_full_max_chars if full else self.settings.max_subtitle_chars
            )
        else:
            limit = max(500, min(int(max_chars), self.settings.caption_full_max_chars))
        key = (
            f"{info.key}:{self._cookie_fingerprint()}:{limit}:{self.settings.subtitle_language}:"
            f"{self.settings.subtitle_fallback}"
        )
        cached = self._get_cache(key)
        if cached is not None:
            return cached
        if self._negative_hit(key):
            return None

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._get_cache(key)
            if cached is not None:
                return cached
            if self._negative_hit(key):
                return None

            official = await self._official(info, limit)
            if official is not None:
                self._put_cache(key, official)
                return official

            if not allow_fallback or self.settings.subtitle_fallback != "bcut":
                self._mark_negative(key)
                return None
            try:
                fallback = await self._bcut(info, limit)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log(f"B 站必剪转写失败（{info.key}）：{type(exc).__name__}")
                self._mark_negative(key)
                return None
            if fallback is None:
                self._mark_negative(key)
                return None
            self._put_cache(key, fallback)
            return fallback

    async def _official(self, info: BilibiliInfo, limit: int) -> TranscriptResult | None:
        source_key = f"{info.key}:{self._cookie_fingerprint()}:{self.settings.subtitle_language}"
        cached_source = self._official_source.get(source_key)
        if cached_source is not None:
            expires, text = cached_source
            if expires > time.monotonic():
                self._official_source.move_to_end(source_key)
                return TranscriptResult(
                    text=_sample_timeline(text, limit),
                    full_text=text,
                    source="B 站官方/AI 字幕",
                )
            self._official_source.pop(source_key, None)
        try:
            source_limit = max(limit, self.settings.caption_full_max_chars)
            text = await self.bili._fetch_official_subtitle(info, source_limit)
        except Exception as exc:
            self.log(f"B 站官方字幕读取失败（{info.key}）：{type(exc).__name__}")
            return None
        if not text:
            return None
        self._official_source[source_key] = (time.monotonic() + 600.0, text)
        self._official_source.move_to_end(source_key)
        while len(self._official_source) > 64:
            self._official_source.popitem(last=False)
        return TranscriptResult(
            text=_sample_timeline(text, limit),
            full_text=text,
            source="B 站官方/AI 字幕",
        )

    def _cookie_fingerprint(self) -> str:
        """返回当前登录态的短指纹，不把 Cookie 本身写入缓存键或日志。"""
        provider = getattr(self.bili, "_effective_cookie", None)
        try:
            cookie = str(provider() or "") if callable(provider) else self.settings.cookie
        except Exception:
            cookie = self.settings.cookie
        return hashlib.sha256(cookie.encode("utf-8")).hexdigest()[:12]

    async def _bcut(self, info: BilibiliInfo, limit: int) -> TranscriptResult | None:
        # 下载、抽音和上传都是重活；即使不同视频的字幕 key 不同，也只
        # 允许一个必剪任务同时运行，避免把上游和本机 ffmpeg 一起打满。
        async with self._bcut_work_gate:
            if not self.runner.available:
                return None
            video_path = await self.bili.download(info)
            audio_path = self.store.audio_path()
            parsed: tuple[str, str] | None = None
            try:
                clip = await self.runner.extract_audio(
                    video_path,
                    audio_path,
                    seconds=min(info.duration or 600.0, 600.0),
                    timeout=max(30.0, float(self.settings.bcut_timeout_seconds)),
                )
                parsed = await self._transcribe_audio(clip.path)
            except FfmpegError:
                return None
            finally:
                TempStore.discard(audio_path)
            if parsed is None:
                return None
            full_text, language = parsed
            if not full_text:
                return None
            # 统一约束 ``full`` 工具和 send_file 的上限；官方字幕在客户端
            # 层已经有同样的边界，必剪回退也不能因为识别结果异常变成无界文本。
            bounded_full_text = _sample_timeline(
                full_text,
                max(500, self.settings.caption_full_max_chars),
            )
            return TranscriptResult(
                text=_sample_timeline(bounded_full_text, limit),
                full_text=bounded_full_text,
                language=language,
                source="必剪语音转写（仅依据视频声音）",
                fallback=True,
            )

    async def _transcribe_audio(self, path: Path) -> tuple[str, str]:
        size = path.stat().st_size
        extension = path.suffix.lower().lstrip(".") or "wav"
        create = await self._bcut_json(
            "POST",
            "/resource/create",
            {
                "type": 2,
                "name": f"audio.{extension}",
                "size": size,
                "ResourceFileType": extension,
                "model_id": "8",
            },
        )
        upload_urls = create.get("upload_urls")
        per_size = _safe_int(create.get("per_size"))
        if not isinstance(upload_urls, list) or not upload_urls or per_size <= 0:
            raise RuntimeError("BCut did not return upload chunks")
        if len(upload_urls) > _MAX_UPLOAD_CHUNKS:
            raise RuntimeError("BCut returned too many upload chunks")

        etags: list[str] = []
        for index, raw_url in enumerate(upload_urls):
            upload_url = _secure_bcut_upload_url(str(raw_url or ""))
            chunk = await asyncio.to_thread(_read_chunk, path, index * per_size, per_size)
            response = await self._bcut_upload(upload_url, chunk)
            etags.append(response.headers.get("ETag", "").strip('"'))

        committed = await self._bcut_json(
            "POST",
            "/resource/create/complete",
            {
                "InBossKey": create.get("in_boss_key"),
                "ResourceId": create.get("resource_id"),
                "Etags": ",".join(etags),
                "UploadId": create.get("upload_id"),
                "model_id": "8",
            },
        )
        task = await self._bcut_json(
            "POST",
            "/task",
            {"resource": committed.get("download_url"), "model_id": "8"},
        )
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("BCut did not return a task id")

        deadline = time.monotonic() + self.settings.bcut_timeout_seconds
        while time.monotonic() < deadline:
            data = await self._bcut_json(
                "GET",
                "/task/result",
                None,
                params={"model_id": "7", "task_id": task_id},
            )
            state = _safe_int(data.get("state"))
            if state == 4:
                result = data.get("result") or {}
                parsed = _parse_bcut_result(result)
                if parsed is None:
                    raise RuntimeError("BCut returned an empty transcript")
                return parsed
            if state == 3:
                raise RuntimeError("BCut task failed")
            await asyncio.sleep(2.0)
        raise TimeoutError("BCut transcript timed out")

    async def _bcut_upload(self, url: str, chunk: bytes) -> httpx.Response:
        client = await self._get_bcut_client()
        response = await client.put(
            url,
            content=chunk,
            headers={"Content-Type": "application/octet-stream"},
            timeout=max(30.0, float(self.settings.bcut_timeout_seconds)),
            follow_redirects=False,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"BCut upload HTTP {response.status_code}")
        return response

    async def _bcut_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        async with self._gate:
            wait = 0.8 - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()
            client = await self._get_bcut_client()
            response = await client.request(
                method,
                f"{_BCUT_BASE}{path}",
                params=params,
                json=body,
                headers=_BCUT_HEADERS,
                timeout=max(30.0, float(self.settings.bcut_timeout_seconds)),
                follow_redirects=False,
            )
        if response.status_code == 429:
            raise RuntimeError("BCut HTTP 429")
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"BCut HTTP {response.status_code}")
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("BCut response too large")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("BCut returned invalid JSON") from exc
        if not isinstance(payload, dict) or _safe_int(payload.get("code"), -1) != 0:
            raise RuntimeError("BCut rejected the request")
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise RuntimeError("BCut returned invalid data")
        return data

    async def _get_bcut_client(self) -> httpx.AsyncClient:
        if self._bcut_client is None:
            # 独立客户端故意不继承 B 站登录 Cookie，避免把凭据带到音频 CDN。
            self._bcut_client = httpx.AsyncClient(
                follow_redirects=False,
                limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
            )
        return self._bcut_client

    def _get_cache(self, key: str) -> TranscriptResult | None:
        item = self._cache.get(key)
        if item is None:
            return None
        expires, value = item
        if expires <= time.monotonic():
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return value

    def _put_cache(self, key: str, value: TranscriptResult) -> None:
        self._cache[key] = (time.monotonic() + 600.0, value)
        self._cache.move_to_end(key)
        while len(self._cache) > 64:
            self._cache.popitem(last=False)

    def _negative_hit(self, key: str) -> bool:
        expires = self._negative.get(key)
        if expires is None:
            return False
        if expires <= time.monotonic():
            self._negative.pop(key, None)
            return False
        self._negative.move_to_end(key)
        return True

    def _mark_negative(self, key: str) -> None:
        self._negative[key] = time.monotonic() + 30.0
        self._negative.move_to_end(key)
        while len(self._negative) > 64:
            self._negative.popitem(last=False)


def _parse_bcut_result(value: Any) -> tuple[str, str] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return (value.strip(), "") if value.strip() else None
    if not isinstance(value, dict):
        return None
    segments: list[str] = []
    for item in value.get("utterances") or value.get("segments") or []:
        if not isinstance(item, dict):
            continue
        text = _clean_segment(item.get("transcript") or item.get("text"))
        if not text:
            continue
        start = _safe_float(item.get("start_time", item.get("start", 0)))
        if start > 10000:
            start /= 1000.0
        segments.append(f"[{_format_time(start)}] {text}")
    if not segments:
        text = _clean_segment(value.get("text") or value.get("transcript"))
        if text:
            segments.append(text)
    if not segments:
        return None
    return "\n".join(segments), str(value.get("language") or "zh")


def _sample_timeline(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    head_budget = int(limit * 0.42)
    tail_budget = int(limit * 0.32)
    head: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > head_budget:
            break
        head.append(line)
        used += len(line) + 1
    tail: list[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) + 1 > tail_budget:
            break
        tail.append(line)
        used += len(line) + 1
    marker = "[中间部分因长度限制已省略]"
    output = "\n".join([*head, marker, *reversed(tail)])
    if len(output) <= limit:
        return output
    return output[: max(0, limit - 1)].rstrip() + "…"


def _secure_bcut_upload_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError as exc:
        raise RuntimeError("BCut upload URL is invalid") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not any(
        host == suffix or host.endswith("." + suffix) for suffix in _BCUT_UPLOAD_SUFFIXES
    ):
        raise RuntimeError("BCut upload URL is outside the allowlist")
    return parsed._replace(scheme="https").geturl()


def _read_chunk(path: Path, start: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(size)


def _clean_segment(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


__all__ = [
    "BilibiliTranscriptService",
    "TranscriptResult",
    "_secure_bcut_upload_url",
]
