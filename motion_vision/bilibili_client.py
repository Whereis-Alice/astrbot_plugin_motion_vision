"""B 站元数据、字幕与下载的共享客户端。

客户端集中处理三件容易失控的事情：API 串行限流、短生命周期缓存和 yt-dlp 下载
的有界并发。解析规则与同步下载实现分别位于 ``bilibili_parser`` 和
``bilibili_download``，避免所有逻辑再次堆进一个入口文件。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from .bilibili_download import download_video_sync
from .bilibili_parser import (
    SUBTITLE_HOST_SUFFIXES,
    USER_AGENT,
    BilibiliError,
    BilibiliInfo,
    BilibiliReference,
    _allowed_host,
    _clean_text,
    _positive_float,
    _safe_int,
    _safe_subtitle_url,
    _subtitle_lines,
    _truncate_timeline,
    build_context,
    parse_reference,
)
from .models import MediaItem, MediaKind
from .settings import MB, BilibiliSettings

BILIBILI_API = "https://api.bilibili.com"
VIEW_ENDPOINT = f"{BILIBILI_API}/x/web-interface/view"
PLAYER_ENDPOINT = f"{BILIBILI_API}/x/player/v2"
NAV_ENDPOINT = f"{BILIBILI_API}/x/web-interface/nav"

MAX_API_BYTES = 4 * MB
MAX_SUBTITLE_BYTES = 8 * MB
MAX_SHORT_BODY_BYTES = 512 * 1024
API_MIN_INTERVAL = 0.45
SUBTITLE_FAILURE_TTL = 30.0
DOWNLOAD_TAIL_CLOSE_TIMEOUT = 5.0

RequestInterval = float | Callable[[], float]


class BilibiliClient:
    """共享 B 站元数据、字幕和下载服务。"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        download_path_factory: Callable[[str], Path],
        settings: BilibiliSettings,
        *,
        ffmpeg_path: str = "",
        timeout_seconds: int = 180,
        max_download_mb: int = 100,
        log: Callable[[str], None] | None = None,
        request_interval: RequestInterval | None = None,
        saved_cookie_provider: Callable[[], str] | None = None,
    ) -> None:
        self.client = client
        self._download_path = download_path_factory
        self.settings = settings
        self.ffmpeg_path = ffmpeg_path
        self.timeout_seconds = max(30, timeout_seconds)
        self.max_download_mb = max(1, max_download_mb)
        self._log = log or (lambda _message: None)
        self._saved_cookie_provider = saved_cookie_provider
        self._request_interval = (
            request_interval if request_interval is not None else API_MIN_INTERVAL
        )
        self._api_gate = asyncio.Semaphore(1)
        self._download_gate = asyncio.Semaphore(1)
        self._last_api_request = 0.0
        self._metadata: OrderedDict[str, tuple[float, BilibiliInfo]] = OrderedDict()
        self._subtitles: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._subtitle_failures: OrderedDict[str, float] = OrderedDict()
        self._downloads: OrderedDict[str, tuple[float, Path]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._download_tail: asyncio.Task[Any] | None = None
        self._transcript_service: Any = None

    def set_transcript_service(self, service: Any) -> None:
        """挂接可选的必剪回退服务，避免 B 站客户端反向依赖其实现。"""
        self._transcript_service = service

    def configure(
        self,
        settings: BilibiliSettings,
        *,
        ffmpeg_path: str = "",
        timeout_seconds: int = 180,
        max_download_mb: int = 100,
        saved_cookie_provider: Callable[[], str] | None = None,
    ) -> None:
        old_cookie = self._effective_cookie()
        old_subtitle_options = (
            self.settings.fetch_subtitles,
            self.settings.max_subtitle_chars,
            self.settings.subtitle_language,
            self.settings.subtitle_fallback,
        )
        self.settings = settings
        self.ffmpeg_path = ffmpeg_path
        self.timeout_seconds = max(30, timeout_seconds)
        self.max_download_mb = max(1, max_download_mb)
        self._saved_cookie_provider = saved_cookie_provider
        if old_cookie != self._effective_cookie():
            self._metadata.clear()
            self._subtitles.clear()
            self._subtitle_failures.clear()
            # 登录态可能改变可见清晰度、字幕和受限视频内容；不要复用旧
            # Cookie 生成的下载文件。文件本身交给 TempStore 按 TTL 回收，
            # 避免删除仍被会话回看记录引用的源文件。
            self._downloads.clear()
        if old_subtitle_options != (
            settings.fetch_subtitles,
            settings.max_subtitle_chars,
            settings.subtitle_language,
            settings.subtitle_fallback,
        ):
            self._subtitles.clear()
            self._subtitle_failures.clear()

    async def close(self) -> None:
        tail = self._download_tail
        self._download_tail = None
        if tail is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(tail),
                    timeout=DOWNLOAD_TAIL_CLOSE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # yt-dlp 在线程里运行，不能被 asyncio 强行取消。插件卸载不能无限
                # 等待一个失联的下载；它结束后仍会由 done callback 清理目标文件。
                self._log("B 站下载线程仍在退出，插件先完成卸载")
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._metadata.clear()
        self._subtitles.clear()
        self._subtitle_failures.clear()
        self._downloads.clear()
        self._locks.clear()

    async def resolve(self, reference: BilibiliReference) -> BilibiliInfo:
        """解析短链、标题、分 P、CID 和时长。"""

        key = reference.key
        cached = self._cache_get(self._metadata, key)
        if cached is not None:
            return cached
        lock = self._locks.setdefault("meta:" + key, asyncio.Lock())
        async with lock:
            cached = self._cache_get(self._metadata, key)
            if cached is not None:
                return cached
            resolved = reference
            if reference.kind == "short_url":
                target = await self._resolve_short_url(reference.value)
                resolved = (
                    parse_reference(
                        target,
                        quoted=reference.quoted,
                        title=reference.title,
                        description=reference.description,
                        author=reference.author,
                    )
                    or reference
                )
            params = {"p": str(max(1, resolved.page))}
            if resolved.kind == "bvid":
                params["bvid"] = resolved.value
            elif resolved.kind == "aid":
                params["aid"] = resolved.value
            else:
                raise BilibiliError(
                    "short URL did not resolve to a Bilibili video",
                    "这个 B 站短链没有跳转到可识别的视频。",
                )
            payload = await self._api_json(VIEW_ENDPOINT, params=params)
            if _safe_int(payload.get("code"), -1) != 0:
                raise BilibiliError(
                    f"view API returned {payload.get('code')}",
                    "B 站视频信息读取失败，链接可能已失效或需要登录。",
                )
            data = payload.get("data")
            if not isinstance(data, dict):
                raise BilibiliError(
                    "view API returned invalid data", "B 站没有返回有效的视频信息。"
                )
            pages = data.get("pages") if isinstance(data.get("pages"), list) else []
            page = max(1, resolved.page)
            if pages and page > len(pages):
                raise BilibiliError(
                    "requested page is out of range",
                    f"这个视频没有第 {page} 个分 P（共 {len(pages)} 个分 P）。",
                )
            page_data = pages[page - 1] if pages else {}
            if not isinstance(page_data, dict):
                page_data = {}
            bvid = _clean_text(data.get("bvid")) or (
                resolved.value if resolved.kind == "bvid" else ""
            )
            cid = _safe_int(page_data.get("cid")) or _safe_int(data.get("cid"))
            if not bvid or cid <= 0:
                raise BilibiliError(
                    "view API omitted bvid/cid", "B 站视频信息不完整，暂时无法读取。"
                )
            title = _clean_text(data.get("title")) or resolved.title or bvid
            page_name = _clean_text(page_data.get("part"))
            if page_name and page > 1:
                title = f"{title} (P{page}: {page_name})"
            owner = data.get("owner")
            dimension = page_data.get("dimension")
            if not isinstance(dimension, dict):
                dimension = data.get("dimension")
            if not isinstance(dimension, dict):
                dimension = {}
            stat = data.get("stat")
            info = BilibiliInfo(
                bvid=bvid,
                aid=_safe_int(data.get("aid")) or None,
                cid=cid,
                page=page,
                title=title,
                description=_clip_text(_clean_text(data.get("desc")) or resolved.description, 2000),
                author=(
                    _clean_text(owner.get("name")) if isinstance(owner, dict) else resolved.author
                ),
                duration=_positive_float(page_data.get("duration"))
                or _positive_float(data.get("duration")),
                canonical_url=f"https://www.bilibili.com/video/{bvid}"
                + (f"?p={page}" if page > 1 else ""),
                page_count=max(1, len(pages)),
                part_title=page_name,
                pubdate=_safe_int(data.get("pubdate")) or None,
                category=_clean_text(data.get("tname")),
                cover_url=_clean_text(data.get("pic")),
                width=_safe_int(dimension.get("width")),
                height=_safe_int(dimension.get("height")),
                stats=_public_stats(stat),
            )
            self._cache_put(self._metadata, key, info)
            self._cache_put(self._metadata, info.key, info)
            return info

    async def fetch_subtitle(self, info: BilibiliInfo, max_chars: int | None = None) -> str:
        """读取并压缩带时间点的官方/AI 字幕；无字幕返回空字符串。"""

        if not self.settings.fetch_subtitles:
            return ""
        return await self.fetch_subtitle_with_limit(
            info,
            self.settings.max_subtitle_chars if max_chars is None else max_chars,
        )

    async def fetch_subtitle_with_limit(self, info: BilibiliInfo, max_chars: int) -> str:
        """读取字幕并按调用方上限截断；工具的完整模式复用同一实现。"""
        max_chars = max(500, min(int(max_chars), 200000))
        service = self._transcript_service
        if service is not None:
            # 统一走 TranscriptService，避免先在这里请求一次官方字幕，随后
            # 为判断是否需要必剪回退又重复请求一次。官方字幕仍由本类的
            # `_fetch_official_subtitle` 提供，依赖方向不会反过来。
            with contextlib.suppress(Exception):
                result = await service.fetch(
                    info,
                    full=max_chars >= self.settings.caption_full_max_chars,
                    max_chars=max_chars,
                    allow_fallback=True,
                )
                return result.text if result is not None else ""
        official = await self._fetch_official_subtitle(info, max_chars)
        if official or self.settings.subtitle_fallback != "bcut":
            return official
        return ""

    async def _fetch_official_subtitle(self, info: BilibiliInfo, max_chars: int) -> str:
        """只访问 B 站官方字幕接口；必剪服务通过外部挂接点调用。"""
        cache_key = (
            f"{info.key}:{_cookie_fingerprint(self._effective_cookie())}:"
            f"{self.settings.subtitle_language}"
        )
        cached = self._cache_get(self._subtitles, cache_key)
        if cached is not None:
            return _truncate_timeline(cached.splitlines(), max_chars) if cached else ""
        if self._subtitle_failure_cached(cache_key):
            return ""

        lock = self._locks.setdefault("subtitle:" + cache_key, asyncio.Lock())
        async with lock:
            cached = self._cache_get(self._subtitles, cache_key)
            if cached is not None:
                return _truncate_timeline(cached.splitlines(), max_chars) if cached else ""
            if self._subtitle_failure_cached(cache_key):
                return ""
            try:
                payload = await self._api_json(
                    PLAYER_ENDPOINT,
                    params={"bvid": info.bvid, "cid": str(info.cid)},
                )
                if _safe_int(payload.get("code"), -1) != 0:
                    self._mark_subtitle_failure(cache_key)
                    return ""
                data = payload.get("data")
                subtitle_root = data.get("subtitle") if isinstance(data, dict) else None
                candidates = (
                    subtitle_root.get("subtitles") if isinstance(subtitle_root, dict) else None
                )
                if not isinstance(candidates, list):
                    self._cache_put(self._subtitles, cache_key, "")
                    return ""
                ordered = self._ordered_subtitles(
                    candidates,
                    self.settings.subtitle_language,
                )
                if not ordered:
                    self._cache_put(self._subtitles, cache_key, "")
                    return ""
                for candidate in ordered[:8]:
                    raw_url = _clean_text(candidate.get("subtitle_url"))
                    try:
                        subtitle_url = _safe_subtitle_url(raw_url)
                        body = await self._bounded_request(
                            subtitle_url,
                            headers={
                                "User-Agent": USER_AGENT,
                                "Referer": "https://www.bilibili.com/",
                            },
                            max_bytes=MAX_SUBTITLE_BYTES,
                            allow_cookie=False,
                            redirect_suffixes=SUBTITLE_HOST_SUFFIXES,
                        )
                        subtitle_payload = json.loads(body.decode("utf-8"))
                        lines = _subtitle_lines(subtitle_payload)
                    except (BilibiliError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        self._log(f"B 站候选字幕不可用（{info.key}）：{type(exc).__name__}")
                        continue
                    if lines:
                        # 缓存一份有界的时间线原文，之后切换普通/完整模式时
                        # 只在本地重截断，不再为同一视频重复打 B 站接口。
                        raw_text = _truncate_timeline(lines, 200000)
                        self._cache_put(self._subtitles, cache_key, raw_text)
                        return _truncate_timeline(raw_text.splitlines(), max_chars)
                self._cache_put(self._subtitles, cache_key, "")
                return ""
            except (BilibiliError, httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                self._log(f"B 站字幕读取失败（{info.key}）：{type(exc).__name__}")
                self._mark_subtitle_failure(cache_key)
                return ""

    async def verify_cookie(self) -> tuple[bool, str]:
        """验证当前有效 Cookie；返回 ``(是否登录, 用户名)``，不回显凭据。"""
        cookie = self._effective_cookie()
        if not cookie:
            return False, ""
        try:
            payload = await self._api_json(NAV_ENDPOINT, params={})
        except BilibiliError:
            return False, ""
        if _safe_int(payload.get("code"), -1) != 0:
            return False, ""
        data = payload.get("data")
        if not isinstance(data, dict) or not data.get("isLogin"):
            return False, ""
        name = _clean_text(data.get("uname")) or _clean_text(data.get("mid"))
        return True, name

    async def prepare(
        self,
        reference: BilibiliReference,
        *,
        download_video: bool = True,
    ) -> MediaItem:
        """把一个引用准备成通用 ``MediaItem``。"""

        try:
            info = await self.resolve(reference)
        except BilibiliError as exc:
            # 卡片里通常已经有标题、简介和作者。即使 B 站接口临时限流，
            # 也不要把这些可验证的资料一起丢掉；返回一个文字型 MediaItem，
            # 下一轮回看时仍可凭 source_url 重试下载。
            return self._fallback_item(reference, exc.user_message)
        subtitle = await self.fetch_subtitle(info)
        context_text = build_context(info, reference, subtitle)
        path: Path | None = None
        notice = ""
        if download_video:
            try:
                path = await self.download(info)
            except BilibiliError as exc:
                notice = exc.user_message
        elif not subtitle:
            notice = "视频链接已识别，但当前视频解析开关已关闭。"
        return MediaItem(
            kind=MediaKind.VIDEO,
            name=info.title or f"B 站视频 {info.bvid}",
            identity=f"bilibili:{info.key}",
            path=path,
            quoted=reference.quoted,
            source_url=info.canonical_url,
            owned_temp=path is not None,
            context_text=context_text,
            context_label="B 站视频资料",
            source_notice=notice,
        )

    @staticmethod
    def _fallback_item(reference: BilibiliReference, notice: str) -> MediaItem:
        value = reference.value or "未知链接"
        title = reference.title or f"B 站视频 {value}"
        bvid = value if reference.kind == "bvid" else ""
        info = BilibiliInfo(
            bvid=bvid,
            aid=int(value) if reference.kind == "aid" and value.isdigit() else None,
            cid=0,
            page=max(1, reference.page),
            title=title,
            description=reference.description,
            author=reference.author,
            duration=None,
            canonical_url=reference.canonical_url,
        )
        return MediaItem(
            kind=MediaKind.VIDEO,
            name=title,
            identity=f"bilibili:{reference.key}",
            quoted=reference.quoted,
            source_url=reference.canonical_url,
            context_text=build_context(info, reference, ""),
            context_label="B 站卡片资料",
            source_notice=notice,
        )

    async def download(self, info: BilibiliInfo) -> Path:
        """使用 yt-dlp 下载一个分 P，返回可交给 ffmpeg 的本地文件。"""

        cache_key = f"{info.key}:{self.max_download_mb}"
        cached = self._cache_get(self._downloads, cache_key)
        if cached is not None and cached.exists():
            return cached
        lock = self._locks.setdefault("download:" + info.key, asyncio.Lock())
        async with lock:
            cached = self._cache_get(self._downloads, cache_key)
            if cached is not None and cached.exists():
                return cached
            async with self._download_gate:
                await self._drain_download_tail()
                target = self._download_path(".mp4")
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        download_video_sync,
                        info,
                        target,
                        cookie=self._effective_cookie(),
                        ffmpeg_path=self.ffmpeg_path,
                        max_download_mb=self.max_download_mb,
                        timeout_seconds=self.timeout_seconds,
                        log=self._log,
                    )
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(worker),
                        timeout=self.timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    # 取消 wait_for 不会停止已经进入线程的 yt-dlp。把它记成 tail，
                    # 下一次下载先等它退出，避免超时后出现两个并行下载；同时清掉
                    # 这次可能留下的目标文件，避免把半成品交给回看流程。
                    self._download_tail = worker
                    worker.add_done_callback(lambda task: _discard_timed_out_file(task, target))
                    raise BilibiliError(
                        "yt-dlp download timed out",
                        f"下载 B 站视频超过 {self.timeout_seconds} 秒，已跳过本次画面解析。",
                    ) from exc
                except asyncio.CancelledError:
                    # 取消当前协程不会自动停止已经进入线程的 yt-dlp；把它
                    # 记为尾任务，下一次下载先等待其退出，避免后台叠加请求。
                    self._download_tail = worker
                    worker.add_done_callback(lambda task: _discard_timed_out_file(task, target))
                    raise
                except BilibiliError:
                    target.unlink(missing_ok=True)
                    raise
                finally:
                    if worker.done() and self._download_tail is worker:
                        self._download_tail = None
                self._cache_put(self._downloads, cache_key, target)
                return target

    async def _drain_download_tail(self) -> None:
        tail = self._download_tail
        if tail is None:
            return
        self._download_tail = None
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await tail

    async def download_from_url(self, value: str) -> Path:
        reference = parse_reference(value)
        if reference is None:
            raise BilibiliError("not a Bilibili reference", "这不是可识别的 B 站视频链接。")
        info = await self.resolve(reference)
        return await self.download(info)

    async def caption_from_value(self, value: str, page: int = 1) -> tuple[BilibiliInfo, str]:
        reference = parse_reference(value)
        if reference is None:
            raise BilibiliError(
                "not a Bilibili reference",
                "请提供 B 站完整链接、BV 号、av 号或 b23.tv 短链。",
            )
        if page > 1:
            reference = replace(reference, page=page)
        info = await self.resolve(reference)
        text = await self.fetch_subtitle(info)
        return info, text

    async def _resolve_short_url(self, url: str) -> str:
        current = url
        for _ in range(6):
            if not _allowed_host(current):
                raise BilibiliError(
                    "short URL left the Bilibili allowlist",
                    "B 站短链跳转到了非 B 站域名，已为安全起见拒绝访问。",
                )
            response_body, headers, status = await self._bounded_request_with_headers(
                current,
                max_bytes=MAX_SHORT_BODY_BYTES,
                allow_cookie=False,
            )
            if status in {301, 302, 303, 307, 308}:
                location = headers.get("location", "")
                if not location:
                    break
                current = urllib.parse.urljoin(current, location)
                ref = parse_reference(current)
                if ref is not None and ref.kind != "short_url":
                    return ref.canonical_url
                continue
            ref = parse_reference(current)
            if ref is not None and ref.kind != "short_url":
                return ref.canonical_url
            body_text = response_body.decode("utf-8", errors="ignore")
            ref = parse_reference(body_text)
            if ref is not None and ref.kind != "short_url":
                return ref.canonical_url
            break
        raise BilibiliError(
            "short URL could not be resolved",
            "这个 B 站短链经过多次跳转后仍无法识别对应视频。",
        )

    async def _api_json(self, url: str, *, params: dict[str, str]) -> dict[str, Any]:
        body = await self._bounded_request(
            url,
            params=params,
            max_bytes=MAX_API_BYTES,
            allow_cookie=True,
        )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BilibiliError(
                "Bilibili API returned invalid JSON", "B 站返回的数据暂时无法解析。"
            ) from exc
        if not isinstance(payload, dict):
            raise BilibiliError("Bilibili API returned a non-object", "B 站返回了无效数据。")
        return payload

    async def _bounded_request(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int,
        allow_cookie: bool,
        redirect_suffixes: tuple[str, ...] = (),
    ) -> bytes:
        body, _headers, _status = await self._bounded_request_with_headers(
            url,
            params=params,
            headers=headers,
            max_bytes=max_bytes,
            allow_cookie=allow_cookie,
            redirect_suffixes=redirect_suffixes,
        )
        return body

    async def _bounded_request_with_headers(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int,
        allow_cookie: bool,
        redirect_suffixes: tuple[str, ...] = (),
    ) -> tuple[bytes, httpx.Headers, int]:
        request_headers = {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"}
        if headers:
            request_headers.update(headers)
        cookie = self._effective_cookie()
        if allow_cookie and cookie:
            request_headers["Cookie"] = cookie

        current_url = url
        current_params = params
        for redirect_count in range(4):
            result = await self._request_once(
                current_url,
                params=current_params,
                headers=request_headers,
                max_bytes=max_bytes,
                allow_redirect_retry=redirect_count == 0,
            )
            _body, response_headers, status = result
            if status not in {301, 302, 303, 307, 308}:
                return result
            location = response_headers.get("location", "")
            if not location or not redirect_suffixes:
                return result
            next_url = urllib.parse.urljoin(current_url, location)
            if not _trusted_redirect(next_url, redirect_suffixes):
                raise BilibiliError(
                    "redirect left the trusted Bilibili CDN allowlist",
                    "B 站字幕跳转到了不受信任的地址，已为安全起见拒绝访问。",
                )
            current_url = next_url
            current_params = None
        raise BilibiliError(
            "too many Bilibili CDN redirects",
            "B 站字幕跳转次数过多，已停止读取。",
        )

    async def _request_once(
        self,
        url: str,
        *,
        params: dict[str, str] | None,
        headers: dict[str, str],
        max_bytes: int,
        allow_redirect_retry: bool,
    ) -> tuple[bytes, httpx.Headers, int]:
        async with self._api_gate:
            for attempt in range(2 if allow_redirect_retry else 1):
                interval = self._interval_value()
                wait = interval - (time.monotonic() - self._last_api_request)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_api_request = time.monotonic()
                try:
                    async with self.client.stream(
                        "GET",
                        url,
                        params=params,
                        headers=headers,
                        follow_redirects=False,
                    ) as response:
                        if response.status_code == 429 and attempt == 0:
                            retry_after = _retry_after(response.headers.get("retry-after"))
                            await asyncio.sleep(retry_after)
                            continue
                        declared = response.headers.get("content-length")
                        if declared and declared.isdigit() and int(declared) > max_bytes:
                            raise BilibiliError(
                                "Bilibili response exceeded the configured limit",
                                "B 站返回的数据过大，已停止读取。",
                            )
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > max_bytes:
                                raise BilibiliError(
                                    "Bilibili response exceeded the configured limit",
                                    "B 站返回的数据过大，已停止读取。",
                                )
                            chunks.append(chunk)
                        if response.status_code >= 400:
                            if response.status_code == 429:
                                raise BilibiliError(
                                    "Bilibili returned HTTP 429",
                                    "B 站请求过于频繁，请稍后再试。",
                                )
                            raise BilibiliError(
                                f"Bilibili returned HTTP {response.status_code}",
                                f"B 站接口返回 HTTP {response.status_code}。",
                            )
                        return b"".join(chunks), response.headers, response.status_code
                except httpx.HTTPError as exc:
                    raise BilibiliError(
                        f"Bilibili request failed: {type(exc).__name__}",
                        "访问 B 站时网络异常，请稍后重试。",
                    ) from exc
        raise BilibiliError("Bilibili request retry exhausted", "B 站请求频率受限，请稍后再试。")

    def _interval_value(self) -> float:
        value = (
            self._request_interval() if callable(self._request_interval) else self._request_interval
        )
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return API_MIN_INTERVAL

    @staticmethod
    def _ordered_subtitles(
        candidates: list[Any], language_preference: str = "auto"
    ) -> list[dict[str, Any]]:
        valid = [item for item in candidates if isinstance(item, dict) and item.get("subtitle_url")]
        if not valid:
            return []
        preference = (language_preference or "auto").casefold()
        if preference == "en":
            preferred = ("en-us", "en", "zh-cn", "zh-hans", "zh-hant", "ai-zh")
        elif preference not in {"auto", "zh"} and preference:
            preferred = (preference, "zh-cn", "zh-hans", "zh-hant", "ai-zh", "en-us", "en")
        else:
            preferred = ("zh-cn", "zh-hans", "zh-hant", "zh-hk", "ai-zh", "en-us", "en")

        def rank(item: dict[str, Any]) -> tuple[int, int]:
            language = _clean_text(item.get("lan")).casefold()
            doc = _clean_text(item.get("lan_doc")).casefold()
            position = next(
                (
                    index
                    for index, wanted in enumerate(preferred)
                    if wanted in language or wanted in doc
                ),
                len(preferred) + 1,
            )
            return position, _safe_int(item.get("ai_type"))

        return sorted(valid, key=rank)

    def _effective_cookie(self) -> str:
        if self.settings.use_saved_cookie and self._saved_cookie_provider is not None:
            with contextlib.suppress(Exception):
                saved = str(self._saved_cookie_provider() or "").strip()
                if saved:
                    return saved
        return self.settings.cookie

    @staticmethod
    def _cache_get(cache: OrderedDict[str, tuple[float, Any]], key: str) -> Any | None:
        entry = cache.get(key)
        if entry is None:
            return None
        expires, value = entry
        if expires <= time.monotonic():
            cache.pop(key, None)
            return None
        cache.move_to_end(key)
        if isinstance(value, Path) and not value.exists():
            cache.pop(key, None)
            return None
        return value

    @staticmethod
    def _cache_put(cache: OrderedDict[str, tuple[float, Any]], key: str, value: Any) -> None:
        cache[key] = (time.monotonic() + 600, value)
        cache.move_to_end(key)
        while len(cache) > 64:
            cache.popitem(last=False)

    def _subtitle_failure_cached(self, key: str) -> bool:
        expires = self._subtitle_failures.get(key)
        if expires is None:
            return False
        if expires <= time.monotonic():
            self._subtitle_failures.pop(key, None)
            return False
        self._subtitle_failures.move_to_end(key)
        return True

    def _mark_subtitle_failure(self, key: str) -> None:
        self._subtitle_failures[key] = time.monotonic() + SUBTITLE_FAILURE_TTL
        self._subtitle_failures.move_to_end(key)
        while len(self._subtitle_failures) > 64:
            self._subtitle_failures.popitem(last=False)


def _clip_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "…"


def _retry_after(value: str | None) -> float:
    try:
        return min(8.0, max(1.0, float(value or 2)))
    except (TypeError, ValueError):
        return 2.0


def _cookie_fingerprint(cookie: str) -> str:
    return hashlib.sha1(cookie.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _trusted_redirect(value: str, suffixes: tuple[str, ...]) -> bool:
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return any(host == suffix or host.endswith("." + suffix) for suffix in suffixes)


def _public_stats(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    fields = ("view", "like", "coin", "favorite", "share", "reply", "danmaku")
    stats = {key: _safe_int(value.get(key), -1) for key in fields}
    stats = {key: number for key, number in stats.items() if number >= 0}
    return stats or None


def _discard_timed_out_file(task: asyncio.Task[Any], target: Path) -> None:
    """回收超时线程的结果并消费异常，避免事件循环打印未取回异常。"""
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()
    target.unlink(missing_ok=True)


__all__ = ["API_MIN_INTERVAL", "BilibiliClient"]
