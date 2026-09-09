"""收集本次请求里的视频。

QQ 的视频有很多种到达方式，因此按可靠性从高到低逐层降级：

1. 提示词里的 [Video/File Attachment: ...] 标记（AstrBot 已经落好盘）；
2. 消息链里的 Video / File 组件（含引用消息）；
3. OneBot 原始上报 raw_message 里的 file / video 段；
4. 群文件 / 私聊文件：调 get_group_file_url 拿直链再限量下载；
5. 引用消息回查：调 get_msg 拿到被引用消息，再走第 3 步。

任何一层拿到本地文件就停止继续降级。
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..cards import REMOTE_ATTEMPTED_ATTR, REMOTE_PAYLOADS_ATTR
from ..models import MediaItem, MediaKind
from .common import (
    MB,
    DownloadTooLarge,
    download_to_file,
    is_http_url,
    looks_like_video,
    parse_markers,
    redact_url,
    resolve_local_path,
)

_CQ_SEGMENT_PATTERN = re.compile(
    r"\[CQ:(?P<type>[A-Za-z0-9_-]+)(?:,(?P<body>[^\]]*))?\]",
    re.IGNORECASE,
)
_MAX_RAW_SEGMENTS = 64


@dataclass
class VideoCandidate:
    """一个「可能是视频」的线索。"""

    name: str = ""
    raw_path: str = ""
    url: str = ""
    file_id: str = ""
    busid: Any = None
    quoted: bool = False
    part_index: int | None = None
    origin: str = ""
    marker_raw: str = ""
    component: Any = None
    """对应的消息组件，必要时让它自己去落盘。"""

    def identity_keys(self) -> list[str]:
        keys: list[str] = []
        if self.file_id:
            keys.append(f"file_id:{self.file_id}")
        if self.url:
            keys.append(f"url:{self.url}")
        if self.raw_path:
            keys.append(f"path:{self.raw_path.casefold()}")
        # 文件名只能在没有更可靠标识时参与去重；不同来源完全可能都叫
        # ``video.mp4``，不能因为同名就把第二个视频吞掉。
        if self.name and not keys:
            keys.append(f"name:{Path(self.name).name.casefold()}")
        return keys


@dataclass
class CollectResult:
    items: list[MediaItem] = field(default_factory=list)
    notices: list[tuple[str, str]] = field(default_factory=list)
    """(名称, 原因) —— 会作为中文说明注入，让模型知道为什么没有画面。"""


class VideoCollector:
    """把散落在各处的视频线索汇总成本地文件。"""

    def __init__(
        self,
        request: Any,
        event: Any,
        client: httpx.AsyncClient,
        download_path_factory: Any,
        max_videos: int = 2,
        max_download_mb: int = 100,
        include_group_files: bool = True,
        search_dirs: tuple[Path, ...] = (),
    ) -> None:
        self.request = request
        self.event = event
        self.client = client
        self._download_path = download_path_factory
        self.max_videos = max(1, max_videos)
        self.max_download_bytes = max(1, max_download_mb) * MB
        self.include_group_files = include_group_files
        self.search_dirs = search_dirs
        self._seen: set[str] = set()
        self._seen_paths: set[str] = set()

    # --- 对外入口 -----------------------------------------------------------

    async def collect(self) -> CollectResult:
        result = CollectResult()

        candidates = self._from_markers()
        # 标记和消息链经常是同一附件的两份表示，但也可能各自包含不同的
        # 视频；全部收集后按规范化本地路径去重，不用“有 marker 就跳过整条链”
        # 这种会漏掉第二个视频的捷径。
        candidates += self._from_chain(skip_video_components=False)
        candidates += self._from_raw_message()
        # 一条消息可以同时包含直接发送的视频和一个未展开的引用视频。不能因为
        # 前者已经找到，就跳过对后者的一次有界回查。
        remote_lookup_done = bool(_event_value(self.event, REMOTE_ATTEMPTED_ATTR, False))
        if (
            not candidates or not any(candidate.quoted for candidate in candidates)
        ) and not remote_lookup_done:
            candidates += await self._from_reply_lookup()

        for candidate in candidates:
            if len(result.items) >= self.max_videos:
                break
            if self._already_seen(candidate):
                continue
            item = await self._resolve(candidate, result)
            if item is not None:
                self._remember(candidate)
                result.items.append(item)

        return result

    # --- 各层来源 -----------------------------------------------------------

    def _from_markers(self) -> list[VideoCandidate]:
        found: list[VideoCandidate] = []
        parts = getattr(self.request, "extra_user_content_parts", None) or []
        for part_index, part in enumerate(parts):
            text = getattr(part, "text", None)
            if not isinstance(text, str) or not text:
                continue
            for marker in parse_markers(text, part_index):
                if marker.kind == "image":
                    continue
                if marker.kind == "file" and not looks_like_video(marker.name):
                    continue
                found.append(
                    VideoCandidate(
                        name=marker.name,
                        raw_path=marker.path if not is_http_url(marker.path) else "",
                        url=marker.path if is_http_url(marker.path) else "",
                        quoted=marker.quoted,
                        part_index=part_index,
                        origin="marker",
                        marker_raw=marker.raw,
                    )
                )
        return found

    def _from_chain(self, skip_video_components: bool = False) -> list[VideoCandidate]:
        from astrbot.api.message_components import File as AstrFile
        from astrbot.api.message_components import Reply
        from astrbot.api.message_components import Video as AstrVideo

        try:
            chain = self.event.get_messages() or []
        except Exception:
            return []

        direct: list[VideoCandidate] = []
        quoted: list[VideoCandidate] = []

        def handle(component: Any, bucket: list[VideoCandidate], is_quoted: bool) -> None:
            if isinstance(component, AstrVideo):
                if skip_video_components:
                    return
                values = [
                    str(getattr(component, name, "") or "") for name in ("path", "file", "url")
                ]
                url = next((value for value in values if is_http_url(value)), "")
                raw = next((value for value in values if value and not is_http_url(value)), "")
                source = url or raw
                bucket.append(
                    VideoCandidate(
                        name=Path(source.split("?", 1)[0]).name or "video",
                        raw_path=raw,
                        url=url,
                        quoted=is_quoted,
                        origin="chain:video",
                        component=component,
                    )
                )
            elif isinstance(component, AstrFile):
                name = str(getattr(component, "name", "") or "")
                raw = str(
                    getattr(component, "file_", "")
                    or getattr(component, "file", "")
                    or getattr(component, "path", "")
                    or ""
                )
                url = str(getattr(component, "url", "") or "")
                if (
                    not looks_like_video(name, str(getattr(component, "mime", "") or ""))
                    and not looks_like_video(raw)
                    and not looks_like_video(url)
                ):
                    return
                bucket.append(
                    VideoCandidate(
                        name=name or Path((url or raw).split("?", 1)[0]).name or "视频",
                        raw_path="" if is_http_url(raw) else raw,
                        url=url if is_http_url(url) else "",
                        file_id=str(getattr(component, "file_id", "") or ""),
                        quoted=is_quoted,
                        origin="chain:file",
                        component=component,
                    )
                )

        for component in chain:
            if isinstance(component, Reply):
                for inner in component.chain or []:
                    handle(inner, quoted, True)
            else:
                handle(component, direct, False)

        return direct + quoted

    def _from_raw_message(self) -> list[VideoCandidate]:
        """OneBot 原始上报里常常带着消息链丢掉的 file_id 与 file_size。"""
        message_obj = getattr(self.event, "message_obj", None)
        raw = getattr(message_obj, "raw_message", None)
        if raw is None:
            raw = getattr(self.event, "raw_message", None)
        segments = _raw_segments(raw)
        remote_payloads = _event_value(self.event, REMOTE_PAYLOADS_ATTR, None) or []
        candidates = [
            candidate
            for candidate in (self._candidate_from_segment(segment) for segment in segments)
            if candidate is not None
        ]
        remote_candidates: list[VideoCandidate] = []
        if isinstance(remote_payloads, list):
            for payload in remote_payloads:
                for segment in _raw_segments(payload):
                    candidate = self._candidate_from_segment(segment)
                    if candidate is not None:
                        candidate.quoted = True
                        candidate.origin = "remote-card"
                        remote_candidates.append(candidate)
        return candidates + remote_candidates

    async def _from_reply_lookup(self) -> list[VideoCandidate]:
        """前面都没找到时，用 get_msg 回查被引用的那条消息。"""
        from astrbot.api.message_components import Reply

        if not self._is_onebot():
            return []
        message_id = None
        try:
            for component in self.event.get_messages() or []:
                if isinstance(component, Reply) and component.id:
                    message_id = component.id
                    break
        except Exception:
            message_id = None
        if message_id is None:
            message_id = _reply_id_from_raw(self.event)
        if message_id is None:
            return []

        payload = await self._call_action_variants("get_msg", "message_id", message_id)
        if not isinstance(payload, dict):
            return []
        segments = _raw_segments(payload)
        found = []
        for segment in segments:
            candidate = self._candidate_from_segment(segment)
            if candidate:
                candidate.quoted = True
                candidate.origin = "get_msg"
                found.append(candidate)
        return found

    def _candidate_from_segment(self, segment: Any) -> VideoCandidate | None:
        if not isinstance(segment, dict):
            return None
        seg_type = str(segment.get("type") or "").lower()
        if seg_type not in ("file", "video"):
            return None
        data = segment.get("data")
        if not isinstance(data, dict):
            data = {}

        name = str(data.get("file_name") or data.get("name") or data.get("file") or "")
        mime = str(data.get("mime") or data.get("mime_type") or data.get("content_type") or "")
        if seg_type == "file" and not looks_like_video(name, mime):
            return None

        raw = str(data.get("path") or data.get("file") or "")
        url = str(data.get("url") or data.get("download_url") or "")
        return VideoCandidate(
            name=Path(name).name or "视频",
            raw_path="" if is_http_url(raw) else raw,
            url=url if is_http_url(url) else "",
            file_id=str(data.get("file_id") or ""),
            busid=data.get("busid"),
            origin=f"raw:{seg_type}",
        )

    # --- 线索 -> 本地文件 ---------------------------------------------------

    async def _resolve(self, candidate: VideoCandidate, result: CollectResult) -> MediaItem | None:
        label = candidate.name or "视频"

        local = resolve_local_path(candidate.raw_path, self.search_dirs)
        if local is not None:
            if self._path_seen(local):
                return None
            return self._make_item(candidate, local, owned=False)

        component_path, component_url = await self._ask_component(candidate)
        if component_path is not None:
            if self._path_seen(component_path):
                return None
            return self._make_item(candidate, component_path, owned=False, url=component_url)

        url = candidate.url or component_url
        if not url and candidate.file_id and self.include_group_files:
            url = await self._group_file_url(candidate)

        if not url:
            result.notices.append((label, "没能拿到可下载的文件地址"))
            return None

        dest = self._download_path(Path(candidate.name or "video").suffix or ".mp4")
        try:
            await download_to_file(self.client, url, dest, self.max_download_bytes)
        except DownloadTooLarge as exc:
            result.notices.append((label, str(exc)))
            return None
        except (RuntimeError, httpx.HTTPError, OSError):
            result.notices.append((label, f"下载失败（{redact_url(url)}）"))
            return None

        return self._make_item(candidate, dest, owned=True, url=url)

    async def _ask_component(self, candidate: VideoCandidate) -> tuple[Path | None, str]:
        """交给组件自己的 get_file / convert_to_file_path 去落盘。

        这条路径覆盖了非 OneBot 平台（Telegram、Discord 等）的视频附件。
        """
        component = candidate.component
        if component is None:
            return (None, "")

        raw = ""
        getter = getattr(component, "get_file", None)
        if callable(getter):
            try:
                raw = await getter(allow_return_url=True)
            except TypeError:
                try:
                    raw = await getter()
                except Exception:
                    raw = ""
            except Exception:
                raw = ""
        if not raw:
            converter = getattr(component, "convert_to_file_path", None)
            if callable(converter):
                try:
                    raw = await converter()
                except Exception:
                    raw = ""

        raw = str(raw or "")
        if not raw:
            return (None, "")
        if is_http_url(raw):
            return (None, raw)
        return (resolve_local_path(raw, self.search_dirs), "")

    async def _group_file_url(self, candidate: VideoCandidate) -> str:
        if not self._is_onebot():
            return ""
        group_id = getattr(self.event, "get_group_id", lambda: "")() or ""
        if group_id:
            params: dict[str, Any] = {"group_id": group_id, "file_id": candidate.file_id}
            if candidate.busid is not None:
                params["busid"] = candidate.busid
            payload = await self._call_action("get_group_file_url", params)
            if payload is None and str(group_id).isdigit():
                params["group_id"] = int(group_id)
                payload = await self._call_action("get_group_file_url", params)
        else:
            payload = await self._call_action(
                "get_private_file_url", {"file_id": candidate.file_id}
            )

        if not isinstance(payload, dict):
            return ""
        for key in ("url", "download_url"):
            value = payload.get(key)
            if isinstance(value, str) and is_http_url(value):
                return value
        return ""

    # --- OneBot 调用 --------------------------------------------------------

    def _is_onebot(self) -> bool:
        try:
            name = str(self.event.get_platform_name() or "").casefold()
            if any(token in name for token in ("onebot", "aiocqhttp", "llbot", "llonebot")):
                return True
        except Exception:
            pass
        return self._bot_call() is not None

    def _bot_call(self) -> Any:
        bot = getattr(self.event, "bot", None)
        for holder in (bot, getattr(bot, "api", None)):
            call = getattr(holder, "call_action", None)
            if callable(call):
                return call
        return None

    async def _call_action(self, action: str, params: dict[str, Any]) -> Any:
        call = self._bot_call()
        if call is None:
            return None
        try:
            response = await call(action, **params)
        except Exception:
            return None
        if isinstance(response, dict):
            data = response.get("data")
            return data if isinstance(data, dict) else response
        return response

    async def _call_action_variants(self, action: str, key: str, value: Any) -> Any:
        """OneBot 实现对 id 的类型要求不一致，int / str 都试一次。"""
        variants: list[Any] = [value]
        text = str(value)
        if text.isdigit() and not isinstance(value, int):
            variants.append(int(text))
        elif isinstance(value, int):
            variants.append(text)
        for variant in variants:
            payload = await self._call_action(action, {key: variant})
            if payload is not None:
                return payload
        return None

    # --- 去重与打包 ---------------------------------------------------------

    def _already_seen(self, candidate: VideoCandidate) -> bool:
        return any(key in self._seen for key in candidate.identity_keys())

    def _remember(self, candidate: VideoCandidate) -> None:
        self._seen.update(candidate.identity_keys())

    def _make_item(
        self, candidate: VideoCandidate, path: Path, owned: bool, url: str = ""
    ) -> MediaItem:
        identity = f"path:{str(path).casefold()}"
        self._seen.add(identity)
        self._seen_paths.add(str(path).casefold())
        return MediaItem(
            kind=MediaKind.VIDEO,
            name=candidate.name or path.name,
            identity=identity,
            path=path,
            part_index=candidate.part_index,
            marker_raw=candidate.marker_raw,
            quoted=candidate.quoted,
            owned_temp=owned,
            source_url=url or candidate.url,
        )

    def _path_seen(self, path: Path) -> bool:
        return str(path).casefold() in self._seen_paths


def _raw_segments(raw: Any) -> list[Any]:
    """从 OneBot 字典、JSON 字符串或 CQ 码中取出消息段列表。"""
    if isinstance(raw, dict):
        for key in ("message", "messages"):
            value = raw.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)[:_MAX_RAW_SEGMENTS]
            if isinstance(value, str):
                segments = _raw_segments(value)
                if segments:
                    return segments
        if str(raw.get("type") or "").casefold() in {"file", "video", "reply"}:
            return [raw]
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)[:_MAX_RAW_SEGMENTS]
    if not isinstance(raw, str):
        return []

    text = raw.strip()
    if not text:
        return []
    if text.startswith(("{", "[")):
        try:
            decoded = json.loads(html.unescape(text))
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if decoded is not None and decoded is not raw:
            segments = _raw_segments(decoded)
            if segments:
                return segments

    segments: list[dict[str, Any]] = []
    for match in _CQ_SEGMENT_PATTERN.finditer(text):
        body: dict[str, str] = {}
        for token in (match.group("body") or "").split(","):
            key, separator, value = token.partition("=")
            if separator and key.strip():
                body[key.strip()] = html.unescape(value)
        segments.append({"type": match.group("type").casefold(), "data": body})
        if len(segments) >= _MAX_RAW_SEGMENTS:
            break
    return segments


def _reply_id_from_raw(event: Any) -> str | None:
    message_obj = getattr(event, "message_obj", None)
    raw = getattr(message_obj, "raw_message", None)
    if raw is None:
        raw = getattr(event, "raw_message", None)
    for segment in _raw_segments(raw):
        if str(segment.get("type") or "").casefold() != "reply":
            continue
        data = segment.get("data")
        if isinstance(data, dict):
            for key in ("id", "message_id", "messageId"):
                value = str(data.get(key) or "").strip()
                if value:
                    return value
    return None


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    """读取事件或其 message_obj 上的共享状态。"""

    for owner in (event, getattr(event, "message_obj", None)):
        if owner is None:
            continue
        try:
            value = getattr(owner, name, default)
        except Exception:
            continue
        if value is not default:
            return value
    return default
