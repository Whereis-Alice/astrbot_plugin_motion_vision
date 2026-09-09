"""从消息正文和引用卡片中收集 B 站视频。"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..bilibili import BilibiliClient, BilibiliError, extract_event_references
from ..models import MediaItem
from ..settings import BilibiliSettings
from .video import CollectResult


class BilibiliCollector:
    """把 B 站链接变成通用视频 ``MediaItem``。

    下载失败并不会丢掉字幕和卡片资料：只要元数据或字幕拿到了，仍然返回一个
    只有文字上下文的项目，让模型至少能回答“视频讲了什么”。
    """

    def __init__(
        self,
        event: Any,
        client: BilibiliClient,
        settings: BilibiliSettings,
        *,
        max_videos: int = 2,
        download_video: bool = True,
        existing_items: list[MediaItem] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.event = event
        self.client = client
        self.settings = settings
        self.max_videos = max(0, max_videos)
        self.download_video = download_video
        self.existing_items = existing_items or []
        self._log = log or (lambda _message: None)

    async def collect(self) -> CollectResult:
        result = CollectResult()
        if not self.settings.enabled or self.max_videos <= 0:
            return result

        references = await self._references()
        seen: set[str] = set()
        for reference in references:
            if len(result.items) >= self.max_videos:
                break
            if reference.key in seen:
                continue
            seen.add(reference.key)
            try:
                item = await self.client.prepare(
                    reference,
                    download_video=self.download_video,
                )
            except BilibiliError as exc:
                result.notices.append((reference.value, exc.user_message))
                continue
            except Exception as exc:
                self._log(f"B 站视频解析异常（{reference.value}）：{type(exc).__name__}: {exc}")
                result.notices.append((reference.value, "B 站视频解析出现异常，已跳过本次读取。"))
                continue
            if self._duplicates_existing(item):
                continue
            result.items.append(item)
        return result

    async def _references(self) -> list[Any]:
        """读取已展开的消息；必要时只回查一次未展开的引用消息。

        有些 OneBot 网关只给插件 ``reply.id``，不把被引用消息的卡片内容挂到
        当前事件上。这里复用只读的 ``get_msg``，并限制为最多一个引用、三秒超时，
        避免为了找卡片给平台和 B 站都增加额外压力。
        """
        references = extract_event_references(self.event)
        if any(reference.quoted for reference in references):
            return references
        message_id = _first_reply_id(self.event)
        if not message_id:
            return references
        payload = await _fetch_reply(self.event, message_id)
        if payload is None:
            return references
        remote_event = _RawPayloadEvent(payload)
        remote = extract_event_references(remote_event)
        if not remote:
            return references
        merged: OrderedDict[str, Any] = OrderedDict((item.key, item) for item in references)
        for item in remote:
            item = replace(item, quoted=True)
            old = merged.get(item.key)
            if old is None:
                merged[item.key] = item
            else:
                merged[item.key] = replace(
                    old,
                    title=old.title or item.title,
                    description=old.description or item.description,
                    author=old.author or item.author,
                    quoted=True,
                )
        return list(merged.values())

    def _duplicates_existing(self, item: MediaItem) -> bool:
        for existing in self.existing_items:
            if item.source_url and item.source_url == existing.source_url:
                return True
            if item.path is not None and item.path == existing.path:
                return True
        return False


__all__ = ["BilibiliCollector"]


class _RawPayloadEvent:
    """让通用事件解析器读取 OneBot ``get_msg`` 的返回值。"""

    def __init__(self, payload: Any) -> None:
        self.raw_message = payload


def _first_reply_id(event: Any) -> str:
    active: set[int] = set()

    def walk(value: Any, depth: int = 0) -> str:
        if value is None or depth > 8:
            return ""
        if isinstance(value, str):
            text = value.strip()
            match = re.search(
                r"\[CQ:reply\b[^\]]*(?:^|,)(?:id|message_id)=([^,\]]+)",
                text,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
            if text.startswith(("{", "[")):
                with contextlib.suppress(TypeError, ValueError, json.JSONDecodeError):
                    return walk(json.loads(text), depth + 1)
            return ""
        if isinstance(value, (bytes, bytearray, int, float, bool)):
            return ""
        object_id = id(value)
        if object_id in active:
            return ""
        active.add(object_id)
        try:
            if isinstance(value, dict):
                kind = str(value.get("type", "")).casefold()
                data = value.get("data")
                if kind in {"reply", "replyelement"}:
                    if isinstance(data, dict):
                        for key in ("id", "message_id", "messageId"):
                            candidate = str(data.get(key) or "").strip()
                            if candidate:
                                return candidate
                    for key in ("id", "message_id", "messageId"):
                        candidate = str(value.get(key) or "").strip()
                        if candidate:
                            return candidate
                for nested in value.values():
                    found = walk(nested, depth + 1)
                    if found:
                        return found
                return ""
            if isinstance(value, (list, tuple, set)):
                for nested in value:
                    found = walk(nested, depth + 1)
                    if found:
                        return found
                return ""
            class_name = type(value).__name__.casefold()
            if "reply" in class_name:
                for key in ("id", "message_id", "messageId"):
                    candidate = str(getattr(value, key, "") or "").strip()
                    if candidate:
                        return candidate
            for key in ("chain", "message", "data", "raw_message"):
                with contextlib.suppress(Exception):
                    found = walk(getattr(value, key, None), depth + 1)
                    if found:
                        return found
            return ""
        finally:
            active.discard(object_id)

    return (
        walk(getattr(event, "message_obj", None))
        or walk(getattr(event, "raw_message", None))
        or walk(getattr(event, "message", None))
        or walk(_safe_get_messages(event))
    )


def _safe_get_messages(event: Any) -> Any:
    with contextlib.suppress(Exception):
        return event.get_messages()
    return None


async def _fetch_reply(event: Any, message_id: str) -> Any:
    bot = getattr(event, "bot", None)
    owners = [getattr(bot, "api", None), bot]
    callables: list[Callable[..., Any]] = []
    for owner in owners:
        call = getattr(owner, "call_action", None)
        if callable(call) and call not in callables:
            callables.append(call)
    direct = getattr(bot, "get_msg", None)
    if callable(direct) and direct not in callables:
        callables.append(direct)
    if not callables:
        return None

    variants: list[str | int] = [message_id]
    if message_id.isdigit():
        variants.append(int(message_id))
    deadline = asyncio.get_running_loop().time() + 3.0
    for call in callables:
        for value in variants:
            attempts = (
                ((), {"action": "get_msg", "message_id": value}),
                ((), {"action": "get_msg", "id": value}),
                (("get_msg",), {"message_id": value}),
                (("get_msg",), {"id": value}),
                ((), {"message_id": value}),
                ((), {"id": value}),
            )
            for args, kwargs in attempts:
                if deadline <= asyncio.get_running_loop().time():
                    return None
                try:
                    result = call(*args, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    break
                try:
                    if inspect.isawaitable(result):
                        result = await asyncio.wait_for(
                            result,
                            timeout=max(0.01, deadline - asyncio.get_running_loop().time()),
                        )
                except (asyncio.TimeoutError, TypeError, ValueError):
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    continue
                unwrapped = _unwrap_reply_result(result)
                if unwrapped is not None:
                    return unwrapped
    return None


def _unwrap_reply_result(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict) and "data" in value:
        return value.get("data")
    return value
