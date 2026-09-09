"""B 站适配的稳定公开入口。

实现按职责拆在三个模块中；这个文件只保留旧 import 路径，避免插件升级后已有
调用方需要改成内部模块路径。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from . import bilibili_client as _client
from .bilibili_client import BilibiliClient as _BilibiliClient
from .bilibili_parser import (
    BilibiliError,
    BilibiliInfo,
    BilibiliReference,
    build_context,
    extract_event_references,
    extract_references,
    has_bilibili_reference,
    parse_reference,
)
from .settings import BilibiliSettings

# 保留这个名字是为了兼容旧的测试和扩展；门面类的请求间隔通过闭包动态读取它，
# 因而 monkeypatch / 热更新不会被拆模块后悄悄绕过。
API_MIN_INTERVAL = _client.API_MIN_INTERVAL


class BilibiliClient(_BilibiliClient):
    """兼容旧构造函数的 B 站客户端门面。"""

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
        saved_cookie_provider: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(
            client,
            download_path_factory,
            settings,
            ffmpeg_path=ffmpeg_path,
            timeout_seconds=timeout_seconds,
            max_download_mb=max_download_mb,
            log=log,
            request_interval=lambda: API_MIN_INTERVAL,
            saved_cookie_provider=saved_cookie_provider,
        )


__all__ = [
    "API_MIN_INTERVAL",
    "BilibiliClient",
    "BilibiliError",
    "BilibiliInfo",
    "BilibiliReference",
    "build_context",
    "extract_event_references",
    "extract_references",
    "has_bilibili_reference",
    "parse_reference",
]
