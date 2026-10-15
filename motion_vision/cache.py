"""抽帧结果缓存。

同一个视频/动图在对话里被连续追问几轮是很常见的（「他在干什么」→「那背景呢」），
每轮都重新抽帧既慢又费 CPU。这里用一个有界 LRU + TTL 缓存帧文件路径，
命中时校验文件仍然存在，被淘汰的条目立刻删掉目录。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from .models import AudioClip, FrameSheet, SampledFrame


@dataclass
class CacheEntry:
    frames: list[SampledFrame] = field(default_factory=list)
    duration: float | None = None
    source_frame_count: int | None = None
    transcript: str = ""
    native_report: str = ""
    audio: AudioClip | None = None
    notice: str = ""
    audio_status: str = ""
    sheets: list[FrameSheet] = field(default_factory=list)
    sheets_dir: Path | None = None
    frames_dir: Path | None = None
    created_at: float = field(default_factory=time.time)


class ResultCache:
    """很小的一个 LRU，容量默认 24 条，足够覆盖连续追问。"""

    def __init__(
        self,
        capacity: int = 24,
        ttl_seconds: float = 3600.0,
        on_evict: Callable[[Path | None], None] | None = None,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.capacity = max(1, capacity)
        self.ttl_seconds = max(60.0, ttl_seconds)
        self.max_bytes = max(1, max_bytes)
        self._on_evict = on_evict
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._inflight: dict[str, tuple[asyncio.Lock, int]] = {}
        self._sizes: dict[str, int] = {}

    @asynccontextmanager
    async def processing(self, key: str) -> AsyncIterator[None]:
        """相同内容只生成一次；等待者随后重新检查缓存。"""
        lock, users = self._inflight.get(key, (asyncio.Lock(), 0))
        self._inflight[key] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            _, users = self._inflight[key]
            if users == 1:
                del self._inflight[key]
            else:
                self._inflight[key] = (lock, users - 1)

    def get(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() - entry.created_at > self.ttl_seconds or not self._intact(entry):
            self._drop(key)
            return None
        self._store.move_to_end(key)
        return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        if key in self._store:
            self._drop(key)
        self._store[key] = entry
        paths = {frame.path for frame in entry.frames} | {sheet.path for sheet in entry.sheets}
        if entry.audio is not None:
            paths.add(entry.audio.path)
        size = 0
        for path in paths:
            with suppress(OSError):
                size += path.stat().st_size
        self._sizes[key] = size
        self._store.move_to_end(key)
        while len(self._store) > self.capacity or sum(self._sizes.values()) > self.max_bytes:
            oldest, _ = next(iter(self._store.items()))
            self._drop(oldest)

    def clear(self) -> None:
        for key in list(self._store):
            self._drop(key)

    def __len__(self) -> int:
        return len(self._store)

    # --- 内部 ---------------------------------------------------------------

    def _drop(self, key: str) -> None:
        entry = self._store.pop(key, None)
        self._sizes.pop(key, None)
        if entry is not None and self._on_evict is not None:
            self._on_evict(entry.frames_dir)
            self._on_evict(entry.sheets_dir)
            if entry.audio is not None:
                self._on_evict(entry.audio.path)

    @staticmethod
    def _intact(entry: CacheEntry) -> bool:
        for frame in entry.frames:
            if not frame.path.exists():
                return False
        if any(not sheet.path.exists() for sheet in entry.sheets):
            return False
        return entry.audio is None or entry.audio.path.exists()


def fingerprint(path: Path | None, data: bytes | None, signature: str) -> str:
    """流式全内容 SHA-256；路径和时间戳不同不妨碍复用。在线程中调用。"""
    digest = hashlib.sha256()
    if path is not None:
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return f"{str(path).casefold()}|?|?|{signature}"
    elif data is not None:
        digest.update(data)
    else:
        return f"unknown|{signature}"
    return f"sha256:{digest.hexdigest()}|{signature}"
