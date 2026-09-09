"""抽帧结果缓存。

同一个视频/动图在对话里被连续追问几轮是很常见的（「他在干什么」→「那背景呢」），
每轮都重新抽帧既慢又费 CPU。这里用一个有界 LRU + TTL 缓存帧文件路径，
命中时校验文件仍然存在，被淘汰的条目立刻删掉目录。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import AudioClip, SampledFrame


@dataclass
class CacheEntry:
    frames: list[SampledFrame] = field(default_factory=list)
    duration: float | None = None
    source_frame_count: int | None = None
    transcript: str = ""
    native_report: str = ""
    audio: AudioClip | None = None
    notice: str = ""
    frames_dir: Path | None = None
    created_at: float = field(default_factory=time.time)


class ResultCache:
    """很小的一个 LRU，容量默认 24 条，足够覆盖连续追问。"""

    def __init__(
        self,
        capacity: int = 24,
        ttl_seconds: float = 3600.0,
        on_evict: Callable[[Path | None], None] | None = None,
    ) -> None:
        self.capacity = max(1, capacity)
        self.ttl_seconds = max(60.0, ttl_seconds)
        self._on_evict = on_evict
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

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
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
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
        if entry is not None and self._on_evict is not None:
            self._on_evict(entry.frames_dir)
            if entry.audio is not None:
                self._on_evict(entry.audio.path)

    @staticmethod
    def _intact(entry: CacheEntry) -> bool:
        for frame in entry.frames:
            if not frame.path.exists():
                return False
        return entry.audio is None or entry.audio.path.exists()


def fingerprint(path: Path | None, data: bytes | None, signature: str) -> str:
    """为一个媒体生成缓存键：内容特征 + 影响产物的配置指纹。"""
    if path is not None:
        try:
            stat = path.stat()
            return f"{str(path).casefold()}|{stat.st_size}|{int(stat.st_mtime)}|{signature}"
        except OSError:
            return f"{str(path).casefold()}|?|?|{signature}"
    if data is not None:
        import hashlib

        return f"bytes|{len(data)}|{hashlib.sha1(data[:262144]).hexdigest()}|{signature}"
    return f"unknown|{signature}"
