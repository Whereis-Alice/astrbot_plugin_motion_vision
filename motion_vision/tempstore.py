"""临时文件管理。

所有中间产物（抽出的帧、剥离的音轨、下载的视频）都放在 AstrBot 临时目录下的
motion_vision 子目录里，按用途再分子目录：

    <astrbot_temp>/motion_vision/
        frames/<key>/000.jpg ...
        audio/<uuid>.wav
        downloads/<uuid>.mp4

帧目录会被结果缓存引用，所以不能在请求结束时立刻删；改为按保留时长清理，
并在进程重启后由 sweep() 回收孤儿目录。
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

DIR_NAME = "motion_vision"


class TempStore:
    """带 TTL 清理的临时目录。"""

    def __init__(self, base_dir: Path, retention_seconds: float = 6 * 3600) -> None:
        self.base = Path(base_dir) / DIR_NAME
        self.retention_seconds = max(60.0, float(retention_seconds))
        self._last_sweep = 0.0

    # --- 目录 ---------------------------------------------------------------

    def _sub(self, name: str) -> Path:
        target = self.base / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def frames_dir(self, key: str) -> Path:
        safe = "".join(ch if ch.isalnum() else "_" for ch in key)[:64]
        target = self._sub("frames") / f"{safe}_{uuid.uuid4().hex[:8]}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def audio_path(self, suffix: str = ".wav") -> Path:
        return self._sub("audio") / f"{uuid.uuid4().hex}{suffix}"

    def download_path(self, suffix: str = "") -> Path:
        clean = "".join(ch for ch in suffix if ch.isalnum() or ch == ".")[:16]
        return self._sub("downloads") / f"{uuid.uuid4().hex}{clean}"

    # --- 清理 ---------------------------------------------------------------

    @staticmethod
    def discard(path: Path | str | None) -> None:
        """安静地删除一个文件或目录，失败不抛。"""
        if not path:
            return
        target = Path(path)
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        except OSError:
            pass

    def sweep(self, min_interval: float = 300.0) -> int:
        """删除超过保留时长的条目，返回删除数量。

        min_interval 用于避免每个请求都去遍历磁盘。
        """
        now = time.time()
        if now - self._last_sweep < min_interval:
            return 0
        self._last_sweep = now

        if not self.base.exists():
            return 0

        deadline = now - self.retention_seconds
        removed = 0
        for group in ("frames", "audio", "downloads"):
            group_dir = self.base / group
            if not group_dir.is_dir():
                continue
            for entry in self._iter(group_dir):
                try:
                    if entry.stat().st_mtime >= deadline:
                        continue
                except OSError:
                    continue
                self.discard(entry)
                removed += 1
        return removed

    def usage(self) -> tuple[int, int]:
        """返回 (文件数, 总字节数)，用于诊断命令。"""
        count = 0
        total = 0
        if not self.base.exists():
            return (0, 0)
        for path in self.base.rglob("*"):
            try:
                if path.is_file():
                    count += 1
                    total += path.stat().st_size
            except OSError:
                continue
        return (count, total)

    @staticmethod
    def _iter(directory: Path) -> list[Path]:
        try:
            return list(directory.iterdir())
        except OSError:
            return []
