"""文件租约：清理可以立即请求，实际删除必须等所有使用者结束。"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

_lock = threading.RLock()
_pins: dict[Path, int] = {}
_pending: set[Path] = set()
_T = TypeVar("_T")


def _key(path: Path | str) -> Path:
    # 不跟随清理目标的符号链接，避免误删链接指向的外部目录。
    return Path(os.path.abspath(path))


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _delete(path: Path) -> None:
    try:
        if path.is_symlink():
            path.unlink(missing_ok=True)
        elif getattr(path, "is_junction", lambda: False)():
            path.rmdir()
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def discard(path: Path | str | None) -> None:
    """删除文件；存在父目录或子文件租约时延迟到租约释放。"""
    if not path:
        return
    target = _key(path)
    with _lock:
        if any(_overlaps(target, pinned) for pinned in _pins):
            _pending.add(target)
        else:
            _delete(target)


class ResourceLease:
    """一个请求/后台任务持有的路径集合；close 可重复调用。"""

    def __init__(self) -> None:
        self._paths: set[Path] = set()
        self._closed = False

    def add(self, path: Path | str | None) -> None:
        if not path:
            return
        target = _key(path)
        with _lock:
            if self._closed:
                raise RuntimeError("文件租约已经释放")
            if target not in self._paths:
                _pins[target] = _pins.get(target, 0) + 1
                self._paths.add(target)

    def close(self) -> None:
        with _lock:
            if self._closed:
                return
            self._closed = True
            for path in self._paths:
                count = _pins[path] - 1
                if count:
                    _pins[path] = count
                else:
                    del _pins[path]
            self._paths.clear()
            for path in list(_pending):
                if not any(_overlaps(path, pinned) for pinned in _pins):
                    _pending.remove(path)
                    _delete(path)

    def __enter__(self) -> ResourceLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


async def protected_thread(
    function: Callable[..., _T],
    *args: object,
    paths: Iterable[Path] = (),
    on_finished: Callable[[], None] | None = None,
) -> _T:
    """取消协程不等于停止线程；线程自己在 finally 里释放文件保护。"""
    loop = asyncio.get_running_loop()
    lease = ResourceLease()
    for path in paths:
        lease.add(path)

    def work() -> _T:
        try:
            return function(*args)
        finally:
            lease.close()
            if on_finished is not None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(on_finished)

    try:
        future = loop.run_in_executor(None, work)
    except BaseException:
        lease.close()
        if on_finished is not None:
            on_finished()
        raise

    def consume(done: asyncio.Future[_T]) -> None:
        if not done.cancelled():
            done.exception()

    future.add_done_callback(consume)
    return await asyncio.shield(future)
