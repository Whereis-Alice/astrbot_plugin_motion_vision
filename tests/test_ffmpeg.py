from __future__ import annotations

import asyncio

import pytest

from motion_vision.ffmpeg import FfmpegError, FfmpegRunner, FfmpegTools


class _SlowProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if not self.killed:
            await asyncio.sleep(10)
        self.returncode = -9
        return b"", b""

    def kill(self) -> None:
        self.killed = True


def test_ffmpeg_timeout_kills_the_child_process(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _SlowProcess()

    async def create_process(*_args: object, **_kwargs: object) -> _SlowProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    runner = FfmpegRunner(FfmpegTools(ffmpeg="fake-ffmpeg"))

    with pytest.raises(FfmpegError, match="超时"):
        asyncio.run(runner._run(["fake-ffmpeg"], timeout=0.01))

    assert process.killed is True
