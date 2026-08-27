"""整轮请求的图片预算保护。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from motion_vision.models import MediaItem, MediaKind, MediaResult, SampledFrame
from motion_vision.pipeline import MediaPipeline
from motion_vision.settings import MB, AdvancedSettings, Settings


def _pipeline(**advanced: object) -> MediaPipeline:
    settings = dataclasses.replace(Settings(), advanced=AdvancedSettings(**advanced))
    # 预算裁剪是纯计算，不需要 ffmpeg / 缓存 / 网络依赖。
    return MediaPipeline(settings, None, None, None, None, None, None)


def _result(name: str, count: int, size: int) -> MediaResult:
    item = MediaItem(kind=MediaKind.VIDEO, name=name, identity=name)
    frames = [
        SampledFrame(path=Path(f"{name}-{index}.jpg"), index=index, size_bytes=size)
        for index in range(count)
    ]
    return MediaResult(item=item, frames=frames)


def test_frame_count_budget_keeps_the_first_media_intact() -> None:
    pipeline = _pipeline(max_images_per_request=4)
    first, second = _result("a.mp4", 3, 1024), _result("b.mp4", 3, 1024)

    pipeline._apply_payload_budget([first, second])

    assert len(first.frames) == 3
    assert len(second.frames) == 1
    assert "丢弃 2 张帧" in second.notice
    assert first.notice == ""


def test_byte_budget_drops_oversized_frames() -> None:
    pipeline = _pipeline(max_frame_payload_mb=1)
    result = _result("a.mp4", 4, int(0.6 * MB))

    pipeline._apply_payload_budget([result])

    assert len(result.frames) == 1
    assert "图片预算" in result.notice


def test_existing_notice_is_kept() -> None:
    pipeline = _pipeline(max_images_per_request=1)
    result = _result("a.mp4", 2, 10)
    result.notice = "音频转写失败"

    pipeline._apply_payload_budget([result])

    assert result.notice.startswith("音频转写失败；")


def test_generous_budget_changes_nothing() -> None:
    pipeline = _pipeline()
    result = _result("a.mp4", 8, 1024)

    pipeline._apply_payload_budget([result])

    assert len(result.frames) == 8
    assert result.notice == ""
