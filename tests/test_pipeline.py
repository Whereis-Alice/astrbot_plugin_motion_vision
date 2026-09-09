"""整轮请求的图片预算保护。"""

from __future__ import annotations

import asyncio
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


def test_frame_count_budget_is_shared_fairly() -> None:
    pipeline = _pipeline(max_images_per_request=4)
    first, second = _result("a.mp4", 3, 1024), _result("b.mp4", 3, 1024)

    pipeline._apply_payload_budget([first, second])

    # 关键：后面的媒体不会被前面的吃光
    assert len(first.frames) == 2
    assert len(second.frames) == 2
    assert "只保留了 2 帧" in first.notice
    assert "只保留了 2 帧" in second.notice


def test_budget_leftovers_go_to_whoever_still_needs_them() -> None:
    pipeline = _pipeline(max_images_per_request=10)
    small, big = _result("a.gif", 2, 1024), _result("b.mp4", 20, 1024)

    pipeline._apply_payload_budget([small, big])

    # 小媒体只要 2 帧，剩下的名额全给还需要的那个
    assert len(small.frames) == 2
    assert len(big.frames) == 8
    assert small.notice == ""


def test_trimming_keeps_both_ends_of_the_timeline() -> None:
    pipeline = _pipeline(max_images_per_request=3)
    result = _result("a.mp4", 9, 1024)

    pipeline._apply_payload_budget([result])

    # 抽稀而不是截断：片头片尾都要留下
    assert [frame.path.name for frame in result.frames] == [
        "a.mp4-0.jpg",
        "a.mp4-4.jpg",
        "a.mp4-8.jpg",
    ]
    assert [frame.index for frame in result.frames] == [0, 1, 2]


def test_byte_budget_thins_frames() -> None:
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


def test_fair_allocation_water_fills() -> None:
    from motion_vision.sampling import fair_allocation

    assert fair_allocation([3, 3], 4) == [2, 2]
    assert fair_allocation([2, 20], 10) == [2, 8]
    assert fair_allocation([5, 5], 100) == [5, 5]
    assert fair_allocation([4, 4, 4], 2) == [1, 1, 0]
    assert fair_allocation([0, 6], 4) == [0, 4]
    assert fair_allocation([], 5) == []


def test_generous_budget_changes_nothing() -> None:
    pipeline = _pipeline()
    result = _result("a.mp4", 8, 1024)

    pipeline._apply_payload_budget([result])

    assert len(result.frames) == 8
    assert result.notice == ""


def test_text_only_video_source_does_not_report_missing_file() -> None:
    pipeline = _pipeline()
    item = MediaItem(
        kind=MediaKind.VIDEO,
        name="B站视频",
        identity="bilibili:test",
        context_text="【B站视频资料】带时间点字幕",
    )

    result, cache_entry = asyncio.run(pipeline._process_video(item))

    assert result.notice == ""
    assert result.item.context_text
    assert cache_entry is None
