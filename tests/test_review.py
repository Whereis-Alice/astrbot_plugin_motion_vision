"""回看参数的清洗与返回值组装。"""

from __future__ import annotations

from pathlib import Path

from motion_vision.models import MediaItem, MediaKind, MediaResult, SampledFrame, TimeSpan
from motion_vision.review import (
    MAX_FRAMES,
    build_payload,
    coerce_frames,
    make_span,
    summarize,
    tune_settings,
)
from motion_vision.settings import Settings


def _result(tmp_path: Path, count: int = 3, **kwargs: object) -> MediaResult:
    frames = []
    for index in range(count):
        target = tmp_path / f"f{index}.jpg"
        target.write_bytes(b"jpeg-bytes")
        frames.append(SampledFrame(path=target, index=index, timestamp=index * 1.5, size_bytes=10))
    item = MediaItem(kind=MediaKind.VIDEO, name="clip.mp4", identity="review:a3f1")
    return MediaResult(item=item, frames=frames, duration=12.0, **kwargs)  # type: ignore[arg-type]


# --- 参数清洗 ---------------------------------------------------------------


def test_frames_request_is_clamped():
    assert coerce_frames(0) == 0
    assert coerce_frames("18") == 18
    assert coerce_frames(18.7) == 18
    assert coerce_frames(999) == MAX_FRAMES
    assert coerce_frames(-4) == 0
    assert coerce_frames("很多") == 0
    assert coerce_frames(None) == 0


def test_span_needs_a_video():
    assert make_span(MediaKind.ANIMATION, 3, 9) is None
    assert make_span(MediaKind.VIDEO, 3, 9) == TimeSpan(3.0, 9.0)


def test_span_treats_zeroes_as_the_whole_clip():
    assert make_span(MediaKind.VIDEO, 0, 0) is None
    assert make_span(MediaKind.VIDEO, -5, 0) is None


def test_span_open_ended_when_only_start_is_given():
    span = make_span(MediaKind.VIDEO, 30, 0)
    assert span == TimeSpan(30.0, None)
    assert span.active is True
    assert span.label == "30.0 秒 ~ 结尾"


def test_span_reversed_range_falls_back_to_the_whole_clip():
    assert make_span(MediaKind.VIDEO, 40, 10) is None
    assert make_span(MediaKind.VIDEO, "x", "y") is None


# --- 配置改写 ---------------------------------------------------------------


def test_tuned_settings_drop_audio_and_cap_images():
    tuned = tune_settings(Settings(), 0)
    assert tuned.audio.mode == "off"
    assert tuned.advanced.max_images_per_request == MAX_FRAMES
    assert tuned.video_frames_override == 0
    assert tuned.animation_frames_override == 0


def test_tuned_settings_apply_an_explicit_frame_count():
    tuned = tune_settings(Settings(), 20)
    assert tuned.video_frames_override == 20
    assert tuned.animation_frames_override == 20


def test_tuned_settings_never_raise_an_existing_lower_cap():
    base = Settings()
    lowered = tune_settings(
        base.__class__(advanced=base.advanced.__class__(max_images_per_request=8)), 0
    )
    assert lowered.advanced.max_images_per_request == 8


# --- 说明文字 ---------------------------------------------------------------


def test_summary_mentions_the_token_and_the_window(tmp_path: Path) -> None:
    text = summarize("clip.mp4", "a3f1", _result(tmp_path), TimeSpan(30.0, 45.0))
    assert "编号 a3f1" in text
    assert "只看" in text
    assert "30.0 秒 ~ 45.0 秒" in text
    assert "0.0s" in text


def test_summary_mentions_the_duration_when_reviewing_the_whole_clip(tmp_path: Path) -> None:
    text = summarize("clip.mp4", "a3f1", _result(tmp_path), None)
    assert "整段时长约 12.0 秒" in text
    assert "只看" not in text


def test_summary_passes_notices_through(tmp_path: Path) -> None:
    text = summarize("g.gif", "b2", _result(tmp_path, notice="动图不支持只看某一段"), None)
    assert "动图不支持只看某一段" in text


# --- 返回值 -----------------------------------------------------------------


def test_payload_carries_one_image_per_frame(tmp_path: Path) -> None:
    payload = build_payload("clip.mp4", "a3f1", _result(tmp_path, count=3), None)
    if isinstance(payload, str):  # 没装 mcp 的环境退回纯文本
        assert "编号 a3f1" in payload
        return

    kinds = [item.type for item in payload.content]
    assert kinds == ["text", "image", "image", "image"]
    assert all(item.mimeType == "image/jpeg" for item in payload.content[1:])
    assert "编号 a3f1" in payload.content[0].text


def test_payload_degrades_to_text_when_frames_vanished(tmp_path: Path) -> None:
    result = _result(tmp_path, count=2)
    for frame in result.frames:
        frame.path.unlink()

    payload = build_payload("clip.mp4", "a3f1", result, None)
    assert isinstance(payload, str)
    assert "编号 a3f1" in payload
