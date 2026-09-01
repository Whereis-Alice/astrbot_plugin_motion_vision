"""注入层：帧、说明、清理与兜底。"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from astrbot.core.agent.message import TextPart

from motion_vision.inject import FALLBACK_PROMPT, HEADER, inject
from motion_vision.models import AudioClip, MediaItem, MediaKind, MediaResult, SampledFrame
from motion_vision.settings import InjectionSettings, Settings


class FakeRequest:
    """只保留插件真正会碰到的 ProviderRequest 字段。"""

    def __init__(
        self,
        prompt: str = "这是什么",
        image_urls: list[str] | None = None,
        parts: list[Any] | None = None,
    ) -> None:
        self.prompt = prompt
        self.image_urls = list(image_urls or [])
        self.audio_urls: list[str] = []
        self.extra_user_content_parts: list[Any] = list(parts or [])


def _settings(**injection: Any) -> Settings:
    return dataclasses.replace(Settings(), injection=InjectionSettings(**injection))


def _frames(tmp_path: Path, count: int, size: int = 1024) -> list[SampledFrame]:
    frames = []
    for index in range(count):
        target = tmp_path / f"frame{index}.jpg"
        target.write_bytes(b"x" * size)
        frames.append(
            SampledFrame(path=target, index=index, timestamp=index * 0.5, size_bytes=size)
        )
    return frames


def _result(tmp_path: Path, frames: int = 3, **kwargs: Any) -> MediaResult:
    item = MediaItem(
        kind=kwargs.pop("kind", MediaKind.VIDEO),
        name=kwargs.pop("name", "clip.mp4"),
        identity=kwargs.pop("identity", "id-1"),
        image_url_index=kwargs.pop("image_url_index", None),
        part_index=kwargs.pop("part_index", None),
        marker_raw=kwargs.pop("marker_raw", ""),
        quoted=kwargs.pop("quoted", False),
    )
    return MediaResult(item=item, frames=_frames(tmp_path, frames), duration=6.0, **kwargs)


def _texts(request: FakeRequest) -> list[str]:
    return [
        part.text
        for part in request.extra_user_content_parts
        if getattr(part, "type", "") == "text"
    ]


def _images(request: FakeRequest) -> list[Any]:
    return [
        part
        for part in request.extra_user_content_parts
        if getattr(part, "type", "") == "image_url"
    ]


def test_frames_are_injected_as_temporary_parts(tmp_path: Path) -> None:
    request = FakeRequest()
    report = inject(request, [_result(tmp_path, frames=4)], _settings())

    images = _images(request)
    assert report.media == 1
    assert report.frames == 4
    assert len(images) == 4
    assert request.image_urls == []
    assert HEADER in _texts(request)
    assert all(part._no_save for part in request.extra_user_content_parts)


def test_frame_ids_carry_timestamps(tmp_path: Path) -> None:
    request = FakeRequest()
    inject(request, [_result(tmp_path, frames=2)], _settings())

    ids = [part.image_url.id for part in _images(request)]
    assert ids == ["clip.mp4 #1@0.0s", "clip.mp4 #2@0.5s"]


def test_description_mentions_duration_and_sample_size(tmp_path: Path) -> None:
    request = FakeRequest()
    inject(request, [_result(tmp_path, frames=3, quoted=True)], _settings())

    described = [text for text in _texts(request) if "clip.mp4" in text]
    assert described
    assert "引用消息里的视频" in described[0]
    assert "取样 3 帧" in described[0]


def test_original_image_is_dropped_from_request(tmp_path: Path) -> None:
    request = FakeRequest(image_urls=["keep-me.png", "the-gif.gif"])
    result = _result(tmp_path, frames=2, kind=MediaKind.ANIMATION, image_url_index=1)

    inject(request, [result], _settings())

    assert request.image_urls == ["keep-me.png"]


def test_marker_is_stripped_from_prompt_part(tmp_path: Path) -> None:
    marker = "[Video Attachment: name clip.mp4, path C:/tmp/clip.mp4]"
    part = TextPart(text=f"这个 {marker} 讲了什么")
    request = FakeRequest(parts=[part])
    result = _result(tmp_path, frames=2, part_index=0, marker_raw=marker)

    inject(request, [result], _settings())

    remaining = _texts(request)
    assert all(marker not in text for text in remaining)
    assert any("这个" in text and "讲了什么" in text for text in remaining)


def test_marker_only_part_is_removed_entirely(tmp_path: Path) -> None:
    marker = "[Video Attachment: path C:/tmp/clip.mp4]"
    request = FakeRequest(parts=[TextPart(text=marker)])
    result = _result(tmp_path, frames=1, part_index=0, marker_raw=marker)

    inject(request, [result], _settings())

    assert marker not in "".join(_texts(request))


def test_empty_prompt_gets_a_fallback_question(tmp_path: Path) -> None:
    request = FakeRequest(prompt="   ")
    inject(request, [_result(tmp_path, frames=1)], _settings())

    assert request.prompt == FALLBACK_PROMPT


def test_existing_prompt_is_preserved(tmp_path: Path) -> None:
    request = FakeRequest(prompt="他在干什么")
    inject(request, [_result(tmp_path, frames=1)], _settings())

    assert request.prompt == "他在干什么"


def test_keep_frames_in_history_uses_image_urls(tmp_path: Path) -> None:
    request = FakeRequest()
    result = _result(tmp_path, frames=3)

    report = inject(request, [result], _settings(keep_frames_in_history=True))

    assert report.frames == 3
    assert len(request.image_urls) == 3
    assert _images(request) == []
    assert not any(part._no_save for part in request.extra_user_content_parts)


def test_audio_and_transcript_are_attached(tmp_path: Path) -> None:
    clip = tmp_path / "audio.wav"
    clip.write_bytes(b"wav")
    result = _result(
        tmp_path, frames=1, transcript="你好世界", audio=AudioClip(path=clip, seconds=6.0)
    )
    request = FakeRequest()

    report = inject(request, [result], _settings())

    assert report.audio == 1
    assert report.transcripts == 1
    assert request.audio_urls == [str(clip)]
    assert any("你好世界" in text for text in _texts(request))


def test_notice_only_result_still_informs_the_model(tmp_path: Path) -> None:
    item = MediaItem(kind=MediaKind.VIDEO, name="big.mp4", identity="id-2")
    request = FakeRequest()

    report = inject(request, [MediaResult(item=item, notice="文件太大，已跳过")], _settings())

    assert report.media == 0
    assert report.notices == 1
    assert any("文件太大" in text for text in _texts(request))


def test_notice_can_be_silenced(tmp_path: Path) -> None:
    item = MediaItem(kind=MediaKind.VIDEO, name="big.mp4", identity="id-2")
    request = FakeRequest()

    report = inject(
        request, [MediaResult(item=item, notice="文件太大")], _settings(notice_enabled=False)
    )

    assert report.touched is False
    assert request.extra_user_content_parts == []


def test_nothing_to_do_leaves_request_untouched() -> None:
    request = FakeRequest(prompt="", image_urls=["a.png"])

    report = inject(request, [], _settings())

    assert report.touched is False
    assert request.prompt == ""
    assert request.image_urls == ["a.png"]
    assert request.extra_user_content_parts == []


def test_extra_guidance_is_appended(tmp_path: Path) -> None:
    request = FakeRequest()
    inject(request, [_result(tmp_path, frames=1)], _settings(extra_guidance="请用中文吐槽一句"))

    assert "请用中文吐槽一句" in _texts(request)


def test_memo_is_the_only_thing_that_stays_in_history(tmp_path: Path) -> None:
    request = FakeRequest()
    report = inject(request, [_result(tmp_path, frames=2)], _settings(), memo="【档案】编号 a3f1")

    assert report.memo is True
    kept = [part for part in request.extra_user_content_parts if not part._no_save]
    assert [part.text for part in kept] == ["【档案】编号 a3f1"]


def test_memo_is_skipped_when_nothing_was_readable() -> None:
    item = MediaItem(kind=MediaKind.VIDEO, name="big.mp4", identity="id-3")
    request = FakeRequest()

    report = inject(request, [MediaResult(item=item, notice="太大")], _settings(), memo="【档案】x")

    assert report.memo is False
    assert "【档案】x" not in _texts(request)
