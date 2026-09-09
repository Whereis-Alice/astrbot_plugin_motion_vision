"""附件标记解析与路径/URL 处理。"""

from __future__ import annotations

from pathlib import Path

from motion_vision.sources.common import (
    decode_inline_image,
    is_http_url,
    looks_like_video,
    parse_markers,
    redact_url,
    resolve_local_path,
    strip_file_scheme,
)
from motion_vision.sources.video import VideoCandidate, _raw_segments


def test_parse_marker_with_name_and_path() -> None:
    text = "看看这个 [Video Attachment: name clip.mp4, path C:/tmp/clip.mp4] 讲了什么"
    (marker,) = parse_markers(text, part_index=2)

    assert marker.kind == "video"
    assert marker.name == "clip.mp4"
    assert marker.path == "C:/tmp/clip.mp4"
    assert marker.quoted is False
    assert marker.part_index == 2
    assert marker.raw == "[Video Attachment: name clip.mp4, path C:/tmp/clip.mp4]"
    assert marker.raw in text


def test_parse_marker_from_quoted_message() -> None:
    (marker,) = parse_markers("[Image Attachment in quoted message: name a.gif, path /tmp/a.gif]")

    assert marker.kind == "image"
    assert marker.quoted is True
    assert marker.path == "/tmp/a.gif"


def test_parse_marker_without_name_falls_back_to_filename() -> None:
    (marker,) = parse_markers("[File Attachment: /tmp/sub/report.bin]")

    assert marker.kind == "file"
    assert marker.path == "/tmp/sub/report.bin"
    assert marker.name == "report.bin"


def test_parse_markers_collects_all_and_skips_empty() -> None:
    markers = parse_markers(
        "[Image Attachment: ] [Image Attachment: path /a.gif] [Video Attachment: path /b.mp4]"
    )

    assert [marker.kind for marker in markers] == ["image", "video"]


def test_parse_markers_ignores_plain_text() -> None:
    assert parse_markers("这里没有附件") == []
    assert parse_markers("") == []


def test_looks_like_video() -> None:
    assert looks_like_video(name="A.MP4") is True
    assert looks_like_video(name="x.mkv") is True
    assert looks_like_video(mime="video/quicktime") is True
    assert looks_like_video(name="x.gif") is False
    assert looks_like_video() is False


def test_is_http_url() -> None:
    assert is_http_url("HTTPS://a.com/x") is True
    assert is_http_url("file:///tmp/a") is False
    assert is_http_url("") is False


def test_redact_url_hides_signed_query() -> None:
    redacted = redact_url("https://cdn.qq.com/get?rkey=secret&sig=abc")

    assert redacted == "https://cdn.qq.com/get?<query-redacted>"
    assert "secret" not in redacted
    assert redact_url("https://cdn.qq.com/get") == "https://cdn.qq.com/get"
    assert redact_url("C:/tmp/a.mp4") == "C:/tmp/a.mp4"


def test_decode_inline_image() -> None:
    assert decode_inline_image("base64://aGVsbG8=") == b"hello"
    assert decode_inline_image("data:image/gif;base64,aGVsbG8=") == b"hello"
    assert decode_inline_image("") is None
    assert decode_inline_image("https://a.com/x.gif") is None
    assert decode_inline_image("/tmp/a.gif") is None


def test_strip_file_scheme_handles_windows_drive() -> None:
    assert strip_file_scheme("file:///C:/tmp/a%20b.mp4") == "C:/tmp/a b.mp4"
    assert strip_file_scheme("/tmp/a.mp4") == "/tmp/a.mp4"


def test_resolve_local_path_accepts_plain_path(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"data")

    assert resolve_local_path(str(target)) == target.resolve()


def test_resolve_local_path_accepts_file_url(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"data")

    assert resolve_local_path("file:///" + target.as_posix()) == target.resolve()


def test_resolve_local_path_searches_known_dirs(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"data")

    assert resolve_local_path("clip.mp4", search_dirs=(tmp_path,)) == target.resolve()
    assert resolve_local_path("nope/clip.mp4", search_dirs=(tmp_path,)) == target.resolve()


def test_resolve_local_path_rejects_remote_and_missing(tmp_path: Path) -> None:
    assert resolve_local_path("https://a.com/x.mp4") is None
    assert resolve_local_path("base64://aGk=") is None
    assert resolve_local_path("") is None
    assert resolve_local_path(str(tmp_path / "missing.mp4")) is None


def test_video_candidate_does_not_deduplicate_distinct_urls_by_filename() -> None:
    first = VideoCandidate(name="video.mp4", url="https://a.example/video.mp4")
    second = VideoCandidate(name="video.mp4", url="https://b.example/video.mp4")

    assert set(first.identity_keys()).isdisjoint(second.identity_keys())


def test_raw_cq_message_is_normalized_into_onebot_segments() -> None:
    raw = "[CQ:reply,id=42][CQ:file,name=clip&#44;one.mp4,file_id=abc]"

    segments = _raw_segments(raw)

    assert [segment["type"] for segment in segments] == ["reply", "file"]
    assert segments[0]["data"]["id"] == "42"
    assert segments[1]["data"]["name"] == "clip,one.mp4"
    assert segments[1]["data"]["file_id"] == "abc"
