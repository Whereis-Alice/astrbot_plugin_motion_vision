"""会话级媒体档案：编号、查找、淘汰与备忘文案。"""

from __future__ import annotations

from pathlib import Path

from motion_vision.models import MediaKind, format_duration
from motion_vision.registry import (
    MediaRegistry,
    build_memo,
    describe_age,
)


def _registry(**kwargs: object) -> MediaRegistry:
    return MediaRegistry(**kwargs)  # type: ignore[arg-type]


def test_tokens_are_unique_within_a_session(tmp_path: Path) -> None:
    registry = _registry()
    tokens = {
        registry.remember("s1", MediaKind.VIDEO, f"v{i}.mp4", path=tmp_path / f"v{i}.mp4").token
        for i in range(4)
    }
    assert len(tokens) == 4


def test_same_file_reuses_its_token_and_moves_to_front(tmp_path: Path) -> None:
    registry = _registry()
    clip = tmp_path / "clip.mp4"
    first = registry.remember("s1", MediaKind.VIDEO, "clip.mp4", path=clip, duration=12.0)
    registry.remember("s1", MediaKind.VIDEO, "other.mp4", path=tmp_path / "other.mp4")
    again = registry.remember("s1", MediaKind.VIDEO, "clip.mp4", path=clip, frames_seen=8)

    assert again is first
    assert again.token == first.token
    assert again.duration == 12.0
    assert again.frames_seen == 8
    assert registry.records("s1")[0] is again
    assert len(registry) == 2


def test_same_url_reuses_its_token() -> None:
    registry = _registry()
    first = registry.remember("s1", MediaKind.VIDEO, "a.mp4", source_url="https://x/a.mp4")
    again = registry.remember("s1", MediaKind.VIDEO, "a.mp4", source_url="https://x/a.mp4")
    assert again is first
    assert len(registry) == 1


def test_find_defaults_to_the_latest(tmp_path: Path) -> None:
    registry = _registry()
    registry.remember("s1", MediaKind.VIDEO, "old.mp4", path=tmp_path / "old.mp4")
    newest = registry.remember("s1", MediaKind.VIDEO, "new.mp4", path=tmp_path / "new.mp4")

    assert registry.find("s1", "") is newest
    assert registry.find("s1", "最近") is newest
    assert registry.find("s1", "LATEST") is newest


def test_find_tolerates_decorated_tokens(tmp_path: Path) -> None:
    registry = _registry()
    record = registry.remember("s1", MediaKind.VIDEO, "clip.mp4", path=tmp_path / "clip.mp4")

    assert registry.find("s1", record.token) is record
    assert registry.find("s1", f"编号 {record.token}。") is record
    assert registry.find("s1", f"#{record.token.upper()}") is record
    assert registry.find("s1", "ffff" if record.token != "ffff" else "0000") is None


def test_find_on_unknown_session_returns_none() -> None:
    assert _registry().find("nobody", "abcd") is None


def test_per_session_limit_discards_the_oldest(tmp_path: Path) -> None:
    dropped: list[Path | None] = []
    registry = _registry(per_session=2, on_discard=dropped.append)

    for index in range(3):
        registry.remember(
            "s1",
            MediaKind.VIDEO,
            f"{index}.mp4",
            path=tmp_path / f"{index}.mp4",
            owned_temp=True,
        )

    assert len(registry.records("s1")) == 2
    assert dropped == [tmp_path / "0.mp4"]


def test_borrowed_files_are_never_deleted(tmp_path: Path) -> None:
    dropped: list[Path | None] = []
    registry = _registry(per_session=1, on_discard=dropped.append)

    registry.remember("s1", MediaKind.VIDEO, "a.mp4", path=tmp_path / "a.mp4", owned_temp=False)
    registry.remember("s1", MediaKind.VIDEO, "b.mp4", path=tmp_path / "b.mp4")

    assert dropped == []


def test_session_limit_evicts_whole_sessions(tmp_path: Path) -> None:
    dropped: list[Path | None] = []
    registry = _registry(max_sessions=2, on_discard=dropped.append)

    for index in range(3):
        registry.remember(
            f"s{index}",
            MediaKind.VIDEO,
            f"{index}.mp4",
            path=tmp_path / f"{index}.mp4",
            owned_temp=True,
        )

    assert registry.records("s0") == []
    assert dropped == [tmp_path / "0.mp4"]
    assert len(registry) == 2


def test_clear_reports_how_many_were_forgotten(tmp_path: Path) -> None:
    registry = _registry()
    registry.remember("s1", MediaKind.VIDEO, "a.mp4", path=tmp_path / "a.mp4")
    registry.remember("s2", MediaKind.VIDEO, "b.mp4", path=tmp_path / "b.mp4")

    assert registry.clear("s1") == 1
    assert registry.clear() == 1
    assert len(registry) == 0


def test_record_summary_is_human_readable(tmp_path: Path) -> None:
    registry = _registry()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"data")
    record = registry.remember("s1", MediaKind.VIDEO, "clip.mp4", path=clip, duration=72.0)

    summary = record.summary()
    assert f"编号 {record.token}" in summary
    assert "视频" in summary
    assert format_duration(72.0) in summary
    assert "刚刚" in summary
    assert "源文件已清理" not in summary


def test_record_without_source_is_marked_unreadable() -> None:
    registry = _registry()
    record = registry.remember("s1", MediaKind.ANIMATION, "gone.gif")
    assert record.readable is False
    assert record.kind_label == "动图"
    assert "源文件已清理" in record.summary()


def test_memo_lists_tokens_and_the_tool_name(tmp_path: Path) -> None:
    registry = _registry()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"data")
    record = registry.remember("s1", MediaKind.VIDEO, "clip.mp4", path=clip, duration=72.0)

    memo = build_memo(registry.records("s1"), "review_it")
    assert record.token in memo
    assert "review_it" in memo
    assert "clip.mp4" in memo
    assert format_duration(72.0) in memo


def test_memo_is_empty_when_nothing_can_be_reviewed() -> None:
    registry = _registry()
    registry.remember("s1", MediaKind.VIDEO, "gone.mp4")
    assert build_memo(registry.records("s1"), "review_it") == ""
    assert build_memo([], "review_it") == ""


def test_describe_age_reads_naturally() -> None:
    assert describe_age(3) == "刚刚"
    assert describe_age(59.9) == "刚刚"
    assert describe_age(600) == "10 分钟前"
    assert describe_age(7200) == "2 小时前"
    assert describe_age(200000) == "2 天前"
