"""混合媒体的优先级、预算硬上限和真实体积回归。"""

from __future__ import annotations

import random

import pytest

from motion_vision.budget import frame_allowances
from motion_vision.models import MediaKind
from motion_vision.sampling import thin_indices

VIDEO = MediaKind.VIDEO
GIF = MediaKind.ANIMATION


@pytest.mark.parametrize(
    ("wants", "kinds", "limit", "expected"),
    [
        ([30, 30, 30, 30], [VIDEO, GIF, GIF, GIF], 48, [30, 6, 6, 6]),
        ([30, 30, 30, 30], [GIF, GIF, GIF, VIDEO], 48, [6, 6, 6, 30]),
        ([40, 30, 30, 30], [VIDEO, GIF, GIF, GIF], 48, [40, 3, 3, 2]),
        ([48, 30, 30, 30], [VIDEO, GIF, GIF, GIF], 48, [42, 2, 2, 2]),
        ([30, 30, 30, 30, 30], [VIDEO, GIF, VIDEO, GIF, GIF], 48, [21, 2, 21, 2, 2]),
        ([6, 6, 6, 20], [GIF, GIF, GIF, VIDEO], 48, [6, 6, 6, 20]),
        ([1, 30, 30, 30], [GIF, GIF, GIF, VIDEO], 48, [1, 9, 8, 30]),
        ([30, 30, 30], [GIF, GIF, GIF], 48, [16, 16, 16]),
        ([30, 30], [VIDEO, VIDEO], 48, [24, 24]),
        ([30, 30], [GIF, VIDEO], 1, [0, 1]),
        ([30, 30], [GIF, VIDEO], 0, [0, 0]),
        ([], [], 48, []),
    ],
)
def test_video_priority_and_animation_reserve(wants, kinds, limit, expected):
    assert frame_allowances([[1] * n for n in wants], kinds, limit, 10000) == expected


def test_bytes_prioritize_video_even_when_animations_come_first():
    # 张数不超，体积超限；同样应得到视频 30 帧、三个 GIF 各 6 帧。
    sizes = [[100] * 30] * 4
    assert frame_allowances(sizes, [GIF, GIF, GIF, VIDEO], 120, 4800) == [6, 6, 6, 30]


def test_both_limits_apply_without_losing_animation_reserve():
    assert frame_allowances([[100] * 30] * 4, [GIF, GIF, GIF, VIDEO], 48, 2000) == [2, 2, 2, 14]


def test_two_frame_reserve_uses_each_animation_actual_size():
    assert frame_allowances([[100] * 30, [400] * 30, [100] * 30], [GIF, GIF, VIDEO], 48, 3000) == [
        2,
        2,
        20,
    ]


def test_byte_rounding_leftovers_are_reused():
    # 同类初分各 150 字节只够各一帧，但两个零头还能多放一帧。
    assert frame_allowances([[100] * 6] * 2, [VIDEO, VIDEO], 12, 300) == [2, 1]


def test_variable_size_frames_do_not_break_byte_cap():
    # 首尾很大：按平均体积推算保留 2 帧，会实际超过 100 字节。
    sizes = [[70, 1, 1, 70]]
    assert frame_allowances(sizes, [VIDEO], 4, 100) == [1]


def test_budget_uses_original_timeline_for_final_thinning():
    # count=4 会选 0,3,5,8；预算应直接在原来的九帧上选最终的 0,4,8。
    sizes = [[10] * 9]
    assert frame_allowances(sizes, [VIDEO], 4, 30) == [3]


def test_unaffordable_video_does_not_waste_animation_budget():
    assert frame_allowances([[1000] * 4, [10] * 4], [VIDEO, GIF], 8, 100) == [0, 4]


def test_unaffordable_video_releases_its_frame_slots_to_gifs():
    sizes = [[10000] * 48, *([[1] * 30] * 3)]
    assert frame_allowances(sizes, [VIDEO, GIF, GIF, GIF], 48, 1000) == [0, 16, 16, 16]


def test_unaffordable_gif_releases_reserve_to_video():
    assert frame_allowances([[10000] * 30, [1] * 48], [GIF, VIDEO], 48, 1000) == [0, 48]


def test_frame_and_byte_limits_hold_for_varied_media():
    rng = random.Random(319)
    for _ in range(500):
        sizes = [[rng.randrange(1, 500) for _ in range(rng.randrange(0, 65))] for _ in range(5)]
        kinds = [rng.choice([VIDEO, GIF]) for _ in sizes]
        max_frames, max_bytes = rng.randrange(0, 100), rng.randrange(0, 10000)
        before = [list(media) for media in sizes]
        counts = frame_allowances(sizes, kinds, max_frames, max_bytes)
        assert sizes == before
        assert all(0 <= count <= len(media) for media, count in zip(sizes, counts, strict=True))
        assert sum(counts) <= max_frames
        assert (
            sum(
                sum(media[i] for i in thin_indices(len(media), count))
                for media, count in zip(sizes, counts, strict=True)
            )
            <= max_bytes
        )
