"""采样算法。"""

from __future__ import annotations

from motion_vision.sampling import (
    BURST_WINDOW_SECONDS,
    SINGLE_PASS_MAX_SECONDS,
    ExtractionWindow,
    animation_frame_budget,
    audio_clip_seconds,
    distribute,
    plan_extraction,
    sample_indices,
    video_frame_budget,
)
from motion_vision.settings import DETAIL_PRESETS, MB


def test_sample_indices_covers_both_ends():
    assert sample_indices(10, 4) == [0, 3, 6, 9]
    assert sample_indices(5, 5) == [0, 1, 2, 3, 4]
    assert sample_indices(3, 9) == [0, 1, 2]
    assert sample_indices(7, 1) == [0]
    assert sample_indices(0, 5) == []
    assert sample_indices(5, 0) == []


def test_animation_budget_respects_source_length():
    preset = DETAIL_PRESETS["balanced"]
    assert animation_frame_budget(1, 0, preset) == 1
    assert animation_frame_budget(0, 0, preset) == 0
    # 只有 3 帧的动图不可能抽出 5 帧
    assert animation_frame_budget(3, 0, preset) == 3


def test_animation_budget_backs_off_for_large_files():
    preset = DETAIL_PRESETS["detailed"]
    plain = animation_frame_budget(60, 0, preset)
    medium = animation_frame_budget(60, 6 * MB, preset)
    huge = animation_frame_budget(60, 12 * MB, preset)

    assert plain > medium > huge
    assert huge >= 2


def test_animation_budget_never_drops_below_two():
    preset = DETAIL_PRESETS["frugal"]
    assert animation_frame_budget(30, 50 * MB, preset) == 2


def test_animation_override_wins():
    preset = DETAIL_PRESETS["frugal"]
    assert animation_frame_budget(40, 0, preset, override=9) == 9


def test_video_budget_grows_for_long_videos():
    preset = DETAIL_PRESETS["balanced"]
    assert video_frame_budget(10.0, preset) == preset.video_frames
    assert video_frame_budget(600.0, preset) == preset.long_video_frames
    assert video_frame_budget(None, preset) == preset.video_frames
    assert video_frame_budget(600.0, preset, override=4) == 4


def test_distribute_spreads_remainder_to_the_front():
    assert distribute(7, 3) == [3, 2, 2]
    assert distribute(6, 3) == [2, 2, 2]
    assert distribute(2, 5) == [1, 1, 0, 0, 0]
    assert distribute(5, 0) == []


def test_short_video_uses_a_single_pass():
    windows = plan_extraction(30.0, 8)
    assert windows == [ExtractionWindow(0.0, 30.0, 8)]


def test_unknown_duration_scans_whole_file():
    windows = plan_extraction(None, 6)
    assert len(windows) == 1
    assert windows[0].length is None
    assert windows[0].count == 6


def test_long_video_is_split_into_bursts():
    duration = 900.0
    budget = 20
    windows = plan_extraction(duration, budget)

    assert len(windows) > 1
    assert sum(window.count for window in windows) == budget
    assert windows[0].start == 0.0
    for window in windows:
        assert window.length is not None
        assert window.length <= BURST_WINDOW_SECONDS + 0.001
        assert window.start + window.length <= duration + 0.001
    starts = [window.start for window in windows]
    assert starts == sorted(starts)
    # 最后一个窗口应该贴近片尾，否则等于没覆盖全片
    assert starts[-1] > duration * 0.8


def test_boundary_duration_stays_single_pass():
    windows = plan_extraction(SINGLE_PASS_MAX_SECONDS, 12)
    assert len(windows) == 1


def test_window_timestamps_stay_inside_the_window():
    window = ExtractionWindow(10.0, 5.0, 5)
    stamps = window.timestamps()

    assert len(stamps) == 5
    assert stamps[0] == 10.0
    assert all(10.0 <= value < 15.0 for value in stamps)
    assert ExtractionWindow(0.0, None, 3).timestamps() == [0.0]
    assert ExtractionWindow(0.0, 5.0, 0).timestamps() == []


def test_audio_clip_is_capped():
    assert audio_clip_seconds(12.0) == 12.0
    assert audio_clip_seconds(9999.0) == 600.0
    assert audio_clip_seconds(None) == 600.0
