"""采样算法。"""

from __future__ import annotations

from motion_vision.sampling import (
    BURST_WINDOW_SECONDS,
    SINGLE_PASS_MAX_SECONDS,
    UNKNOWN_DURATION_INTERVAL,
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
    # 只有 3 帧的动图不可能抽出 6 帧
    assert animation_frame_budget(3, 0, preset) == 3


def test_animation_budget_follows_duration_not_just_a_flat_count():
    preset = DETAIL_PRESETS["balanced"]
    short = animation_frame_budget(20, 0, preset, duration=2.0)
    long = animation_frame_budget(200, 0, preset, duration=20.0)

    # 2 秒的表情包按下限给，20 秒的短动画必须明显更多，否则等于每 3 秒看一眼
    assert short == preset.min_animation_frames
    assert long > short * 2
    assert long <= preset.max_animation_frames


def test_animation_budget_reaches_thirty_on_detailed():
    preset = DETAIL_PRESETS["detailed"]
    assert preset.max_animation_frames == 30
    assert animation_frame_budget(300, 0, preset, duration=20.0) == 30


def test_animation_budget_estimates_duration_from_frame_count():
    preset = DETAIL_PRESETS["balanced"]
    # 时长未知时按假定帧率倒推：200 帧 ≈ 20 秒，不该被当成两秒的表情包
    assert animation_frame_budget(200, 0, preset) > animation_frame_budget(20, 0, preset)


def test_animation_budget_is_capped_by_the_preset_ceiling():
    for name, preset in DETAIL_PRESETS.items():
        budget = animation_frame_budget(600, 0, preset, duration=600.0)
        assert budget == preset.max_animation_frames, name


def test_animation_budget_backs_off_for_large_files():
    preset = DETAIL_PRESETS["detailed"]
    plain = animation_frame_budget(60, 0, preset, duration=6.0)
    medium = animation_frame_budget(60, 6 * MB, preset, duration=6.0)
    huge = animation_frame_budget(60, 12 * MB, preset, duration=6.0)

    assert plain > medium > huge
    assert huge >= 2


def test_animation_budget_never_drops_below_two():
    preset = DETAIL_PRESETS["frugal"]
    assert animation_frame_budget(30, 50 * MB, preset) == 2


def test_animation_override_wins():
    preset = DETAIL_PRESETS["frugal"]
    assert animation_frame_budget(40, 0, preset, override=9) == 9
    # 覆盖值也不能超过源帧数
    assert animation_frame_budget(5, 0, preset, override=30) == 5


def test_animation_and_video_densities_are_independent():
    for name, preset in DETAIL_PRESETS.items():
        assert preset.animation_seconds_per_frame < preset.video_seconds_per_frame, name


def test_video_budget_follows_target_interval():
    preset = DETAIL_PRESETS["balanced"]
    # 60 秒 / 目标 3 秒一帧 = 20 帧，落在上下限之间时严格按密度给
    assert video_frame_budget(60.0, preset) == 20
    # 极短片段不会低于下限
    assert video_frame_budget(3.0, preset) == preset.min_video_frames
    # 超长片段封顶在上限
    assert video_frame_budget(36000.0, preset) == preset.max_video_frames
    assert video_frame_budget(600.0, preset, override=4) == 4


def test_video_budget_is_monotonic_in_duration():
    for preset in DETAIL_PRESETS.values():
        budgets = [video_frame_budget(d, preset) for d in (5, 20, 60, 180, 600, 3600, 7200)]
        assert budgets == sorted(budgets)
        assert budgets[0] >= preset.min_video_frames
        assert budgets[-1] <= preset.max_video_frames


def test_one_minute_video_is_denser_on_higher_detail():
    frugal = video_frame_budget(60.0, DETAIL_PRESETS["frugal"])
    balanced = video_frame_budget(60.0, DETAIL_PRESETS["balanced"])
    detailed = video_frame_budget(60.0, DETAIL_PRESETS["detailed"])
    assert frugal < balanced < detailed
    # 「精细」档对 1 分钟视频至少要做到 2 秒 1 帧
    assert detailed >= 30


def test_unknown_duration_falls_back_to_a_fixed_interval():
    window = ExtractionWindow(0.0, None, 6)
    assert window.step == UNKNOWN_DURATION_INTERVAL
    # 时间戳必须真的铺开，而不是全部堆在 0 秒
    assert window.timestamps() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_very_long_videos_prefer_more_sampling_points():
    duration = 7200.0
    budget = 24
    windows = plan_extraction(duration, budget)
    assert sum(w.count for w in windows) == budget
    # 每窗口 2 帧 -> 12 个采样点，比每窗口 3 帧的 8 个点覆盖更广
    assert len(windows) >= 12


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
    assert ExtractionWindow(0.0, None, 3).timestamps() == [0.0, 2.0, 4.0]
    assert ExtractionWindow(0.0, 5.0, 0).timestamps() == []


def test_audio_clip_is_capped():
    assert audio_clip_seconds(12.0) == 12.0
    assert audio_clip_seconds(9999.0) == 600.0
    assert audio_clip_seconds(None) == 600.0
