"""采样算法（纯函数，无 IO，便于单测）。

两条主线：

* 动图：源帧数已知，直接按下标均匀取样。
* 视频：只知道时长，需要先决定「取多少帧」，再决定「在哪些时间点取」。
  短视频一次顺序解码即可；长视频改用「多个短窗口」——既覆盖全片，
  又不必解码整部片子，同时窗口内的连续帧能真正体现动作。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .settings import HARD_MAX_FRAMES_PER_MEDIA, MB, DetailPreset

SINGLE_PASS_MAX_SECONDS = 120.0
"""不超过这个时长就整片顺序解码一次取帧。"""

BURST_WINDOW_SECONDS = 5.0
"""长视频每个采样窗口的长度。"""

FRAMES_PER_BURST = 3
"""长视频每个采样窗口期望的帧数。"""

COVERAGE_FIRST_SECONDS = 900.0
"""超过这个时长时优先铺开覆盖范围：每个窗口少给一帧、换更多的窗口。"""

UNKNOWN_DURATION_INTERVAL = 2.0
"""时长探测失败时的兜底采样间隔（秒）。"""

MIN_ANIMATION_FRAMES = 2

ASSUMED_ANIMATION_FPS = 10.0
"""动图没写帧延时时的假定帧率，用来把帧数换算成时长。"""


@dataclass(frozen=True)
class ExtractionWindow:
    """一次 ffmpeg 调用要覆盖的时间窗口。

    length 为 None 表示「一直到文件结尾」（时长探测失败时使用）；这种情况下
    退回固定的 UNKNOWN_DURATION_INTERVAL 间隔，而不是让 ffmpeg 连续输出
    原生帧——那样抽出来的几张图几乎一模一样，等于白花钱。
    """

    start: float
    length: float | None
    count: int

    @property
    def step(self) -> float:
        """相邻两帧的时间间隔（秒）。"""
        if self.count <= 0:
            return 0.0
        if self.length is None or self.length <= 0:
            return UNKNOWN_DURATION_INTERVAL
        return self.length / self.count

    def timestamps(self) -> list[float]:
        """窗口内实际会落在哪些时间点（与 fps 滤镜的行为一致）。"""
        if self.count <= 0:
            return []
        step = self.step
        return [self.start + step * i for i in range(self.count)]


def sample_indices(total: int, target: int) -> list[int]:
    """从 [0, total) 里均匀取 target 个下标，必定包含首尾。"""
    if total <= 0 or target <= 0:
        return []
    if target >= total:
        return list(range(total))
    if target == 1:
        return [0]
    step = (total - 1) / (target - 1)
    picked = sorted({round(i * step) for i in range(target)})
    picked[0] = 0
    picked[-1] = total - 1
    return sorted(set(picked))


def animation_frame_budget(
    total_frames: int,
    size_bytes: int,
    preset: DetailPreset,
    override: int = 0,
    duration: float | None = None,
) -> int:
    """决定一张动图抽多少帧。

    和视频一样按密度算：动图的时长差别很大，两秒的表情包和二十秒的短动画
    如果都只给 6 帧，后者等于每 3 秒才看一眼。时长未知时用帧数按假定帧率
    倒推，至少不会把长动图当成短表情包。

    额外的两道保护：源帧数不够时以源为准（4 帧的图抽 6 帧没有意义），
    文件体量大时按比例收敛，但至少保留 2 帧——只剩 1 帧就等于没抽帧了。
    """
    if total_frames <= 1:
        return min(1, max(total_frames, 0))

    ceiling = min(preset.max_animation_frames, HARD_MAX_FRAMES_PER_MEDIA, total_frames)
    floor = min(preset.min_animation_frames, ceiling)

    if override > 0:
        target = min(override, HARD_MAX_FRAMES_PER_MEDIA, total_frames)
    else:
        seconds = duration if duration and duration > 0 else total_frames / ASSUMED_ANIMATION_FPS
        wanted = math.ceil(seconds / max(preset.animation_seconds_per_frame, 0.05))
        target = max(floor, min(wanted, ceiling))

    if total_frames <= 4:
        target = min(target, 3)
    target = _shrink_for_size(target, size_bytes)

    return max(MIN_ANIMATION_FRAMES, min(target, total_frames))


def _shrink_for_size(target: int, size_bytes: int) -> int:
    """大文件按比例减帧：固定减 1~2 帧对 30 帧的预算等于没减。"""
    if size_bytes >= 10 * MB:
        return round(target * 0.5)
    if size_bytes >= 5 * MB:
        return round(target * 0.75)
    return target


def video_frame_budget(duration: float | None, preset: DetailPreset, override: int = 0) -> int:
    """决定一个视频抽多少帧：按目标采样间隔随时长增长，再夹进上下限。

    这样 10 秒的短片和 10 分钟的长片不会拿到同一个帧数，也不会因为「长视频」
    这一个笼统的档位，让 1 分钟和 2 小时的视频得到完全一样的待遇。
    """
    if override > 0:
        return min(override, HARD_MAX_FRAMES_PER_MEDIA)

    ceiling = min(preset.max_video_frames, HARD_MAX_FRAMES_PER_MEDIA)
    floor = min(preset.min_video_frames, ceiling)

    if duration is None or duration <= 0:
        # 时长未知，按兜底间隔估一个中等规模，别一上来就顶到上限。
        return max(floor, min(ceiling, floor * 2))

    wanted = math.ceil(duration / max(preset.video_seconds_per_frame, 0.1))
    return max(floor, min(wanted, ceiling))


def frames_per_burst(duration: float) -> int:
    """很长的片子改成每个窗口 2 帧，用同样的帧数换更多的采样位置。"""
    return 2 if duration > COVERAGE_FIRST_SECONDS else FRAMES_PER_BURST


def distribute(total: int, buckets: int) -> list[int]:
    """把 total 尽量均匀地分给 buckets 个桶，余数从前往后补。"""
    if buckets <= 0 or total <= 0:
        return []
    base, remainder = divmod(total, buckets)
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def plan_extraction(duration: float | None, frame_budget: int) -> list[ExtractionWindow]:
    """把「抽 N 帧」翻译成若干个 ffmpeg 抽帧窗口。"""
    count = max(1, frame_budget)

    if duration is None or duration <= 0:
        # 时长未知（探测失败）：整片走一遍，让 ffmpeg 自己决定能给多少帧。
        return [ExtractionWindow(0.0, None, count)]

    if duration <= SINGLE_PASS_MAX_SECONDS or count <= 2:
        return [ExtractionWindow(0.0, duration, count)]

    per_burst = frames_per_burst(duration)
    bursts = max(2, min(count, round(count / per_burst)))
    counts = [c for c in distribute(count, bursts) if c > 0]
    bursts = len(counts)
    window = min(BURST_WINDOW_SECONDS, duration / bursts)
    span = max(0.0, duration - window)
    step = span / (bursts - 1) if bursts > 1 else 0.0

    return [
        ExtractionWindow(round(step * i, 3), round(window, 3), counts[i]) for i in range(bursts)
    ]


def fair_allocation(wants: list[int], total: int) -> list[int]:
    """把 total 个名额分给若干需求，谁也不会被前面的媒体饿死。

    做法是注水式均分：每轮把剩余名额平均分给还没吃饱的需求，吃饱的退出，
    直到名额用尽。这样第一个视频不会一口气吃掉整轮预算。
    """
    granted = [0] * len(wants)
    remaining = max(0, total)
    active = [i for i, want in enumerate(wants) if want > 0]

    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            for i in active[:extra]:
                granted[i] += 1
            break
        for i in list(active):
            take = min(share, wants[i] - granted[i])
            granted[i] += take
            remaining -= take
            if granted[i] >= wants[i]:
                active.remove(i)

    return granted


def thin_indices(total: int, keep: int) -> list[int]:
    """要砍帧时均匀地砍，而不是把片尾整段丢掉。"""
    if keep <= 0:
        return []
    return sample_indices(total, keep)


def audio_clip_seconds(duration: float | None, limit: float = 600.0) -> float:
    """决定抽多长的音轨去转写/附带。"""
    if duration is None or duration <= 0:
        return limit
    return min(duration, limit)
