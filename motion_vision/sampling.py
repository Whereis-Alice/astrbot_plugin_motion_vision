"""采样算法（纯函数，无 IO，便于单测）。

两条主线：

* 动图：源帧数已知，直接按下标均匀取样。
* 视频：只知道时长，需要先决定「取多少帧」，再决定「在哪些时间点取」。
  短视频一次顺序解码即可；长视频改用「多个短窗口」——既覆盖全片，
  又不必解码整部片子，同时窗口内的连续帧能真正体现动作。
"""

from __future__ import annotations

from dataclasses import dataclass

from .settings import HARD_MAX_FRAMES_PER_MEDIA, LONG_VIDEO_SECONDS, MB, DetailPreset

SINGLE_PASS_MAX_SECONDS = 120.0
"""不超过这个时长就整片顺序解码一次取帧。"""

BURST_WINDOW_SECONDS = 5.0
"""长视频每个采样窗口的长度。"""

FRAMES_PER_BURST = 3
"""长视频每个采样窗口期望的帧数。"""

MIN_ANIMATION_FRAMES = 2


@dataclass(frozen=True)
class ExtractionWindow:
    """一次 ffmpeg 调用要覆盖的时间窗口。

    length 为 None 表示「一直到文件结尾」（时长未知时使用）。
    """

    start: float
    length: float | None
    count: int

    def timestamps(self) -> list[float]:
        """窗口内实际会落在哪些时间点（与 fps 滤镜的行为一致）。"""
        if self.count <= 0:
            return []
        if self.length is None or self.length <= 0:
            return [self.start]
        step = self.length / self.count
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
    total_frames: int, size_bytes: int, preset: DetailPreset, override: int = 0
) -> int:
    """决定一张动图抽多少帧。

    体量越大越保守：>=5MB 减一帧，>=10MB 减两帧，但至少保留 2 帧
    （只剩 1 帧就等于没抽帧了）。
    """
    if total_frames <= 1:
        return min(1, max(total_frames, 0))

    base = override if override > 0 else preset.animation_frames
    target = min(base, total_frames, HARD_MAX_FRAMES_PER_MEDIA)
    if total_frames <= 4:
        target = min(target, 3)

    if size_bytes >= 10 * MB:
        target -= 2
    elif size_bytes >= 5 * MB:
        target -= 1

    return max(MIN_ANIMATION_FRAMES, min(target, total_frames))


def video_frame_budget(duration: float | None, preset: DetailPreset, override: int = 0) -> int:
    """决定一个视频抽多少帧。长视频给更多帧，因为时间跨度更大。"""
    if override > 0:
        return min(override, HARD_MAX_FRAMES_PER_MEDIA)
    if duration is not None and duration > LONG_VIDEO_SECONDS:
        return min(preset.long_video_frames, HARD_MAX_FRAMES_PER_MEDIA)
    return min(preset.video_frames, HARD_MAX_FRAMES_PER_MEDIA)


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

    bursts = max(2, min(count, round(count / FRAMES_PER_BURST)))
    counts = [c for c in distribute(count, bursts) if c > 0]
    bursts = len(counts)
    window = min(BURST_WINDOW_SECONDS, duration / bursts)
    span = max(0.0, duration - window)
    step = span / (bursts - 1) if bursts > 1 else 0.0

    return [
        ExtractionWindow(round(step * i, 3), round(window, 3), counts[i]) for i in range(bursts)
    ]


def audio_clip_seconds(duration: float | None, limit: float = 600.0) -> float:
    """决定抽多长的音轨去转写/附带。"""
    if duration is None or duration <= 0:
        return limit
    return min(duration, limit)
