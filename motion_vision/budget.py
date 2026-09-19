"""一轮媒体的画面预算：视频优先，同类均分，动图尽量保留首尾。"""

from __future__ import annotations

from .models import MediaKind
from .sampling import fair_allocation, thin_indices

ANIMATION_RESERVE = 2
"""混合消息里，每张动图尽量保留两帧；张数和体积上限始终优先。"""


def _priority_allocation(
    wants: list[int], minimums: list[int], kinds: list[MediaKind], total: int
) -> list[int]:
    videos = [i for i, kind in enumerate(kinds) if kind is MediaKind.VIDEO]
    animations = [i for i, kind in enumerate(kinds) if kind is MediaKind.ANIMATION]
    if not videos or not animations:
        return fair_allocation(wants, total)

    # 只有两类媒体的基础画面都放得下，才为动图预留。极小预算仍优先视频。
    can_reserve = sum(minimums) <= total
    floors = [minimums[i] if can_reserve else 0 for i in animations]
    reserve = sum(floors)
    granted = [0] * len(wants)
    video_grants = fair_allocation([wants[i] for i in videos], total - reserve)
    for i, value in zip(videos, video_grants, strict=True):
        granted[i] = value
    animation_grants = fair_allocation(
        [wants[i] - floor for i, floor in zip(animations, floors, strict=True)],
        total - sum(video_grants) - reserve,
    )
    for i, value, floor in zip(animations, animation_grants, floors, strict=True):
        granted[i] = value + floor
    return granted


def frame_allowances(
    sizes: list[list[int]], kinds: list[MediaKind], max_frames: int, max_bytes: int
) -> list[int]:
    """返回每份媒体最终保留的帧数，不修改原始帧或缓存。

    张数先为动图预留少量画面，再满足视频目标，余量按动图需求均分。
    体积不足时复用同样的优先级，但按均匀取样后真实的字节数计算。
    """
    wants = [len(media) for media in sizes]
    limits = _priority_allocation(
        wants, [min(ANIMATION_RESERVE, n) for n in wants], kinds, max(0, max_frames)
    )
    # 每种帧数对应的实际体积。取样点会随数量变化，体积不一定单调，不能
    # 用平均帧大小推算或二分；单媒体最多 64 帧，枚举成本很小。
    costs = [
        [
            sum(max(0, media[j]) for j in thin_indices(len(media), count))
            for count in range(len(media) + 1)
        ]
        for media in sizes
    ]
    required = [cost[limit] for cost, limit in zip(costs, limits, strict=True)]
    max_bytes = max(0, max_bytes)
    if sum(required) <= max_bytes:
        return limits

    minimums = [
        min(cost[limit], cost[min(ANIMATION_RESERVE, limit)])
        for cost, limit in zip(costs, limits, strict=True)
    ]
    byte_grants = _priority_allocation(required, minimums, kinds, max_bytes)
    counts = [
        next(n for n in range(limit, -1, -1) if cost[n] <= grant)
        for cost, limit, grant in zip(costs, limits, byte_grants, strict=True)
    ]

    # 字节不能像帧数一样整除。回收放不下一帧的零头，优先补给视频，
    # 再补给动图；同类里先补帧数少的，避免顺序靠前的独占剩余预算。
    remaining = max_bytes - sum(cost[n] for cost, n in zip(costs, counts, strict=True))
    while True:
        before = sum(counts)
        for kind in (MediaKind.VIDEO, MediaKind.ANIMATION):
            members = [i for i, current in enumerate(kinds) if current is kind]
            while True:
                slots = max_frames - sum(counts)
                for i in sorted(members, key=lambda index: counts[index]):
                    current = counts[i]
                    # 体积裁剪腾出的张数也可再利用，不受初次张数分配限制。
                    ceiling = min(wants[i], current + slots)
                    target = next(
                        (
                            n
                            for n in range(current + 1, ceiling + 1)
                            if costs[i][n] - costs[i][current] <= remaining
                        ),
                        None,
                    )
                    if target is not None:
                        remaining -= costs[i][target] - costs[i][current]
                        counts[i] = target
                        break
                else:
                    break
        # 改变取样点可能反而节省体积，再优先尝试补视频。每次必须增加
        # 帧数才继续，循环总量受源帧数和张数上限约束。
        if sum(counts) == before:
            break
    return counts
