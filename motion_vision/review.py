"""回看请求的翻译层：模型给的参数 -> 插件内部对象 -> 工具返回值。

模型传过来的东西不能当真：帧数可能是字符串、可能是 999，时间区间可能写反、
可能给动图也带上。这里负责把它们洗成安全值，洗不动的就退回「整段重看」——
宁可多给几帧，也不要甩一个报错回去让模型多绕一轮。

和 main.py 分开是为了能单测：这里没有任何 AstrBot 依赖，也不碰实例状态。
"""

from __future__ import annotations

import base64
from dataclasses import replace
from typing import Any

from .models import MediaKind, MediaResult, TimeSpan, format_duration
from .settings import Settings

MAX_FRAMES = 32
"""一次回看最多回传多少帧。工具结果会直接进上下文，必须有个硬上限。"""

STAMPS_SHOWN = 12
"""说明文字里最多列几个时间点，再多就只是噪音。"""


def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def coerce_frames(value: Any) -> int:
    """把模型给的帧数请求夹到合理范围。0 表示不覆盖，按当前档位自动决定。"""
    try:
        wanted = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(wanted, MAX_FRAMES))


def make_span(kind: MediaKind, start: Any, end: Any) -> TimeSpan | None:
    """把起止秒数变成 TimeSpan，None 表示整段。

    动图没有可靠的时间轴（每帧延时都可能不同），分段意义不大，一律整段重看。
    区间写反、写空、写成负数也都退回整段，而不是报错。
    """
    if kind is not MediaKind.VIDEO:
        return None
    begin = max(0.0, coerce_float(start))
    finish = coerce_float(end)
    if finish <= 0:
        return TimeSpan(begin, None) if begin > 0 else None
    if finish <= begin:
        return None
    return TimeSpan(begin, finish)


def tune_settings(settings: Settings, frames: Any) -> Settings:
    """回看专用配置：只要画面。

    音频关掉，是因为语音在第一次出现时已经转写过了，再来一遍纯属浪费；
    帧数上限也单独收紧 —— 工具结果直接进上下文，不能按整轮预算放开。
    """
    tuned = replace(
        settings,
        audio=replace(settings.audio, mode="off"),
        advanced=replace(
            settings.advanced,
            max_images_per_request=min(settings.advanced.max_images_per_request, MAX_FRAMES),
        ),
    )
    wanted = coerce_frames(frames)
    if wanted > 0:
        tuned = replace(tuned, animation_frames_override=wanted, video_frames_override=wanted)
    return tuned


def summarize(name: str, token: str, result: MediaResult, span: TimeSpan | None) -> str:
    """回看结果的说明文字。模型得知道这批图是什么、覆盖了哪一段。"""
    pieces = [f"《{name}》（编号 {token}）重新取样 {len(result.frames)} 帧"]
    if span is not None and span.active:
        pieces.append(f"这次只看 {span.label} 这一段")
    elif result.duration:
        pieces.append(f"整段时长约 {format_duration(result.duration)}")

    stamps = [frame.timestamp for frame in result.frames if frame.timestamp is not None]
    if stamps:
        shown = "、".join(f"{value:.1f}s" for value in stamps[:STAMPS_SHOWN])
        if len(stamps) > STAMPS_SHOWN:
            shown += " …"
        pieces.append(f"对应时间点 {shown}")
    if result.notice:
        pieces.append(result.notice)

    return "，".join(pieces) + "。这些帧按时间顺序排列，属于同一段画面的连续采样。"


def build_payload(name: str, token: str, result: MediaResult, span: TimeSpan | None) -> Any:
    """把帧编码成工具返回值。

    图片走 mcp 的 ImageContent，AstrBot 会落盘并把它们追加进本轮上下文 ——
    前提是当前模型声明支持图片输入。拿不到 mcp 类型时退回纯文本，至少不报错。

    读文件 + base64 是同步的重活，调用方用 to_thread 包着跑。
    """
    summary = summarize(name, token, result, span)
    try:
        from mcp.types import CallToolResult, ImageContent, TextContent
    except Exception:
        return summary + "（当前环境没法直接回传图片，只能给出这段说明。）"

    content: list[Any] = [TextContent(type="text", text=summary)]
    for frame in result.frames:
        try:
            payload = base64.b64encode(frame.path.read_bytes()).decode("ascii")
        except OSError:
            continue
        content.append(ImageContent(type="image", data=payload, mimeType="image/jpeg"))

    if len(content) == 1:
        return summary + "（帧文件已经不在了，没能回传画面。）"
    return CallToolResult(content=content)
