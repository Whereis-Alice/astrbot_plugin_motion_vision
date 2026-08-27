"""把处理结果注入 ProviderRequest。

这一层是插件与 AstrBot 请求对象唯一的接触面，做四件事：

1. 把抽出来的帧作为图片挂到本轮请求上；
2. 补一段中文说明，告诉模型「这些图是同一段画面的连续采样」——
   否则模型很容易把 12 张帧当成 12 张无关的图；
3. 需要时附上音轨和语音转写文本；
4. 把原始的附件文本标记和无法阅读的原图从请求里摘掉，避免重复占用额度。

默认所有新增内容都标记为临时（`mark_as_temp`），只对本轮生效、不写进
对话历史；这样长对话不会被几十张帧撑爆。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from .models import MediaKind, MediaResult, SampledFrame
from .settings import Settings

HEADER = (
    "【动态媒体解析】下面的图片是从用户发来的动图或视频里按时间顺序抽取的关键帧，"
    "它们属于同一段画面的连续采样，不是多张互不相关的图片。"
    "请综合这些帧理解画面内容和动作变化，像正常看过这段内容一样回答，"
    "不要逐帧罗列，也不必提到抽帧、关键帧之类的技术细节。"
)

FALLBACK_PROMPT = "请看看这段内容里发生了什么。"

MAX_STAMPS_SHOWN = 12

COARSE_INTERVAL_SECONDS = 10.0
"""平均间隔超过这个值就明确提醒模型「中间的过程没看到」。"""


@dataclass
class InjectionReport:
    """注入结果，用于日志和诊断。"""

    media: int = 0
    frames: int = 0
    audio: int = 0
    transcripts: int = 0
    notices: int = 0

    @property
    def touched(self) -> bool:
        return bool(self.media or self.notices)

    def __str__(self) -> str:
        return (
            f"{self.media} 个媒体 / {self.frames} 帧"
            f" / {self.audio} 段音轨 / {self.transcripts} 条转写"
        )


def inject(request: Any, results: list[MediaResult], settings: Settings) -> InjectionReport:
    """把 MediaResult 列表写进 ProviderRequest，返回统计。"""
    report = InjectionReport()
    usable = [result for result in results if result.ok]
    notices = [result for result in results if result.notice] if _notices_on(settings) else []

    if not usable and not notices:
        return report

    keep = settings.injection.keep_frames_in_history
    _cleanup_request(request, usable)

    parts: list[Any] = _ensure_parts(request)
    add = _appender(parts, temporary=not keep)

    if usable:
        add(_text(HEADER))

    for result in usable:
        label = describe(result)
        if label:
            add(_text(label))

        if result.frames:
            if keep:
                _image_urls(request).extend(str(frame.path) for frame in result.frames)
            else:
                for frame in result.frames:
                    add(_image(frame, result))
            report.frames += len(result.frames)

        if result.transcript:
            add(_text(describe_transcript(result)))
            report.transcripts += 1

        if result.audio is not None:
            _audio_urls(request).append(str(result.audio.path))
            report.audio += 1

        report.media += 1

    if notices:
        add(_text(describe_notices(notices)))
        report.notices = len(notices)

    guidance = settings.injection.extra_guidance
    if guidance and usable:
        add(_text(guidance))

    if not _prompt(request):
        _set_prompt(request, FALLBACK_PROMPT)

    return report


# --- 中文说明 ---------------------------------------------------------------


def describe(result: MediaResult) -> str:
    """一句话交代这个媒体的规格，让模型知道时间跨度有多大。"""
    if not result.frames:
        return ""

    name = result.item.display_name
    source = "引用消息里的" if result.item.quoted else ""
    pieces: list[str] = []

    if result.duration and result.duration > 0:
        pieces.append(f"时长约 {_seconds(result.duration)}")
    if result.kind is MediaKind.ANIMATION and result.source_frame_count:
        pieces.append(f"源共 {result.source_frame_count} 帧")
    pieces.append(f"取样 {len(result.frames)} 帧")

    density = _density(result)
    if density:
        pieces.append(density)

    stamps = _stamps(result.frames)
    if stamps:
        pieces.append(f"对应时间点 {stamps}")

    kind = "动图" if result.kind is MediaKind.ANIMATION else "视频"
    return f"{source}{kind}《{name}》：" + "，".join(pieces) + "。"


def describe_transcript(result: MediaResult) -> str:
    return (
        f"《{result.item.display_name}》里的语音内容（机器转写，可能有错字）：{result.transcript}"
    )


def describe_notices(results: list[MediaResult]) -> str:
    lines = [
        f"- 《{result.item.display_name}》：{result.notice}" for result in results if result.notice
    ]
    return (
        "【动态媒体解析】以下内容没能完整读取，回答时请如实说明看不到，不要编造画面：\n"
        + "\n".join(lines)
    )


# --- 请求对象读写 -----------------------------------------------------------


def _cleanup_request(request: Any, usable: list[MediaResult]) -> None:
    """摘掉已经被替换掉的原始素材。

    * 原始动图留在 image_urls 里毫无用处：多数模型只会看到第一帧，
      却要为整张图付出代价，所以直接删掉。
    * 提示词里的附件标记同理，帧已经附上了，标记只会让模型困惑。
    """
    _drop_original_images(request, usable)
    _strip_markers(request, usable)


def _drop_original_images(request: Any, usable: list[MediaResult]) -> None:
    urls = _image_urls(request)
    indexes = sorted(
        {
            result.item.image_url_index
            for result in usable
            if result.item.image_url_index is not None
        },
        reverse=True,
    )
    for index in indexes:
        if 0 <= index < len(urls):
            del urls[index]


def _strip_markers(request: Any, usable: list[MediaResult]) -> None:
    parts = getattr(request, "extra_user_content_parts", None)
    if not isinstance(parts, list) or not parts:
        return

    dirty = False
    for result in usable:
        item = result.item
        if item.part_index is None or not item.marker_raw:
            continue
        if not 0 <= item.part_index < len(parts):
            continue
        part = parts[item.part_index]
        text = getattr(part, "text", None)
        if not isinstance(text, str) or item.marker_raw not in text:
            continue
        try:
            part.text = text.replace(item.marker_raw, "").strip()
        except Exception:
            continue
        dirty = True

    if not dirty:
        return

    survivors = [
        part
        for part in parts
        if getattr(part, "type", "") != "text" or (getattr(part, "text", "") or "").strip()
    ]
    if len(survivors) != len(parts):
        parts[:] = survivors


def _ensure_parts(request: Any) -> list[Any]:
    parts = getattr(request, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        parts = []
        request.extra_user_content_parts = parts
    return parts


def _image_urls(request: Any) -> list[str]:
    urls = getattr(request, "image_urls", None)
    if not isinstance(urls, list):
        urls = []
        request.image_urls = urls
    return urls


def _audio_urls(request: Any) -> list[str]:
    urls = getattr(request, "audio_urls", None)
    if not isinstance(urls, list):
        urls = []
        request.audio_urls = urls
    return urls


def _prompt(request: Any) -> str:
    value = getattr(request, "prompt", "")
    return value.strip() if isinstance(value, str) else ""


def _set_prompt(request: Any, text: str) -> None:
    with contextlib.suppress(Exception):
        request.prompt = text


# --- ContentPart 构造 -------------------------------------------------------


def _appender(parts: list[Any], temporary: bool) -> Any:
    seen: set[str] = set()

    def add(part: Any) -> None:
        if part is None:
            return
        text = getattr(part, "text", None)
        if isinstance(text, str):
            if text in seen:
                return
            seen.add(text)
        if temporary:
            marker = getattr(part, "mark_as_temp", None)
            if callable(marker):
                marker()
        parts.append(part)

    return add


def _text(content: str) -> Any:
    from astrbot.core.agent.message import TextPart

    return TextPart(text=content)


def _image(frame: SampledFrame, result: MediaResult) -> Any:
    from astrbot.core.agent.message import ImageURLPart

    return ImageURLPart(
        image_url=ImageURLPart.ImageURL(url=str(frame.path), id=_frame_id(frame, result))
    )


def _frame_id(frame: SampledFrame, result: MediaResult) -> str:
    """给每帧一个短标签，模型引用某一瞬间时能对上号。"""
    stem = result.item.display_name
    if len(stem) > 24:
        stem = stem[:23] + "…"
    if frame.timestamp is None:
        return f"{stem} #{frame.index + 1}"
    return f"{stem} #{frame.index + 1}@{frame.timestamp:.1f}s"


# --- 格式化小工具 -----------------------------------------------------------


def _notices_on(settings: Settings) -> bool:
    return settings.injection.notice_enabled


def _seconds(value: float) -> str:
    if value >= 60:
        minutes, rest = divmod(value, 60)
        return f"{int(minutes)} 分 {rest:.0f} 秒"
    return f"{value:.1f} 秒"


def _density(result: MediaResult) -> str:
    """说明这段内容被抽得有多稀，避免模型把「看过几帧」当成「看完全片」。"""
    duration = result.duration or 0.0
    count = len(result.frames)
    if duration <= 0 or count <= 1:
        return ""

    interval = duration / count
    text = f"平均每 {interval:.1f} 秒 1 帧"
    if interval >= COARSE_INTERVAL_SECONDS:
        text += "（间隔较大，两帧之间发生的事情没有被记录，不要凭空补全）"
    return text


def _stamps(frames: list[SampledFrame]) -> str:
    values = [frame.timestamp for frame in frames if frame.timestamp is not None]
    if not values:
        return ""
    shown = [f"{value:.1f}s" for value in values[:MAX_STAMPS_SHOWN]]
    text = "、".join(shown)
    if len(values) > MAX_STAMPS_SHOWN:
        text += " …"
    return text
