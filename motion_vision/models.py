"""插件内部使用的数据模型。

这些对象在「来源收集 -> 抽帧/抽音 -> 注入」三个阶段之间传递，
刻意保持成简单的 dataclass，方便单测直接构造。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


def format_duration(seconds: float | None) -> str:
    """把秒数写成人能读的时长，全插件共用一种写法。"""
    if seconds is None or seconds < 0:
        return "未知时长"
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds - hours * 3600) // 60)
        return f"{hours} 小时 {minutes} 分"
    if seconds >= 60:
        minutes = int(seconds // 60)
        rest = round(seconds - minutes * 60)
        if rest >= 60:  # 59.7 秒会四舍五入成 60，别写出「1 分 60 秒」
            minutes += 1
            rest = 0
        return f"{minutes} 分 {rest} 秒"
    return f"{seconds:.1f} 秒"


class MediaKind(str, Enum):
    """插件能处理的两类动态媒体。"""

    ANIMATION = "animation"
    """动图：GIF / 动态 WebP / APNG。"""

    VIDEO = "video"
    """视频文件。"""


@dataclass
class MediaItem:
    """一个待处理的动态媒体。

    path / data 二者至少有一个有值：path 指向本地文件，data 是内联字节
    （来自 base64 图片）。
    """

    kind: MediaKind
    name: str
    identity: str
    path: Path | None = None
    data: bytes | None = None
    image_url_index: int | None = None
    """若来自 ProviderRequest.image_urls，记录下标，便于原位替换。"""
    part_index: int | None = None
    """若来自 extra_user_content_parts 的文本标记，记录下标，便于事后移除。"""
    marker_raw: str = ""
    """对应的原始标记文本，处理完成后可以从提示词里删掉。"""
    quoted: bool = False
    """是否来自引用（回复）消息。"""

    source_url: str = ""
    """原始下载地址。本地临时文件被清理后，靠它才能重新取回来。"""
    owned_temp: bool = False
    """该文件是否由插件下载/生成，处理完可以删除。"""

    @property
    def display_name(self) -> str:
        return self.name or ("动图" if self.kind is MediaKind.ANIMATION else "视频")


@dataclass(frozen=True)
class TimeSpan:
    """一个时间段（秒），用于「只看视频的某一段」。

    end 为 None 表示一直到结尾。start 与 end 都是相对片头的绝对秒数，
    这样抽出来的帧时间戳可以直接对上原片。
    """

    start: float = 0.0
    end: float | None = None

    @property
    def active(self) -> bool:
        """是否真的限定了范围（两端都留空就等于整片）。"""
        return self.start > 0 or (self.end is not None and self.end > 0)

    @property
    def key(self) -> str:
        """写进缓存键的稳定表示。"""
        tail = "end" if self.end is None else f"{self.end:.3f}"
        return f"{self.start:.3f}-{tail}"

    @property
    def label(self) -> str:
        head = format_duration(self.start) if self.start > 0 else "开头"
        tail = format_duration(self.end) if self.end is not None else "结尾"
        return f"{head} ~ {tail}"


@dataclass(frozen=True)
class SampledFrame:
    """抽出的一张静态帧。"""

    path: Path
    index: int
    timestamp: float | None = None
    size_bytes: int = 0


@dataclass(frozen=True)
class AudioClip:
    """从视频里剥离出来的音轨片段。"""

    path: Path
    seconds: float


@dataclass
class MediaResult:
    """单个媒体的处理产物。"""

    item: MediaItem
    frames: list[SampledFrame] = field(default_factory=list)
    duration: float | None = None
    source_frame_count: int | None = None
    audio: AudioClip | None = None
    transcript: str = ""
    notice: str = ""
    """处理失败或被降级时给模型看的中文说明。"""

    @property
    def kind(self) -> MediaKind:
        return self.item.kind

    @property
    def ok(self) -> bool:
        return bool(self.frames or self.audio or self.transcript)
