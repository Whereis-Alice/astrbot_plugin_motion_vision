"""插件内部使用的数据模型。

这些对象在「来源收集 -> 抽帧/抽音 -> 注入」三个阶段之间传递，
刻意保持成简单的 dataclass，方便单测直接构造。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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
    owned_temp: bool = False
    """该文件是否由插件下载/生成，处理完可以删除。"""

    @property
    def display_name(self) -> str:
        return self.name or ("动图" if self.kind is MediaKind.ANIMATION else "视频")


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
