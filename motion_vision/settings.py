"""配置解析。

WebUI 里只暴露少量按模块分组的选项，具体的画质/帧数等数值由
「细节档位」(detail_level) 一次性给出，避免用户面对几十个旋钮。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MB = 1024 * 1024

HARD_MAX_FRAMES_PER_MEDIA = 64
"""单个媒体的绝对帧数上限，防止配置写飞。"""


@dataclass(frozen=True)
class DetailPreset:
    """一档画质/密度预设。

    帧数都不写死，而是由「目标采样间隔」乘时长算出来，再夹在上下限之间：
    短的不浪费额度，长的也不会出现「两小时给 20 帧」这种名不副实的情况。

    动图和视频各有一套密度参数。两者的形态差别很大——表情包往往只有一两秒
    但动作全在这一两秒里，视频则可能长达几十分钟——用同一组数值必然有一头
    照顾不到。
    """

    name: str

    animation_seconds_per_frame: float
    """动图的目标采样间隔：每隔这么多秒希望有 1 帧。"""

    min_animation_frames: int
    """再短的动图也至少给这么多帧（源帧数不够时以源为准）。"""

    max_animation_frames: int
    """帧数再多的动图也不超过这么多帧。"""

    video_seconds_per_frame: float
    """视频的目标采样间隔：每隔这么多秒希望有 1 帧。"""

    min_video_frames: int
    """再短的视频也至少给这么多帧。"""

    max_video_frames: int
    """再长的视频也不超过这么多帧（成本闸门）。"""

    max_side: int
    jpeg_quality: int


DETAIL_PRESETS: dict[str, DetailPreset] = {
    "frugal": DetailPreset("frugal", 1.2, 3, 10, 8.0, 4, 14, 512, 80),
    "balanced": DetailPreset("balanced", 0.6, 6, 20, 3.0, 6, 28, 768, 85),
    "detailed": DetailPreset("detailed", 0.35, 10, 30, 1.5, 10, 48, 1024, 90),
}
DEFAULT_DETAIL_LEVEL = "balanced"

DETAIL_ALIASES: dict[str, str] = {
    "节省": "frugal",
    "均衡": "balanced",
    "精细": "detailed",
}
"""WebUI 里展示的中文档位名 -> 内部键名。"""

DETAIL_LABELS: dict[str, str] = {value: key for key, value in DETAIL_ALIASES.items()}
"""内部键名 -> 中文档位名，用于状态输出。"""

AUDIO_MODES = ("off", "attach", "transcribe", "attach_and_transcribe")

AUDIO_ALIASES: dict[str, str] = {
    "关闭": "off",
    "附带音轨": "attach",
    "转写文字": "transcribe",
    "音轨和文字": "attach_and_transcribe",
}

AUDIO_LABELS: dict[str, str] = {value: key for key, value in AUDIO_ALIASES.items()}


@dataclass(frozen=True)
class AnimationSettings:
    enabled: bool = True


@dataclass(frozen=True)
class VideoSettings:
    enabled: bool = True
    max_videos_per_request: int = 2
    max_download_mb: int = 100
    include_group_files: bool = True


@dataclass(frozen=True)
class AudioSettings:
    mode: str = "off"
    """off / attach / transcribe / attach_and_transcribe"""

    stt_provider_id: str = ""
    api_base: str = ""
    api_key: str = ""
    model: str = "whisper-1"
    max_transcript_chars: int = 800

    @property
    def attach(self) -> bool:
        return self.mode in ("attach", "attach_and_transcribe")

    @property
    def transcribe(self) -> bool:
        return self.mode in ("transcribe", "attach_and_transcribe")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def use_custom_api(self) -> bool:
        return bool(self.api_base and self.api_key)


@dataclass(frozen=True)
class InjectionSettings:
    notice_enabled: bool = True
    keep_frames_in_history: bool = False
    extra_guidance: str = ""


@dataclass(frozen=True)
class AdvancedSettings:
    ffmpeg_path: str = ""
    max_images_per_request: int = 48
    max_frame_payload_mb: int = 20
    max_seconds_per_video: int = 180
    temp_retention_hours: int = 6
    debug_log: bool = False


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    detail_level: str = DEFAULT_DETAIL_LEVEL
    animation_frames_override: int = 0
    video_frames_override: int = 0
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    injection: InjectionSettings = field(default_factory=InjectionSettings)
    advanced: AdvancedSettings = field(default_factory=AdvancedSettings)

    @property
    def preset(self) -> DetailPreset:
        return DETAIL_PRESETS.get(self.detail_level, DETAIL_PRESETS[DEFAULT_DETAIL_LEVEL])

    @property
    def cache_signature(self) -> str:
        """影响抽帧产物的配置指纹，用于缓存键。"""
        preset = self.preset
        return (
            f"{preset.name}:{self.animation_frames_override}:{self.video_frames_override}"
            f":{preset.max_side}:{preset.jpeg_quality}:{self.audio.mode}"
        )


def _section(raw: Any, key: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return default


def _as_int(value: Any, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_choice(
    value: Any,
    choices: tuple[str, ...],
    default: str,
    aliases: dict[str, str] | None = None,
) -> str:
    """把配置值归一到内部键名，同时接受中文选项和历史写法。"""
    raw = _as_str(value)
    if aliases and raw in aliases:
        return aliases[raw]
    lowered = raw.lower()
    return lowered if lowered in choices else default


def load_settings(config: Any) -> Settings:
    """把 AstrBotConfig（或任意嵌套 dict）解析成不可变的 Settings。"""

    raw: dict[str, Any] = config if isinstance(config, dict) else {}

    sampling = _section(raw, "sampling")
    animation = _section(raw, "animation")
    video = _section(raw, "video")
    audio = _section(raw, "audio")
    injection = _section(raw, "injection")
    advanced = _section(raw, "advanced")

    # 0.2.x 只有一个共用的 frames_override，升级上来时沿用它当两边的初值。
    legacy_override = sampling.get("frames_override", 0)

    return Settings(
        enabled=_as_bool(raw.get("enabled"), True),
        detail_level=_as_choice(
            sampling.get("detail_level"),
            tuple(DETAIL_PRESETS),
            DEFAULT_DETAIL_LEVEL,
            DETAIL_ALIASES,
        ),
        animation_frames_override=_as_int(
            sampling.get("animation_frames_override", legacy_override),
            0,
            0,
            HARD_MAX_FRAMES_PER_MEDIA,
        ),
        video_frames_override=_as_int(
            sampling.get("video_frames_override", legacy_override),
            0,
            0,
            HARD_MAX_FRAMES_PER_MEDIA,
        ),
        animation=AnimationSettings(
            enabled=_as_bool(animation.get("enabled"), True),
        ),
        video=VideoSettings(
            enabled=_as_bool(video.get("enabled"), True),
            max_videos_per_request=_as_int(video.get("max_videos_per_request"), 2, 1, 10),
            max_download_mb=_as_int(video.get("max_download_mb"), 100, 1, 4096),
            include_group_files=_as_bool(video.get("include_group_files"), True),
        ),
        audio=AudioSettings(
            mode=_as_choice(audio.get("mode"), AUDIO_MODES, "off", AUDIO_ALIASES),
            stt_provider_id=_as_str(audio.get("stt_provider_id")),
            api_base=_as_str(audio.get("api_base")).rstrip("/"),
            api_key=_as_str(audio.get("api_key")),
            model=_as_str(audio.get("model"), "whisper-1"),
            max_transcript_chars=_as_int(audio.get("max_transcript_chars"), 800, 40, 20000),
        ),
        injection=InjectionSettings(
            notice_enabled=_as_bool(injection.get("notice_enabled"), True),
            keep_frames_in_history=_as_bool(injection.get("keep_frames_in_history"), False),
            extra_guidance=_as_str(injection.get("extra_guidance")),
        ),
        advanced=AdvancedSettings(
            ffmpeg_path=_as_str(advanced.get("ffmpeg_path")),
            max_images_per_request=_as_int(advanced.get("max_images_per_request"), 48, 1, 200),
            max_frame_payload_mb=_as_int(advanced.get("max_frame_payload_mb"), 20, 1, 500),
            max_seconds_per_video=_as_int(advanced.get("max_seconds_per_video"), 180, 15, 1800),
            temp_retention_hours=_as_int(advanced.get("temp_retention_hours"), 6, 1, 168),
            debug_log=_as_bool(advanced.get("debug_log"), False),
        ),
    )
