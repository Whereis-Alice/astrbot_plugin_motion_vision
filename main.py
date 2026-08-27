"""动态视觉 Motion Vision —— 让大模型真正读懂动图和视频。

这个文件只做编排：接住 AstrBot 的钩子、组装依赖、把活派给 motion_vision 包里的
各个模块，然后把结果注入本轮 LLM 请求。所有实际逻辑都在子模块里，方便单测。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .motion_vision.cache import ResultCache
from .motion_vision.ffmpeg import FfmpegRunner, FfmpegTools, discover_tools
from .motion_vision.inject import inject
from .motion_vision.models import MediaItem, MediaKind, MediaResult
from .motion_vision.pipeline import MediaPipeline
from .motion_vision.settings import AUDIO_LABELS, DETAIL_LABELS, Settings, load_settings
from .motion_vision.sources.animation import resolve_animations
from .motion_vision.sources.video import VideoCollector
from .motion_vision.stt import describe_backend
from .motion_vision.tempstore import TempStore

PLUGIN_NAME = "astrbot_plugin_motion_vision"
LOG_TAG = "[MotionVision]"

MAX_CONCURRENT_REQUESTS = 2
"""同时处理媒体的会话数上限。抽帧和转写都是重活，放开了只会互相拖慢并触发上游限流。"""

DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)

EMPTY_MESSAGE_FALLBACK = "请看看我发的这段内容里有什么。"


@register(
    PLUGIN_NAME,
    "Whereis-Alice",
    "让大模型读懂动图和视频：自动抽取关键帧、可选提取语音，再连同说明一起交给模型。",
    "0.2.0",
    "https://github.com/Whereis-Alice/astrbot_plugin_motion_vision",
)
class MotionVisionPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.raw_config: Any = config if config is not None else {}
        self.settings: Settings = load_settings(self.raw_config)

        self.store = TempStore(
            Path(get_astrbot_temp_path()),
            self.settings.advanced.temp_retention_hours * 3600,
        )
        self.cache = ResultCache(on_evict=TempStore.discard)
        self.client = httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        self._gate = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        self._tools: FfmpegTools = discover_tools(self.settings.advanced.ffmpeg_path)
        self._tools_key = self.settings.advanced.ffmpeg_path
        self.runner = FfmpegRunner(self._tools)

        logger.info(
            f"{LOG_TAG} 已加载：细节档位 {self.settings.detail_level}，"
            f"ffmpeg {self._tools.source or '未找到'}"
        )

    # --- 钩子 ---------------------------------------------------------------

    @filter.on_waiting_llm_request()
    async def ensure_question(self, event: AstrMessageEvent) -> None:
        """纯媒体消息（只发了个视频、一句话都没说）补一句默认提问。

        否则请求里没有任何文字，部分模型会直接拒绝，或者答得莫名其妙。
        """
        if not self._refresh().enabled:
            return
        try:
            if (event.message_str or "").strip():
                return
            if not self._has_media(event):
                return
            event.message_str = EMPTY_MESSAGE_FALLBACK
            event.message_obj.message_str = EMPTY_MESSAGE_FALLBACK
        except Exception as exc:
            logger.debug(f"{LOG_TAG} 兜底提问写入失败: {exc}")

    @filter.on_llm_request()
    async def handle(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """本插件的主入口：收集媒体 -> 抽帧/抽音 -> 注入请求。

        全程包在 try 里：解析动态媒体属于增强能力，出任何问题都不该影响对话本身。
        """
        settings = self._refresh()
        if not settings.enabled:
            return
        if not settings.animation.enabled and not settings.video.enabled:
            return

        try:
            async with self._gate:
                await self._handle(event, req, settings)
        except Exception as exc:
            logger.error(
                f"{LOG_TAG} 处理动态媒体时出错: {exc}", exc_info=settings.advanced.debug_log
            )

    async def _handle(
        self, event: AstrMessageEvent, req: ProviderRequest, settings: Settings
    ) -> None:
        self.store.sweep()

        items, notices = await self._collect(event, req, settings)
        if not items and not notices:
            return

        pipeline = MediaPipeline(
            settings=settings,
            runner=self.runner,
            store=self.store,
            cache=self.cache,
            client=self.client,
            context=self.context,
            log=logger,
        )
        results: list[MediaResult] = await pipeline.run(items) if items else []
        results.extend(
            MediaResult(item=MediaItem(MediaKind.VIDEO, name, name, None), notice=reason)
            for name, reason in notices
        )

        report = inject(req, results, settings)
        if report.touched:
            logger.info(f"{LOG_TAG} 已注入 {report}")

        self._discard_owned(results)

    # --- 来源收集 -----------------------------------------------------------

    async def _collect(
        self, event: AstrMessageEvent, req: ProviderRequest, settings: Settings
    ) -> tuple[list[MediaItem], list[tuple[str, str]]]:
        items: list[MediaItem] = []
        notices: list[tuple[str, str]] = []
        search_dirs = self._search_dirs()

        if settings.animation.enabled:
            items.extend(
                await resolve_animations(
                    req,
                    event,
                    self.client,
                    search_dirs=search_dirs,
                    on_skip=lambda name, reason: notices.append((name, reason)),
                )
            )

        if settings.video.enabled:
            collector = VideoCollector(
                request=req,
                event=event,
                client=self.client,
                download_path_factory=self.store.download_path,
                max_videos=settings.video.max_videos_per_request,
                max_download_mb=settings.video.max_download_mb,
                include_group_files=settings.video.include_group_files,
                search_dirs=search_dirs,
            )
            found = await collector.collect()
            items.extend(found.items)
            notices.extend(found.notices)

        return (items, notices)

    @staticmethod
    def _has_media(event: AstrMessageEvent) -> bool:
        from astrbot.api.message_components import File, Image, Video

        try:
            chain = event.get_messages() or []
        except Exception:
            return False
        return any(isinstance(component, (Image, Video, File)) for component in chain)

    @staticmethod
    def _search_dirs() -> tuple[Path, ...]:
        """平台适配器落盘的位置各不相同，给相对路径留几个查找根。"""
        candidates = [Path(get_astrbot_temp_path()), Path.cwd()]
        seen: list[Path] = []
        for candidate in candidates:
            if candidate.is_dir() and candidate not in seen:
                seen.append(candidate)
        return tuple(seen)

    def _discard_owned(self, results: list[MediaResult]) -> None:
        """插件自己下载的原始视频在抽完帧后就没用了，立刻删掉省磁盘。"""
        for result in results:
            item = result.item
            if item.owned_temp and item.path is not None:
                TempStore.discard(item.path)
                item.path = None

    # --- 配置热更新 ---------------------------------------------------------

    def _refresh(self) -> Settings:
        """每轮重新解析配置，这样在 WebUI 改完就生效，不必重载插件。"""
        settings = load_settings(self.raw_config)
        self.settings = settings
        self.store.retention_seconds = max(60.0, settings.advanced.temp_retention_hours * 3600)

        if settings.advanced.ffmpeg_path != self._tools_key:
            self._tools_key = settings.advanced.ffmpeg_path
            self._tools = discover_tools(self._tools_key)
            self.runner = FfmpegRunner(self._tools)
            logger.info(f"{LOG_TAG} ffmpeg 路径已更新：{self._tools.source or '未找到'}")

        return settings

    # --- 指令 ---------------------------------------------------------------

    @filter.command_group("motionvision", alias={"动态视觉"})
    def motionvision(self) -> None:
        """动态视觉插件的管理指令。"""

    @motionvision.command("status", alias={"状态"})
    async def status(self, event: AstrMessageEvent):
        """查看运行状态与依赖检测结果。"""
        settings = self._refresh()
        preset = settings.preset
        files, total_bytes = self.store.usage()
        ffmpeg_state = self._tools.source if self._tools.available else "未找到（视频功能不可用）"

        lines = [
            "动态视觉 Motion Vision",
            f"总开关：{'开' if settings.enabled else '关'}",
            f"动图：{'开' if settings.animation.enabled else '关'}"
            f" / 视频：{'开' if settings.video.enabled else '关'}",
            f"细节档位：{DETAIL_LABELS.get(settings.detail_level, settings.detail_level)}"
            f"（动图 {preset.animation_frames} 帧，视频约每 {preset.seconds_per_frame:g} 秒 1 帧、"
            f"{preset.min_video_frames}~{preset.max_video_frames} 帧，"
            f"最长边 {preset.max_side}px）",
            f"帧数覆盖：{settings.frames_override or '未设置'}",
            f"ffmpeg：{ffmpeg_state}",
            f"音频模式：{AUDIO_LABELS.get(settings.audio.mode, settings.audio.mode)}",
            f"语音转写：{describe_backend(self.context, settings.audio)}",
            f"结果缓存：{len(self.cache)} 条",
            f"临时文件：{files} 个 / {total_bytes / 1048576:.1f} MB",
        ]
        yield event.plain_result("\n".join(lines))

    @motionvision.command("clear", alias={"清理"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def clear(self, event: AstrMessageEvent):
        """清空结果缓存和临时文件。"""
        self.cache.clear()
        removed = self.store.sweep(min_interval=0.0)
        TempStore.discard(self.store.base)
        yield event.plain_result(f"已清空结果缓存，并清理了 {removed} 项临时文件。")

    # --- 生命周期 -----------------------------------------------------------

    async def terminate(self) -> None:
        self.cache.clear()
        with contextlib.suppress(Exception):
            await self.client.aclose()
        logger.info(f"{LOG_TAG} 已卸载")
