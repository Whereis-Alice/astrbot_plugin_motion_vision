"""处理流水线：MediaItem -> MediaResult。

职责边界很清楚：这里只负责「把媒体变成帧/音轨/转写文本」以及各种预算保护，
不碰 AstrBot 的请求对象（那是 inject.py 的事）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .animation import AnimationError, sample_animation
from .cache import CacheEntry, ResultCache, fingerprint
from .ffmpeg import FfmpegError, FfmpegRunner
from .models import MediaItem, MediaKind, MediaResult, SampledFrame
from .sampling import (
    animation_frame_budget,
    audio_clip_seconds,
    plan_extraction,
    video_frame_budget,
)
from .settings import MB, Settings
from .stt import SttError, transcribe
from .tempstore import TempStore


class MediaPipeline:
    def __init__(
        self,
        settings: Settings,
        runner: FfmpegRunner,
        store: TempStore,
        cache: ResultCache,
        client: Any,
        context: Any,
        log: Any,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.store = store
        self.cache = cache
        self.client = client
        self.context = context
        self.log = log

    # --- 入口 ---------------------------------------------------------------

    async def run(self, items: list[MediaItem]) -> list[MediaResult]:
        results: list[MediaResult] = []
        for item in items:
            try:
                results.append(await self._process(item))
            except Exception as exc:  # 任何单个媒体的失败都不该影响整轮对话
                self.log.warning(f"[MotionVision] 处理 {item.display_name} 时出错: {exc}")
                results.append(MediaResult(item=item, notice="处理时发生未预期的错误"))
        self._apply_payload_budget(results)
        return results

    # --- 单个媒体 -----------------------------------------------------------

    async def _process(self, item: MediaItem) -> MediaResult:
        key = fingerprint(item.path, item.data, self.settings.cache_signature)
        cached = self.cache.get(key)
        if cached is not None:
            return MediaResult(
                item=item,
                frames=list(cached.frames),
                duration=cached.duration,
                source_frame_count=cached.source_frame_count,
                audio=cached.audio,
                transcript=cached.transcript,
            )

        if item.kind is MediaKind.ANIMATION:
            result, entry = await self._process_animation(item)
        else:
            result, entry = await self._process_video(item)

        if entry is not None:
            self.cache.put(key, entry)
        return result

    async def _process_animation(self, item: MediaItem) -> tuple[MediaResult, CacheEntry | None]:
        preset = self.settings.preset
        source = item.data if item.data is not None else item.path
        if source is None:
            return (MediaResult(item=item, notice="找不到动图文件"), None)

        out_dir = self.store.frames_dir("anim")

        def budget(total_frames: int, size_bytes: int) -> int:
            return animation_frame_budget(
                total_frames, size_bytes, preset, self.settings.frames_override
            )

        try:
            sample = await sample_animation(
                source, out_dir, preset.max_side, preset.jpeg_quality, budget
            )
        except AnimationError as exc:
            TempStore.discard(out_dir)
            return (MediaResult(item=item, notice=str(exc)), None)

        result = MediaResult(
            item=item,
            frames=sample.frames,
            duration=sample.duration,
            source_frame_count=sample.total_frames,
        )
        entry = CacheEntry(
            frames=list(sample.frames),
            duration=sample.duration,
            source_frame_count=sample.total_frames,
            frames_dir=out_dir,
        )
        return (result, entry)

    async def _process_video(self, item: MediaItem) -> tuple[MediaResult, CacheEntry | None]:
        if item.path is None:
            return (MediaResult(item=item, notice="找不到视频文件"), None)
        if not self.runner.available:
            return (
                MediaResult(
                    item=item,
                    notice="服务器上没有可用的 ffmpeg，无法解析视频画面",
                ),
                None,
            )

        deadline = time.monotonic() + self.settings.advanced.max_seconds_per_video
        preset = self.settings.preset

        def remaining() -> float:
            return deadline - time.monotonic()

        try:
            probe = await self.runner.probe(item.path, timeout=min(30.0, max(5.0, remaining())))
        except FfmpegError as exc:
            return (MediaResult(item=item, notice=str(exc)), None)

        if not probe.has_video and not probe.has_audio:
            return (MediaResult(item=item, notice="这个文件里既没有画面也没有声音"), None)

        frames_dir: Path | None = self.store.frames_dir("video")
        frames: list[SampledFrame] = []
        notice = ""

        if probe.has_video and remaining() > 5:
            budget = video_frame_budget(probe.duration, preset, self.settings.frames_override)
            windows = plan_extraction(probe.duration, budget)
            try:
                frames = await self.runner.extract_frames(
                    item.path,
                    windows,
                    frames_dir,
                    preset.max_side,
                    preset.jpeg_quality,
                    timeout=max(10.0, remaining()),
                )
            except FfmpegError as exc:
                notice = str(exc)
        elif probe.has_video:
            notice = "处理时间已用尽，未能抽取画面"

        result = MediaResult(
            item=item,
            frames=frames,
            duration=probe.duration,
            notice=notice,
        )

        if not frames:
            TempStore.discard(frames_dir)
            frames_dir = None

        await self._attach_audio(item, probe, result, remaining)

        entry = CacheEntry(
            frames=list(result.frames),
            duration=result.duration,
            audio=result.audio,
            transcript=result.transcript,
            frames_dir=frames_dir,
        )
        return (result, entry if result.ok else None)

    # --- 音频 ---------------------------------------------------------------

    async def _attach_audio(
        self, item: MediaItem, probe: Any, result: MediaResult, remaining: Any
    ) -> None:
        audio_cfg = self.settings.audio
        if not audio_cfg.enabled or not probe.has_audio or item.path is None:
            return
        if remaining() < 5:
            return

        audio_path = self.store.audio_path()
        seconds = audio_clip_seconds(probe.duration)
        try:
            clip = await self.runner.extract_audio(
                item.path, audio_path, seconds, timeout=max(10.0, remaining())
            )
        except FfmpegError as exc:
            self.log.debug(f"[MotionVision] 音轨提取失败: {exc}")
            TempStore.discard(audio_path)
            return

        if audio_cfg.transcribe and remaining() > 3:
            try:
                result.transcript = await transcribe(
                    clip.path, audio_cfg, self.context, self.client
                )
            except SttError as exc:
                self.log.debug(f"[MotionVision] 语音转写失败: {exc}")

        if audio_cfg.attach:
            result.audio = clip
        else:
            # 只做转写的话，wav 已经没用了（转写成功与否都一样）。
            TempStore.discard(clip.path)

    # --- 全局预算 -----------------------------------------------------------

    def _apply_payload_budget(self, results: list[MediaResult]) -> None:
        """限制整轮请求送出去的图片张数与总字节数。

        超预算时从后往前砍：先保住第一个媒体的完整性，因为它通常是用户真正在问的。
        """
        max_images = self.settings.advanced.max_images_per_request
        max_bytes = self.settings.advanced.max_frame_payload_mb * MB

        used_images = 0
        used_bytes = 0
        for result in results:
            kept: list[SampledFrame] = []
            for frame in result.frames:
                size = frame.size_bytes or 0
                if used_images + 1 > max_images or used_bytes + size > max_bytes:
                    continue
                used_images += 1
                used_bytes += size
                kept.append(frame)
            if len(kept) != len(result.frames):
                dropped = len(result.frames) - len(kept)
                result.frames = kept
                extra = f"因图片预算限制，已丢弃 {dropped} 张帧"
                result.notice = f"{result.notice}；{extra}" if result.notice else extra
