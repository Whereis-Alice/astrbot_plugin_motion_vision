"""处理流水线：MediaItem -> MediaResult。

职责边界很清楚：这里只负责「把媒体变成帧/音轨/转写文本」以及各种预算保护，
不碰 AstrBot 的请求对象（那是 inject.py 的事）。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from .animation import AnimationError, sample_animation
from .budget import frame_allowances
from .cache import CacheEntry, ResultCache, fingerprint
from .ffmpeg import FfmpegError, FfmpegRunner
from .models import MediaItem, MediaKind, MediaResult, SampledFrame, TimeSpan
from .sampling import (
    animation_frame_budget,
    audio_clip_seconds,
    plan_extraction,
    thin_indices,
    video_frame_budget,
)
from .settings import MB, Settings
from .stt import SttError, transcribe
from .tempstore import TempStore


def _span_length(duration: float | None, span: TimeSpan | None) -> float | None:
    """算出实际要采样的时长：整片时就是片长，只看一段时是那一段的长度。"""
    if span is None or not span.active:
        return duration
    end = span.end
    if duration is not None and duration > 0:
        end = duration if end is None else min(end, duration)
    if end is None:
        return None
    length = end - max(0.0, span.start)
    return length if length > 0 else duration


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
        native: Any = None,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.store = store
        self.cache = cache
        self.client = client
        self.context = context
        self.log = log
        self.native = native
        self._native_auto_used = 0

    # --- 入口 ---------------------------------------------------------------

    async def run(
        self,
        items: list[MediaItem],
        span: TimeSpan | None = None,
        *,
        question: str = "",
    ) -> list[MediaResult]:
        """处理一批媒体。span 只在「回看某一段」时给出，平时是整片。"""
        results: list[MediaResult] = []
        self._native_auto_used = 0
        for item in items:
            try:
                results.append(await self._process(item, span, question=question))
            except Exception as exc:  # 任何单个媒体的失败都不该影响整轮对话
                self.log.warning(f"[MotionVision] 处理 {item.display_name} 时出错: {exc}")
                results.append(MediaResult(item=item, notice="处理时发生未预期的错误"))
        self._apply_payload_budget(results)
        return results

    # --- 单个媒体 -----------------------------------------------------------

    async def _process(
        self,
        item: MediaItem,
        span: TimeSpan | None = None,
        *,
        question: str = "",
    ) -> MediaResult:
        window = span if span is not None and span.active else None
        signature = self.settings.cache_signature
        if question.strip() and self.settings.native_video.automatic:
            signature = f"{signature}:q-{question.strip()[:2000]}"
        if window is not None:
            signature = f"{signature}@{window.key}"

        key = fingerprint(item.path, item.data, signature)
        if item.path is None and item.data is None:
            # 只有文字资料的来源（例如 B 站字幕）没有文件指纹，不能全部落到
            # fingerprint() 的 unknown 键上，否则不同视频会互相串缓存。
            key = f"identity:{item.identity}|{signature}"
        cached = self.cache.get(key)
        if cached is not None:
            return MediaResult(
                item=item,
                frames=list(cached.frames),
                duration=cached.duration,
                source_frame_count=cached.source_frame_count,
                audio=cached.audio,
                transcript=cached.transcript,
                native_report=cached.native_report,
                notice=cached.notice,
            )

        if item.kind is MediaKind.ANIMATION:
            result, entry = await self._process_animation(item)
            if window is not None and result.frames:
                # 动图整体就那么几十帧，按时间段裁反而更容易漏掉关键动作。
                result.notice = "动图不支持只看某一段，这里是整段重新取样的结果"
        else:
            result, entry = await self._process_video(item, window, question=question)

        if entry is not None:
            self.cache.put(key, entry)
        return result

    async def _process_animation(self, item: MediaItem) -> tuple[MediaResult, CacheEntry | None]:
        preset = self.settings.preset
        source = item.data if item.data is not None else item.path
        if source is None:
            return (MediaResult(item=item, notice="找不到动图文件"), None)

        out_dir = self.store.frames_dir("anim")

        def budget(total_frames: int, size_bytes: int, duration: float | None) -> int:
            return animation_frame_budget(
                total_frames,
                size_bytes,
                preset,
                self.settings.animation_frames_override,
                duration,
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
            notice=item.source_notice,
        )
        entry = CacheEntry(
            frames=list(sample.frames),
            duration=sample.duration,
            source_frame_count=sample.total_frames,
            notice=result.notice,
            frames_dir=out_dir,
        )
        return (result, entry)

    async def _process_video(
        self,
        item: MediaItem,
        span: TimeSpan | None = None,
        *,
        question: str = "",
    ) -> tuple[MediaResult, CacheEntry | None]:
        if item.path is None:
            # B 站“只读字幕”模式本来就不会下载视频文件；有文字资料时不要
            # 再伪造一条“找不到视频文件”的失败提示，避免模型误以为字幕也无效。
            notice = item.source_notice
            if not notice and not item.context_text:
                notice = "找不到视频文件"
            return (MediaResult(item=item, notice=notice), None)

        # 原生视频模型是“整片理解”的可选后端。它先于 ffmpeg 探测执行，
        # 因而即使部署环境没有 ffmpeg，配置好的原生后端仍然可以工作。
        native_report = ""
        native_notice = ""
        native = getattr(self, "native", None)
        if (
            native is not None
            and self._native_auto_used < self.settings.native_video.max_auto_videos
            and native.should_attempt(item)
        ):
            self._native_auto_used += 1
            try:
                native_report = await native.analyze(item.path, item.display_name, question)
            except Exception as exc:
                native_notice = native.user_error(exc)
                self.log.debug(f"[MotionVision] 原生视频后端失败：{exc}")
                if native.only_mode:
                    return (
                        MediaResult(
                            item=item,
                            native_report="",
                            notice=_join_notice(item.source_notice, native_notice),
                        ),
                        None,
                    )
            if native_report and native.only_mode:
                result = MediaResult(
                    item=item,
                    native_report=native_report,
                    notice=item.source_notice,
                )
                return result, CacheEntry(
                    native_report=native_report,
                    notice=result.notice,
                )

        if not self.runner.available:
            return (
                MediaResult(
                    item=item,
                    native_report=native_report,
                    notice=_join_notice(
                        item.source_notice,
                        native_notice,
                        "服务器上没有可用的 ffmpeg，无法解析视频画面",
                    ),
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
            return (
                MediaResult(
                    item=item,
                    native_report=native_report,
                    notice=_join_notice(item.source_notice, native_notice, str(exc)),
                ),
                None,
            )

        if not probe.has_video and not probe.has_audio:
            return (
                MediaResult(
                    item=item,
                    native_report=native_report,
                    notice=_join_notice(
                        item.source_notice,
                        native_notice,
                        "这个文件里既没有画面也没有声音",
                    ),
                ),
                None,
            )

        frames_dir: Path | None = self.store.frames_dir("video")
        frames: list[SampledFrame] = []
        notice = item.source_notice

        if probe.has_video and remaining() > 5:
            # 只看一段时，帧数按这一段的长度算——否则 10 秒的片段会拿到整片的预算。
            scope = _span_length(probe.duration, span)
            budget = video_frame_budget(scope, preset, self.settings.video_frames_override)
            windows = plan_extraction(
                probe.duration,
                budget,
                start=span.start if span is not None else 0.0,
                end=span.end if span is not None else None,
            )
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
                notice = _join_notice(notice, str(exc))
        elif probe.has_video:
            notice = "处理时间已用尽，未能抽取画面"

        result = MediaResult(
            item=item,
            frames=frames,
            duration=probe.duration,
            native_report=native_report,
            notice=notice,
        )
        if native_notice:
            result.notice = _join_notice(result.notice, native_notice)

        if not frames:
            TempStore.discard(frames_dir)
            frames_dir = None

        await self._attach_audio(item, probe, result, remaining)

        entry = CacheEntry(
            frames=list(result.frames),
            duration=result.duration,
            audio=result.audio,
            transcript=result.transcript,
            native_report=result.native_report,
            notice=result.notice,
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
            result.notice = _join_notice(result.notice, "未能读取视频音轨，无法确认其中的声音内容")
            return

        if audio_cfg.transcribe and remaining() > 3:
            try:
                result.transcript = await asyncio.wait_for(
                    transcribe(clip.path, audio_cfg, self.context, self.client),
                    timeout=max(1.0, remaining()),
                )
            except asyncio.TimeoutError:
                self.log.debug("[MotionVision] 语音转写超出本视频的处理时间预算")
                result.notice = _join_notice(
                    result.notice, "语音转写超时，无法确认视频中的说话内容"
                )
            except SttError as exc:
                self.log.debug(f"[MotionVision] 语音转写失败: {exc}")
                result.notice = _join_notice(
                    result.notice, "语音转写失败，无法确认视频中的说话内容"
                )

        if audio_cfg.attach:
            result.audio = clip
        else:
            # 只做转写的话，wav 已经没用了（转写成功与否都一样）。
            TempStore.discard(clip.path)

    # --- 全局预算 -----------------------------------------------------------

    def _apply_payload_budget(self, results: list[MediaResult]) -> None:
        """限制整轮请求送出去的图片张数与总字节数。

        * **视频优先**：满足视频目标后，剩余额度分给动图；同类均分。
          预算允许时每张动图预留两帧，避免完全丢失表情信息。
        * **均匀**：需要砍帧时沿时间轴抽稀，而不是把片尾整段丢掉——
          丢掉结尾等于让模型只看了个开头。
        """
        carriers = [result for result in results if result.frames]
        if not carriers:
            return

        original = [len(result.frames) for result in carriers]

        allowance = frame_allowances(
            [[frame.size_bytes or 0 for frame in result.frames] for result in carriers],
            [result.kind for result in carriers],
            self.settings.advanced.max_images_per_request,
            self.settings.advanced.max_frame_payload_mb * MB,
        )
        # 两种预算一并算完后只抽稀一次，保留原始时间轴上的均匀取样点。
        for result, keep in zip(carriers, allowance, strict=True):
            self._thin(result, keep)

        # 如实告知被砍了多少。
        for result, before in zip(carriers, original, strict=True):
            dropped = before - len(result.frames)
            if dropped <= 0:
                continue
            result.budget_dropped_frames += dropped
            extra = (
                f"因本轮图片预算限制，{before} 帧里只保留了 "
                f"{len(result.frames)} 帧（沿时间轴均匀抽稀）"
            )
            result.notice = f"{result.notice}；{extra}" if result.notice else extra

    @staticmethod
    def _thin(result: MediaResult, keep: int) -> None:
        """把某个媒体的帧数降到 keep，索引重新编号以便标签连续。"""
        frames = result.frames
        keep = max(0, min(keep, len(frames)))
        if keep == len(frames):
            return
        picked = [frames[i] for i in thin_indices(len(frames), keep)]
        result.frames = [
            SampledFrame(f.path, position, f.timestamp, f.size_bytes)
            for position, f in enumerate(picked)
        ]


def _join_notice(*parts: str) -> str:
    """合并来源阶段和媒体阶段的降级说明，避免重复分号。"""

    return "；".join(part.strip("； ") for part in parts if part and part.strip("； "))
