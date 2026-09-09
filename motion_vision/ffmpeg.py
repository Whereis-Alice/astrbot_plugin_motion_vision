"""ffmpeg / ffprobe 封装。

设计要点：

* **自动探测**：配置里手填的路径 -> PATH -> imageio-ffmpeg 自带的二进制。
  三者都没有时，视频功能整体优雅降级，只给模型留一句中文说明。
* **一次调用抽多帧**：每个采样窗口只起一个 ffmpeg 进程（用 fps 滤镜），
  而不是「一帧一个进程」，长视频能省掉大量进程启动开销。
* **不阻塞事件循环**：所有子进程都跑在 asyncio.to_thread 里，并用信号量
  把并发压在 2 以内，避免一次多个视频把 CPU 打满。
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import AudioClip, SampledFrame
from .sampling import ExtractionWindow

MAX_CONCURRENT_JOBS = 2
DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

FFMPEG_INSTALL_HINT = """视频功能需要 ffmpeg，当前没有找到。三种装法任选其一：

1) 最省事：AstrBot WebUI -> 控制台 -> 安装 pip 库，填「imageio-ffmpeg」。
   它自带一份 ffmpeg 二进制（约 80MB），装完重启 AstrBot 就能用。
2) 装到系统里：Windows「winget install ffmpeg」、
   Debian/Ubuntu「sudo apt install ffmpeg」、macOS「brew install ffmpeg」，装完重启 AstrBot。
3) 已经装在别处：把 ffmpeg 所在目录（或可执行文件本身）填进插件配置的「高级 - ffmpeg 路径」。

动图（GIF / 动态 WebP / APNG）不依赖 ffmpeg，不受影响。"""
"""ffmpeg 缺失时给出的安装引导。日志和 /motionvision status 共用同一份文案。"""


class FfmpegError(RuntimeError):
    """ffmpeg 相关失败，message 直接可以给用户/模型看。"""


@dataclass(frozen=True)
class FfmpegTools:
    ffmpeg: str | None = None
    ffprobe: str | None = None
    source: str = "未找到"

    @property
    def available(self) -> bool:
        return bool(self.ffmpeg)


@dataclass(frozen=True)
class ProbeResult:
    duration: float | None = None
    has_video: bool = True
    has_audio: bool = False
    width: int = 0
    height: int = 0


def _resolve_explicit(raw: str) -> FfmpegTools | None:
    """支持填 ffmpeg 可执行文件本身，也支持填它所在的目录。"""
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        ffmpeg = _first_existing(candidate, "ffmpeg")
        ffprobe = _first_existing(candidate, "ffprobe")
        if ffmpeg:
            return FfmpegTools(ffmpeg, ffprobe, f"配置目录 {candidate}")
        return None
    if candidate.is_file():
        ffprobe = _first_existing(candidate.parent, "ffprobe")
        return FfmpegTools(str(candidate), ffprobe, f"配置路径 {candidate}")
    return None


def _first_existing(directory: Path, stem: str) -> str | None:
    for name in (f"{stem}.exe", stem):
        target = directory / name
        if target.is_file():
            return str(target)
    return None


def _from_imageio() -> FfmpegTools | None:
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    if not exe or not Path(exe).exists():
        return None
    return FfmpegTools(exe, None, "imageio-ffmpeg 内置")


def discover_tools(explicit_path: str = "") -> FfmpegTools:
    """按优先级找 ffmpeg / ffprobe。"""
    found = _resolve_explicit(explicit_path)
    if found:
        return found

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return FfmpegTools(ffmpeg, shutil.which("ffprobe"), "系统 PATH")

    fallback = _from_imageio()
    if fallback:
        return fallback

    return FfmpegTools()


def _seconds_from_match(match: re.Match[str]) -> float:
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def jpeg_quality_to_qscale(quality: int) -> int:
    """把 0-100 的 JPEG 质量换算成 ffmpeg 的 -q:v（2 最好，31 最差）。"""
    return max(2, min(31, round((100 - quality) / 3) + 2))


def _scale_filter(max_side: int) -> str:
    """等比缩放到最长边不超过 max_side，且绝不放大。"""
    return (
        f"scale=w='min(iw,{max_side})':h='min(ih,{max_side})'"
        ":force_original_aspect_ratio=decrease:flags=lanczos"
    )


class FfmpegRunner:
    """对 ffmpeg / ffprobe 的一层薄封装。"""

    def __init__(self, tools: FfmpegTools, max_concurrency: int = MAX_CONCURRENT_JOBS) -> None:
        self.tools = tools
        self._max_concurrency = max(1, max_concurrency)
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def available(self) -> bool:
        return self.tools.available

    def _gate(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    async def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        async with self._gate():
            process: asyncio.subprocess.Process | None = None
            try:
                spawn_options: dict[str, object] = {
                    "stdout": asyncio.subprocess.PIPE,
                    "stderr": asyncio.subprocess.PIPE,
                }
                if _NO_WINDOW:
                    spawn_options["creationflags"] = _NO_WINDOW
                process = await asyncio.create_subprocess_exec(*args, **spawn_options)
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=max(0.1, timeout)
                )
            except asyncio.TimeoutError as exc:
                await _stop_process(process)
                raise FfmpegError("处理超时，可能是文件过大或机器负载过高") from exc
            except asyncio.CancelledError:
                await _stop_process(process)
                raise
            except FileNotFoundError as exc:
                raise FfmpegError("找不到 ffmpeg 可执行文件") from exc
            except OSError as exc:
                raise FfmpegError(f"调用 ffmpeg 失败：{exc}") from exc

            return subprocess.CompletedProcess(
                args=args,
                returncode=process.returncode if process is not None else -1,
                stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
            )

    # --- 探测 ---------------------------------------------------------------

    async def probe(self, path: Path, timeout: float = 30.0) -> ProbeResult:
        """读取时长与轨道信息。ffprobe 不可用时退化为解析 ffmpeg 的日志。"""
        if self.tools.ffprobe:
            result = await self._run(
                [
                    self.tools.ffprobe,
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-show_format",
                    str(path),
                ],
                timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                parsed = _parse_ffprobe(result.stdout)
                if parsed is not None:
                    return parsed
        return await self._probe_via_ffmpeg(path, timeout)

    async def _probe_via_ffmpeg(self, path: Path, timeout: float) -> ProbeResult:
        if not self.tools.ffmpeg:
            raise FfmpegError("ffmpeg 不可用，无法读取视频信息")
        result = await self._run([self.tools.ffmpeg, "-hide_banner", "-i", str(path)], timeout)
        log = f"{result.stderr}\n{result.stdout}"
        match = DURATION_PATTERN.search(log)
        duration = _seconds_from_match(match) if match else None
        has_video = "Video:" in log
        if duration is None and not has_video:
            raise FfmpegError("无法解析这个文件，可能不是有效视频")
        return ProbeResult(duration=duration, has_video=has_video, has_audio="Audio:" in log)

    # --- 抽帧 ---------------------------------------------------------------

    async def extract_frames(
        self,
        path: Path,
        windows: list[ExtractionWindow],
        out_dir: Path,
        max_side: int,
        quality: int,
        timeout: float = 120.0,
    ) -> list[SampledFrame]:
        """按窗口列表抽帧，返回按时间排序的帧。"""
        if not self.tools.ffmpeg:
            raise FfmpegError("ffmpeg 不可用，无法从视频抽帧")

        out_dir.mkdir(parents=True, exist_ok=True)
        qscale = str(jpeg_quality_to_qscale(quality))
        scale = _scale_filter(max_side)
        frames: list[SampledFrame] = []
        failures: list[str] = []
        deadline = asyncio.get_running_loop().time() + max(0.1, timeout)

        for slot, window in enumerate(windows):
            if window.count <= 0:
                continue
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                failures.append("处理超时，可能是文件过大或机器负载过高")
                break
            pattern = out_dir / f"w{slot:02d}_%03d.jpg"
            args = [self.tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
            if window.start > 0:
                # -ss 放在 -i 之前 = 关键帧快速定位，长视频靠它省时间。
                args += ["-ss", f"{window.start:.3f}"]
            args += ["-i", str(path)]
            if window.length is not None and window.length > 0:
                args += ["-t", f"{window.length:.3f}"]

            filters = scale
            if window.count > 1 and window.step > 0:
                # step 由窗口自己算：时长已知时是 length/count，未知时是兜底间隔。
                fps = 1.0 / window.step
                filters = f"fps={fps:.6f},{scale}"
            args += [
                "-vf",
                filters,
                "-frames:v",
                str(window.count),
                "-q:v",
                qscale,
                "-pix_fmt",
                "yuvj420p",
                str(pattern),
            ]

            result = None
            try:
                result = await self._run(args, remaining)
            except FfmpegError as exc:
                # 长视频由多个窗口组成。某个窗口超时或损坏时，保留前面已经
                # 成功抽出的帧；只有一帧都没有时才把整个媒体判定为失败。
                failures.append(str(exc))
                if "超时" in str(exc) or remaining <= 0.2:
                    break
            produced = sorted(out_dir.glob(f"w{slot:02d}_*.jpg"))
            if result is not None and result.returncode != 0 and not produced:
                failures.append(_last_line(result.stderr))
                continue
            if not produced:
                continue

            stamps = window.timestamps()
            for offset, frame_path in enumerate(produced):
                timestamp = stamps[offset] if offset < len(stamps) else None
                frames.append(
                    SampledFrame(
                        path=frame_path,
                        index=len(frames),
                        timestamp=timestamp,
                        size_bytes=_size_of(frame_path),
                    )
                )

        if not frames:
            detail = failures[0] if failures else "没有可用的视频帧"
            raise FfmpegError(f"抽帧失败：{detail}")

        frames.sort(key=lambda item: (item.timestamp is None, item.timestamp or 0.0))
        return [
            SampledFrame(f.path, position, f.timestamp, f.size_bytes)
            for position, f in enumerate(frames)
        ]

    # --- 抽音轨 -------------------------------------------------------------

    async def extract_audio(
        self,
        path: Path,
        out_path: Path,
        seconds: float,
        timeout: float = 120.0,
    ) -> AudioClip:
        """导出 16kHz 单声道 WAV —— 主流 STT 服务最通吃的格式。"""
        if not self.tools.ffmpeg:
            raise FfmpegError("ffmpeg 不可用，无法提取音轨")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            self.tools.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
        ]
        if seconds > 0:
            args += ["-t", f"{seconds:.3f}"]
        args += ["-f", "wav", str(out_path)]

        result = await self._run(args, timeout)
        if result.returncode != 0 or not out_path.exists() or _size_of(out_path) < 1024:
            raise FfmpegError("音轨提取失败，这个视频可能没有声音")
        return AudioClip(path=out_path, seconds=seconds)

    async def compress_video(
        self,
        path: Path,
        out_path: Path,
        *,
        max_seconds: int = 120,
        height: int = 720,
        crf: int = 28,
        timeout: float = 300.0,
    ) -> Path:
        """把视频压到适合原生视频接口上传的临时 MP4。

        这是可选的降级路径，不参与普通抽帧。单独放在 Runner 里可以复用
        同一套并发闸门、超时和子进程清理逻辑，避免压缩任务偷偷绕过资源限制。
        """
        if not self.tools.ffmpeg:
            raise FfmpegError("ffmpeg 不可用，无法压缩视频")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        height = max(144, int(height))
        crf = max(18, min(40, int(crf)))
        args = [
            self.tools.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
        ]
        if max_seconds > 0:
            args += ["-t", f"{max_seconds:d}"]
        args += [
            "-vf",
            f"scale=w='min(iw,{height})':h='min(ih,{height})':force_original_aspect_ratio=decrease",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = await self._run(args, timeout)
        if result.returncode != 0 or not out_path.exists() or _size_of(out_path) <= 0:
            out_path.unlink(missing_ok=True)
            raise FfmpegError(f"视频压缩失败：{_last_line(result.stderr)}")
        return out_path


def _parse_ffprobe(payload: str) -> ProbeResult | None:
    import json

    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None

    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    duration: float | None = None
    for source in (fmt.get("duration"), *(s.get("duration") for s in streams)):
        try:
            value = float(source)
        except (TypeError, ValueError):
            continue
        if value > 0:
            duration = value
            break

    width = height = 0
    has_video = has_audio = False
    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            has_video = True
            width = width or int(stream.get("width") or 0)
            height = height or int(stream.get("height") or 0)
        elif codec_type == "audio":
            has_audio = True

    return ProbeResult(
        duration=duration,
        has_video=has_video,
        has_audio=has_audio,
        width=width,
        height=height,
    )


def _last_line(text: str | None) -> str:
    """取 ffmpeg 报错的最后一行——通常那句才是真正的原因。"""
    lines = (text or "").strip().splitlines()
    return lines[-1].strip() if lines else "未知错误"


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
    """在超时或取消时终止 ffmpeg，并把管道里的剩余输出消费掉。"""
    if process is None or process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, OSError):
        process.kill()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await process.communicate()
