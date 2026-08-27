"""动图抽帧（GIF / 动态 WebP / APNG），基于 Pillow。

QQ 上的表情包早就不只有 GIF 了，动态 WebP 和 APNG 同样常见，所以这里按
「文件头识别族 -> Pillow 确认帧数」的顺序判断，而不是只看扩展名。
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageSequence, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from .models import SampledFrame
from .sampling import sample_indices

HEAD_BYTES = 1024
"""判断「是不是动图」需要读取的文件头长度。"""

MAX_PIXELS_PER_FRAME = 50_000_000
"""单帧像素上限，防解压炸弹。"""

GIF_MAGICS = (b"GIF87a", b"GIF89a")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class AnimationError(RuntimeError):
    """动图解析失败，message 可直接展示。"""


@dataclass(frozen=True)
class AnimationSample:
    frames: list[SampledFrame]
    total_frames: int
    duration: float | None
    image_format: str


def detect_family(head: bytes) -> str | None:
    """只看文件头判断图片族，返回 gif / webp / png / None。"""
    if not head:
        return None
    if head.startswith(GIF_MAGICS):
        return "gif"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(PNG_MAGIC):
        return "png"
    return None


def maybe_animated(head: bytes) -> bool:
    """在只有文件头时给出「值得进一步处理吗」的判断。

    GIF 无法只靠文件头确定是否多帧，一律放行交给 Pillow；
    WebP / PNG 则分别依赖 ANIM / acTL 标记，能在下载前就筛掉静态图。
    """
    family = detect_family(head)
    if family == "gif":
        return True
    if family == "webp":
        return b"ANIM" in head[:HEAD_BYTES] or b"ANMF" in head[:HEAD_BYTES]
    if family == "png":
        return b"acTL" in head[:HEAD_BYTES]
    return False


def _open(source: Path | bytes) -> Image.Image:
    try:
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source))
        return Image.open(source)
    except DecompressionBombError as exc:
        raise AnimationError("图片尺寸异常巨大，已跳过") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AnimationError(f"无法解析这张图片：{exc}") from exc


def _flatten(frame: Image.Image) -> Image.Image:
    """把带透明通道的帧压到白底，避免透明区域变成一片黑。"""
    if frame.mode in ("RGBA", "LA", "P", "PA"):
        converted = frame.convert("RGBA")
        canvas = Image.new("RGB", converted.size, (255, 255, 255))
        canvas.paste(converted, mask=converted.split()[-1])
        return canvas
    return frame.convert("RGB")


def _sample_sync(
    source: Path | bytes,
    out_dir: Path,
    max_side: int,
    quality: int,
    budget: Callable[[int, int], int],
) -> AnimationSample:
    size_bytes = len(source) if isinstance(source, bytes) else _size_of(source)

    with _open(source) as image:
        image_format = (image.format or "").upper() or "UNKNOWN"
        total_frames = int(getattr(image, "n_frames", 1) or 1)
        if total_frames <= 1:
            raise AnimationError("这张图只有一帧，不需要抽帧")
        if image.size[0] * image.size[1] > MAX_PIXELS_PER_FRAME:
            raise AnimationError("图片分辨率过高，已跳过")

        target = budget(total_frames, size_bytes)
        wanted = set(sample_indices(total_frames, target))
        if not wanted:
            raise AnimationError("采样帧数为 0，已跳过")

        out_dir.mkdir(parents=True, exist_ok=True)
        frames: list[SampledFrame] = []
        elapsed_ms = 0.0
        stamps: list[float] = []

        # 顺序遍历而不是随机 seek：GIF 的帧是差分存储的，顺序读才能正确合成。
        for position, raw_frame in enumerate(ImageSequence.Iterator(image)):
            if position in wanted:
                stamps.append(elapsed_ms / 1000.0)
                flat = _flatten(raw_frame)
                flat.thumbnail((max_side, max_side), Image.LANCZOS)
                target_path = out_dir / f"f{len(frames):03d}.jpg"
                flat.save(target_path, format="JPEG", quality=quality, optimize=True)
                frames.append(
                    SampledFrame(
                        path=target_path,
                        index=len(frames),
                        timestamp=stamps[-1],
                        size_bytes=_size_of(target_path),
                    )
                )
            with contextlib.suppress(TypeError, ValueError):
                elapsed_ms += float(raw_frame.info.get("duration") or 0.0)

    if not frames:
        raise AnimationError("没能抽出任何帧")

    duration = elapsed_ms / 1000.0 if elapsed_ms > 0 else None
    return AnimationSample(
        frames=frames,
        total_frames=total_frames,
        duration=duration,
        image_format=image_format,
    )


async def sample_animation(
    source: Path | bytes,
    out_dir: Path,
    max_side: int,
    quality: int,
    budget: Callable[[int, int], int],
) -> AnimationSample:
    """抽帧并落盘成 JPEG。Pillow 是同步的，整段丢进线程池。"""
    return await asyncio.to_thread(_sample_sync, source, out_dir, max_side, quality, budget)


def _size_of(path: Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0
