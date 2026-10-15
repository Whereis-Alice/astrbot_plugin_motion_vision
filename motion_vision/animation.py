"""动图抽帧（GIF / 动态 WebP / APNG），基于 Pillow。

QQ 上的表情包早就不只有 GIF 了，动态 WebP 和 APNG 同样常见，所以这里按
「文件头识别族 -> Pillow 确认帧数」的顺序判断，而不是只看扩展名。
"""

from __future__ import annotations

import asyncio
import io
import math
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from .models import SampledFrame
from .resources import protected_thread
from .sampling import sample_indices

HEAD_BYTES = 1024
"""判断「是不是动图」需要读取的文件头长度。"""

MAX_PIXELS_PER_FRAME = 50_000_000
"""单帧像素上限，防解压炸弹。"""

GIF_MAGICS = (b"GIF87a", b"GIF89a")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

DEFAULT_FRAME_DELAY_MS = 100.0
"""帧延时缺失或写得离谱时的取值——和浏览器的处理一致。"""

MIN_CREDIBLE_DELAY_MS = 20.0
"""小于这个值的帧延时视为「没写」：不少 GIF 会填 0 或 10ms。"""

MAX_SOURCE_FRAMES = 2000
MAX_DECODE_PIXELS = 400_000_000  # 包括时间线与输出两次遍历
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_DECODE_SECONDS = 30.0
_decode_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


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


def guess_suffix(head: bytes) -> str:
    """按文件头猜一个扩展名，供「把内联字节落盘」时使用。"""
    family = detect_family(head)
    return f".{family}" if family else ".img"


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
    budget: Callable[[int, int, float | None], int],
    cancelled: threading.Event | None = None,
) -> AnimationSample:
    size_bytes = len(source) if isinstance(source, bytes) else _size_of(source)
    if size_bytes > MAX_SOURCE_BYTES:
        raise AnimationError("动图体积超过 32 MB，已跳过")
    deadline = time.monotonic() + MAX_DECODE_SECONDS

    def check() -> None:
        if cancelled is not None and cancelled.is_set():
            raise AnimationError("动图处理已取消")
        if time.monotonic() > deadline:
            raise AnimationError("动图解码超时，已跳过")

    # 不读取 GIF.n_frames：它本身会扫描整个输入，必须让每次 seek 都受保护。
    with _open(source) as image:
        image_format = (image.format or "").upper() or "UNKNOWN"
        pixels = image.width * image.height
        if pixels > MAX_PIXELS_PER_FRAME:
            raise AnimationError("图片分辨率过高，已跳过")
        first = 1 if image.info.get("default_image", False) else 0
        stamps: list[float] = []
        elapsed_ms = 0.0
        for position in range(MAX_SOURCE_FRAMES + 1):
            check()
            try:
                image.seek(first + position)
            except EOFError:
                break
            if position == MAX_SOURCE_FRAMES:
                raise AnimationError("动图源帧数超过安全上限，已跳过")
            if (position + 1) * pixels * 2 > MAX_DECODE_PIXELS:
                raise AnimationError("动图累计解码像素超过安全上限，已跳过")
            image.load()  # WebP 直到 load 后才更新当前帧 duration。
            stamps.append(elapsed_ms / 1000.0)
            elapsed_ms += _frame_delay_ms(image)
        total_frames = len(stamps)
        if total_frames <= 1:
            raise AnimationError("这张图只有一帧，不需要抽帧")
        duration = elapsed_ms / 1000.0
        target = budget(total_frames, size_bytes, duration)
        wanted = set(sample_indices(total_frames, target))
        if not wanted:
            raise AnimationError("采样帧数为 0，已跳过")

        out_dir.mkdir(parents=True, exist_ok=True)
        frames: list[SampledFrame] = []
        # 顺序遍历而不是随机 seek：GIF 的帧是差分存储的，顺序读才能正确合成。
        for position in range(total_frames):
            check()
            image.seek(first + position)
            if position in wanted:
                flat = _flatten(image)
                flat.thumbnail((max_side, max_side), Image.LANCZOS)
                target_path = out_dir / f"f{len(frames):03d}.jpg"
                temporary = target_path.with_suffix(".tmp")
                flat.save(temporary, format="JPEG", quality=quality, optimize=True)
                temporary.replace(target_path)
                frames.append(
                    SampledFrame(
                        path=target_path,
                        index=len(frames),
                        timestamp=stamps[position],
                        size_bytes=_size_of(target_path),
                    )
                )

    if not frames:
        raise AnimationError("没能抽出任何帧")

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
    budget: Callable[[int, int, float | None], int],
) -> AnimationSample:
    """有界解码；取消后后台线程仍持有文件与并发名额，直到真正结束。"""
    loop = asyncio.get_running_loop()
    gate = _decode_gates.setdefault(loop, asyncio.Semaphore(2))
    await gate.acquire()
    cancelled = threading.Event()
    try:
        return await protected_thread(
            _sample_sync,
            source,
            out_dir,
            max_side,
            quality,
            budget,
            cancelled,
            paths=[out_dir, *([source] if isinstance(source, Path) else [])],
            on_finished=gate.release,
        )
    except asyncio.CancelledError:
        cancelled.set()
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise AnimationError("动图损坏或帧数据不完整，已跳过") from exc


def _frame_delay_ms(frame: Image.Image) -> float:
    """单帧显示时长。写 0 或写得离谱的一律按 100ms 算。

    不做这个兜底的话，一整张 GIF 的时间戳会全是 0.0s，模型看到的每个标签都是
    「@0.0s」，等于没有时间信息。
    """
    try:
        delay = float(frame.info.get("duration") or 0.0)
    except (TypeError, ValueError):
        delay = 0.0
    minimum = MIN_CREDIBLE_DELAY_MS if image_format_of(frame) == "GIF" else 0.001
    return delay if math.isfinite(delay) and delay >= minimum else DEFAULT_FRAME_DELAY_MS


def image_format_of(frame: Image.Image) -> str:
    return (frame.format or "").upper()


def _size_of(path: Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0
