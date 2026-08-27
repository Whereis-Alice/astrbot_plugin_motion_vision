"""动图识别与抽帧。"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from PIL import Image

from motion_vision.animation import (
    AnimationError,
    detect_family,
    maybe_animated,
    sample_animation,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _gif_bytes(frames: int = 10, size: tuple[int, int] = (64, 48)) -> bytes:
    images = [Image.new("RGB", size, (index * 20 % 256, 40, 200)) for index in range(frames)]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


def _png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", size, (0, 0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _webp_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format="WEBP")
    return buffer.getvalue()


def test_detect_family_reads_magic_bytes() -> None:
    assert detect_family(_gif_bytes(2)[:64]) == "gif"
    assert detect_family(_png_bytes()[:64]) == "png"
    assert detect_family(_webp_bytes()[:64]) == "webp"
    assert detect_family(b"") is None
    assert detect_family(b"just some text") is None


def test_maybe_animated_lets_every_gif_through() -> None:
    # GIF 无法只看文件头判断帧数，必须放行。
    assert maybe_animated(_gif_bytes(1)[:64]) is True


def test_maybe_animated_filters_static_images() -> None:
    assert maybe_animated(_png_bytes()[:1024]) is False
    assert maybe_animated(_webp_bytes()[:1024]) is False
    assert maybe_animated(b"whatever") is False
    assert maybe_animated(b"") is False


def test_maybe_animated_detects_apng_and_animated_webp() -> None:
    assert maybe_animated(PNG_MAGIC + b"\x00\x00\x00\x08acTL") is True
    assert maybe_animated(b"RIFF\x00\x00\x00\x00WEBPVP8XANIM") is True


def test_sample_animation_writes_downscaled_frames(tmp_path: Path) -> None:
    data = _gif_bytes(frames=12, size=(200, 120))
    out_dir = tmp_path / "frames"

    sample = asyncio.run(
        sample_animation(data, out_dir, max_side=64, quality=80, budget=lambda _t, _s: 5)
    )

    assert sample.total_frames == 12
    assert sample.image_format == "GIF"
    assert len(sample.frames) == 5
    assert sample.duration is not None and sample.duration > 0

    for position, frame in enumerate(sample.frames):
        assert frame.index == position
        assert frame.path.exists()
        assert frame.size_bytes > 0
        with Image.open(frame.path) as image:
            assert max(image.size) <= 64

    stamps = [frame.timestamp for frame in sample.frames]
    assert stamps == sorted(stamps)


def test_sample_animation_honours_budget(tmp_path: Path) -> None:
    data = _gif_bytes(frames=30)
    sample = asyncio.run(
        sample_animation(data, tmp_path / "f", 32, 70, lambda total, _s: min(total, 3))
    )
    assert len(sample.frames) == 3


def test_sample_animation_rejects_single_frame(tmp_path: Path) -> None:
    with pytest.raises(AnimationError):
        asyncio.run(sample_animation(_png_bytes((16, 16)), tmp_path, 64, 80, lambda *_: 3))


def test_sample_animation_rejects_garbage(tmp_path: Path) -> None:
    with pytest.raises(AnimationError):
        asyncio.run(sample_animation(b"not an image at all", tmp_path, 64, 80, lambda *_: 3))
