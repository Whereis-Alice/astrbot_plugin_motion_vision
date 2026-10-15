"""可选的四帧分页宫格：画面保持比例，时间戳写在画面外，不补重复帧。"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from .models import FrameSheet, SampledFrame
from .resources import protected_thread

FRAMES_PER_PAGE = 4
LABEL_HEIGHT = 24
GAP = 4


def _render(
    frames: list[SampledFrame], directory: Path, max_side: int, quality: int
) -> list[FrameSheet]:
    directory.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_index, begin in enumerate(range(0, len(frames), FRAMES_PER_PAGE)):
        group = frames[begin : begin + FRAMES_PER_PAGE]
        images = []
        try:
            for frame in group:
                with Image.open(frame.path) as source:
                    picture = source.convert("RGB")
                picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                images.append(picture)
            columns = min(2, len(group))
            rows = math.ceil(len(group) / columns)
            width = max(picture.width for picture in images)
            height = max(picture.height for picture in images)
            canvas = Image.new(
                "RGB",
                (columns * (width + GAP) - GAP, rows * (height + LABEL_HEIGHT + GAP) - GAP),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for i, (frame, picture) in enumerate(zip(group, images, strict=True)):
                x = (i % columns) * (width + GAP)
                y = (i // columns) * (height + LABEL_HEIGHT + GAP)
                canvas.paste(picture, (x + (width - picture.width) // 2, y))
                stamp = "?" if frame.timestamp is None else f"{frame.timestamp:.2f}s"
                draw.text((x + 4, y + height + 4), f"#{begin + i + 1} @{stamp}", fill="black")
            target = directory / f"page{page_index:03d}.jpg"
            temporary = target.with_suffix(".tmp")
            canvas.save(temporary, format="JPEG", quality=quality, optimize=True)
            temporary.replace(target)
            pages.append(FrameSheet(target, tuple(group), target.stat().st_size))
        finally:
            for picture in images:
                picture.close()
    return pages


async def render_sheets(
    frames: list[SampledFrame],
    directory: Path,
    max_side: int,
    quality: int,
) -> list[FrameSheet]:
    return await protected_thread(
        _render,
        frames,
        directory,
        max_side,
        quality,
        paths=[directory, *(frame.path for frame in frames)],
    )
