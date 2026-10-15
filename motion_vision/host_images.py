"""只读适配宿主图片预处理结果，不修改框架函数或全局 Provider。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_LABEL = re.compile(
    r"^\[Image (?P<number>\d+)(?P<quoted> in quoted message)?: original path "
    r"(?P<original>.+); animation converted to a 3x3 frame montage\]$"
)


def part_image_ref(part: Any) -> str:
    if isinstance(part, dict):
        image = part.get("image_url")
        return str(image.get("url", "")) if isinstance(image, dict) else ""
    return str(getattr(getattr(part, "image_url", None), "url", "") or "")


@dataclass(frozen=True)
class HostAnimation:
    original: str
    prepared: str
    marker: str
    quoted: bool


def host_animations(request: Any) -> list[HostAnimation]:
    """v4.29 的编号顺序：extra 图片在前，image_urls 在后；不凭文件名猜对应。"""
    parts = getattr(request, "extra_user_content_parts", None) or []
    visuals = [ref for part in parts if (ref := part_image_ref(part))]
    visuals.extend(getattr(request, "image_urls", None) or [])
    found = []
    for part in parts:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if not isinstance(text, str) or not (match := _LABEL.fullmatch(text)):
            continue
        index = int(match["number"]) - 1
        if 0 <= index < len(visuals):
            found.append(
                HostAnimation(match["original"], visuals[index], text, bool(match["quoted"]))
            )
    return found
