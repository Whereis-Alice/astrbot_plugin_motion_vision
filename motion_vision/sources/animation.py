"""收集本次请求里的动图。

同一张图可能同时出现在三个地方（LLM 请求的 image_urls、提示词里的附件标记、
原始消息链），所以按「请求 > 标记 > 消息链」的优先级合并并去重。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..animation import HEAD_BYTES, maybe_animated
from ..models import MediaItem, MediaKind
from .common import (
    DownloadTooLarge,
    decode_inline_image,
    download_bytes,
    is_http_url,
    parse_markers,
    peek_head,
    redact_url,
    resolve_local_path,
)

MAX_ANIMATION_BYTES = 32 * 1024 * 1024
"""动图下载上限。动图再大也不该超过这个量级。"""


@dataclass(frozen=True)
class _Candidate:
    raw: str
    image_url_index: int | None = None
    part_index: int | None = None
    quoted: bool = False
    name: str = ""
    marker_raw: str = ""


def collect_candidates(request: Any, event: Any) -> list[_Candidate]:
    """按优先级列出所有「可能是动图」的引用。"""
    candidates: list[_Candidate] = []

    for index, raw in enumerate(getattr(request, "image_urls", None) or []):
        if isinstance(raw, str) and raw:
            candidates.append(_Candidate(raw, image_url_index=index))

    for part_index, part in enumerate(getattr(request, "extra_user_content_parts", None) or []):
        text = getattr(part, "text", None)
        if not isinstance(text, str) or not text:
            continue
        for marker in parse_markers(text, part_index):
            if marker.kind != "image":
                continue
            candidates.append(
                _Candidate(
                    marker.path,
                    part_index=part_index,
                    quoted=marker.quoted,
                    name=marker.name,
                    marker_raw=marker.raw,
                )
            )

    for raw, quoted in _iter_chain_images(event):
        candidates.append(_Candidate(raw, quoted=quoted))

    return candidates


def _iter_chain_images(event: Any) -> list[tuple[str, bool]]:
    """消息链里的图片：直接发的排在引用的前面。"""
    from astrbot.api.message_components import Image as AstrImage
    from astrbot.api.message_components import Reply

    direct: list[tuple[str, bool]] = []
    quoted: list[tuple[str, bool]] = []
    try:
        chain = event.get_messages() or []
    except Exception:
        return []

    for component in chain:
        if isinstance(component, AstrImage):
            for raw in (component.path, component.url, component.file):
                if isinstance(raw, str) and raw:
                    direct.append((raw, False))
                    break
        elif isinstance(component, Reply):
            for inner in component.chain or []:
                if not isinstance(inner, AstrImage):
                    continue
                for raw in (inner.path, inner.url, inner.file):
                    if isinstance(raw, str) and raw:
                        quoted.append((raw, True))
                        break
    return direct + quoted


def _identity(raw: str, path: Path | None, data: bytes | None) -> str:
    if path is not None:
        return f"path:{str(path).casefold()}"
    if data is not None:
        import hashlib

        digest = hashlib.sha1(data[:65536]).hexdigest()
        return f"bytes:{len(data)}:{digest}"
    return f"raw:{raw}"


async def resolve_animations(
    request: Any,
    event: Any,
    client: httpx.AsyncClient,
    search_dirs: tuple[Path, ...] = (),
    limit: int = 8,
    on_skip: Any = None,
) -> list[MediaItem]:
    """把候选引用还原成确实是动图的 MediaItem 列表。"""
    items: list[MediaItem] = []
    seen: set[str] = set()

    for candidate in collect_candidates(request, event):
        if len(items) >= limit:
            break

        data = decode_inline_image(candidate.raw)
        path: Path | None = None

        if data is None:
            path = resolve_local_path(candidate.raw, search_dirs)

        if data is None and path is None and is_http_url(candidate.raw):
            head = await peek_head(client, candidate.raw, HEAD_BYTES)
            if not maybe_animated(head):
                continue
            try:
                data = await download_bytes(client, candidate.raw, MAX_ANIMATION_BYTES)
            except DownloadTooLarge:
                if on_skip:
                    on_skip(candidate.name or "动图", "体积超过下载上限")
                continue
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                if on_skip:
                    on_skip(
                        candidate.name or "动图",
                        f"下载失败：{exc.__class__.__name__}（{redact_url(candidate.raw)}）",
                    )
                continue

        if data is not None:
            if not maybe_animated(data[:HEAD_BYTES]):
                continue
        elif path is not None:
            if not maybe_animated(_read_head(path)):
                continue
        else:
            continue

        identity = _identity(candidate.raw, path, data)
        if identity in seen:
            continue
        seen.add(identity)

        items.append(
            MediaItem(
                kind=MediaKind.ANIMATION,
                name=candidate.name or (path.name if path else "动图"),
                identity=identity,
                path=path,
                data=data,
                image_url_index=candidate.image_url_index,
                part_index=candidate.part_index,
                marker_raw=candidate.marker_raw,
                quoted=candidate.quoted,
                source_url=candidate.raw if is_http_url(candidate.raw) else "",
            )
        )

    return items


def _read_head(path: Path, nbytes: int = HEAD_BYTES) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(nbytes)
    except OSError:
        return b""
