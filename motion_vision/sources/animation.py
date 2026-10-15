"""收集本次请求里的动图。

同一张图可能同时出现在三个地方（LLM 请求的 image_urls、提示词里的附件标记、
原始消息链），所以按「请求 > 标记 > 消息链」的优先级合并并去重。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..animation import HEAD_BYTES, maybe_animated
from ..cache import fingerprint
from ..host_images import host_animations, part_image_ref
from ..models import MediaItem, MediaKind
from ..resources import protected_thread
from ..tempstore import TempStore
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
from .video import VideoCollector

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
    prepared_ref: str = ""
    component: Any = None
    owned_temp: bool = False
    source_url: str = ""


def collect_candidates(request: Any, event: Any, *, include_files: bool = True) -> list[_Candidate]:
    """按优先级列出所有「可能是动图」的引用。"""
    candidates: list[_Candidate] = []

    for animation in host_animations(request):
        candidates.append(
            _Candidate(
                animation.original,
                quoted=animation.quoted,
                marker_raw=animation.marker,
                prepared_ref=animation.prepared,
            )
        )

    for index, raw in enumerate(getattr(request, "image_urls", None) or []):
        if isinstance(raw, str) and raw:
            candidates.append(_Candidate(raw, image_url_index=index))

    for part in getattr(request, "extra_user_content_parts", None) or []:
        if ref := part_image_ref(part):
            candidates.append(_Candidate(ref, prepared_ref=ref))

    for part_index, part in enumerate(getattr(request, "extra_user_content_parts", None) or []):
        text = getattr(part, "text", None)
        if not isinstance(text, str) or not text:
            continue
        for marker in parse_markers(text, part_index):
            if marker.kind != "image" and not (
                include_files
                and marker.kind == "file"
                and _animation_name(marker.name or marker.path)
            ):
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

    candidates.extend(_iter_chain_images(event, include_files=include_files))

    return candidates


def _iter_chain_images(event: Any, *, include_files: bool = True) -> list[_Candidate]:
    """消息链里的图片：直接发的排在引用的前面。"""
    from astrbot.api.message_components import File as AstrFile
    from astrbot.api.message_components import Image as AstrImage
    from astrbot.api.message_components import Reply

    direct: list[_Candidate] = []
    quoted: list[_Candidate] = []
    try:
        chain = event.get_messages() or []
    except Exception:
        return []

    def add(component: Any, bucket: list[_Candidate], is_quoted: bool) -> None:
        if isinstance(component, AstrImage):
            for raw in (getattr(component, key, "") for key in ("path", "url", "file")):
                if isinstance(raw, str) and raw:
                    bucket.append(_Candidate(raw, quoted=is_quoted))
                    break
        elif include_files and isinstance(component, AstrFile):
            name = str(getattr(component, "name", "") or "")
            raw = next(
                (
                    str(value)
                    for key in ("file_", "file", "path", "url")
                    if isinstance(value := getattr(component, key, None), str) and value
                ),
                "",
            )
            if _animation_name(name) or _animation_name(raw):
                bucket.append(_Candidate(raw, name=name, quoted=is_quoted, component=component))

    for component in chain:
        if isinstance(component, Reply):
            for inner in getattr(component, "chain", None) or []:
                add(inner, quoted, True)
        else:
            add(component, direct, False)
    return direct + quoted


def _animation_name(name: str) -> bool:
    return Path(name.split("?", 1)[0]).suffix.casefold() in {".gif", ".webp", ".apng", ".png"}


def _identity(raw: str, path: Path | None, data: bytes | None) -> str:
    if path is not None or data is not None:
        return fingerprint(path, data, "source")
    return f"raw:{raw}"


def _merge_candidate(item: MediaItem, candidate: _Candidate) -> None:
    refs = [candidate.prepared_ref]
    if candidate.image_url_index is not None:
        refs.append(candidate.raw)
        if item.image_url_index is None:
            item.image_url_index = candidate.image_url_index
    for ref in refs:
        if ref and ref not in item.original_image_refs:
            item.original_image_refs.append(ref)
    if candidate.marker_raw and candidate.marker_raw not in item.original_markers:
        item.original_markers.append(candidate.marker_raw)


async def resolve_animations(
    request: Any,
    event: Any,
    client: httpx.AsyncClient,
    search_dirs: tuple[Path, ...] = (),
    limit: int = 8,
    on_skip: Any = None,
    download_path_factory: Any = None,
) -> list[MediaItem]:
    """把候选引用还原成确实是动图的 MediaItem 列表。"""
    items: list[MediaItem] = []
    seen: dict[str, MediaItem] = {}
    resolved_refs: dict[str, MediaItem] = {}

    candidates = collect_candidates(request, event, include_files=download_path_factory is None)
    if download_path_factory is not None:
        attachments = await VideoCollector(
            request,
            event,
            client,
            download_path_factory,
            max_videos=limit,
            max_download_mb=32,
            search_dirs=search_dirs,
            media_kind=MediaKind.ANIMATION,
            media_predicate=lambda name, _mime: _animation_name(name),
        ).collect()
        for name, notice in attachments.notices:
            if on_skip:
                on_skip(name, notice)
        for item in attachments.items:
            if item.path is not None and maybe_animated(_read_head(item.path)):
                candidates.append(
                    _Candidate(
                        str(item.path),
                        part_index=item.part_index,
                        quoted=item.quoted,
                        name=item.name,
                        marker_raw=item.marker_raw,
                        owned_temp=item.owned_temp,
                        source_url=item.source_url,
                    )
                )
            elif item.owned_temp:
                TempStore.discard(item.path)

    for candidate in candidates:
        if candidate.raw in resolved_refs:
            _merge_candidate(resolved_refs[candidate.raw], candidate)
            continue
        if len(items) >= limit:
            if candidate.owned_temp:
                TempStore.discard(resolve_local_path(candidate.raw, search_dirs))
            continue
        if candidate.raw.startswith(("base64://", "data:")) and len(candidate.raw) > (
            MAX_ANIMATION_BYTES * 4 // 3 + 1024
        ):
            if on_skip:
                on_skip(candidate.name or "动图", "体积超过 32 MB 上限")
            continue
        data = decode_inline_image(candidate.raw)
        path: Path | None = None

        if data is None:
            path = resolve_local_path(candidate.raw, search_dirs)

        if data is None and path is None and candidate.component is not None:
            getter = getattr(candidate.component, "get_file", None)
            if callable(getter):
                try:
                    raw = await asyncio.wait_for(getter(), timeout=30.0)
                    if isinstance(raw, str):
                        path = resolve_local_path(raw, search_dirs)
                        if path is None and is_http_url(raw):
                            data = await download_bytes(client, raw, MAX_ANIMATION_BYTES)
                except (TimeoutError, OSError, RuntimeError, httpx.HTTPError):
                    if on_skip:
                        on_skip(candidate.name or "动图文件", "附件读取失败或超时")
                    continue

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
            if len(data) > MAX_ANIMATION_BYTES:
                if on_skip:
                    on_skip(candidate.name or "动图", "体积超过 32 MB 上限")
                continue
            if not maybe_animated(data[:HEAD_BYTES]):
                continue
        elif path is not None:
            if path.stat().st_size > MAX_ANIMATION_BYTES:
                if on_skip:
                    on_skip(candidate.name or "动图", "体积超过 32 MB 上限")
                continue
            if not maybe_animated(_read_head(path)):
                continue
        else:
            continue

        identity = await protected_thread(
            _identity,
            candidate.raw,
            path,
            data,
            paths=[path] if path is not None else [],
        )
        item = seen.get(identity)
        if item is None:
            if len(items) >= limit:
                continue
            item = MediaItem(
                kind=MediaKind.ANIMATION,
                name=candidate.name or (path.name if path else "动图"),
                identity=identity,
                path=path,
                data=data,
                image_url_index=candidate.image_url_index,
                part_index=candidate.part_index,
                marker_raw=candidate.marker_raw,
                quoted=candidate.quoted,
                source_url=candidate.source_url
                or (candidate.raw if is_http_url(candidate.raw) else ""),
                owned_temp=candidate.owned_temp,
            )
            items.append(item)
            seen[identity] = item
        elif candidate.owned_temp and path != item.path:
            TempStore.discard(path)
        resolved_refs[candidate.raw] = item
        _merge_candidate(item, candidate)

    return items


def _read_head(path: Path, nbytes: int = HEAD_BYTES) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(nbytes)
    except OSError:
        return b""
