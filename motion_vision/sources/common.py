"""来源解析的公共工具：路径还原、URL 处理、附件标记解析、限量下载。"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

MB = 1024 * 1024
DOWNLOAD_CHUNK = 64 * 1024

MARKER_PATTERN = re.compile(
    r"\[(Image|Video|File) Attachment(?P<quoted> in quoted message)?:\s*(?P<body>[^\]]*)\]"
)
DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[\w./+-]+)?;base64,(?P<payload>.*)$", re.S)

VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".m4v",
        ".mkv",
        ".webm",
        ".avi",
        ".flv",
        ".wmv",
        ".mpg",
        ".mpeg",
        ".ts",
        ".m2ts",
        ".3gp",
        ".ogv",
    }
)


@dataclass(frozen=True)
class AttachmentMarker:
    """AstrBot 注入到提示词里的附件文本标记。"""

    kind: str
    """image / video / file"""

    name: str
    path: str
    quoted: bool
    part_index: int

    raw: str = ""
    """标记在原文里的完整文本，用于处理完成后原位移除。"""


def parse_markers(text: str, part_index: int = 0) -> list[AttachmentMarker]:
    """解析形如 [Video Attachment: name a.mp4, path C:/x/a.mp4] 的标记。"""
    markers: list[AttachmentMarker] = []
    for match in MARKER_PATTERN.finditer(text or ""):
        kind = match.group(1).lower()
        body = (match.group("body") or "").strip()
        if not body:
            continue
        name = ""
        path = body
        if ", path " in body:
            name, _, path = body.rpartition(", path ")
            name = name.removeprefix("name ").strip()
        path = path.strip()
        if not path:
            continue
        markers.append(
            AttachmentMarker(
                kind=kind,
                name=name or Path(path).name,
                path=path,
                quoted=bool(match.group("quoted")),
                part_index=part_index,
                raw=match.group(0),
            )
        )
    return markers


def is_http_url(text: str) -> bool:
    return bool(text) and text.lower().startswith(("http://", "https://"))


def redact_url(url: str) -> str:
    """日志脱敏：QQ 的文件直链带签名参数，不能原样打出来。"""
    if not is_http_url(url):
        return url
    try:
        parsed = urlparse(url)
    except ValueError:
        return "<invalid-url>"
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return f"{base}?<query-redacted>" if parsed.query else base


def decode_inline_image(text: str) -> bytes | None:
    """还原 base64:// 或 data:image/...;base64, 形式的内联图片。"""
    if not text:
        return None
    payload: str | None = None
    if text.startswith("base64://"):
        payload = text[len("base64://") :]
    else:
        match = DATA_URL_PATTERN.match(text)
        if match:
            payload = match.group("payload")
    if not payload:
        return None
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None


def strip_file_scheme(text: str) -> str:
    if text.lower().startswith("file://"):
        parsed = urlparse(text)
        raw = unquote(parsed.path or "")
        # Windows 下 file:///C:/x 会解析出 /C:/x
        if re.match(r"^/[A-Za-z]:", raw):
            raw = raw[1:]
        return raw or text
    return text


def resolve_local_path(raw: str, search_dirs: tuple[Path, ...] = ()) -> Path | None:
    """把各种形态的路径字符串还原成真实存在的本地文件。

    协议直链一律返回 None，交给下载分支处理。
    """
    if not raw or is_http_url(raw) or raw.startswith(("base64://", "data:")):
        return None

    candidate = strip_file_scheme(raw).strip().strip('"')
    if not candidate:
        return None

    seen: list[Path] = []
    try:
        primary = Path(candidate)
    except (OSError, ValueError):
        return None

    seen.append(primary)
    if not primary.is_absolute():
        name = primary.name
        for directory in search_dirs:
            seen.append(directory / candidate)
            if name and name != candidate:
                seen.append(directory / name)

    for path in seen:
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            continue
    return None


def looks_like_video(name: str = "", mime: str = "") -> bool:
    if mime and mime.lower().startswith("video/"):
        return True
    if not name:
        return False
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


async def peek_head(
    client: httpx.AsyncClient, url: str, nbytes: int, timeout: float = 10.0
) -> bytes:
    """用 Range 请求只取文件头，避免为了判类型下载整个文件。"""
    try:
        response = await client.get(
            url, headers={"Range": f"bytes=0-{max(0, nbytes - 1)}"}, timeout=timeout
        )
        if response.status_code >= 400:
            return b""
        return response.content[:nbytes]
    except (httpx.HTTPError, ValueError):
        return b""


class DownloadTooLarge(RuntimeError):
    """远端文件超过允许的下载体积。"""


async def download_to_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    max_bytes: int,
    timeout: float = 120.0,
) -> int:
    """流式下载并强制体积上限，超限立刻中止并删除半成品。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise DownloadTooLarge(f"文件约 {int(declared) / MB:.1f} MB，超过限制")
            with dest.open("wb") as handle:
                async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK):
                    written += len(chunk)
                    if written > max_bytes:
                        raise DownloadTooLarge("文件超过下载体积限制")
                    handle.write(chunk)
    except DownloadTooLarge:
        dest.unlink(missing_ok=True)
        raise
    except (httpx.HTTPError, OSError) as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"下载失败：{exc}") from exc
    return written


async def download_bytes(
    client: httpx.AsyncClient, url: str, max_bytes: int, timeout: float = 60.0
) -> bytes:
    """下载到内存，同样强制体积上限（用于动图，通常只有几 MB）。"""
    buffer = bytearray()
    async with client.stream("GET", url, timeout=timeout) as response:
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise DownloadTooLarge("文件超过下载体积限制")
        async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK):
            buffer.extend(chunk)
            if len(buffer) > max_bytes:
                raise DownloadTooLarge("文件超过下载体积限制")
    return bytes(buffer)
