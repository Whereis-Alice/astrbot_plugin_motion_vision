"""B 站视频下载适配。

下载是唯一需要在线程中运行的 B 站工作：``yt-dlp`` 本身是同步库，且可能调用
ffmpeg 合并音视频。这里把它隔离出来，客户端只负责限流、缓存和生命周期。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .bilibili_parser import USER_AGENT, BilibiliError, BilibiliInfo, _clip_text
from .settings import MB


def download_video_sync(
    info: BilibiliInfo,
    target: Path,
    *,
    cookie: str,
    ffmpeg_path: str,
    max_download_mb: int,
    timeout_seconds: int,
    log: Callable[[str], None],
) -> None:
    """同步下载一个 B 站分 P，并把最终文件移动到 ``target``。

    函数不负责并发控制；调用方必须在后台线程和有界信号量中调用它。下载过程会
    同时使用 yt-dlp 的预检限制、进度回调和最终文件大小检查，避免合并阶段绕过
    配置中的大小预算。
    """

    try:
        import yt_dlp
    except ImportError as exc:
        raise BilibiliError(
            "yt-dlp is not installed",
            "解析 B 站链接需要 yt-dlp 依赖，请在插件管理页重新安装依赖。",
        ) from exc

    limit_bytes = max(1, max_download_mb) * MB
    target.parent.mkdir(parents=True, exist_ok=True)
    work_dir = target.parent / f"{target.stem}_bilibili"
    work_dir.mkdir(parents=True, exist_ok=True)
    exceeded = False

    def progress(status: dict[str, Any]) -> None:
        nonlocal exceeded
        downloaded = int(status.get("downloaded_bytes") or 0)
        if downloaded > limit_bytes:
            exceeded = True
            raise yt_dlp.utils.DownloadError("motion vision size limit exceeded")

    try:
        cookie_file = _write_cookie_file(work_dir, cookie)
        options: dict[str, Any] = {
            "format": "bv*[height<=720]+ba/b[height<=720]/best",
            "merge_output_format": "mp4",
            "outtmpl": str(work_dir / "media.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 1,
            "fragment_retries": 1,
            "concurrent_fragment_downloads": 1,
            "max_filesize": limit_bytes,
            "socket_timeout": min(60, max(10, timeout_seconds)),
            "overwrites": True,
            "http_headers": {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"},
            "progress_hooks": [progress],
            "logger": _YtdlpLogger(log),
        }
        if cookie_file is not None:
            options["cookiefile"] = str(cookie_file)
        if ffmpeg_path:
            options["ffmpeg_location"] = ffmpeg_path

        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([info.canonical_url])

        output = _find_downloaded_file(work_dir)
        if output is None:
            if exceeded:
                raise _size_limit_error(max_download_mb)
            raise BilibiliError(
                "yt-dlp produced no media file",
                "B 站视频下载后没有找到媒体文件。",
            )
        if output.stat().st_size > limit_bytes:
            raise _size_limit_error(max_download_mb)
        target.unlink(missing_ok=True)
        output.replace(target)
    except BilibiliError:
        raise
    except Exception as exc:
        if exceeded or "size limit" in str(exc).lower():
            raise _size_limit_error(max_download_mb) from exc
        raise BilibiliError(
            f"yt-dlp download failed: {type(exc).__name__}",
            "B 站视频下载失败，可能需要登录、视频已失效或当前网络无法访问 B 站。",
        ) from exc
    finally:
        # 临时 Cookie 只在 yt-dlp 工作目录中存在，下载结束后连同分片一起删除。
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


def _size_limit_error(max_download_mb: int) -> BilibiliError:
    return BilibiliError(
        "Bilibili download exceeded size limit",
        f"B 站视频超过配置的 {max_download_mb} MB 大小限制。",
    )


class _YtdlpLogger:
    """把 yt-dlp 的警告压缩后交给 AstrBot debug 日志。"""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log

    def debug(self, _message: str) -> None:
        return

    def info(self, _message: str) -> None:
        return

    def warning(self, message: str) -> None:
        self._log(f"B 站下载提示：{_clip_text(message, 180)}")

    def error(self, message: str) -> None:
        self._log(f"B 站下载错误：{_clip_text(message, 180)}")


def _write_cookie_file(directory: Path, cookie: str) -> Path | None:
    pairs: list[tuple[str, str]] = []
    for token in cookie.split(";"):
        name, separator, value = token.strip().partition("=")
        name = name.strip().replace("\t", "").replace("\r", "").replace("\n", "")
        value = value.strip().replace("\t", "").replace("\r", "").replace("\n", "")
        if separator and name and value:
            pairs.append((name, value))
    if not pairs:
        return None
    target = directory / "cookies.txt"
    lines = ["# Netscape HTTP Cookie File"]
    lines.extend(f".bilibili.com\tTRUE\t/\tFALSE\t0\t{name}\t{value}" for name, value in pairs)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _find_downloaded_file(directory: Path) -> Path | None:
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "cookies.txt" and path.suffix not in {".part", ".ytdl"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


__all__ = ["download_video_sync"]
