from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from motion_vision.bilibili_parser import BilibiliInfo
from motion_vision.bilibili_transcript import (
    BilibiliTranscriptService,
    _secure_bcut_upload_url,
)
from motion_vision.settings import BilibiliSettings
from motion_vision.tempstore import TempStore


class _FakeBilibili:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.cookie = "SESSDATA=one"

    def _effective_cookie(self) -> str:
        return self.cookie

    async def _fetch_official_subtitle(self, _info: BilibiliInfo, limit: int) -> str:
        self.calls.append(limit)
        return "\n".join(f"[{index}] line {index}" for index in range(20))


def _info() -> BilibiliInfo:
    return BilibiliInfo(
        bvid="BV1xx411c7mD",
        aid=1,
        cid=2,
        page=1,
        title="title",
        description="",
        author="",
        duration=10,
        canonical_url="https://www.bilibili.com/video/BV1xx411c7mD",
    )


def test_transcript_cache_is_separated_by_cookie_and_reuses_full_source(tmp_path: Path) -> None:
    bili = _FakeBilibili()
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200)))
    service = BilibiliTranscriptService(
        client,
        bili,
        runner=None,  # official subtitles do not need ffmpeg
        store=TempStore(tmp_path),
        settings=BilibiliSettings(max_subtitle_chars=100, caption_full_max_chars=500),
    )
    try:
        first = asyncio.run(service.fetch(_info()))
        second = asyncio.run(service.fetch(_info(), full=True))
        bili.cookie = "SESSDATA=two"
        third = asyncio.run(service.fetch(_info()))
    finally:
        asyncio.run(client.aclose())

    assert first is not None and second is not None and third is not None
    assert len(bili.calls) == 2
    assert len(second.full_text) >= len(first.text)


def test_bcut_upload_allowlist() -> None:
    assert _secure_bcut_upload_url("http://upload.biliapi.net/chunk").startswith("https://")
    with pytest.raises(RuntimeError):
        _secure_bcut_upload_url("https://example.com/chunk")
