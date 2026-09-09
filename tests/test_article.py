from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from motion_vision.bilibili_article import (
    BilibiliArticleService,
    parse_article_reference,
)
from motion_vision.settings import BilibiliSettings
from motion_vision.tempstore import TempStore


def test_parse_article_reference_supports_read_opus_and_short_links() -> None:
    read = parse_article_reference("https://www.bilibili.com/read/cv12345")
    opus = parse_article_reference("/opus/67890")
    short = parse_article_reference("https://b23.tv/article")

    assert read is not None and read.article_id == "12345"
    assert opus is not None and opus.article_id == "67890"
    assert short is not None and short.article_id == ""


def test_article_service_reads_multiple_articles_and_keeps_cookie_off_cover(
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("cookie", "")))
        if request.url.path.endswith("viewinfo"):
            article_id = request.url.params.get("id")
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "title": f"Article {article_id}",
                        "author_name": "author",
                        "summary": "summary",
                        "content": "<p>正文</p>",
                        "banner_url": "https://i0.hdslb.com/cover.jpg",
                    },
                },
            )
        if request.url.path.endswith("cover.jpg"):
            return httpx.Response(200, content=b"not-an-image")
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliArticleService(
        client,
        TempStore(tmp_path),
        BilibiliSettings(cookie="SESSDATA=secret"),
    )
    event = SimpleNamespace(
        message_str=("https://www.bilibili.com/read/cv123 https://www.bilibili.com/opus/456")
    )
    try:
        evidence = asyncio.run(service.collect_many(event))
    finally:
        asyncio.run(client.aclose())

    assert len(evidence) == 2
    assert all("Article" in item.text for item in evidence)
    assert any(path.endswith("viewinfo") and cookie for path, cookie in seen)
    assert all(not cookie for path, cookie in seen if path.endswith("cover.jpg"))
