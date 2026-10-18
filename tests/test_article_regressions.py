from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from motion_vision.article_parse import parse_article_page
from motion_vision.bilibili_article import (
    RESOLVED_SHORTS_ATTR,
    ArticleError,
    BilibiliArticleService,
    _allowed,
    parse_article_reference,
)
from motion_vision.cards import REMOTE_PAYLOADS_ATTR
from motion_vision.inject import inject
from motion_vision.models import MediaItem, MediaKind
from motion_vision.settings import BilibiliSettings, Settings, load_settings
from motion_vision.sources.bilibili import BilibiliCollector
from motion_vision.tempstore import TempStore


def opus_html(text: str = "正文第一段", tail: str = "正文最后一段") -> str:
    def paragraph(value: str) -> dict:
        return {"para_type": 1, "text": {"nodes": [{"word": {"words": value}}]}}

    state = {
        "detail": {
            "modules": [
                {"module_title": {"text": "专栏标题"}},
                {"module_author": {"name": "专栏作者"}},
                {"module_content": {"paragraphs": [paragraph(text), paragraph(tail)]}},
            ]
        }
    }
    return f'<nav>导航噪音</nav><script>window.__INITIAL_STATE__={json.dumps(state)};(function(){{throw "不要执行"}})()</script>'


def test_nested_html_keeps_all_paragraphs_and_inline_words() -> None:
    page = '<meta content="标题" property="og:title"><nav>不属于正文</nav><div class="opus-module-content"><div><p>第一段<b>加粗</b>末尾</p></div><div><p>第二段</p></div></div><footer>页脚</footer>'
    result = parse_article_page(page)
    assert result.title == "标题"
    assert result.content == "第一段加粗末尾\n第二段"
    assert "页脚" not in result.content


def test_opus_initial_state_extracts_title_author_and_tail() -> None:
    result = parse_article_page(opus_html())
    assert result.title == "专栏标题"
    assert result.author == "专栏作者"
    assert result.content == "正文第一段\n正文最后一段"
    assert result.source == "Opus 结构化正文"
    assert "导航" not in result.content


def test_page_shell_is_never_used_as_article_body() -> None:
    result = parse_article_page(
        "<title>安全验证</title><nav>首页</nav><script>login()</script>请登录"
    )
    assert not result.content


def test_legacy_read_initial_state_is_supported() -> None:
    result = parse_article_page(
        '<script>window.__INITIAL_STATE__={"readInfo":{"title":"旧文","author":{"name":"作者"},"content":"<p>第一段</p><p>最后一段</p>"}}</script>'
    )
    assert result.content == "第一段\n最后一段"
    assert result.author == "作者"


@pytest.mark.parametrize(
    ("value", "url"),
    [
        ("cv123", "https://www.bilibili.com/read/cv123"),
        ("read/cv123", "https://www.bilibili.com/read/cv123"),
        ("/read/123", "https://www.bilibili.com/read/cv123"),
        ("bilibili.com/opus/123?share_source=QQ", "https://www.bilibili.com/opus/123"),
        ("https://m.bilibili.com/read/mobile?id=123", "https://www.bilibili.com/read/cv123"),
        ("https%3A%2F%2Fwww.bilibili.com%2Fopus%2F123", "https://www.bilibili.com/opus/123"),
    ],
)
def test_article_reference_formats(value: str, url: str) -> None:
    reference = parse_article_reference(value)
    assert reference is not None and reference.url == url


def test_unrelated_urls_are_not_rewritten_as_bilibili() -> None:
    assert parse_article_reference("https://example.com/opus/123") is None
    assert parse_article_reference("https://bilibili.com.evil.test/read/cv123") is None
    assert parse_article_reference("https://www.bilibili.com/video/BV1xx411c7mD") is None


def test_raw_plain_reply_and_cv_opus_id_collision() -> None:
    event = SimpleNamespace(
        message_obj=SimpleNamespace(
            raw_message={"message": [{"type": "text", "data": {"text": "cv123 /opus/123"}}]}
        ),
        **{REMOTE_PAYLOADS_ATTR: [{"message": [{"type": "text", "data": {"text": "cv456"}}]}]},
    )
    references = BilibiliArticleService.find_references(event)
    assert [item.key for item in references] == ["cv:123", "opus:123", "cv:456"]


def test_qq_json_card_with_empty_plain_text() -> None:
    event = SimpleNamespace(
        message_str="",
        message_obj=SimpleNamespace(
            raw_message={
                "message": [
                    {
                        "type": "json",
                        "data": {
                            "data": json.dumps(
                                {
                                    "meta": {
                                        "detail_1": {
                                            "title": "分享",
                                            "qqdocurl": "https://www.bilibili.com/opus/123?share_source=QQ",
                                        }
                                    }
                                }
                            )
                        },
                    }
                ]
            }
        ),
    )
    references = BilibiliArticleService.find_references(event)
    assert len(references) == 1 and references[0].key == "opus:123"


def test_metadata_only_api_falls_back_to_redirected_opus(tmp_path: Path) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("viewinfo"):
            return httpx.Response(
                200, json={"code": 0, "data": {"title": "只有标题", "author_name": "旧作者"}}
            )
        if request.url.path == "/read/cv123":
            return httpx.Response(302, headers={"location": "/opus/987"})
        return httpx.Response(200, text=opus_html())

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = BilibiliArticleService(
                client, TempStore(tmp_path), BilibiliSettings(article_cover_enabled=False)
            )
            evidence = await service.collect_many(SimpleNamespace(message_str="cv123"))
        assert evidence[0].status == "body"
        assert "正文最后一段" in evidence[0].text
        assert "/opus/987" in evidence[0].text
        assert seen == ["/x/article/viewinfo", "/read/cv123", "/opus/987"]

    asyncio.run(run())


def test_http_failure_on_metadata_does_not_prevent_page_fallback(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("viewinfo"):
            raise httpx.ReadTimeout("failed", request=request)
        return httpx.Response(200, text=opus_html())

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = BilibiliArticleService(
                client, TempStore(tmp_path), BilibiliSettings(article_cover_enabled=False)
            )
            evidence = await service.collect_many(SimpleNamespace(message_str="cv123"))
            assert evidence[0].status == "body"

    asyncio.run(run())


def test_direct_opus_skips_cv_metadata_api_and_marks_missing_body(tmp_path: Path) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200, text='<meta property="og:title" content="仅标题"><nav>首页</nav>'
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = BilibiliArticleService(
                client, TempStore(tmp_path), BilibiliSettings(article_cover_enabled=False)
            )
            evidence = await service.collect_many(SimpleNamespace(message_str="/opus/123"))
        assert evidence[0].status == "metadata"
        assert "正文未读取到" in evidence[0].text
        assert "专栏正文：" not in evidence[0].text
        assert seen == ["/opus/123"]

    asyncio.run(run())


def test_video_short_link_is_routed_once_without_article_failure(tmp_path: Path) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        assert "cookie" not in request.headers if request.url.host == "b23.tv" else True
        if request.url.host == "b23.tv":
            return httpx.Response(
                302, headers={"location": "https://www.bilibili.com/video/BV1xx411c7mD"}
            )
        return httpx.Response(200, text="x" * 600000)

    class VideoClient:
        async def prepare(self, reference, **kwargs):
            assert reference.kind == "bvid"
            return MediaItem(MediaKind.VIDEO, "video", reference.key, context_text="元数据")

    async def run() -> None:
        event = SimpleNamespace(message_str="https://b23.tv/video")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = BilibiliArticleService(
                client, TempStore(tmp_path), BilibiliSettings(cookie="SESSDATA=secret")
            )
            assert await service.collect_many(event) == []
            found = await BilibiliCollector(
                event, VideoClient(), BilibiliSettings(), download_video=False
            ).collect()
            assert len(found.items) == 1 and not found.notices
        assert seen == ["b23.tv", "www.bilibili.com"]

    asyncio.run(run())


def test_article_short_link_is_excluded_from_video_collector(tmp_path: Path) -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "b23.tv":
                return httpx.Response(
                    302, headers={"location": "https://www.bilibili.com/opus/123"}
                )
            return httpx.Response(200, text=opus_html())

        event = SimpleNamespace(message_str="https://b23.tv/article")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = BilibiliArticleService(
                client, TempStore(tmp_path), BilibiliSettings(article_cover_enabled=False)
            )
            evidence = await service.collect_many(event)
            found = await BilibiliCollector(event, object(), BilibiliSettings()).collect()
        assert evidence[0].status == "body"
        assert not found.items and not found.notices
        assert getattr(event, RESOLVED_SHORTS_ATTR)["https://b23.tv/article"].endswith("/opus/123")

    asyncio.run(run())


def test_long_article_injection_is_bounded_and_tool_reads_consecutive_pages(tmp_path: Path) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, text=opus_html("正文" * 2000))
            )
        ) as client:
            settings = BilibiliSettings(article_cover_enabled=False, article_max_chars=1000)
            service = BilibiliArticleService(client, TempStore(tmp_path), settings)
            evidence = await service.collect_many(SimpleNamespace(message_str="/opus/123"))
            request = SimpleNamespace(prompt="读专栏", extra_user_content_parts=[], image_urls=[])
            report = inject(request, [], replace(Settings(), bilibili=settings), evidence=evidence)
            assert report.article_bodies == 1 and report.evidence_chars > 4000
            assert "中间内容已省略" in evidence[0].text
            assert "read_bilibili_article" in evidence[0].text
            text_parts = request.extra_user_content_parts
            assert text_parts[0]._no_save
            assert not text_parts[-1]._no_save
            first = await service.read("/opus/123", offset=0, limit=1000)
            tail = await service.read("/opus/123", offset=4000, limit=1000)
            assert "offset=1000" in first
            assert "正文最后一段" in tail and "已到结尾" in tail

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "https://evilhdslb.com/a.jpg",
        "file://www.bilibili.com/x",
        "https://www.bilibili.com:123/x",
        "https://user:pass@www.bilibili.com/x",
    ],
)
def test_article_requests_reject_unsafe_domains_and_urls(url: str) -> None:
    assert not _allowed(url, image_only=True)


def test_oversized_response_is_stopped_during_streaming(tmp_path: Path) -> None:
    yielded = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(10):
                yielded.append(i)
                yield b"x" * 100

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, stream=Body()))
        ) as client:
            service = BilibiliArticleService(client, TempStore(tmp_path), BilibiliSettings())
            with pytest.raises(ArticleError, match="大小限制"):
                await service._bytes("https://www.bilibili.com/opus/123", max_bytes=250)
        assert yielded == [0, 1, 2]

    asyncio.run(run())


def test_compressed_page_is_decoded_once(tmp_path: Path) -> None:
    body = opus_html().encode()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield gzip.compress(body)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Body())
            )
        ) as client:
            service = BilibiliArticleService(
                client, TempStore(tmp_path), BilibiliSettings(article_cover_enabled=False)
            )
            evidence = await service.collect_many(SimpleNamespace(message_str="/opus/123"))
            assert evidence[0].status == "body"
            assert "正文最后一段" in evidence[0].text

    asyncio.run(run())


def test_article_config_limits() -> None:
    assert load_settings({}).bilibili.article_max_chars == 20000
    assert load_settings({"bilibili": {"article_max_chars": 1}}).bilibili.article_max_chars == 1000
    assert (
        load_settings({"bilibili": {"article_max_chars": 999999}}).bilibili.article_max_chars
        == 60000
    )
