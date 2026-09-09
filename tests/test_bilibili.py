from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from motion_vision import bilibili, bilibili_client
from motion_vision.bilibili import (
    BilibiliClient,
    BilibiliError,
    extract_event_references,
    extract_references,
    parse_reference,
)
from motion_vision.settings import BilibiliSettings
from motion_vision.sources.bilibili import BilibiliCollector


def test_parse_reference_supports_bvid_av_and_page() -> None:
    bvid = parse_reference("分享：https://www.bilibili.com/video/BV1xx411c7mD?p=3")
    assert bvid is not None
    assert bvid.kind == "bvid"
    assert bvid.value == "BV1xx411c7mD"
    assert bvid.page == 3

    aid = parse_reference("av123456 p=2")
    assert aid is not None
    assert aid.kind == "aid"
    assert aid.value == "123456"
    assert aid.page == 2


def test_extract_references_deduplicates_multiple_links() -> None:
    refs = extract_references(
        "BV1xx411c7mD https://www.bilibili.com/video/BV1xx411c7mD "
        "https://www.bilibili.com/video/BV2xx411c7mD"
    )
    assert [ref.value for ref in refs] == ["BV1xx411c7mD", "BV2xx411c7mD"]


def test_extract_references_supports_multiple_bare_identifiers() -> None:
    refs = extract_references("BV1xx411c7mD BV2xx411c7mD av123456")
    assert [(ref.kind, ref.value) for ref in refs] == [
        ("bvid", "BV1xx411c7mD"),
        ("bvid", "BV2xx411c7mD"),
        ("aid", "123456"),
    ]


def test_page_parameter_is_scoped_to_nearby_bare_identifier() -> None:
    refs = extract_references("BV1xx411c7mD?p=2 BV2xx411c7mD")
    assert [ref.page for ref in refs] == [2, 1]


def test_parser_does_not_treat_unrelated_bilibili_pages_as_videos() -> None:
    assert parse_reference("https://www.bilibili.com/space/123456") is None
    assert parse_reference("https://www.bilibili.com/read/cv123") is None
    assert parse_reference("https://www.bilibili.com/read/av123456") is None


def test_extract_event_references_reads_card_metadata_and_reply() -> None:
    event = type("Event", (), {})()
    event.message_str = ""
    event.message_obj = type("Message", (), {})()
    event.message_obj.message_str = ""
    event.get_messages = lambda: [
        {
            "type": "reply",
            "data": {
                "title": "卡片标题",
                "description": "卡片简介",
                "url": "https://www.bilibili.com/video/BV1xx411c7mD",
            },
        }
    ]

    refs = extract_event_references(event)

    assert len(refs) == 1
    assert refs[0].quoted is True
    assert refs[0].title == "卡片标题"
    assert refs[0].description == "卡片简介"


def test_extract_event_references_reads_encoded_cq_json_card() -> None:
    payload = {
        "meta": {
            "detail_1": {
                "title": "CQ 卡片视频",
                "desc": "带逗号、方括号 [测试]",
                "qqdocurl": "https://b23.tv/cq-example",
            }
        }
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("&", "&amp;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace(",", "&#44;")
    )
    event = type("Event", (), {})()
    event.message_str = ""
    event.message_obj = type("Message", (), {})()
    event.message_obj.raw_message = f"[CQ:reply,id=9002][CQ:json,data={encoded}]"

    refs = extract_event_references(event)

    assert len(refs) == 1
    assert refs[0].quoted is True
    assert refs[0].value == "https://b23.tv/cq-example"
    assert refs[0].title == "CQ 卡片视频"


def test_client_reads_metadata_and_timestamped_subtitle(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bilibili, "API_MIN_INTERVAL", 0.0)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/x/web-interface/view":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "aid": 12,
                        "bvid": "BV1xx411c7mD",
                        "title": "测试视频",
                        "desc": "测试简介",
                        "owner": {"name": "测试 UP"},
                        "pages": [
                            {
                                "cid": 345,
                                "duration": 12,
                                "part": "正片",
                                "dimension": {"width": 1920, "height": 1080},
                            }
                        ],
                        "tname": "科技",
                        "pubdate": 1_700_000_000,
                        "stat": {"view": 123, "like": 45},
                    },
                },
            )
        if request.url.path == "/x/player/v2":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "zh-CN",
                                    "subtitle_url": "https://aisubtitle.hdslb.com/test.json",
                                }
                            ]
                        }
                    },
                },
            )
        if request.url.host == "aisubtitle.hdslb.com":
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "body": [
                            {"from": 0.2, "content": "第一句"},
                            {"from": 4.5, "content": "第二句"},
                        ]
                    }
                ).encode(),
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(),
    )
    try:
        info, caption = asyncio.run(service.caption_from_value("BV1xx411c7mD"))
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())

    assert info.title == "测试视频"
    assert info.cid == 345
    assert info.page_count == 1
    assert info.part_title == "正片"
    assert info.category == "科技"
    assert info.width == 1920
    assert info.height == 1080
    assert info.stats == {"view": 123, "like": 45}
    assert "[0:00] 第一句" in caption
    assert "[0:04] 第二句" in caption


def test_subtitle_cdn_redirect_stays_within_allowlist(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bilibili, "API_MIN_INTERVAL", 0.0)
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/x/web-interface/view":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "bvid": "BV1xx411c7mD",
                        "title": "重定向测试",
                        "pages": [{"cid": 345, "duration": 12}],
                    },
                },
            )
        if request.url.path == "/x/player/v2":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "zh-CN",
                                    "subtitle_url": "https://aisubtitle.hdslb.com/redirect.json",
                                }
                            ]
                        }
                    },
                },
            )
        if request.url.path == "/redirect.json":
            return httpx.Response(
                302,
                headers={"location": "https://aisubtitle.hdslb.com/final.json"},
            )
        if request.url.path == "/final.json":
            return httpx.Response(
                200,
                json={"body": [{"from": 1, "content": "重定向后的字幕"}]},
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(),
    )
    try:
        _info, caption = asyncio.run(service.caption_from_value("BV1xx411c7mD"))
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())

    assert "重定向后的字幕" in caption
    assert any("final.json" in value for value in calls)


def test_missing_subtitle_is_negative_cached(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bilibili, "API_MIN_INTERVAL", 0.0)
    player_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal player_calls
        if request.url.path == "/x/web-interface/view":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "bvid": "BV1xx411c7mD",
                        "title": "无字幕测试",
                        "pages": [{"cid": 345, "duration": 12}],
                    },
                },
            )
        if request.url.path == "/x/player/v2":
            player_calls += 1
            return httpx.Response(200, json={"code": 0, "data": {"subtitle": {}}})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(),
    )
    try:

        async def run() -> None:
            info = await service.resolve(parse_reference("BV1xx411c7mD"))
            assert info is not None
            assert await service.fetch_subtitle(info) == ""
            assert await service.fetch_subtitle(info) == ""

        asyncio.run(run())
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())

    assert player_calls == 1


def test_transient_subtitle_failure_is_briefly_cached(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bilibili, "API_MIN_INTERVAL", 0.0)
    player_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal player_calls
        if request.url.path == "/x/web-interface/view":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "bvid": "BV1xx411c7mD",
                        "title": "字幕失败缓存",
                        "pages": [{"cid": 345, "duration": 12}],
                    },
                },
            )
        if request.url.path == "/x/player/v2":
            player_calls += 1
            return httpx.Response(503)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(),
    )
    try:

        async def run() -> None:
            info = await service.resolve(parse_reference("BV1xx411c7mD"))
            assert await service.fetch_subtitle(info) == ""
            assert await service.fetch_subtitle(info) == ""

        asyncio.run(run())
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())

    assert player_calls == 1


def test_bilibili_api_retries_http_429_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bilibili, "API_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(bilibili_client, "_retry_after", lambda _value: 0.0)
    view_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal view_calls
        if request.url.path != "/x/web-interface/view":
            return httpx.Response(404)
        view_calls += 1
        if view_calls == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "bvid": "BV1xx411c7mD",
                    "title": "429 重试成功",
                    "pages": [{"cid": 345, "duration": 12}],
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(fetch_subtitles=False),
    )
    try:
        info = asyncio.run(service.resolve(parse_reference("BV1xx411c7mD")))
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())

    assert info.title == "429 重试成功"
    assert view_calls == 2


def test_collector_hydrates_an_unexpanded_reply_card(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bilibili, "API_MIN_INTERVAL", 0.0)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/x/web-interface/view":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "bvid": "BV1xx411c7mD",
                        "title": "引用卡片视频",
                        "pages": [{"cid": 345, "duration": 12}],
                    },
                },
            )
        if request.url.path == "/x/player/v2":
            return httpx.Response(200, json={"code": 0, "data": {"subtitle": {}}})
        return httpx.Response(404)

    class Bot:
        async def call_action(self, action: str, **kwargs: object) -> dict[str, object]:
            assert action == "get_msg"
            assert kwargs["message_id"] in {"42", 42}
            return {
                "data": {
                    "message": [
                        {
                            "type": "json",
                            "data": {
                                "title": "引用卡片视频",
                                "url": "https://www.bilibili.com/video/BV1xx411c7mD",
                            },
                        }
                    ]
                }
            }

    event = type("Event", (), {})()
    event.message_str = ""
    event.get_messages = lambda: [{"type": "reply", "data": {"id": "42"}}]
    event.bot = Bot()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(),
    )
    try:

        async def run() -> list[object]:
            found = await BilibiliCollector(
                event,
                service,
                BilibiliSettings(),
                download_video=False,
            ).collect()
            return found.items

        items = asyncio.run(run())
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())

    assert len(items) == 1
    assert items[0].quoted is True
    assert items[0].name == "引用卡片视频"


def test_prepare_keeps_card_text_when_bilibili_api_is_temporarily_unavailable():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(503)))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(),
    )
    reference = parse_reference(
        "https://www.bilibili.com/video/BV1xx411c7mD",
        title="卡片标题",
        description="卡片简介",
        quoted=True,
    )
    assert reference is not None
    try:
        item = asyncio.run(service.prepare(reference, download_video=False))
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())

    assert item.name == "卡片标题"
    assert item.quoted is True
    assert "卡片简介" in item.context_text
    assert item.source_notice


def test_short_url_redirect_leaving_allowlist_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bilibili, "API_MIN_INTERVAL", 0.0)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/private"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = BilibiliClient(
        client,
        lambda suffix: Path("unused") / suffix,
        BilibiliSettings(),
    )
    try:
        with pytest.raises(BilibiliError, match="allowlist"):
            asyncio.run(service.caption_from_value("https://b23.tv/abc"))
    finally:
        asyncio.run(service.close())
        asyncio.run(client.aclose())
