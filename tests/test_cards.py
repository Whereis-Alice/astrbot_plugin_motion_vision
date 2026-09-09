from __future__ import annotations

import asyncio
from types import SimpleNamespace

from motion_vision.cards import extract_event_cards, hydrate_event_cards, render_event_cards


def test_card_reader_decodes_escaped_ark_fields() -> None:
    event = SimpleNamespace(
        message_str=(
            r"[CQ:reply,id=42][CQ:json,data={\"app\":\"com.tencent.miniapp\","
            r"\"meta\":{\"detail_1\":{\"title\":\"title&#44;second\","
            r"\"desc\":\"description\",\"qqdocurl\":\"https%3A%2F%2Fexample.com%2Fa\"}}}]"
        )
    )

    cards = extract_event_cards(event)

    assert len(cards) == 1
    assert cards[0].title == "title,second"
    assert cards[0].url == "https://example.com/a"
    assert cards[0].quoted is True


def test_card_reader_includes_location_and_contact_segments() -> None:
    event = SimpleNamespace(
        raw_message={
            "message": [
                {"type": "location", "data": {"lat": "31.2", "lon": "121.5", "title": "上海"}},
                {"type": "contact", "data": {"nickname": "小明", "uin": "10001"}},
            ]
        }
    )

    rendered = render_event_cards(event)

    assert "位置卡片" in rendered
    assert "31.2, 121.5" in rendered
    assert "联系人或群名片" in rendered
    assert "小明" in rendered


def test_bilibili_cards_can_be_excluded_from_generic_context() -> None:
    event = SimpleNamespace(
        message_str='[CQ:json,data={"title":"视频","url":"https://www.bilibili.com/video/BV1xx411c7mD"}]'
    )

    assert extract_event_cards(event, exclude_bilibili=True) == []


def test_remote_reply_card_is_fetched_once_and_shared() -> None:
    calls: list[tuple[str, str]] = []

    class Bot:
        async def call_action(self, action: str, **params: object) -> dict[str, object]:
            calls.append((action, str(params.get("message_id") or params.get("id"))))
            return {
                "data": {
                    "message": [
                        {
                            "type": "json",
                            "data": {
                                "meta": {
                                    "detail_1": {
                                        "title": "远程卡片",
                                        "qqdocurl": "https://example.com/remote",
                                    }
                                }
                            },
                        }
                    ]
                }
            }

    event = SimpleNamespace(
        bot=Bot(),
        raw_message={"message": [{"type": "reply", "data": {"id": "42"}}]},
    )

    assert asyncio.run(hydrate_event_cards(event)) == 1
    assert asyncio.run(hydrate_event_cards(event)) == 0
    cards = extract_event_cards(event)

    assert len(calls) == 1
    assert cards[0].title == "远程卡片"
    assert cards[0].quoted is True
