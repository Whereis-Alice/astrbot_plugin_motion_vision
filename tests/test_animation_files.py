from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
from astrbot.api.message_components import File, Reply
from astrbot.core.agent.message import TextPart

from motion_vision.sources.animation import resolve_animations
from motion_vision.tempstore import TempStore
from tests.test_animation_timeline import animation_bytes
from tests.test_host_images import request


def component(kind, **fields):
    if kind is File and "file_" in fields:
        fields["file"] = fields.pop("file_")
    if kind is Reply:
        fields.setdefault("id", "1")
    try:
        return kind(**fields)
    except TypeError:
        value = kind()
        for key, field in fields.items():
            setattr(value, key, field)
        return value


def test_file_and_quoted_file_are_collected_and_markers_deduplicated(tmp_path):
    first = tmp_path / "first.gif"
    second = tmp_path / "second.apng"
    first.write_bytes(animation_bytes("GIF", [100, 100]))
    second.write_bytes(animation_bytes("PNG", [100, 100, 100], cover=True))
    files = [
        component(File, name=first.name, file_=str(first)),
        component(File, name=second.name, file_=str(second)),
    ]
    event = SimpleNamespace(get_messages=lambda: [files[0], component(Reply, chain=[files[1]])])
    req = request(parts=[TextPart(text=f"[File Attachment: name first.gif, path {first}]")])

    async def scenario():
        async with httpx.AsyncClient() as client:
            return await resolve_animations(
                req, event, client, download_path_factory=TempStore(tmp_path).download_path
            )

    items = asyncio.run(scenario())
    assert [item.path for item in items] == [first, second]
    assert [item.quoted for item in items] == [False, True]
    assert items[0].marker_raw


def test_onebot_group_file_id_is_resolved_and_verified(tmp_path):
    calls = []
    content = animation_bytes("WEBP", [100, 200, 300])

    async def action(name, **params):
        calls.append((name, params))
        return {"data": {"url": "https://files.test/a.webp"}}

    event = SimpleNamespace(
        get_messages=lambda: [],
        get_group_id=lambda: "42",
        get_platform_name=lambda: "aiocqhttp",
        bot=SimpleNamespace(call_action=action),
        message_obj=SimpleNamespace(
            raw_message={
                "message": [{"type": "file", "data": {"name": "a.webp", "file_id": "abc"}}]
            }
        ),
    )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
        ) as client:
            return await resolve_animations(
                request(), event, client, download_path_factory=TempStore(tmp_path).download_path
            )

    (item,) = asyncio.run(scenario())
    assert calls[0][0] == "get_group_file_url"
    assert item.owned_temp and item.path.read_bytes() == content


def test_static_file_is_not_treated_as_animation(tmp_path):
    from PIL import Image

    path = tmp_path / "still.png"
    Image.new("RGB", (16, 16)).save(path)
    event = SimpleNamespace(get_messages=lambda: [component(File, name=path.name, file_=str(path))])

    async def scenario():
        async with httpx.AsyncClient() as client:
            return await resolve_animations(
                request(), event, client, download_path_factory=TempStore(tmp_path).download_path
            )

    assert asyncio.run(scenario()) == []
    assert path.exists()


def test_unknown_documents_do_not_trigger_attachment_download(tmp_path):
    event = SimpleNamespace(get_messages=lambda: [component(File, name="document.zip", file_="id")])

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: (_ for _ in ()).throw(AssertionError("unexpected download"))
            )
        ) as client:
            return await resolve_animations(
                request(), event, client, download_path_factory=TempStore(tmp_path).download_path
            )

    assert asyncio.run(scenario()) == []
