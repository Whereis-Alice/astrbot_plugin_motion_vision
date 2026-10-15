from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
from astrbot.core.agent.message import ImageURLPart, TextPart

from motion_vision.host_images import host_animations
from motion_vision.inject import inject
from motion_vision.models import MediaResult, SampledFrame
from motion_vision.settings import Settings
from motion_vision.sources.animation import resolve_animations
from tests.test_animation_timeline import animation_bytes


def request(urls=(), parts=()):
    return SimpleNamespace(
        image_urls=list(urls), extra_user_content_parts=list(parts), audio_urls=[], prompt="看看"
    )


def test_host_numbering_accounts_for_extra_images_before_urls(tmp_path):
    original = tmp_path / "source.gif"
    original.write_bytes(animation_bytes("GIF", [100, 100, 100]))
    extra = ImageURLPart(image_url=ImageURLPart.ImageURL(url="keep-extra.png"))
    label = f"[Image 3 in quoted message: original path {original}; animation converted to a 3x3 frame montage]"
    req = request(["keep.png", "host-grid.jpg"], [extra, TextPart(text=label)])
    (binding,) = host_animations(req)
    assert binding.prepared == "host-grid.jpg" and binding.quoted
    event = SimpleNamespace(get_messages=lambda: [])

    async def scenario():
        async with httpx.AsyncClient() as client:
            return await resolve_animations(req, event, client)

    (item,) = asyncio.run(scenario())
    assert item.path == original
    frame = tmp_path / "new.jpg"
    frame.write_bytes(b"jpeg")
    inject(req, [MediaResult(item, frames=[SampledFrame(frame, 0)])], Settings())
    assert req.image_urls == ["keep.png"]
    assert extra in req.extra_user_content_parts
    assert all(getattr(p, "text", "") != label for p in req.extra_user_content_parts)


def test_host_extra_part_is_replaced_and_budget_drop_removes_it(tmp_path):
    original = tmp_path / "source.gif"
    original.write_bytes(animation_bytes("GIF", [100, 100]))
    label = f"[Image 1: original path {original}; animation converted to a 3x3 frame montage]"
    visual = ImageURLPart(image_url=ImageURLPart.ImageURL(url="prepared.jpg"))
    req = request(parts=[visual, TextPart(text=label)])

    async def scenario():
        async with httpx.AsyncClient() as client:
            return await resolve_animations(req, SimpleNamespace(get_messages=lambda: []), client)

    (item,) = asyncio.run(scenario())
    inject(req, [MediaResult(item, budget_dropped_frames=4)], Settings())
    assert visual not in req.extra_user_content_parts


def test_unrecognized_or_failed_host_binding_keeps_original_visual():
    label = (
        "[Image 1: original path /does/not/exist.gif; animation converted to a 3x3 frame montage]"
    )
    req = request(["prepared.jpg"], [TextPart(text=label)])
    inject(req, [], Settings())
    assert req.image_urls == ["prepared.jpg"]
    assert not host_animations(
        request(["x"], [TextPart(text=label.replace("Image 1", "Image 99"))])
    )
