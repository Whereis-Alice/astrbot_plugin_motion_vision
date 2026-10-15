from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest
from PIL import Image

from motion_vision.cache import ResultCache
from motion_vision.models import MediaItem, MediaKind, SampledFrame
from motion_vision.montage import render_sheets
from motion_vision.pipeline import MediaPipeline
from motion_vision.settings import AdvancedSettings, InjectionSettings, Settings, load_settings
from motion_vision.tempstore import TempStore
from tests.test_animation_timeline import animation_bytes


@pytest.mark.parametrize("count", [2, 3, 4, 5, 9])
def test_montage_pages_keep_real_frames_without_padding(tmp_path, count):
    frames = []
    for i in range(count):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (128, 64), (20 + i * 10, 40, 200)).save(path)
        frames.append(SampledFrame(path, i, i / 10, path.stat().st_size))
    pages = asyncio.run(render_sheets(frames, tmp_path / "pages", 128, 90))
    assert len(pages) == (count + 3) // 4
    assert [frame for page in pages for frame in page.frames] == frames
    assert all(0 < len(page.frames) <= 4 for page in pages)
    for page in pages:
        with Image.open(page.path) as picture:
            assert picture.height > 64  # 额外的标签区不遮挡原图。
        assert page.size_bytes == page.path.stat().st_size


def test_montage_budget_counts_real_images_and_preserves_full_cache(tmp_path):
    cache = ResultCache(on_evict=TempStore.discard)
    store = TempStore(tmp_path)
    source = tmp_path / "original.gif"
    source.write_bytes(animation_bytes("GIF", [100] * 8))
    copied = tmp_path / "new-download.gif"
    copied.write_bytes(source.read_bytes())
    settings = dataclasses.replace(
        Settings(),
        animation_frames_override=8,
        injection=InjectionSettings(frame_layout="montage"),
        advanced=AdvancedSettings(max_images_per_request=1),
    )

    def pipeline(config):
        return MediaPipeline(config, None, store, cache, None, None, logging.getLogger())

    async def scenario():
        first = (
            await pipeline(settings).run(
                [MediaItem(MediaKind.ANIMATION, "first.gif", "first", path=source)]
            )
        )[0]
        higher = dataclasses.replace(settings, advanced=AdvancedSettings(max_images_per_request=10))
        second = (
            await pipeline(higher).run(
                [MediaItem(MediaKind.ANIMATION, "second.gif", "second", path=copied)]
            )
        )[0]
        return first, second

    first, second = asyncio.run(scenario())
    assert len(first.images) == 1 and len(first.frames) == 4
    assert first.budget_dropped_frames == 4
    assert len(second.images) == 2 and len(second.frames) == 8
    assert len(cache) == 1
    assert first.frames[0].timestamp == second.frames[0].timestamp
    assert first.frames[-1].timestamp == second.frames[-1].timestamp


def test_layout_default_and_aliases_are_simple():
    assert load_settings({}).injection.frame_layout == "individual"
    assert (
        load_settings({"injection": {"frame_layout": "小宫格"}}).injection.frame_layout == "montage"
    )
    assert (
        load_settings({"injection": {"frame_layout": "bad"}}).injection.frame_layout == "individual"
    )


def test_content_cache_does_not_reuse_source_notices(tmp_path):
    store, cache = TempStore(tmp_path), ResultCache()
    content = animation_bytes("GIF", [100] * 3)

    async def scenario():
        results = []
        for notice in ("旧来源说明", "当前来源说明"):
            pipeline = MediaPipeline(
                Settings(), None, store, cache, None, None, logging.getLogger()
            )
            item = MediaItem(
                MediaKind.ANIMATION, notice, notice, data=content, source_notice=notice
            )
            results.append((await pipeline.run([item]))[0])
        return results

    first, second = asyncio.run(scenario())
    assert first.notice == "旧来源说明"
    assert second.notice == "当前来源说明"
    assert len(cache) == 1
