from __future__ import annotations

import asyncio
import os
import time

from motion_vision.cache import CacheEntry, ResultCache, fingerprint
from motion_vision.models import SampledFrame
from motion_vision.resources import ResourceLease, discard
from motion_vision.tempstore import TempStore


def test_fingerprint_uses_all_bytes_and_not_paths(tmp_path):
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    content = b"a" * 300000 + b"tail1"
    first.write_bytes(content)
    second.write_bytes(content)
    assert fingerprint(first, None, "cfg") == fingerprint(second, None, "cfg")
    assert fingerprint(first, None, "cfg") == fingerprint(None, content, "cfg")
    assert fingerprint(None, content, "cfg") != fingerprint(None, content[:-1] + b"2", "cfg")
    assert fingerprint(first, None, "cfg") != fingerprint(first, None, "other")


def test_cache_eviction_is_deferred_until_last_reader(tmp_path):
    directory = tmp_path / "frames"
    directory.mkdir()
    path = directory / "a.jpg"
    path.write_bytes(b"jpeg")
    cache = ResultCache(capacity=1, on_evict=discard)
    cache.put("one", CacheEntry(frames=[SampledFrame(path, 0)], frames_dir=directory))
    first, second = ResourceLease(), ResourceLease()
    first.add(path)
    second.add(directory)
    cache.put("two", CacheEntry())
    assert path.exists() and cache.get("one") is None
    first.close()
    assert path.exists()
    second.close()
    second.close()
    assert not directory.exists()


def test_sweep_and_clear_protect_active_files_but_do_not_delete_future_files(tmp_path):
    store = TempStore(tmp_path, retention_seconds=60)
    old = store.download_path(".mp4")
    old.write_bytes(b"old")
    os.utime(old, (time.time() - 3600,) * 2)
    lease = ResourceLease()
    lease.add(old)
    store.sweep(min_interval=0)
    store.clear()
    new = store.download_path(".mp4")
    new.write_bytes(b"new")
    assert old.exists()
    lease.close()
    assert not old.exists()
    assert new.exists()


def test_same_content_build_is_serial_and_lock_table_is_reclaimed():
    cache = ResultCache()
    builds = 0

    async def worker():
        nonlocal builds
        async with cache.processing("key"):
            if cache.get("key") is None:
                builds += 1
                await asyncio.sleep(0.01)
                cache.put("key", CacheEntry())

    async def scenario():
        await asyncio.gather(worker(), worker(), worker())

    asyncio.run(scenario())
    assert builds == 1
    assert cache._inflight == {}


def test_byte_limit_evicts_oldest_products(tmp_path):
    cache = ResultCache(max_bytes=6)
    for i in range(2):
        path = tmp_path / f"{i}.jpg"
        path.write_bytes(b"1234")
        cache.put(str(i), CacheEntry(frames=[SampledFrame(path, 0)]))
    assert cache.get("0") is None
    assert cache.get("1") is not None


def test_discard_does_not_follow_directory_symlinks(tmp_path):
    import pytest

    target = tmp_path / "external"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink is unavailable")
    discard(link)
    assert (target / "keep.txt").exists()
    assert not link.is_symlink()
