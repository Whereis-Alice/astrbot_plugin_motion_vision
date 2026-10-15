from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from motion_vision import animation


def animation_bytes(fmt: str, delays: list[int], *, cover: bool = False) -> bytes:
    frames = [
        Image.new("RGB", (24, 16), (i * 50 % 256, 30, 100)) for i in range(len(delays) + cover)
    ]
    output = io.BytesIO()
    frames[0].save(
        output,
        format=fmt,
        save_all=True,
        append_images=frames[1:],
        duration=delays,
        default_image=cover,
        loop=0,
    )
    return output.getvalue()


@pytest.mark.parametrize(
    ("fmt", "delays", "cover", "stamps"),
    [
        ("GIF", [2000, 100, 100, 100], False, [0, 2, 2.1, 2.2]),
        ("PNG", [100, 200, 300], True, [0, 0.1, 0.3]),
        ("WEBP", [250, 500, 750, 1000], False, [0, 0.25, 0.75, 1.5]),
    ],
)
def test_exact_timeline_and_budget(tmp_path, fmt, delays, cover, stamps):
    seen = []

    def budget(count, size, duration):
        seen.append((count, duration))
        return count

    sample = asyncio.run(
        animation.sample_animation(
            animation_bytes(fmt, delays, cover=cover),
            tmp_path,
            64,
            90,
            budget,
        )
    )
    assert sample.total_frames == len(delays)
    assert sample.duration == pytest.approx(sum(delays) / 1000)
    assert seen == [(len(delays), sample.duration)]
    assert [f.timestamp for f in sample.frames] == pytest.approx(stamps)
    if cover:
        with Image.open(sample.frames[0].path) as first:
            assert first.getpixel((0, 0))[0] > 40  # 没有把独立封面当成第一帧。


def test_webp_sparse_sampling_still_reads_every_delay(tmp_path):
    sample = asyncio.run(
        animation.sample_animation(
            animation_bytes("WEBP", [250, 500, 750, 1000]),
            tmp_path,
            64,
            90,
            lambda *_: 2,
        )
    )
    assert sample.duration == 2.5
    assert [f.timestamp for f in sample.frames] == [0, 1.5]


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("MAX_SOURCE_FRAMES", 2, "源帧数"),
        ("MAX_DECODE_PIXELS", 500, "像素"),
        ("MAX_PIXELS_PER_FRAME", 50, "分辨率"),
        ("MAX_SOURCE_BYTES", 10, "体积"),
        ("MAX_DECODE_SECONDS", -1, "超时"),
    ],
)
def test_decode_guards_are_independent_of_output_count(
    tmp_path, monkeypatch, setting, value, message
):
    monkeypatch.setattr(animation, setting, value)
    with pytest.raises(animation.AnimationError, match=message):
        asyncio.run(
            animation.sample_animation(
                animation_bytes("GIF", [100] * 4),
                tmp_path,
                64,
                80,
                lambda *_: 1,
            )
        )


def test_cancelled_decoder_keeps_its_files_until_worker_ends(tmp_path, monkeypatch):
    import threading

    from motion_vision.resources import discard

    entered, finish = threading.Event(), threading.Event()
    source = tmp_path / "source.gif"
    source.write_bytes(b"input")
    directory = tmp_path / "output"
    directory.mkdir()

    def slow(*_args):
        entered.set()
        finish.wait(5)
        assert source.exists() and directory.exists()
        return animation.AnimationSample([], 2, 0.2, "GIF")

    monkeypatch.setattr(animation, "_sample_sync", slow)

    async def scenario():
        task = asyncio.create_task(
            animation.sample_animation(source, directory, 64, 80, lambda *_: 1)
        )
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        discard(source)
        discard(directory)
        assert source.exists() and directory.exists()
        finish.set()
        for _ in range(200):
            if not source.exists():
                break
            await asyncio.sleep(0.005)
        assert not source.exists() and not directory.exists()

    try:
        asyncio.run(scenario())
    finally:
        finish.set()
