from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import struct
import wave
from types import SimpleNamespace

import pytest

from motion_vision.audio import is_silent_pcm, provider_supports_audio
from motion_vision.ffmpeg import FfmpegRunner, FfmpegTools, ProbeResult, _parse_ffprobe
from motion_vision.inject import inject
from motion_vision.models import AudioClip, MediaItem, MediaKind, MediaResult
from motion_vision.pipeline import MediaPipeline
from motion_vision.settings import AudioSettings, Settings
from motion_vision.tempstore import TempStore
from tests.test_host_images import request


def wav(path, *, silent=False, channels=2):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(
            (
                bytes(2 * channels)
                if silent
                else struct.pack("<" + "h" * channels, *([1, -1][:channels]))
            )
            * 240
        )


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [(None, False), ([], True), (["image"], False), (["audio", "image"], True)],
)
def test_audio_capability_uses_provider_metadata(modalities, expected):
    provider = SimpleNamespace(provider_config={"modalities": modalities})
    context = SimpleNamespace(get_using_provider=lambda **_: provider)
    assert provider_supports_audio(context, SimpleNamespace(unified_msg_origin="group")) is expected


def test_silence_detection_does_not_drop_quiet_or_opposite_phase_channels(tmp_path):
    silent, quiet = tmp_path / "silent.wav", tmp_path / "quiet.wav"
    wav(silent, silent=True)
    wav(quiet)
    assert is_silent_pcm(silent)
    assert not is_silent_pcm(quiet)


def test_probe_prefers_video_timeline_and_tracks_audio_offset():
    probe = _parse_ffprobe(
        json.dumps(
            {
                "format": {"duration": "20"},
                "streams": [
                    {
                        "codec_type": "video",
                        "duration": "10",
                        "start_time": "1.0",
                        "width": 640,
                        "height": 360,
                    },
                    {"codec_type": "audio", "duration": "18", "start_time": "1.25"},
                ],
            }
        )
    )
    assert probe.duration == 10
    assert probe.audio_start - probe.video_start == 0.25
    assert probe.timing_known


class Runner:
    def __init__(self, silent=False):
        self.silent = silent
        self.calls = []

    async def extract_audio(self, source, destination, seconds, **kwargs):
        self.calls.append(kwargs)
        wav(destination, silent=self.silent, channels=kwargs.get("channels", 1))
        return AudioClip(destination, seconds, kwargs.get("start_seconds", 0))


@pytest.mark.parametrize(
    ("supports", "silent", "expected"),
    [(False, False, "unsupported"), (True, True, "silent"), (True, False, "ready")],
)
def test_audio_attachment_states(tmp_path, supports, silent, expected):
    settings = dataclasses.replace(Settings(), audio=AudioSettings(mode="attach"))
    runner = Runner(silent)
    pipeline = MediaPipeline(
        settings,
        runner,
        TempStore(tmp_path),
        None,
        None,
        None,
        logging.getLogger(),
        supports_audio=supports,
    )
    item = MediaItem(MediaKind.VIDEO, "clip.mp4", "clip", path=tmp_path / "video.mp4")
    result = MediaResult(item)
    probe = ProbeResult(
        duration=10, has_audio=True, video_start=1, audio_start=1.25, timing_known=True
    )
    try:
        asyncio.run(pipeline._attach_audio(item, probe, result, lambda: 100))
        assert result.audio_status == expected
        assert bool(result.audio) == (expected == "ready")
        if supports:
            assert runner.calls[0]["channels"] == 2
            assert runner.calls[0]["audio_offset"] == 0.25
        else:
            assert not runner.calls
    finally:
        pipeline.lease.close()


def test_unsupported_original_audio_still_allows_stt(tmp_path, monkeypatch):
    settings = dataclasses.replace(Settings(), audio=AudioSettings(mode="attach_and_transcribe"))
    runner = Runner()
    pipeline = MediaPipeline(
        settings, runner, TempStore(tmp_path), None, None, None, logging.getLogger()
    )

    async def stt(path, *_):
        assert path.exists()
        return "语音内容"

    monkeypatch.setattr("motion_vision.pipeline.transcribe", stt)
    item = MediaItem(MediaKind.VIDEO, "clip.mp4", "clip", path=tmp_path / "video.mp4")
    result = MediaResult(item)
    try:
        asyncio.run(
            pipeline._attach_audio(
                item, ProbeResult(duration=10, has_audio=True), result, lambda: 100
            )
        )
        assert result.transcript == "语音内容"
        assert result.audio is None and result.audio_status == "unsupported"
        assert len(runner.calls) == 2  # 双声道原音和 STT 专用单声道分开。
    finally:
        pipeline.lease.close()


def test_multiple_audio_attachments_have_unambiguous_media_ids(tmp_path):
    clips = []
    for token in ("aaaa", "bbbb"):
        item = MediaItem(MediaKind.VIDEO, "same.mp4", token, media_id=token)
        clips.append(
            MediaResult(item, audio=AudioClip(tmp_path / f"{token}.wav", 5), audio_status="ready")
        )
    req = request()
    req.audio_urls = ["earlier.wav"]
    inject(req, clips, Settings())
    texts = "\n".join(getattr(p, "text", "") for p in req.extra_user_content_parts)
    assert "第 2 个音频附件对应媒体 aaaa" in texts
    assert "第 3 个音频附件对应媒体 bbbb" in texts


def test_audio_extraction_builds_offset_filter_without_shell(tmp_path, monkeypatch):
    import subprocess

    runner = FfmpegRunner(FfmpegTools(ffmpeg="ffmpeg"))
    captured = []
    target = tmp_path / "sound.wav"

    async def run(args, _timeout):
        captured.extend(args)
        target.write_bytes(bytes(2048))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(runner, "_run", run)
    asyncio.run(
        runner.extract_audio(
            tmp_path / "source.mp4", target, 5, channels=2, sample_rate=24000, audio_offset=0.25
        )
    )
    assert "adelay=250:all=1" in captured[captured.index("-af") + 1]
    assert captured[captured.index("-ac") + 1] == "2"
