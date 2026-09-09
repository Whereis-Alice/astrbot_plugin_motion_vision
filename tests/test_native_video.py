from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from motion_vision.native_video import NativeVideoAnalyzer
from motion_vision.settings import NativeVideoSettings


def test_openai_compatible_video_request_contains_question(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "事实报告"}}]})

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"small-video")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    analyzer = NativeVideoAnalyzer(
        client,
        NativeVideoSettings(
            mode="on_demand",
            provider="openai",
            api_key="secret",
            api_base="https://example.test/v1",
            model="video-model",
            inline_mb=1,
        ),
    )
    try:
        report = asyncio.run(analyzer.analyze(path, "clip.mp4", "里面有什么？"))
    finally:
        asyncio.run(client.aclose())

    assert report == "事实报告"
    payload = json.loads(requests[0].content)
    assert payload["messages"][0]["content"].endswith("里面有什么？")
    assert payload["messages"][1]["content"][1]["type"] == "video_url"


def test_kimi_file_cleanup_does_not_duplicate_files_path(tmp_path: Path) -> None:
    paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file-1"})
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        if request.method == "DELETE" and request.url.path == "/v1/files/file-1":
            return httpx.Response(204)
        return httpx.Response(404)

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    analyzer = NativeVideoAnalyzer(
        client,
        NativeVideoSettings(
            mode="on_demand",
            provider="kimi",
            api_key="secret",
            api_base="https://api.moonshot.cn/v1",
            model="kimi-k3",
        ),
    )
    try:
        assert asyncio.run(analyzer.analyze(path)) == "ok"
    finally:
        asyncio.run(client.aclose())

    assert ("DELETE", "/v1/files/file-1") in paths
    assert all("/files/files/" not in path for _method, path in paths)
