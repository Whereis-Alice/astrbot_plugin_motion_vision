from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from motion_vision.bilibili_qr_login import (
    BilibiliCredentialStore,
    BilibiliQrLoginService,
)


def test_credential_store_keeps_only_safe_cookie_names(tmp_path: Path) -> None:
    store = BilibiliCredentialStore(tmp_path)
    asyncio.run(
        store.save_cookie_pairs(
            {
                "SESSDATA": "abc",
                "bili_jct": "jct",
                "not_a_bilibili_cookie": "drop",
                "sid": "with;semicolon",
            }
        )
    )

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["cookies"] == {"SESSDATA": "abc", "bili_jct": "jct"}
    assert store.cookie_header() == "SESSDATA=abc; bili_jct=jct"


def test_qr_login_polls_serially_and_saves_cookie(tmp_path: Path) -> None:
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path.endswith("/generate"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "url": "https://passport.bilibili.com/h5/abc",
                        "qrcode_key": "key",
                    },
                },
            )
        poll_count += 1
        return httpx.Response(
            200,
            json={"code": 0, "data": {"code": 0, "message": "ok"}},
            headers={"set-cookie": "SESSDATA=abc; Domain=.bilibili.com; Path=/"},
        )

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    credentials = BilibiliCredentialStore(tmp_path)
    service = BilibiliQrLoginService(
        tmp_path,
        credentials,
        poll_interval_seconds=1,
        timeout_seconds=30,
        client_factory=factory,
    )

    async def run_login():
        started = await service.start_login()
        outcome = await service.wait_for_login(started)
        image_exists = started.qr_image_path.is_file()
        await service.close()
        return started, outcome, image_exists

    _started, outcome, image_exists = asyncio.run(run_login())

    assert outcome.status == "success"
    assert poll_count == 1
    assert credentials.has_credentials()
    assert image_exists
