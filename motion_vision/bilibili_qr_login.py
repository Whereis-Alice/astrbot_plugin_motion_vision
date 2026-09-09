"""B 站扫码登录。

登录是可选的辅助能力：它只为拿到需要登录才能看到的字幕/元数据服务，
不会把 Cookie 写回插件配置，也不会在状态消息或日志里回显 Cookie。
整个状态机只有一个活动二维码，轮询串行进行，避免重复执行登录请求。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

QR_GENERATE_ENDPOINT = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
QR_POLL_ENDPOINT = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
MAX_RESPONSE_BYTES = 512 * 1024
_COOKIE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_SAVED_COOKIE_NAMES = frozenset(
    {
        "SESSDATA",
        "bili_jct",
        "DedeUserID",
        "DedeUserID__ckMd5",
        "sid",
        "buvid3",
        "buvid4",
        "b_nut",
        "b_lsid",
    }
)
_BILIBILI_HOST_SUFFIXES = ("bilibili.com",)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class BilibiliQrLoginError(RuntimeError):
    """可以安全展示给管理员的扫码登录错误。"""


@dataclass(frozen=True, slots=True)
class QrLoginOutcome:
    status: str
    message: str


@dataclass(frozen=True, slots=True)
class QrLoginStart:
    qr_image_path: Path
    reused_existing_qr: bool
    task: asyncio.Task[QrLoginOutcome]


@dataclass(slots=True)
class _ActiveLogin:
    qrcode_key: str
    client: httpx.AsyncClient
    cancel_event: asyncio.Event
    task: asyncio.Task[QrLoginOutcome] | None = None


class BilibiliCredentialStore:
    """只保存扫码登录得到的 Cookie 对，且使用插件数据目录。"""

    source_label = "扫码登录保存的凭据"
    filename = "bilibili_qr_credentials.json"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.path = data_dir / self.filename

    def cookie_header(self) -> str:
        payload = self._load()
        cookies = payload.get("cookies") if isinstance(payload, dict) else None
        if not isinstance(cookies, dict):
            return ""
        return self.format_cookie_header(self.normalize_cookie_pairs(cookies))

    def has_credentials(self) -> bool:
        return bool(self.cookie_header())

    async def save_cookie_pairs(self, pairs: dict[str, Any]) -> None:
        normalized = self.normalize_cookie_pairs(pairs)
        if not normalized.get("SESSDATA"):
            raise BilibiliQrLoginError("登录成功但没有取得 SESSDATA，未保存凭据。")
        await asyncio.to_thread(self._save_sync, normalized)

    async def clear(self) -> bool:
        return await asyncio.to_thread(self._clear_sync)

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save_sync(self, cookies: dict[str, str]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        payload = {"version": 1, "saved_at": int(time.time()), "cookies": cookies}
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._restrict_permissions(temporary)
            os.replace(temporary, self.path)
            self._restrict_permissions(self.path)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

    def _clear_sync(self) -> bool:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    @staticmethod
    def normalize_cookie_pairs(values: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for raw_name, raw_value in values.items():
            name = str(raw_name).strip()
            value = str(raw_value).strip()
            if (
                name not in _SAVED_COOKIE_NAMES
                or not _COOKIE_NAME_RE.fullmatch(name)
                or not value
                or any(char in value for char in (";", "\r", "\n"))
            ):
                continue
            result[name] = value
        return result

    @staticmethod
    def format_cookie_header(values: dict[str, str]) -> str:
        return "; ".join(f"{name}={value}" for name, value in values.items())

    @staticmethod
    def _restrict_permissions(path: Path) -> None:
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)


class BilibiliQrLoginService:
    """生成二维码、低频轮询并保存扫码登录凭据。"""

    def __init__(
        self,
        data_dir: Path | httpx.AsyncClient | Any,
        credentials: BilibiliCredentialStore | Path | Any,
        *legacy_args: Any,
        enabled: bool = True,
        private_chat_only: bool = True,
        poll_interval_seconds: int = 2,
        timeout_seconds: int = 180,
        log: Any = None,
        client_factory: Any = None,
    ) -> None:
        legacy_client: httpx.AsyncClient | None = None
        if isinstance(data_dir, httpx.AsyncClient) or callable(getattr(data_dir, "request", None)):
            # 0.5.x 的内部构造函数曾把共享 AsyncClient 作为第一个参数；
            # 兼容这个调用形态，正式路径仍优先使用独立 Cookie Jar。
            legacy_client = data_dir
            if not isinstance(credentials, (str, bytes, os.PathLike)) or not legacy_args:
                raise TypeError("旧版扫码登录构造参数不完整")
            actual_data_dir = credentials
            actual_credentials = legacy_args[0]
        elif isinstance(data_dir, (str, bytes, Path)):
            actual_data_dir = data_dir
            actual_credentials = credentials
        elif legacy_args and isinstance(credentials, (str, bytes, Path)):
            # 早期版本还曾把 config 作为第一个位置参数；config 对扫码
            # 服务本身并不需要，但保留这个形态可以平滑升级旧测试/部署。
            actual_data_dir = credentials
            actual_credentials = legacy_args[0]
        else:
            raise TypeError("扫码登录需要数据目录和凭据存储对象")
        try:
            self.data_dir = Path(actual_data_dir)
        except (TypeError, ValueError) as exc:
            raise TypeError("扫码登录数据目录无效") from exc
        if not callable(getattr(actual_credentials, "save_cookie_pairs", None)):
            raise TypeError("扫码登录需要凭据存储对象")
        self.credentials = actual_credentials
        self.enabled = enabled
        self.private_chat_only = private_chat_only
        self.poll_interval_seconds = max(1, min(15, int(poll_interval_seconds)))
        self.timeout_seconds = max(30, min(600, int(timeout_seconds)))
        self.log = log or (lambda _message: None)
        self._owns_client = legacy_client is None
        self._client_factory = client_factory or (
            (lambda: legacy_client) if legacy_client is not None else self._create_client
        )
        self.qr_image_path = self.data_dir / "bilibili_qr_login.png"
        self._lock = asyncio.Lock()
        self._active: _ActiveLogin | None = None
        self._last_request = 0.0

    def configure(
        self,
        *,
        enabled: bool,
        private_chat_only: bool,
        poll_interval_seconds: int,
        timeout_seconds: int,
    ) -> None:
        self.enabled = enabled
        self.private_chat_only = private_chat_only
        self.poll_interval_seconds = max(1, min(15, int(poll_interval_seconds)))
        self.timeout_seconds = max(30, min(600, int(timeout_seconds)))

    def is_active(self) -> bool:
        active = self._active
        return bool(active and active.task and not active.task.done())

    async def start_login(self) -> QrLoginStart:
        if not self.enabled:
            raise BilibiliQrLoginError("B 站扫码登录功能当前未启用。")

        async with self._lock:
            active = self._active
            if active and active.task and not active.task.done():
                return QrLoginStart(self.qr_image_path, True, active.task)

            client = self._client_factory()
            try:
                qrcode_key, qr_url = await self._generate(client)
                await asyncio.to_thread(self._write_qr_image, qr_url, self.qr_image_path)
            except BaseException:
                if self._owns_client:
                    with contextlib.suppress(Exception):
                        await client.aclose()
                raise
            active = _ActiveLogin(qrcode_key, client, asyncio.Event())
            active.task = asyncio.create_task(
                self._complete(active),
                name="motion-vision-bilibili-qr-login",
            )
            self._active = active
            return QrLoginStart(self.qr_image_path, False, active.task)

    async def wait_for_login(self, started: QrLoginStart) -> QrLoginOutcome:
        return await asyncio.shield(started.task)

    async def cancel_login(self) -> bool:
        async with self._lock:
            active = self._active
            if not active or not active.task or active.task.done():
                return False
            active.cancel_event.set()
            return True

    async def cancel_login_and_wait(self) -> bool:
        async with self._lock:
            active = self._active
            if not active or not active.task or active.task.done():
                return False
            active.cancel_event.set()
            task = active.task
        await asyncio.shield(task)
        return True

    async def close(self) -> None:
        async with self._lock:
            active = self._active
            self._active = None
        if active and active.task and not active.task.done():
            active.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await active.task
        with contextlib.suppress(OSError):
            self.qr_image_path.unlink(missing_ok=True)

    async def clear_qr_image(self) -> None:
        with contextlib.suppress(OSError):
            await asyncio.to_thread(self.qr_image_path.unlink, missing_ok=True)

    @staticmethod
    def _create_client() -> httpx.AsyncClient:
        """为扫码流程创建独立 Cookie Jar，避免污染下载客户端。"""
        return httpx.AsyncClient(
            follow_redirects=False,
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

    async def _generate(self, client: httpx.AsyncClient) -> tuple[str, str]:
        response = await self._request(client, "GET", QR_GENERATE_ENDPOINT)
        payload = _json_payload(response, "获取二维码")
        if _safe_int(payload.get("code"), -1) != 0:
            raise BilibiliQrLoginError(_api_message(payload, "获取二维码失败"))
        data = payload.get("data")
        if not isinstance(data, dict):
            raise BilibiliQrLoginError("B 站没有返回二维码数据，请稍后重试。")
        qr_url = str(data.get("url") or "").strip()
        qrcode_key = str(data.get("qrcode_key") or "").strip()
        if not qrcode_key or not _trusted_qr_url(qr_url):
            raise BilibiliQrLoginError("B 站返回的二维码数据不完整或不受信任。")
        return qrcode_key, qr_url

    async def _complete(self, active: _ActiveLogin) -> QrLoginOutcome:
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while time.monotonic() < deadline:
                if active.cancel_event.is_set():
                    return QrLoginOutcome("cancelled", "扫码登录已取消。")
                response = await self._request(
                    active.client,
                    "GET",
                    QR_POLL_ENDPOINT,
                    params={"qrcode_key": active.qrcode_key},
                )
                payload = _json_payload(response, "检查扫码状态")
                if _safe_int(payload.get("code"), -1) != 0:
                    return QrLoginOutcome("failed", _api_message(payload, "检查扫码状态失败"))
                data = payload.get("data")
                if not isinstance(data, dict):
                    return QrLoginOutcome("failed", "B 站没有返回有效的扫码状态。")
                state = _safe_int(data.get("code"), -1)
                if state == 0:
                    cookies = _cookie_pairs(active.client)
                    await self.credentials.save_cookie_pairs(cookies)
                    return QrLoginOutcome("success", "扫码登录成功，凭据已安全保存。")
                if state == 86038:
                    return QrLoginOutcome("expired", "登录二维码已过期，请重新执行登录命令。")
                if state not in {86101, 86090}:
                    return QrLoginOutcome(
                        "failed",
                        str(data.get("message") or payload.get("message") or "扫码登录失败。"),
                    )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        active.cancel_event.wait(),
                        timeout=min(
                            self.poll_interval_seconds,
                            max(0.1, deadline - time.monotonic()),
                        ),
                    )
            return QrLoginOutcome("timeout", "等待扫码超时，请重新执行登录命令。")
        except asyncio.CancelledError:
            raise
        except BilibiliQrLoginError as exc:
            return QrLoginOutcome("failed", str(exc))
        except Exception as exc:
            self.log(f"B 站扫码登录失败：{type(exc).__name__}")
            return QrLoginOutcome("failed", "扫码登录请求失败，请稍后重试。")
        finally:
            if self._owns_client:
                with contextlib.suppress(Exception):
                    await active.client.aclose()
            async with self._lock:
                if self._active is active:
                    self._active = None

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        wait = 0.8 - (time.monotonic() - self._last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()
        try:
            response = await client.request(
                method,
                url,
                params=params,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Referer": "https://www.bilibili.com/",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Encoding": "gzip, deflate",
                },
                follow_redirects=False,
                timeout=20.0,
            )
        except httpx.HTTPError as exc:
            raise BilibiliQrLoginError("访问 B 站扫码接口时网络异常。") from exc
        if response.status_code == 429:
            raise BilibiliQrLoginError("B 站扫码接口暂时限流，请稍后再试。")
        if response.status_code < 200 or response.status_code >= 300:
            raise BilibiliQrLoginError(f"B 站扫码接口返回 HTTP {response.status_code}。")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise BilibiliQrLoginError("B 站扫码接口返回内容过大，已停止读取。")
        return response

    @staticmethod
    def _write_qr_image(qr_url: str, path: Path) -> None:
        try:
            import qrcode
        except ImportError as exc:
            raise BilibiliQrLoginError("生成二维码需要 qrcode 依赖，请先安装插件依赖。") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.png")
        try:
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=4,
            )
            qr.add_data(qr_url)
            qr.make(fit=True)
            qr.make_image(fill_color="black", back_color="white").save(temporary, "PNG")
            BilibiliCredentialStore._restrict_permissions(temporary)
            os.replace(temporary, path)
            BilibiliCredentialStore._restrict_permissions(path)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)


def _json_payload(response: httpx.Response, action: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise BilibiliQrLoginError(f"B 站{action}接口返回的不是有效 JSON。") from exc
    if not isinstance(payload, dict):
        raise BilibiliQrLoginError(f"B 站{action}接口返回了无效数据。")
    return payload


def _api_message(payload: dict[str, Any], prefix: str) -> str:
    detail = str(payload.get("message") or payload.get("msg") or "").strip()
    return f"{prefix}：{detail}" if detail else f"{prefix}。"


def _trusted_qr_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme == "https" and any(
        host == suffix or host.endswith("." + suffix) for suffix in _BILIBILI_HOST_SUFFIXES
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cookie_pairs(client: httpx.AsyncClient) -> dict[str, str]:
    """只从 B 站域名的 Cookie Jar 中取登录所需字段。"""
    result: dict[str, str] = {}
    try:
        cookies = client.cookies.jar
    except AttributeError:
        return result
    for cookie in cookies:
        domain = str(getattr(cookie, "domain", "") or "").lower().lstrip(".")
        name = str(getattr(cookie, "name", "") or "")
        value = str(getattr(cookie, "value", "") or "")
        if (
            (domain and not (domain == "bilibili.com" or domain.endswith(".bilibili.com")))
            or name not in _SAVED_COOKIE_NAMES
            or not value
        ):
            continue
        result[name] = value
    return result


__all__ = [
    "BilibiliCredentialStore",
    "BilibiliQrLoginError",
    "BilibiliQrLoginService",
    "QrLoginOutcome",
    "QrLoginStart",
]
