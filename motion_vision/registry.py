"""会话级媒体登记表 —— 让模型有「回看」的可能。

帧默认只在当轮可见，用完就从上下文里撤掉，否则十几张图会一直躺在历史里，
长对话很快就被撑爆。代价是模型看过就忘：下一轮再问「他左手拿的是什么」，
手里已经没有画面了。

这个模块补上缺的那一环：每条动态媒体登记一个短编号，源文件在临时目录里多留
一会儿，之后模型可以拿着编号主动申请回看（见 main.py 的回看工具），还能指定
只看某一段、或者要求更多帧。

刻意保持成纯逻辑：没有 IO，不依赖 AstrBot，删文件通过 on_discard 回调外包出去，
这样可以直接单测。
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import MediaKind, format_duration

TOKEN_LENGTH = 4
"""编号长度。会话内只需要区分几条媒体，4 位十六进制足够，也方便模型原样抄写。"""

LATEST_KEYWORDS = frozenset(
    {"latest", "last", "recent", "最近", "最新", "最后", "上一个", "刚才", "刚刚"}
)
"""这些写法都当成「最近那一条」。模型不一定记得住编号，得给个兜底。"""


@dataclass
class MediaRecord:
    """一条被登记的动态媒体。"""

    token: str
    kind: MediaKind
    name: str
    path: Path | None = None
    source_url: str = ""
    context_text: str = ""
    context_label: str = ""
    duration: float | None = None
    frames_seen: int = 0
    """已经给模型看过多少帧，用于状态输出和日志。"""

    reviews: int = 0
    """被回看过几次。"""

    received_at: float = field(default_factory=time.time)
    owned_temp: bool = False
    """源文件是插件自己下载/落盘的，淘汰这条记录时可以删。"""

    @property
    def kind_label(self) -> str:
        return "动图" if self.kind is MediaKind.ANIMATION else "视频"

    @property
    def display_name(self) -> str:
        return self.name or self.kind_label

    @property
    def has_file(self) -> bool:
        return self.path is not None and self.path.exists()

    @property
    def readable(self) -> bool:
        """还能不能再看一次：本地文件还在，或者能重新下载。"""
        return self.has_file or bool(self.source_url)

    def summary(self, now: float | None = None) -> str:
        """给人看的一行摘要，用于 /motionvision status 与工具的错误提示。"""
        moment = describe_age((now if now is not None else time.time()) - self.received_at)
        pieces = [f"编号 {self.token}", self.kind_label, f"《{self.display_name}》"]
        if self.duration and self.duration > 0:
            pieces.append(format_duration(self.duration))
        pieces.append(moment)
        if not self.readable:
            pieces.append("源文件已清理")
        return " · ".join(pieces)


class MediaRegistry:
    """按会话保存最近出现过的媒体，容量有限、先进先出。

    只保留少量记录是故意的：回看的价值集中在最近几条，留太多只会让临时目录
    一直涨，也让模型面对一长串编号无从选择。
    """

    def __init__(
        self,
        per_session: int = 4,
        max_sessions: int = 64,
        on_discard: Callable[[Path | None], None] | None = None,
    ) -> None:
        self.per_session = max(1, per_session)
        self.max_sessions = max(1, max_sessions)
        self._on_discard = on_discard
        self._sessions: OrderedDict[str, list[MediaRecord]] = OrderedDict()

    # --- 写入 ---------------------------------------------------------------

    def remember(
        self,
        session: str,
        kind: MediaKind,
        name: str,
        path: Path | None = None,
        source_url: str = "",
        context_text: str = "",
        context_label: str = "",
        duration: float | None = None,
        frames_seen: int = 0,
        owned_temp: bool = False,
    ) -> MediaRecord:
        """登记一条媒体并返回记录。同一个文件重复出现时复用原编号。"""
        bucket = self._sessions.setdefault(session, [])
        self._sessions.move_to_end(session)

        existing = self._match(bucket, path, source_url)
        if existing is not None:
            existing.name = name or existing.name
            existing.path = path or existing.path
            existing.source_url = source_url or existing.source_url
            existing.context_text = context_text or existing.context_text
            existing.context_label = context_label or existing.context_label
            existing.duration = duration if duration is not None else existing.duration
            existing.frames_seen = frames_seen or existing.frames_seen
            existing.owned_temp = existing.owned_temp or owned_temp
            existing.received_at = time.time()
            bucket.remove(existing)
            bucket.insert(0, existing)
            return existing

        record = MediaRecord(
            token=self._mint(bucket),
            kind=kind,
            name=name,
            path=path,
            source_url=source_url,
            context_text=context_text,
            context_label=context_label,
            duration=duration,
            frames_seen=frames_seen,
            owned_temp=owned_temp,
        )
        bucket.insert(0, record)

        while len(bucket) > self.per_session:
            self._retire(bucket.pop())
        while len(self._sessions) > self.max_sessions:
            _, dropped = self._sessions.popitem(last=False)
            for record_out in dropped:
                self._retire(record_out)

        return record

    # --- 读取 ---------------------------------------------------------------

    def records(self, session: str) -> list[MediaRecord]:
        """该会话的记录，新的在前。"""
        return list(self._sessions.get(session, ()))

    def find(self, session: str, token: str) -> MediaRecord | None:
        """按编号取记录。留空或写「最近」都返回最新那一条。"""
        bucket = self._sessions.get(session)
        if not bucket:
            return None

        wanted = _normalize(token)
        if not wanted or wanted in LATEST_KEYWORDS:
            return bucket[0]

        for record in bucket:
            if record.token == wanted:
                return record
        # 模型可能连着别的字一起写回来（「编号 a3f1」「a3f1。」），宽容一点。
        for record in bucket:
            if record.token in wanted:
                return record
        return None

    def clear(self, session: str | None = None) -> int:
        """清空登记表，返回清掉的条数。"""
        if session is not None:
            bucket = self._sessions.pop(session, [])
            for record in bucket:
                self._retire(record)
            return len(bucket)

        total = 0
        for bucket in self._sessions.values():
            for record in bucket:
                self._retire(record)
            total += len(bucket)
        self._sessions.clear()
        return total

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._sessions.values())

    # --- 内部 ---------------------------------------------------------------

    def _mint(self, bucket: list[MediaRecord]) -> str:
        taken = {record.token for record in bucket}
        for _ in range(32):
            token = secrets.token_hex(TOKEN_LENGTH // 2)
            if token not in taken:
                return token
        return secrets.token_hex(TOKEN_LENGTH)

    @staticmethod
    def _match(bucket: list[MediaRecord], path: Path | None, source_url: str) -> MediaRecord | None:
        for record in bucket:
            if path is not None and record.path is not None and record.path == path:
                return record
            if source_url and record.source_url == source_url:
                return record
        return None

    def _retire(self, record: MediaRecord) -> None:
        if record.owned_temp and self._on_discard is not None:
            self._on_discard(record.path)
        record.path = None


def _normalize(token: str) -> str:
    """把模型可能写成各种样子的编号洗成纯 token。"""
    text = (token or "").strip().lower()
    for junk in ("编号", "id", "：", ":", "#", "《", "》", "【", "】", "。", "，", ",", ".", " "):
        text = text.replace(junk, "")
    return text


def describe_age(seconds: float) -> str:
    """把「多久以前」写成人话。"""
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时前"
    return f"{int(seconds // 86400)} 天前"


def build_memo(records: list[MediaRecord], tool_name: str) -> str:
    """生成写进对话历史的一行备忘。

    这行文字（而不是帧本身）会留在历史里：几十个字的成本，换来模型日后知道
    「这里曾经有过一个视频，编号是什么，还能再看」。

    编号用的是随机短串而不是「第 N 个」——序号会随着新媒体到达而错位，
    过两轮就指向别的东西了。
    """
    usable = [record for record in records if record.readable]
    if not usable:
        return ""

    pieces = "、".join(_memo_piece(record) for record in usable)
    return (
        f"【动态媒体档案】本条消息里有 {pieces}。"
        "画面只在当轮可见，之后就看不到了；"
        f"如果后面需要重新查看，或者想把某一段看得更细，调用 {tool_name} 工具并传入对应编号。"
    )


def _memo_piece(record: MediaRecord) -> str:
    details = []
    if record.duration and record.duration > 0:
        details.append(format_duration(record.duration))
    details.append(f"编号 {record.token}")
    return f"{record.kind_label}《{record.display_name}》（{'，'.join(details)}）"
