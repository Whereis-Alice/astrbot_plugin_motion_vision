"""声音能力与静音检查，不从模型名字猜测能力。"""

from __future__ import annotations

import contextlib
import wave
from pathlib import Path
from typing import Any


def provider_supports_audio(context: Any, event: Any) -> bool:
    """遵循 AstrBot modalities 约定；未知能力保守不附音轨，STT 不受影响。"""
    try:
        provider = context.get_using_provider(umo=event.unified_msg_origin)
        config = getattr(provider, "provider_config", {}) or {}
        modalities = config.get("modalities")
        if "anthropic" in type(provider).__module__.casefold():
            return False  # 此适配器不会把音轨提交为可听输入。
        return modalities == [] or (isinstance(modalities, list) and "audio" in modalities)
    except Exception:
        return False


def is_silent_pcm(path: Path) -> bool:
    """只判定整段 PCM 完全为零，不把极轻声音误报为静音。"""
    with contextlib.suppress(OSError, EOFError, wave.Error), wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            return False
        while chunk := audio.readframes(16384):
            if any(chunk):
                return False
        return True
    return False
