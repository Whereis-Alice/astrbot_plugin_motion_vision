"""测试环境准备。

插件源码按 AstrBot 插件目录组织，测试时把插件根目录加入 sys.path，
这样可以直接 `from motion_vision.xxx import`。

另外，纯逻辑模块不该因为「测试机上没装 AstrBot」而跑不起来，
所以这里在缺失时补一份最小替身。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _install_stubs() -> None:
    try:
        import astrbot.core.agent.message  # noqa: F401
    except Exception:
        pass
    else:
        return

    class ContentPart:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._no_save = False

        def mark_as_temp(self) -> None:
            self._no_save = True

    class TextPart(ContentPart):
        type = "text"

        def __init__(self, text: str = "") -> None:
            super().__init__(text=text)

    class _URL:
        def __init__(self, url: str = "", id: str | None = None) -> None:
            self.url = url
            self.id = id

    class ImageURLPart(ContentPart):
        type = "image_url"
        ImageURL = _URL

        def __init__(self, image_url: _URL) -> None:
            super().__init__(image_url=image_url)

    class AudioURLPart(ContentPart):
        type = "audio_url"
        AudioURL = _URL

        def __init__(self, audio_url: _URL) -> None:
            super().__init__(audio_url=audio_url)

    for name in ("astrbot", "astrbot.core", "astrbot.core.agent"):
        if name not in sys.modules:
            _module(name)

    message = _module("astrbot.core.agent.message")
    message.ContentPart = ContentPart
    message.TextPart = TextPart
    message.ImageURLPart = ImageURLPart
    message.AudioURLPart = AudioURLPart

    components = _module("astrbot.api.message_components")
    _module("astrbot.api")
    for component in ("Image", "Video", "File", "Reply", "Record"):
        setattr(components, component, type(component, (), {}))


_install_stubs()
