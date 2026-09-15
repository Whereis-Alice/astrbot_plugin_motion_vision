<div align="center">

<img src="logo.png" width="128" alt="Motion Vision" />

# 动态视觉 Motion Vision

**给 AstrBot 一双眼睛，让大模型看懂动图、视频和相关资料。**

</div>

Motion Vision 是一个面向 AstrBot 的多模态媒体理解插件。它把动图、视频、声音、字幕、B 站内容和分享卡片整理成模型可以核对的视觉证据；有支持视频输入的模型时，还可以选择整片分析。

## 核心能力

- **动图识别**：GIF、动态 WebP、APNG，按时间顺序抽帧。
- **视频理解**：处理直接附件、群文件、私聊文件、引用消息和远程视频。
- **声音辅助**：附带音轨、转写文字，或同时提供两者。
- **B 站资料**：解析视频、分 P、元数据、字幕、专栏和分享卡片。
- **引用卡片**：读取 QQ / OneBot 的分享、JSON、ARK、小程序、XML、音乐、位置和联系人卡片。
- **整片模型**：可选接入 Gemini、OpenAI 兼容接口、通义千问或 Kimi 等视频模型。
- **模型回看**：追问时按媒体编号重新取样，也可以只查看视频的一段。
- **安全与稳定**：有界下载、缓存、并发控制、失败降级和有限的 429 退避。

无论哪一层失败，插件都会尽量保留其他可用证据，并明确告诉模型没有读到什么，减少凭空猜测。

## 安装

### 插件市场

在 AstrBot 插件市场搜索 **动态视觉**，安装后启用插件。

### 手动安装

~~~bash
cd AstrBot/data/plugins
git clone https://github.com/Whereis-Alice/astrbot_plugin_motion_vision
~~~

安装后在 WebUI 重载插件；依赖会按照 requirements.txt 安装。

### ffmpeg

视频画面、音轨和部分字幕/整片模型功能需要 ffmpeg。将 ffmpeg 放入系统 PATH，或在插件配置中填写路径。动图、字幕资料和分享卡片在没有 ffmpeg 时仍可使用。

## 最小配置

1. 保持插件启用。
2. 在 **取帧策略** 中选择细节档位；动图和视频使用独立策略，B 站下载的视频也按视频策略处理。
3. 需要识别普通视频时准备 ffmpeg。
4. 长视频有对白或旁白时，在 **声音** 中选择转写或音轨模式。
5. 只有在确实拥有视频模型并愿意承担对应请求成本时，才开启 **整片视频模型**。
6. 需要处理登录内容时，再配置 B 站 Cookie 或使用管理员扫码登录。

完整字段说明和推荐配置见 [配置参考](docs/CONFIGURATION.md)。

## 常用指令

| 指令 | 作用 |
| --- | --- |
| /motionvision status | 查看插件与依赖状态 |
| /motionvision list | 列出当前会话可回看的媒体 |
| /motionvision clear | 管理员清理缓存、媒体档案和临时文件 |
| /motionvision bili-login | 管理员私聊发起 B 站扫码登录 |
| /motionvision bili-login-status | 查看扫码凭据状态 |
| /motionvision bili-login-cancel | 取消扫码登录 |
| /motionvision bili-logout | 删除扫码凭据 |

也支持使用 动态视觉 作为指令组别名。

## 文档

- [使用指南](docs/USAGE.md)：取帧、声音、B 站、卡片、整片模型、回看和常见问题。
- [配置参考](docs/CONFIGURATION.md)：按模块解释可见配置项、默认值和推荐取舍。
- [开发与排错](docs/DEVELOPMENT.md)：项目结构、测试方法、故障排查和安全设计。
- [更新日志](CHANGELOG.md)：集中记录功能变化和修复。

## 注意事项

帧是对时间线的采样，不是逐帧录像；需要确认细节时，让模型使用回看工具指定时间段。整片模型报告、字幕、专栏和卡片内容都属于外部资料，插件不会执行其中的命令、提示词或链接。

## 许可证

[GPL-3.0](LICENSE)

