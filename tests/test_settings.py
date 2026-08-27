"""配置解析。"""

from __future__ import annotations

from motion_vision.settings import DETAIL_PRESETS, load_settings


def test_empty_config_falls_back_to_defaults():
    settings = load_settings({})

    assert settings.enabled is True
    assert settings.detail_level == "balanced"
    assert settings.animation_frames_override == 0
    assert settings.video_frames_override == 0
    assert settings.animation.enabled is True
    assert settings.video.max_videos_per_request == 2
    assert settings.audio.mode == "off"
    assert settings.injection.keep_frames_in_history is False
    assert settings.advanced.max_images_per_request == 48


def test_chinese_options_are_normalised():
    settings = load_settings(
        {
            "sampling": {"detail_level": "精细"},
            "audio": {"mode": "音轨和文字"},
        }
    )

    assert settings.detail_level == "detailed"
    assert settings.preset is DETAIL_PRESETS["detailed"]
    assert settings.audio.mode == "attach_and_transcribe"
    assert settings.audio.attach is True
    assert settings.audio.transcribe is True


def test_unknown_option_keeps_default():
    settings = load_settings({"sampling": {"detail_level": "超清"}})
    assert settings.detail_level == "balanced"


def test_numbers_are_clamped_and_coerced():
    settings = load_settings(
        {
            "sampling": {"video_frames_override": "999"},
            "video": {"max_videos_per_request": 0, "max_download_mb": "not a number"},
            "advanced": {"temp_retention_hours": -5},
        }
    )

    assert settings.video_frames_override == 64
    assert settings.video.max_videos_per_request == 1
    assert settings.video.max_download_mb == 100
    assert settings.advanced.temp_retention_hours == 1


def test_overrides_are_independent():
    settings = load_settings({"sampling": {"animation_frames_override": 30}})
    assert settings.animation_frames_override == 30
    assert settings.video_frames_override == 0


def test_legacy_shared_override_still_applies_to_both():
    settings = load_settings({"sampling": {"frames_override": 12}})
    assert settings.animation_frames_override == 12
    assert settings.video_frames_override == 12


def test_bool_accepts_strings():
    settings = load_settings({"enabled": "false", "animation": {"enabled": "on"}})
    assert settings.enabled is False
    assert settings.animation.enabled is True


def test_cache_signature_tracks_output_affecting_options():
    frugal = load_settings({"sampling": {"detail_level": "节省"}})
    detailed = load_settings({"sampling": {"detail_level": "精细"}})
    same = load_settings({"sampling": {"detail_level": "节省"}, "advanced": {"debug_log": True}})

    assert frugal.cache_signature != detailed.cache_signature
    # 只影响日志的选项不该让缓存失效
    assert frugal.cache_signature == same.cache_signature

    # 两个覆盖值互相独立，都要能让缓存失效
    animation_only = load_settings({"sampling": {"animation_frames_override": 30}})
    video_only = load_settings({"sampling": {"video_frames_override": 30}})
    base = load_settings({})
    assert (
        len({base.cache_signature, animation_only.cache_signature, video_only.cache_signature}) == 3
    )


def test_custom_api_requires_both_base_and_key():
    partial = load_settings({"audio": {"api_base": "https://example.com/v1"}})
    full = load_settings({"audio": {"api_base": "https://example.com/v1/", "api_key": "sk-x"}})

    assert partial.audio.use_custom_api is False
    assert full.audio.use_custom_api is True
    assert full.audio.api_base == "https://example.com/v1"
