import json
import os
import pytest

from security_dashboard.utils import _overlay_secrets


class TestOverlaySecrets:

    def test_telegram_token_from_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token_123")
        config = {"alert_manager": {"telegram_bot_token": ""}}
        result = _overlay_secrets(config)
        assert result["alert_manager"]["telegram_bot_token"] == "test_token_123"

    def test_telegram_chat_ids_split(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_CHAT_IDS", "111,222,333")
        config = {"alert_manager": {"telegram_chat_ids": []}}
        result = _overlay_secrets(config)
        assert result["alert_manager"]["telegram_chat_ids"] == ["111", "222", "333"]

    def test_email_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv("SENDER_EMAIL", "sender@test.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret123")
        monkeypatch.setenv("EMAIL_RECIPIENTS", "a@test.com, b@test.com")
        config = {"alert_manager": {"sender_email": "", "email_password": "", "recipient_list": []}}
        result = _overlay_secrets(config)
        assert result["alert_manager"]["sender_email"] == "sender@test.com"
        assert result["alert_manager"]["email_password"] == "secret123"
        assert result["alert_manager"]["recipient_list"] == ["a@test.com", "b@test.com"]

    def test_camera_urls_from_env(self, monkeypatch):
        monkeypatch.setenv("CAMERA_1_URL", "rtsp://cam1")
        monkeypatch.setenv("CAMERA_2_URL", "rtsp://cam2")
        config = {
            "alert_manager": {},
            "cameras": [
                {"camera_id": "", "camera_name": "Cam1"},
                {"camera_id": "", "camera_name": "Cam2"},
            ],
        }
        result = _overlay_secrets(config)
        assert result["cameras"][0]["camera_id"] == "rtsp://cam1"
        assert result["cameras"][1]["camera_id"] == "rtsp://cam2"

    def test_env_vars_not_set_preserves_config(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("SENDER_EMAIL", raising=False)
        monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
        monkeypatch.delenv("EMAIL_RECIPIENTS", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_IDS", raising=False)
        monkeypatch.delenv("CAMERA_1_URL", raising=False)
        config = {
            "alert_manager": {
                "telegram_bot_token": "original_token",
                "sender_email": "original@test.com",
            },
            "cameras": [{"camera_id": "rtsp://original"}],
        }
        result = _overlay_secrets(config)
        assert result["alert_manager"]["telegram_bot_token"] == "original_token"
        assert result["cameras"][0]["camera_id"] == "rtsp://original"
