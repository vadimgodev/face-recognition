"""Unit tests for src/triggers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.triggers.base import MatchEvent, TriggerResult
from src.triggers.providers import GpioTrigger, LogTrigger, WebhookTrigger, create_trigger


def _event(confidence=0.95):
    return MatchEvent(user_name="alice", confidence=confidence, processor="antelopev2")


class TestLogTrigger:
    async def test_fires(self):
        result = await LogTrigger().fire(_event())
        assert result == TriggerResult(success=True, action="fired")


class TestWebhookTrigger:
    async def test_posts_event_payload(self):
        trigger = WebhookTrigger(url="http://hub.local/hook", timeout=1.0)
        response = MagicMock(status_code=200)
        with patch.object(trigger._client, "post", AsyncMock(return_value=response)) as post:
            result = await trigger.fire(_event())
        assert result.success is True and result.action == "fired"
        payload = post.call_args.kwargs["json"]
        assert payload["user_name"] == "alice" and payload["confidence"] == 0.95

    async def test_error_status_is_error_result(self):
        trigger = WebhookTrigger(url="http://hub.local/hook", timeout=1.0)
        with patch.object(
            trigger._client, "post", AsyncMock(return_value=MagicMock(status_code=500))
        ):
            result = await trigger.fire(_event())
        assert result.success is False and result.action == "error"

    async def test_exception_is_error_result(self):
        trigger = WebhookTrigger(url="http://hub.local/hook", timeout=1.0)
        with patch.object(trigger._client, "post", AsyncMock(side_effect=OSError("down"))):
            result = await trigger.fire(_event())
        assert result.success is False and result.action == "error"


class TestGpioTrigger:
    async def test_fires_without_hardware_mocked(self):
        trigger = GpioTrigger(pin=17, pulse_seconds=0)
        trigger._pulse = MagicMock()
        result = await trigger.fire(_event())
        assert result.success is True and result.action == "fired"


class TestCreateTrigger:
    def test_default_is_log(self):
        assert isinstance(create_trigger("log"), LogTrigger)

    def test_legacy_names_map(self):
        assert isinstance(create_trigger("mock"), LogTrigger)
        with patch("src.triggers.providers.settings") as s:
            s.trigger_webhook_url = "http://x/y"
            assert isinstance(create_trigger("http"), WebhookTrigger)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown trigger provider"):
            create_trigger("carrier-pigeon")

    def test_webhook_requires_url(self):
        with patch("src.triggers.providers.settings") as s:
            s.trigger_webhook_url = ""
            with pytest.raises(ValueError, match="TRIGGER_WEBHOOK_URL"):
                create_trigger("webhook")


class TestSettingsAliases:
    def _make(self, **env):
        from src.config.settings import Settings

        return Settings(_env_file=None, **env)

    def test_legacy_door_vars(self):
        s = self._make(
            DOOR_UNLOCK_PROVIDER="http",
            DOOR_UNLOCK_URL="http://d/u",
            DOOR_UNLOCK_CONFIDENCE_THRESHOLD="0.9",
        )
        assert s.trigger_provider == "webhook"
        assert s.trigger_webhook_url == "http://d/u"
        assert s.trigger_confidence_threshold == 0.9

    def test_mock_maps_to_log(self):
        assert self._make(DOOR_UNLOCK_PROVIDER="mock").trigger_provider == "log"


class TestTriggerService:
    async def test_denied_below_threshold(self):
        from src.triggers.service import TriggerService

        service = TriggerService(trigger=LogTrigger())
        with patch("src.triggers.service.settings") as s:
            s.trigger_confidence_threshold = 0.85
            result = await service.fire_if_authorized(_event(confidence=0.5))
        assert result.action == "denied" and result.success is False

    async def test_fires_at_threshold(self):
        from src.triggers.service import TriggerService

        service = TriggerService(trigger=LogTrigger())
        with patch("src.triggers.service.settings") as s:
            s.trigger_confidence_threshold = 0.85
            result = await service.fire_if_authorized(_event(confidence=0.85))
        assert result.action == "fired" and result.success is True

    async def test_misconfigured_provider_returns_error_result(self):
        from src.triggers.service import TriggerService

        service = TriggerService()
        with (
            patch("src.triggers.service.settings") as svc_settings,
            patch("src.triggers.providers.settings") as prov_settings,
        ):
            svc_settings.trigger_confidence_threshold = 0.85
            prov_settings.trigger_provider = "carrier-pigeon"
            result = await service.fire_if_authorized(_event(confidence=0.95))
        assert result.success is False and result.action == "error"
        assert "carrier-pigeon" in (result.detail or "")
