"""Built-in trigger providers: log, webhook, gpio."""

from __future__ import annotations

import asyncio
import logging

import httpx

from src.config.settings import settings
from src.triggers.base import MatchEvent, Trigger, TriggerResult

logger = logging.getLogger(__name__)


class LogTrigger(Trigger):
    name = "log"

    async def fire(self, event: MatchEvent) -> TriggerResult:
        logger.info("[TRIGGER:log] match %s (confidence: %.2f)", event.user_name, event.confidence)
        return TriggerResult(success=True, action="fired")


class WebhookTrigger(Trigger):
    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def fire(self, event: MatchEvent) -> TriggerResult:
        try:
            response = await self._client.post(self.url, json=event.to_payload())
        except Exception as exc:
            logger.error("Webhook trigger failed: %s", exc)
            return TriggerResult(success=False, action="error", detail=str(exc))
        if response.status_code >= 400:
            return TriggerResult(
                success=False, action="error", detail=f"HTTP {response.status_code}"
            )
        return TriggerResult(success=True, action="fired")


class GpioTrigger(Trigger):
    name = "gpio"

    def __init__(self, pin: int = 17, pulse_seconds: float = 1.0):
        self.pin = pin
        self.pulse_seconds = pulse_seconds

    def _pulse(self) -> None:
        import RPi.GPIO as GPIO

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, GPIO.HIGH)
        try:
            import time

            time.sleep(self.pulse_seconds)
        finally:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup(self.pin)

    async def fire(self, event: MatchEvent) -> TriggerResult:
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._pulse)
        except Exception as exc:
            logger.error("GPIO trigger failed: %s", exc)
            return TriggerResult(success=False, action="error", detail=str(exc))
        return TriggerResult(success=True, action="fired")


_LEGACY_NAMES = {"mock": "log", "http": "webhook"}


def create_trigger(name: str | None = None) -> Trigger:
    resolved = _LEGACY_NAMES.get(
        name or settings.trigger_provider, name or settings.trigger_provider
    )
    if resolved == "log":
        return LogTrigger()
    if resolved == "webhook":
        if not settings.trigger_webhook_url:
            raise ValueError("TRIGGER_WEBHOOK_URL must be set for the webhook trigger")
        return WebhookTrigger(url=settings.trigger_webhook_url)
    if resolved == "gpio":
        return GpioTrigger(pin=settings.trigger_gpio_pin)
    raise ValueError(f"Unknown trigger provider: {resolved!r}")
