"""Confidence-gated trigger dispatch."""

from __future__ import annotations

import logging

from src.config.settings import settings
from src.triggers.base import MatchEvent, Trigger, TriggerResult
from src.triggers.providers import create_trigger

logger = logging.getLogger(__name__)


class TriggerService:
    def __init__(self, trigger: Trigger | None = None):
        self._trigger = trigger

    @property
    def trigger(self) -> Trigger:
        if self._trigger is None:
            self._trigger = create_trigger()
        return self._trigger

    async def fire_if_authorized(self, event: MatchEvent) -> TriggerResult:
        if event.confidence < settings.trigger_confidence_threshold:
            return TriggerResult(success=False, action="denied", detail="below threshold")
        try:
            return await self.trigger.fire(event)
        except Exception as exc:
            logger.error("Trigger dispatch failed: %s", exc)
            return TriggerResult(success=False, action="error", detail=str(exc))


_service: TriggerService | None = None


def get_trigger_service() -> TriggerService:
    global _service
    if _service is None:
        _service = TriggerService()
    return _service
