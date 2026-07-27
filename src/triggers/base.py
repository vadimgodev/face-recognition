"""On-match trigger abstractions: a confident recognition fires a pluggable action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass
class MatchEvent:
    user_name: str
    confidence: float
    processor: str
    user_email: str | None = None
    camera_id: int | None = None
    liveness_passed: bool | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(frozen=True)
class TriggerResult:
    success: bool
    action: str
    detail: str | None = None


class Trigger(ABC):
    name: str = "base"

    @abstractmethod
    async def fire(self, event: MatchEvent) -> TriggerResult: ...
