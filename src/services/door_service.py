"""Door unlock service with provider abstraction."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


class DoorUnlockProvider(ABC):
    """Abstract base class for door unlock providers."""

    @abstractmethod
    async def unlock(self, user_name: str, confidence: float) -> bool:
        """
        Trigger door unlock.

        Args:
            user_name: Name of the recognized user
            confidence: Recognition confidence score

        Returns:
            True if unlock was successful, False otherwise
        """
        pass


class MockDoorProvider(DoorUnlockProvider):
    """Mock provider for development/testing."""

    async def unlock(self, user_name: str, confidence: float) -> bool:
        """Simulate door unlock."""
        logger.info(
            f"[MOCK] Door unlocked for {user_name} (confidence: {confidence:.2f})"
        )
        return True


class HttpDoorProvider(DoorUnlockProvider):
    """HTTP-based door unlock provider."""

    def __init__(self, unlock_url: str, timeout: int = 5):
        """
        Initialize HTTP provider.

        Args:
            unlock_url: URL to send unlock request to
            timeout: Request timeout in seconds
        """
        self.unlock_url = unlock_url
        self.timeout = timeout

    async def unlock(self, user_name: str, confidence: float) -> bool:
        """
        Send HTTP request to unlock door.

        Args:
            user_name: Name of the recognized user
            confidence: Recognition confidence score

        Returns:
            True if unlock was successful, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.unlock_url,
                    json={
                        "user_name": user_name,
                        "confidence": confidence,
                        "action": "unlock",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                logger.info(
                    f"Door unlocked via HTTP for {user_name} (confidence: {confidence:.2f})"
                )
                return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to unlock door via HTTP: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error unlocking door: {e}")
            return False


class GpioDoorProvider(DoorUnlockProvider):
    """GPIO-based door unlock provider for Raspberry Pi."""

    def __init__(self, pin: int = 17, pulse_duration: float = 1.0):
        """
        Initialize GPIO provider.

        Args:
            pin: GPIO pin number
            pulse_duration: Duration to hold the unlock signal (seconds)
        """
        self.pin = pin
        self.pulse_duration = pulse_duration
        self._gpio_available = False

        try:
            import RPi.GPIO as GPIO

            self.GPIO = GPIO
            self.GPIO.setmode(GPIO.BCM)
            self.GPIO.setup(self.pin, GPIO.OUT)
            self.GPIO.output(self.pin, GPIO.LOW)
            self._gpio_available = True
            logger.info(f"GPIO provider initialized on pin {pin}")
        except ImportError:
            logger.warning(
                "RPi.GPIO not available. GPIO provider will not function."
            )
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")

    async def unlock(self, user_name: str, confidence: float) -> bool:
        """
        Trigger GPIO pin to unlock door.

        Args:
            user_name: Name of the recognized user
            confidence: Recognition confidence score

        Returns:
            True if unlock was successful, False otherwise
        """
        if not self._gpio_available:
            logger.error("GPIO not available")
            return False

        try:
            import asyncio

            # Set pin HIGH to unlock
            self.GPIO.output(self.pin, self.GPIO.HIGH)
            logger.info(
                f"Door unlocked via GPIO pin {self.pin} for {user_name} "
                f"(confidence: {confidence:.2f})"
            )

            # Wait for pulse duration
            await asyncio.sleep(self.pulse_duration)

            # Set pin back to LOW
            self.GPIO.output(self.pin, self.GPIO.LOW)
            return True
        except Exception as e:
            logger.error(f"Failed to unlock door via GPIO: {e}")
            return False

    def cleanup(self):
        """Clean up GPIO resources."""
        if self._gpio_available:
            self.GPIO.cleanup()


class DoorService:
    """Service for managing door unlock operations."""

    def __init__(self, provider: Optional[DoorUnlockProvider] = None):
        """
        Initialize door service.

        Args:
            provider: Door unlock provider instance (auto-detected from settings if None)
        """
        if provider is None:
            provider = self._create_provider_from_settings()
        self.provider = provider

    def _create_provider_from_settings(self) -> DoorUnlockProvider:
        """Create provider instance based on settings."""
        provider_type = settings.door_unlock_provider.lower()

        if provider_type == "mock":
            return MockDoorProvider()
        elif provider_type == "http":
            return HttpDoorProvider(
                unlock_url=settings.door_unlock_url,
                timeout=5,
            )
        elif provider_type == "gpio":
            return GpioDoorProvider(pin=17, pulse_duration=1.0)
        else:
            logger.warning(
                f"Unknown door provider '{provider_type}', using mock provider"
            )
            return MockDoorProvider()

    async def unlock_if_authorized(
        self, user_name: str, confidence: float
    ) -> tuple[bool, str]:
        """
        Unlock door if confidence meets threshold.

        Args:
            user_name: Name of the recognized user
            confidence: Recognition confidence score

        Returns:
            Tuple of (success: bool, action: str)
            action is one of: "unlocked", "denied"
        """
        if confidence >= settings.door_unlock_confidence_threshold:
            success = await self.provider.unlock(user_name, confidence)
            return success, "unlocked" if success else "denied"
        else:
            logger.info(
                f"Door unlock denied for {user_name} - "
                f"confidence {confidence:.2f} below threshold "
                f"{settings.door_unlock_confidence_threshold}"
            )
            return False, "denied"


# Global door service instance
door_service = DoorService()
