"""Structured logging for access events."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pythonjsonlogger.json import JsonFormatter

from src.config.settings import settings


class AccessLogger:
    """Logger for face recognition access events."""

    def __init__(self):
        """Initialize the access logger."""
        self.logger = logging.getLogger("access_events")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear any existing handlers

        # Configure output based on settings
        if settings.access_log_output in ("stdout", "both"):
            self._add_stdout_handler()

        if settings.access_log_output in ("file", "both"):
            self._add_file_handler()

    def _create_formatter(self) -> logging.Formatter:
        """Create the appropriate log formatter based on settings."""
        if settings.access_log_format == "json":
            return JsonFormatter(
                "%(timestamp)s %(event_type)s %(message)s",
                timestamp=True,
            )
        return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def _add_stdout_handler(self):
        """Add stdout handler for logging."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._create_formatter())
        self.logger.addHandler(handler)

    def _add_file_handler(self):
        """Add file handler for logging."""
        log_path = Path(settings.access_log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_path)
        handler.setFormatter(self._create_formatter())
        self.logger.addHandler(handler)

    def log_recognition_event(
        self,
        result: str,
        confidence: float,
        execution_time_ms: int,
        user_name: str | None = None,
        user_email: str | None = None,
        processor: str | None = None,
        door_action: str | None = None,
        camera_id: int = 0,
        **extra_fields: Any,
    ):
        """
        Log a face recognition event.

        Args:
            result: Recognition result (success/failure)
            confidence: Recognition confidence score
            execution_time_ms: Recognition execution time in milliseconds
            user_name: Recognized user's name (if successful)
            user_email: Recognized user's email (if successful)
            processor: Recognition processor used
            door_action: Door action taken (unlocked/denied)
            camera_id: Camera device ID
            **extra_fields: Additional fields to include in log
        """
        event_data = {
            "event_type": "face_recognition",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result": result,
            "confidence": round(confidence, 4),
            "execution_time_ms": execution_time_ms,
            "camera_id": camera_id,
        }

        if user_name:
            event_data["user_name"] = user_name
        if user_email:
            event_data["user_email"] = user_email
        if processor:
            event_data["processor"] = processor
        if door_action:
            event_data["door_action"] = door_action

        # Add any extra fields
        event_data.update(extra_fields)

        if settings.access_log_format == "json":
            self.logger.info("", extra=event_data)
        else:
            # Format as readable text
            text_msg = f"Recognition {result}: "
            if user_name:
                text_msg += f"{user_name} (confidence: {confidence:.2f})"
            else:
                text_msg += f"Unknown (confidence: {confidence:.2f})"
            if door_action:
                text_msg += f" - Door {door_action}"
            self.logger.info(text_msg)

    def log_cooldown_event(
        self,
        cooldown_remaining_seconds: float,
        last_recognized_user: str | None = None,
    ):
        """
        Log a cooldown status event.

        Args:
            cooldown_remaining_seconds: Seconds remaining in cooldown
            last_recognized_user: Name of last recognized user
        """
        if not settings.access_log_include_cooldown_events:
            return

        event_data = {
            "event_type": "cooldown_active",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cooldown_remaining_seconds": round(cooldown_remaining_seconds, 1),
        }

        if last_recognized_user:
            event_data["last_recognized_user"] = last_recognized_user

        if settings.access_log_format == "json":
            self.logger.info("", extra=event_data)
        else:
            text_msg = f"Cooldown active: {cooldown_remaining_seconds:.1f}s remaining"
            if last_recognized_user:
                text_msg += f" (last: {last_recognized_user})"
            self.logger.info(text_msg)

    def log_error(self, error_message: str, **extra_fields: Any):
        """
        Log an error event.

        Args:
            error_message: Error message
            **extra_fields: Additional fields to include in log
        """
        event_data = {
            "event_type": "error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error_message": error_message,
        }
        event_data.update(extra_fields)

        if settings.access_log_format == "json":
            self.logger.error("", extra=event_data)
        else:
            self.logger.error(f"Error: {error_message}")


# Global logger instance
access_logger = AccessLogger()
