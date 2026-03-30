"""Tests for modules with low/zero coverage:

- src/utils/access_logger.py (AccessLogger)
- src/cache/redis_client.py (RedisCache)
- src/utils/startup_validation.py (validate_liveness_configuration, validate_startup_requirements)
- src/providers/liveness_base.py (SpoofingType, LivenessResult)
- src/storage/factory.py (StorageFactory)
- src/storage/s3.py (S3StorageBackend)
"""

import hashlib
import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# SpoofingType + LivenessResult (src/providers/liveness_base.py)
# ============================================================================
class TestSpoofingType:
    """SpoofingType is a StrEnum with known members."""

    def test_real_value(self):
        from src.providers.liveness_base import SpoofingType

        assert SpoofingType.REAL == "real"

    def test_print_value(self):
        from src.providers.liveness_base import SpoofingType

        assert SpoofingType.PRINT == "print"

    def test_video_value(self):
        from src.providers.liveness_base import SpoofingType

        assert SpoofingType.VIDEO == "video"

    def test_mask_value(self):
        from src.providers.liveness_base import SpoofingType

        assert SpoofingType.MASK == "mask"

    def test_unknown_value(self):
        from src.providers.liveness_base import SpoofingType

        assert SpoofingType.UNKNOWN == "unknown"

    def test_is_str_subclass(self):
        from src.providers.liveness_base import SpoofingType

        assert isinstance(SpoofingType.REAL, str)

    def test_all_members_count(self):
        from src.providers.liveness_base import SpoofingType

        assert len(SpoofingType) == 5

    def test_string_comparison(self):
        from src.providers.liveness_base import SpoofingType

        assert SpoofingType.REAL == "real"
        assert SpoofingType.MASK != "print"


class TestLivenessResult:
    """LivenessResult is a dataclass."""

    def test_create_real_result(self):
        from src.providers.liveness_base import LivenessResult, SpoofingType

        result = LivenessResult(
            is_real=True,
            confidence=0.95,
            spoofing_type=SpoofingType.REAL,
        )
        assert result.is_real is True
        assert result.confidence == 0.95
        assert result.spoofing_type == SpoofingType.REAL
        assert result.details is None

    def test_create_spoof_result_with_details(self):
        from src.providers.liveness_base import LivenessResult, SpoofingType

        details = {"model": "MiniFASNet", "score": 0.12}
        result = LivenessResult(
            is_real=False,
            confidence=0.12,
            spoofing_type=SpoofingType.PRINT,
            details=details,
        )
        assert result.is_real is False
        assert result.confidence == 0.12
        assert result.spoofing_type == SpoofingType.PRINT
        assert result.details == details

    def test_details_default_none(self):
        from src.providers.liveness_base import LivenessResult, SpoofingType

        result = LivenessResult(is_real=True, confidence=0.9, spoofing_type=SpoofingType.REAL)
        assert result.details is None

    def test_equality(self):
        from src.providers.liveness_base import LivenessResult, SpoofingType

        a = LivenessResult(is_real=True, confidence=0.9, spoofing_type=SpoofingType.REAL)
        b = LivenessResult(is_real=True, confidence=0.9, spoofing_type=SpoofingType.REAL)
        assert a == b

    def test_inequality_different_confidence(self):
        from src.providers.liveness_base import LivenessResult, SpoofingType

        a = LivenessResult(is_real=True, confidence=0.9, spoofing_type=SpoofingType.REAL)
        b = LivenessResult(is_real=True, confidence=0.5, spoofing_type=SpoofingType.REAL)
        assert a != b


# ============================================================================
# AccessLogger (src/utils/access_logger.py)
# ============================================================================
class TestAccessLoggerInit:
    """AccessLogger configures handlers based on settings."""

    @patch("src.utils.access_logger.settings")
    def test_stdout_handler_added_for_stdout(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "json"

        from src.utils.access_logger import AccessLogger

        logger = AccessLogger()
        assert len(logger.logger.handlers) == 1
        assert isinstance(logger.logger.handlers[0], logging.StreamHandler)

    @patch("src.utils.access_logger.settings")
    def test_file_handler_added_for_file(self, mock_settings, tmp_path):
        log_file = tmp_path / "logs" / "access.log"
        mock_settings.access_log_output = "file"
        mock_settings.access_log_format = "text"
        mock_settings.access_log_file_path = str(log_file)

        from src.utils.access_logger import AccessLogger

        logger = AccessLogger()
        assert len(logger.logger.handlers) == 1
        assert isinstance(logger.logger.handlers[0], logging.FileHandler)

    @patch("src.utils.access_logger.settings")
    def test_both_handlers_added(self, mock_settings, tmp_path):
        log_file = tmp_path / "logs" / "access.log"
        mock_settings.access_log_output = "both"
        mock_settings.access_log_format = "json"
        mock_settings.access_log_file_path = str(log_file)

        from src.utils.access_logger import AccessLogger

        logger = AccessLogger()
        assert len(logger.logger.handlers) == 2

    @patch("src.utils.access_logger.settings")
    def test_no_handlers_for_unknown_output(self, mock_settings):
        mock_settings.access_log_output = "none"
        mock_settings.access_log_format = "text"

        from src.utils.access_logger import AccessLogger

        logger = AccessLogger()
        assert len(logger.logger.handlers) == 0


class TestAccessLoggerFormatter:
    """_create_formatter returns JsonFormatter or text Formatter."""

    @patch("src.utils.access_logger.settings")
    def test_json_formatter(self, mock_settings):
        mock_settings.access_log_output = "none"
        mock_settings.access_log_format = "json"

        from pythonjsonlogger.json import JsonFormatter

        from src.utils.access_logger import AccessLogger

        logger = AccessLogger()
        fmt = logger._create_formatter()
        assert isinstance(fmt, JsonFormatter)

    @patch("src.utils.access_logger.settings")
    def test_text_formatter(self, mock_settings):
        mock_settings.access_log_output = "none"
        mock_settings.access_log_format = "text"

        from src.utils.access_logger import AccessLogger

        logger = AccessLogger()
        fmt = logger._create_formatter()
        assert isinstance(fmt, logging.Formatter)
        # Not a JsonFormatter — just a plain Formatter
        from pythonjsonlogger.json import JsonFormatter

        assert not isinstance(fmt, JsonFormatter)


class TestAccessLoggerRecognitionEvent:
    """log_recognition_event in both JSON and text modes."""

    @patch("src.utils.access_logger.settings")
    def test_json_format_logs_info(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "json"

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_recognition_event(
                result="success",
                confidence=0.9512,
                execution_time_ms=42,
                user_name="Alice",
                user_email="alice@example.com",
                processor="aws_rekognition",
                door_action="unlocked",
                camera_id=1,
                custom_field="value",
            )
            mock_info.assert_called_once()
            _, kwargs = mock_info.call_args
            extra = kwargs["extra"]
            assert extra["event_type"] == "face_recognition"
            assert extra["result"] == "success"
            assert extra["confidence"] == 0.9512
            assert extra["execution_time_ms"] == 42
            assert extra["user_name"] == "Alice"
            assert extra["user_email"] == "alice@example.com"
            assert extra["processor"] == "aws_rekognition"
            assert extra["door_action"] == "unlocked"
            assert extra["camera_id"] == 1
            assert extra["custom_field"] == "value"

    @patch("src.utils.access_logger.settings")
    def test_text_format_with_user_name(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "text"

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_recognition_event(
                result="success",
                confidence=0.95,
                execution_time_ms=100,
                user_name="Bob",
                door_action="unlocked",
            )
            msg = mock_info.call_args[0][0]
            assert "Recognition success" in msg
            assert "Bob" in msg
            assert "0.95" in msg
            assert "Door unlocked" in msg

    @patch("src.utils.access_logger.settings")
    def test_text_format_unknown_user(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "text"

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_recognition_event(
                result="failure",
                confidence=0.32,
                execution_time_ms=50,
            )
            msg = mock_info.call_args[0][0]
            assert "Unknown" in msg
            assert "0.32" in msg

    @patch("src.utils.access_logger.settings")
    def test_text_format_no_door_action(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "text"

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_recognition_event(
                result="success",
                confidence=0.88,
                execution_time_ms=60,
                user_name="Charlie",
            )
            msg = mock_info.call_args[0][0]
            assert "Door" not in msg

    @patch("src.utils.access_logger.settings")
    def test_json_format_optional_fields_omitted(self, mock_settings):
        """When optional fields are None, they are not included in extra."""
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "json"

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_recognition_event(
                result="failure",
                confidence=0.2,
                execution_time_ms=30,
            )
            extra = mock_info.call_args[1]["extra"]
            assert "user_name" not in extra
            assert "user_email" not in extra
            assert "processor" not in extra
            assert "door_action" not in extra


class TestAccessLoggerCooldownEvent:
    """log_cooldown_event in both modes."""

    @patch("src.utils.access_logger.settings")
    def test_cooldown_skipped_when_disabled(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "json"
        mock_settings.access_log_include_cooldown_events = False

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_cooldown_event(cooldown_remaining_seconds=3.5)
            mock_info.assert_not_called()

    @patch("src.utils.access_logger.settings")
    def test_cooldown_json_format(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "json"
        mock_settings.access_log_include_cooldown_events = True

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_cooldown_event(
                cooldown_remaining_seconds=3.567,
                last_recognized_user="Dave",
            )
            extra = mock_info.call_args[1]["extra"]
            assert extra["event_type"] == "cooldown_active"
            assert extra["cooldown_remaining_seconds"] == 3.6
            assert extra["last_recognized_user"] == "Dave"

    @patch("src.utils.access_logger.settings")
    def test_cooldown_json_no_last_user(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "json"
        mock_settings.access_log_include_cooldown_events = True

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_cooldown_event(cooldown_remaining_seconds=2.0)
            extra = mock_info.call_args[1]["extra"]
            assert "last_recognized_user" not in extra

    @patch("src.utils.access_logger.settings")
    def test_cooldown_text_format_with_user(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "text"
        mock_settings.access_log_include_cooldown_events = True

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_cooldown_event(
                cooldown_remaining_seconds=4.2,
                last_recognized_user="Eve",
            )
            msg = mock_info.call_args[0][0]
            assert "4.2" in msg
            assert "Eve" in msg

    @patch("src.utils.access_logger.settings")
    def test_cooldown_text_format_without_user(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "text"
        mock_settings.access_log_include_cooldown_events = True

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "info") as mock_info:
            al.log_cooldown_event(cooldown_remaining_seconds=1.0)
            msg = mock_info.call_args[0][0]
            assert "1.0" in msg
            assert "last" not in msg.lower()


class TestAccessLoggerError:
    """log_error in both modes."""

    @patch("src.utils.access_logger.settings")
    def test_error_json_format(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "json"

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "error") as mock_error:
            al.log_error("Provider timeout", provider="aws", retries=3)
            extra = mock_error.call_args[1]["extra"]
            assert extra["event_type"] == "error"
            assert extra["error_message"] == "Provider timeout"
            assert extra["provider"] == "aws"
            assert extra["retries"] == 3

    @patch("src.utils.access_logger.settings")
    def test_error_text_format(self, mock_settings):
        mock_settings.access_log_output = "stdout"
        mock_settings.access_log_format = "text"

        from src.utils.access_logger import AccessLogger

        al = AccessLogger()
        with patch.object(al.logger, "error") as mock_error:
            al.log_error("Something went wrong")
            msg = mock_error.call_args[0][0]
            assert "Error: Something went wrong" in msg


# ============================================================================
# RedisCache (src/cache/redis_client.py)
# ============================================================================
class TestRedisCacheInit:
    """RedisCache constructor and initialize/close lifecycle."""

    @patch("src.cache.redis_client.settings")
    def test_init_disabled(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        assert cache._enabled is False
        assert cache._client is None
        assert cache._pool is None

    @patch("src.cache.redis_client.settings")
    def test_init_enabled(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        assert cache._enabled is True

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_initialize_when_disabled(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        await cache.initialize()
        assert cache._client is None

    @patch("src.cache.redis_client.ConnectionPool")
    @patch("src.cache.redis_client.redis.Redis")
    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_settings, mock_redis_cls, mock_pool_cls):
        mock_settings.redis_enabled = True
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.redis_max_connections = 10

        mock_pool = MagicMock()
        mock_pool_cls.from_url.return_value = mock_pool

        mock_client = AsyncMock()
        mock_redis_cls.return_value = mock_client

        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        await cache.initialize()

        assert cache._client is mock_client
        assert cache._pool is mock_pool
        mock_client.ping.assert_awaited_once()

    @patch("src.cache.redis_client.ConnectionPool")
    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, mock_settings, mock_pool_cls):
        mock_settings.redis_enabled = True
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.redis_max_connections = 10

        mock_pool_cls.from_url.side_effect = Exception("Connection refused")

        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        await cache.initialize()

        assert cache._enabled is False
        assert cache._client is None


class TestRedisCacheClose:
    """close() shuts down client and pool."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_close_with_client_and_pool(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._pool = AsyncMock()

        await cache.close()
        cache._client.close.assert_awaited_once()
        cache._pool.disconnect.assert_awaited_once()

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_close_without_client(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        # Should not raise
        await cache.close()


class TestRedisCacheIsAvailable:
    """_is_available returns True only when enabled AND client exists."""

    @patch("src.cache.redis_client.settings")
    def test_available_when_enabled_with_client(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = MagicMock()
        assert cache._is_available() is True

    @patch("src.cache.redis_client.settings")
    def test_not_available_when_disabled(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        assert cache._is_available() is False

    @patch("src.cache.redis_client.settings")
    def test_not_available_when_no_client(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = None
        assert cache._is_available() is False


class TestRedisCacheGet:
    """get() with available/unavailable states and error handling."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_when_unavailable(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        result = await cache.get("key")
        assert result is None

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_returns_value(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.return_value = "cached_value"

        result = await cache.get("my_key")
        assert result == "cached_value"
        cache._client.get.assert_awaited_once_with("my_key")

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.return_value = None

        result = await cache.get("missing_key")
        assert result is None

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_handles_exception(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.side_effect = Exception("Redis error")

        result = await cache.get("key")
        assert result is None


class TestRedisCacheSet:
    """set() with available/unavailable states and error handling."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_set_when_unavailable(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        result = await cache.set("key", "value")
        assert result is False

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_set_success(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        result = await cache.set("key", "value", ex=300)
        assert result is True
        cache._client.set.assert_awaited_once_with("key", "value", ex=300)

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_set_without_expiration(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        result = await cache.set("key", "value")
        assert result is True
        cache._client.set.assert_awaited_once_with("key", "value", ex=None)

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_set_handles_exception(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.set.side_effect = Exception("Redis write error")

        result = await cache.set("key", "value")
        assert result is False


class TestRedisCacheSetex:
    """setex() delegates to set()."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_setex_delegates_to_set(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        result = await cache.setex("key", 60, "value")
        assert result is True
        cache._client.set.assert_awaited_once_with("key", "value", ex=60)


class TestRedisCacheDelete:
    """delete() with available/unavailable states."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_delete_when_unavailable(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        result = await cache.delete("key1", "key2")
        assert result is False

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_delete_success(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        result = await cache.delete("key1", "key2")
        assert result is True
        cache._client.delete.assert_awaited_once_with("key1", "key2")

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_delete_handles_exception(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.delete.side_effect = Exception("Redis delete error")

        result = await cache.delete("key1")
        assert result is False


class TestRedisCacheExists:
    """exists() with available/unavailable states."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_exists_when_unavailable(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        result = await cache.exists("key")
        assert result is False

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_exists_returns_true(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.exists.return_value = 1

        result = await cache.exists("key")
        assert result is True

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_exists_returns_false(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.exists.return_value = 0

        result = await cache.exists("key")
        assert result is False

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_exists_handles_exception(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.exists.side_effect = Exception("Redis error")

        result = await cache.exists("key")
        assert result is False


class TestRedisCacheGetJson:
    """get_json() parses JSON or returns None."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_json_returns_parsed(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.return_value = '{"name": "Alice", "score": 0.95}'

        result = await cache.get_json("user:1")
        assert result == {"name": "Alice", "score": 0.95}

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_json_returns_none_on_miss(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.return_value = None

        result = await cache.get_json("missing")
        assert result is None

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_json_returns_none_on_invalid_json(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.return_value = "not valid json{"

        result = await cache.get_json("bad_json_key")
        assert result is None

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_get_json_when_unavailable(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        result = await cache.get_json("key")
        assert result is None


class TestRedisCacheSetJson:
    """set_json() serializes and stores."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_set_json_success(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        result = await cache.set_json("key", {"data": [1, 2, 3]}, ex=600)
        assert result is True
        call_args = cache._client.set.call_args
        stored_value = call_args[0][1]
        assert json.loads(stored_value) == {"data": [1, 2, 3]}

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_set_json_non_serializable(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        # set() is not callable for non-serializable, but json.dumps fails first
        result = await cache.set_json("key", {1, 2, 3})  # sets are not JSON serializable
        assert result is False

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_set_json_when_unavailable(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        result = await cache.set_json("key", {"a": 1})
        assert result is False


class TestRedisCacheHashBytes:
    """hash_bytes() is a static method returning SHA256 hex."""

    def test_hash_bytes_returns_sha256(self):
        from src.cache.redis_client import RedisCache

        data = b"test image data"
        expected = hashlib.sha256(data).hexdigest()
        assert RedisCache.hash_bytes(data) == expected

    def test_hash_bytes_empty(self):
        from src.cache.redis_client import RedisCache

        expected = hashlib.sha256(b"").hexdigest()
        assert RedisCache.hash_bytes(b"") == expected

    def test_hash_bytes_deterministic(self):
        from src.cache.redis_client import RedisCache

        data = b"\x89PNG\r\n\x1a\n"
        assert RedisCache.hash_bytes(data) == RedisCache.hash_bytes(data)


class TestRedisCacheInvalidatePattern:
    """invalidate_pattern() scans and deletes matching keys."""

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_invalidate_when_unavailable(self, mock_settings):
        mock_settings.redis_enabled = False
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        result = await cache.invalidate_pattern("face:*")
        assert result == 0

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_invalidate_with_matching_keys(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        # Mock scan_iter to yield keys
        async def fake_scan_iter(match=None):
            for key in ["face:1", "face:2", "face:3"]:
                yield key

        cache._client.scan_iter = fake_scan_iter

        result = await cache.invalidate_pattern("face:*")
        assert result == 3
        cache._client.delete.assert_awaited_once_with("face:1", "face:2", "face:3")

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_invalidate_no_matching_keys(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        async def fake_scan_iter(match=None):
            return
            yield  # make it an async generator

        cache._client.scan_iter = fake_scan_iter

        result = await cache.invalidate_pattern("nonexistent:*")
        assert result == 0

    @patch("src.cache.redis_client.settings")
    @pytest.mark.asyncio
    async def test_invalidate_handles_exception(self, mock_settings):
        mock_settings.redis_enabled = True
        from src.cache.redis_client import RedisCache

        cache = RedisCache()
        cache._client = AsyncMock()

        async def failing_scan_iter(match=None):
            raise Exception("Scan failed")
            yield  # noqa: F841 -- unreachable but makes it an async generator

        cache._client.scan_iter = failing_scan_iter

        result = await cache.invalidate_pattern("fail:*")
        assert result == 0


class TestGetRedisClient:
    """get_redis_client() returns singleton."""

    @patch("src.cache.redis_client.settings")
    def test_returns_redis_cache_instance(self, mock_settings):
        mock_settings.redis_enabled = False
        import src.cache.redis_client as mod

        # Reset the global
        mod._redis_cache = None
        client = mod.get_redis_client()
        assert isinstance(client, mod.RedisCache)

    @patch("src.cache.redis_client.settings")
    def test_returns_same_instance(self, mock_settings):
        mock_settings.redis_enabled = False
        import src.cache.redis_client as mod

        mod._redis_cache = None
        a = mod.get_redis_client()
        b = mod.get_redis_client()
        assert a is b


# ============================================================================
# Startup Validation (src/utils/startup_validation.py)
# ============================================================================
class TestValidateLivenessConfiguration:
    """validate_liveness_configuration() checks liveness settings."""

    @patch("src.utils.startup_validation.settings")
    def test_liveness_disabled_returns_valid(self, mock_settings):
        mock_settings.liveness_enabled = False

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert is_valid is True
        assert errors == []

    @patch("src.utils.startup_validation.settings")
    def test_invalid_provider(self, mock_settings, tmp_path):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "unsupported_provider"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(tmp_path / "models")
        mock_settings.liveness_detector_path = str(tmp_path / "detector")

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert is_valid is False
        assert any("Invalid liveness provider" in e for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_invalid_threshold_above_1(self, mock_settings, tmp_path):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 1.5
        mock_settings.liveness_model_dir = str(tmp_path / "models")
        mock_settings.liveness_detector_path = str(tmp_path / "detector")

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert is_valid is False
        assert any("Invalid liveness threshold" in e for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_invalid_threshold_below_0(self, mock_settings, tmp_path):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = -0.1
        mock_settings.liveness_model_dir = str(tmp_path / "models")
        mock_settings.liveness_detector_path = str(tmp_path / "detector")

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert is_valid is False
        assert any("Invalid liveness threshold" in e for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_model_dir_not_found(self, mock_settings, tmp_path):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(tmp_path / "nonexistent_models")
        mock_settings.liveness_detector_path = str(tmp_path / "nonexistent_detector")

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert is_valid is False
        assert any("model directory not found" in e for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_model_dir_is_file_not_directory(self, mock_settings, tmp_path):
        # Create a file where a directory is expected
        model_file = tmp_path / "model_dir"
        model_file.touch()
        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_file)
        mock_settings.liveness_detector_path = str(detector_dir)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("not a directory" in e for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_detector_path_is_file(self, mock_settings, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        detector_file = tmp_path / "detector"
        detector_file.touch()

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_file)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("detector path is not a directory" in e.lower() for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_no_face_detector_available(self, mock_settings, tmp_path):
        """Neither InsightFace nor RetinaFace is available."""
        import sys

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        # Create a .pth file so anti-spoofing model check passes
        pth_file = model_dir / "model.pth"
        pth_file.write_bytes(b"x" * 200_000)

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()
        # No caffemodel or prototxt files

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_dir)

        # Temporarily hide insightface from sys.modules so the import fails
        saved = sys.modules.pop("insightface", None)
        try:
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "insightface":
                    raise ImportError("No module named 'insightface'")
                return real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=fake_import):
                from src.utils.startup_validation import validate_liveness_configuration

                is_valid, errors = validate_liveness_configuration()
                assert any("No face detector available" in e for e in errors)
        finally:
            if saved is not None:
                sys.modules["insightface"] = saved

    @patch("src.utils.startup_validation.settings")
    def test_no_pth_files_in_model_dir(self, mock_settings, tmp_path):
        """Model directory exists but has no .pth files."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_dir)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("No anti-spoofing models" in e for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_empty_pth_file(self, mock_settings, tmp_path):
        """A .pth file exists but is empty."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        pth_file = model_dir / "model.pth"
        pth_file.write_bytes(b"")  # empty

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_dir)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("empty" in e.lower() for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_too_small_pth_file(self, mock_settings, tmp_path):
        """A .pth file exists but is suspiciously small."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        pth_file = model_dir / "model.pth"
        pth_file.write_bytes(b"x" * 50_000)  # 50KB, less than 100KB threshold

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_dir)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("too small" in e.lower() for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_empty_caffemodel(self, mock_settings, tmp_path):
        """RetinaFace caffemodel exists but is empty."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        pth_file = model_dir / "model.pth"
        pth_file.write_bytes(b"x" * 200_000)

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()
        caffemodel = detector_dir / "Widerface-RetinaFace.caffemodel"
        caffemodel.write_bytes(b"")  # empty
        prototxt = detector_dir / "deploy.prototxt"
        prototxt.write_bytes(b"layer { name: 'test' }")

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_dir)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("caffemodel is empty" in e for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_too_small_caffemodel(self, mock_settings, tmp_path):
        """RetinaFace caffemodel is too small."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        pth_file = model_dir / "model.pth"
        pth_file.write_bytes(b"x" * 200_000)

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()
        caffemodel = detector_dir / "Widerface-RetinaFace.caffemodel"
        caffemodel.write_bytes(b"x" * 500_000)  # 500KB, less than 1MB
        prototxt = detector_dir / "deploy.prototxt"
        prototxt.write_bytes(b"layer { name: 'test' }")

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_dir)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("too small" in e.lower() for e in errors)

    @patch("src.utils.startup_validation.settings")
    def test_empty_prototxt(self, mock_settings, tmp_path):
        """RetinaFace prototxt exists but is empty."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        pth_file = model_dir / "model.pth"
        pth_file.write_bytes(b"x" * 200_000)

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()
        caffemodel = detector_dir / "Widerface-RetinaFace.caffemodel"
        caffemodel.write_bytes(b"x" * 5_000_000)  # 5MB
        prototxt = detector_dir / "deploy.prototxt"
        prototxt.write_bytes(b"")  # empty

        mock_settings.liveness_enabled = True
        mock_settings.liveness_provider = "silent_face"
        mock_settings.liveness_threshold = 0.5
        mock_settings.liveness_model_dir = str(model_dir)
        mock_settings.liveness_detector_path = str(detector_dir)

        from src.utils.startup_validation import validate_liveness_configuration

        is_valid, errors = validate_liveness_configuration()
        assert any("prototxt is empty" in e for e in errors)


class TestValidateStartupRequirements:
    """validate_startup_requirements() aggregates validations."""

    @patch("src.utils.startup_validation.validate_liveness_configuration")
    def test_all_valid_returns_true(self, mock_validate_liveness):
        mock_validate_liveness.return_value = (True, [])

        from src.utils.startup_validation import validate_startup_requirements

        result = validate_startup_requirements(fail_on_error=True)
        assert result is True

    @patch("src.utils.startup_validation.validate_liveness_configuration")
    def test_failure_raises_when_fail_on_error_true(self, mock_validate_liveness):
        mock_validate_liveness.return_value = (False, ["Error 1", "Error 2"])

        from src.utils.startup_validation import validate_startup_requirements

        with pytest.raises(RuntimeError, match="2 error"):
            validate_startup_requirements(fail_on_error=True)

    @patch("src.utils.startup_validation.validate_liveness_configuration")
    def test_failure_returns_false_when_fail_on_error_false(self, mock_validate_liveness):
        mock_validate_liveness.return_value = (False, ["Error 1"])

        from src.utils.startup_validation import validate_startup_requirements

        result = validate_startup_requirements(fail_on_error=False)
        assert result is False


# ============================================================================
# StorageFactory (src/storage/factory.py)
# ============================================================================
class TestStorageFactory:
    """StorageFactory.create_storage returns the correct backend."""

    @patch("src.storage.factory.settings")
    def test_create_local_storage(self, mock_settings, tmp_path):
        mock_settings.storage_backend = "local"
        mock_settings.storage_local_path = str(tmp_path)

        from src.storage.factory import StorageFactory
        from src.storage.local import LocalStorageBackend

        storage = StorageFactory.create_storage("local")
        assert isinstance(storage, LocalStorageBackend)

    @patch("src.storage.factory.S3StorageBackend")
    @patch("src.storage.factory.settings")
    def test_create_s3_storage(self, mock_settings, mock_s3_cls):
        mock_settings.storage_backend = "s3"
        mock_settings.storage_s3_bucket = "my-bucket"
        mock_settings.storage_s3_region = "us-west-2"
        mock_settings.aws_access_key_id = "AKID"
        mock_settings.aws_secret_access_key = "SECRET"

        mock_s3_instance = MagicMock()
        mock_s3_cls.return_value = mock_s3_instance

        from src.storage.factory import StorageFactory

        storage = StorageFactory.create_storage("s3")
        assert storage is mock_s3_instance
        mock_s3_cls.assert_called_once_with(
            bucket_name="my-bucket",
            region="us-west-2",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
        )

    @patch("src.storage.factory.settings")
    def test_create_s3_without_bucket_raises(self, mock_settings):
        mock_settings.storage_s3_bucket = ""

        from src.storage.factory import StorageFactory

        with pytest.raises(ValueError, match="S3 bucket name not configured"):
            StorageFactory.create_storage("s3")

    @patch("src.storage.factory.settings")
    def test_create_unknown_backend_raises(self, mock_settings):
        from src.storage.factory import StorageFactory

        with pytest.raises(ValueError, match="Unsupported storage backend"):
            StorageFactory.create_storage("gcs")

    @patch("src.storage.factory.settings")
    def test_defaults_to_settings_backend(self, mock_settings, tmp_path):
        mock_settings.storage_backend = "local"
        mock_settings.storage_local_path = str(tmp_path)

        from src.storage.factory import StorageFactory

        storage = StorageFactory.create_storage()  # No argument
        from src.storage.local import LocalStorageBackend

        assert isinstance(storage, LocalStorageBackend)


class TestGetStorage:
    """get_storage() convenience function."""

    @patch("src.storage.factory.settings")
    def test_get_storage_returns_backend(self, mock_settings, tmp_path):
        mock_settings.storage_backend = "local"
        mock_settings.storage_local_path = str(tmp_path)

        from src.storage.factory import get_storage
        from src.storage.local import LocalStorageBackend

        storage = get_storage()
        assert isinstance(storage, LocalStorageBackend)


# ============================================================================
# S3StorageBackend (src/storage/s3.py)
# ============================================================================
class TestS3StorageBackendInit:
    """S3StorageBackend constructor creates S3FS with correct params."""

    @patch("src.storage.s3.S3FS")
    def test_init_with_credentials(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(
            bucket_name="my-bucket",
            region="eu-west-1",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
        )
        assert storage.bucket_name == "my-bucket"
        assert storage.region == "eu-west-1"
        mock_s3fs_cls.assert_called_once_with(
            "my-bucket",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            region="eu-west-1",
        )

    @patch("src.storage.s3.S3FS")
    def test_init_without_credentials(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        S3StorageBackend(
            bucket_name="my-bucket",
            region="us-east-1",
        )
        mock_s3fs_cls.assert_called_once_with("my-bucket", region="us-east-1")


class TestS3StorageBackendSave:
    """save() writes bytes and returns s3:// URL."""

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_save_returns_s3_url(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        result = await storage.save("faces/photo.jpg", b"image data")

        assert result == "s3://bucket/faces/photo.jpg"
        mock_fs.writebytes.assert_called_once()


class TestS3StorageBackendRead:
    """read() returns bytes or raises FileNotFoundError."""

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_read_existing_file(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_fs.readbytes.return_value = b"file contents"
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        result = await storage.read("faces/photo.jpg")
        assert result == b"file contents"

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_read_nonexistent_raises(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(FileNotFoundError):
            await storage.read("nonexistent.jpg")


class TestS3StorageBackendDelete:
    """delete() removes file or returns False."""

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_delete_existing(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        result = await storage.delete("photo.jpg")
        assert result is True
        mock_fs.remove.assert_called_once()

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        result = await storage.delete("no_such_file.jpg")
        assert result is False

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_delete_exception_returns_false(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_fs.exists.side_effect = Exception("S3 error")
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        result = await storage.delete("photo.jpg")
        assert result is False


class TestS3StorageBackendExists:
    """exists() checks S3 for the key."""

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_exists_true(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        assert await storage.exists("photo.jpg") is True

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_exists_false(self, mock_s3fs_cls):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        mock_s3fs_cls.return_value = mock_fs

        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        assert await storage.exists("nope.jpg") is False


class TestS3StorageBackendUrls:
    """get_url() and get_https_url() return properly formatted URLs."""

    @patch("src.storage.s3.S3FS")
    def test_get_url(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="my-bucket", region="us-east-1")
        url = storage.get_url("faces/photo.jpg")
        assert url == "s3://my-bucket/faces/photo.jpg"

    @patch("src.storage.s3.S3FS")
    def test_get_https_url(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="my-bucket", region="eu-west-1")
        url = storage.get_https_url("faces/photo.jpg")
        assert url == "https://my-bucket.s3.eu-west-1.amazonaws.com/faces/photo.jpg"


class TestS3StorageBackendPathValidation:
    """_validate_path prevents path traversal in S3 backend."""

    @patch("src.storage.s3.S3FS")
    def test_rejects_path_traversal(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("../../etc/passwd")

    @patch("src.storage.s3.S3FS")
    def test_rejects_absolute_path(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("/etc/passwd")

    @patch("src.storage.s3.S3FS")
    def test_accepts_valid_path(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        result = storage._validate_path("faces/user1/photo.jpg")
        assert result == os.path.normpath("faces/user1/photo.jpg")

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_save_rejects_traversal(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(ValueError, match="path traversal"):
            await storage.save("../escape.jpg", b"data")

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_read_rejects_traversal(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(ValueError, match="path traversal"):
            await storage.read("../escape.jpg")

    @patch("src.storage.s3.S3FS")
    @pytest.mark.asyncio
    async def test_exists_rejects_traversal(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(ValueError, match="path traversal"):
            await storage.exists("../escape.jpg")

    @patch("src.storage.s3.S3FS")
    def test_get_url_rejects_traversal(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(ValueError, match="path traversal"):
            storage.get_url("../escape.jpg")

    @patch("src.storage.s3.S3FS")
    def test_get_https_url_rejects_traversal(self, mock_s3fs_cls):
        from src.storage.s3 import S3StorageBackend

        storage = S3StorageBackend(bucket_name="bucket", region="us-east-1")
        with pytest.raises(ValueError, match="path traversal"):
            storage.get_https_url("../escape.jpg")
