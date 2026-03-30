"""Unit tests for application Settings.

pydantic_settings loads values from environment variables, so we use
monkeypatch / os.environ manipulation to control inputs.  Each test
constructs a fresh Settings instance to avoid cross-test pollution.
"""
import os
import pytest
from unittest.mock import patch


def _make_settings(**env_overrides):
    """Create a Settings instance with controlled environment.

    Clears all env vars that Settings might read, applies overrides,
    then constructs a fresh instance with _env_file=None so no .env
    file is loaded.
    """
    # Minimal set of env vars to clear (covers every Field alias)
    clear_keys = [
        "APP_NAME", "APP_ENV", "DEBUG", "API_HOST", "API_PORT",
        "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT",
        "POSTGRES_USER", "POSTGRES_DB", "DATABASE_POOL_SIZE",
        "DATABASE_MAX_OVERFLOW",
        "REDIS_ENABLED", "REDIS_PASSWORD", "REDIS_HOST", "REDIS_PORT",
        "REDIS_DB", "REDIS_MAX_CONNECTIONS", "REDIS_CACHE_TTL",
        "STORAGE_BACKEND", "STORAGE_LOCAL_PATH", "STORAGE_S3_BUCKET",
        "STORAGE_S3_REGION",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
        "AWS_REKOGNITION_COLLECTION_ID",
        "FACE_PROVIDER",
        "LIVENESS_ENABLED", "LIVENESS_PROVIDER", "LIVENESS_THRESHOLD",
        "LIVENESS_DEVICE_ID", "LIVENESS_MODEL_DIR", "LIVENESS_DETECTOR_PATH",
        "LIVENESS_ON_ENROLLMENT", "LIVENESS_ON_RECOGNITION",
        "USE_HYBRID_RECOGNITION", "HYBRID_MODE", "INSIGHTFACE_MODEL",
        "INSIGHTFACE_DET_SIZE", "INSIGHTFACE_CTX_ID",
        "SIMILARITY_THRESHOLD", "VECTOR_SEARCH_CANDIDATES",
        "AWS_VERIFICATION_COUNT",
        "INSIGHTFACE_HIGH_CONFIDENCE", "INSIGHTFACE_MEDIUM_CONFIDENCE",
        "AUTO_CAPTURE_ENABLED", "AUTO_CAPTURE_CONFIDENCE_THRESHOLD",
        "AUTO_CAPTURE_MAX_VERIFIED_PHOTOS",
        "WEBCAM_ENABLED", "WEBCAM_DEVICE_ID", "WEBCAM_FPS",
        "WEBCAM_SUCCESS_COOLDOWN_SECONDS", "WEBCAM_API_URL",
        "MULTIFACE_ENABLED", "FACE_DETECTION_METHOD",
        "INSIGHTFACE_DETECTION_MODEL", "DETECTION_CONFIDENCE_THRESHOLD",
        "MAX_FACES_PER_FRAME", "MIN_FACE_SIZE", "FACE_CROP_PADDING",
        "SAVE_ALL_DETECTED_FACES",
        "FACE_QUALITY_MIN_SIZE", "FACE_QUALITY_MAX_BLUR",
        "FACE_QUALITY_MIN_BRIGHTNESS", "FACE_QUALITY_MAX_BRIGHTNESS",
        "ROI_ENABLED", "ROI_X", "ROI_Y", "ROI_WIDTH", "ROI_HEIGHT",
        "ROI_MIN_OVERLAP",
        "DOOR_UNLOCK_PROVIDER", "DOOR_UNLOCK_URL",
        "DOOR_UNLOCK_CONFIDENCE_THRESHOLD",
        "ACCESS_LOG_OUTPUT", "ACCESS_LOG_FILE_PATH", "ACCESS_LOG_FORMAT",
        "ACCESS_LOG_INCLUDE_COOLDOWN_EVENTS",
        "SECRET_KEY", "ALLOWED_ORIGINS", "LOG_LEVEL",
    ]

    env = {k: v for k, v in os.environ.items()}  # snapshot
    for k in clear_keys:
        env.pop(k, None)
        env.pop(k.lower(), None)

    # Apply caller overrides
    env.update(env_overrides)

    with patch.dict(os.environ, env, clear=True):
        from src.config.settings import Settings
        return Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    """Tests for default settings values."""

    def test_app_version_is_set(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.APP_VERSION == "1.0.0"

    def test_face_quality_min_size_default(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.face_quality_min_size == 80

    def test_face_quality_max_blur_default(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.face_quality_max_blur == 100.0

    def test_face_quality_min_brightness_default(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.face_quality_min_brightness == 40.0

    def test_face_quality_max_brightness_default(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.face_quality_max_brightness == 220.0

    def test_auto_capture_defaults(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.auto_capture_enabled is True
        assert s.auto_capture_confidence_threshold == 0.85
        assert s.auto_capture_max_verified_photos == 4

    def test_storage_backend_default(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.storage_backend == "local"

    def test_hybrid_mode_default(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert s.hybrid_mode == "insightface_only"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestSettingsValidation:
    """Tests for model_validator behavior."""

    def test_s3_without_bucket_raises(self):
        with pytest.raises(ValueError, match="STORAGE_S3_BUCKET"):
            _make_settings(
                SECRET_KEY="x",
                DEBUG="true",
                STORAGE_BACKEND="s3",
                STORAGE_S3_BUCKET="",
            )

    def test_s3_with_bucket_ok(self):
        s = _make_settings(
            SECRET_KEY="x",
            DEBUG="true",
            STORAGE_BACKEND="s3",
            STORAGE_S3_BUCKET="my-bucket",
        )
        assert s.storage_s3_bucket == "my-bucket"

    def test_empty_secret_key_with_debug_no_error(self):
        """Empty secret_key with debug=True should not error (just warn)."""
        s = _make_settings(SECRET_KEY="", DEBUG="true")
        assert s.secret_key == ""


# ---------------------------------------------------------------------------
# Computed properties
# ---------------------------------------------------------------------------

class TestSettingsProperties:
    """Tests for computed properties."""

    def test_database_url(self):
        s = _make_settings(
            SECRET_KEY="x", DEBUG="true",
            POSTGRES_USER="myuser",
            POSTGRES_PASSWORD="mypass",
            POSTGRES_HOST="localhost",
            POSTGRES_PORT="5432",
            POSTGRES_DB="testdb",
        )
        assert s.database_url == (
            "postgresql+asyncpg://myuser:mypass@localhost:5432/testdb"
        )

    def test_redis_url_without_password(self):
        s = _make_settings(
            SECRET_KEY="x", DEBUG="true",
            REDIS_PASSWORD="",
            REDIS_HOST="localhost",
            REDIS_PORT="6379",
            REDIS_DB="0",
        )
        assert s.redis_url == "redis://localhost:6379/0"

    def test_redis_url_with_password(self):
        s = _make_settings(
            SECRET_KEY="x", DEBUG="true",
            REDIS_PASSWORD="secret",
            REDIS_HOST="redis-host",
            REDIS_PORT="6380",
            REDIS_DB="1",
        )
        assert s.redis_url == "redis://:secret@redis-host:6380/1"

    def test_is_production(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true", APP_ENV="production")
        assert s.is_production is True
        assert s.is_development is False

    def test_is_development(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true", APP_ENV="development")
        assert s.is_development is True
        assert s.is_production is False


# ---------------------------------------------------------------------------
# allowed_origins field_validator
# ---------------------------------------------------------------------------

class TestAllowedOrigins:
    """Tests for the allowed_origins field_validator."""

    def test_parses_comma_separated_string(self):
        s = _make_settings(
            SECRET_KEY="x", DEBUG="true",
            ALLOWED_ORIGINS="http://a.com, http://b.com",
        )
        assert s.allowed_origins == ["http://a.com", "http://b.com"]

    def test_default_origins(self):
        s = _make_settings(SECRET_KEY="x", DEBUG="true")
        assert isinstance(s.allowed_origins, list)
        assert len(s.allowed_origins) >= 1
