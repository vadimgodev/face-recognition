"""Integration tests for Face Recognition API endpoints.

Tests all major API endpoints using httpx.AsyncClient with mocked services.
No Docker, database, or model loading required.
"""

import io
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from src.exceptions import (
    FaceRecognitionError,
    FaceNotFoundError,
    NoFaceDetectedError,
    InvalidImageError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_TOKEN = "test-api-token-for-testing"


def _make_jpeg_bytes() -> bytes:
    """Return minimal valid JPEG bytes for upload tests."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def _make_face(**overrides) -> MagicMock:
    """Build a mock Face ORM object accepted by FaceResponse.model_validate."""
    now = datetime.utcnow()
    defaults = dict(
        id=1,
        user_name="alice",
        user_email="alice@example.com",
        provider_name="insightface",
        provider_face_id="face-uuid-001",
        image_path="/data/images/face_1.jpg",
        image_storage="local",
        quality_score=0.95,
        confidence_score=0.92,
        photo_type="enrolled",
        verified_at=None,
        verified_confidence=None,
        verified_by_processor=None,
        created_at=now,
        updated_at=now,
    )
    defaults.update(overrides)

    face = MagicMock()
    for k, v in defaults.items():
        setattr(face, k, v)

    # Pydantic model_validate(face) uses from_attributes, so __dict__ must work
    face.__dict__.update(defaults)
    return face


# ---------------------------------------------------------------------------
# App with bypassed lifespan & patched secret_key
# ---------------------------------------------------------------------------

@pytest.fixture()
def _patch_settings():
    """Patch settings.secret_key so the test token is accepted."""
    with patch("src.middleware.auth.settings") as mock_settings:
        mock_settings.secret_key = TEST_TOKEN
        yield mock_settings


@pytest.fixture()
def mock_face_service():
    """Return a fully-mocked FaceService."""
    service = AsyncMock()
    return service


@pytest.fixture()
def _override_service(mock_face_service, _patch_settings):
    """Override the get_face_service dependency with the mock and patch settings."""
    # Import here so the module-level `app` is already loaded
    from src.api.routes import get_face_service
    from src.main import app

    app.dependency_overrides[get_face_service] = lambda: mock_face_service
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def test_image_bytes():
    return _make_jpeg_bytes()


def _build_app_no_lifespan():
    """Return the FastAPI app with lifespan disabled for testing."""
    from src.main import app
    # Disable the lifespan so tests don't trigger startup validation,
    # Redis, model warmup, etc.
    app.router.lifespan_context = None  # type: ignore[assignment]
    return app


@pytest.fixture(autouse=True)
def _disable_lifespan():
    """Disable the lifespan for every test in this module."""
    from src.main import app

    original = app.router.lifespan_context
    app.router.lifespan_context = None  # type: ignore[assignment]
    yield
    app.router.lifespan_context = original


# ---------------------------------------------------------------------------
# Transport helper
# ---------------------------------------------------------------------------

def _transport():
    from src.main import app
    return ASGITransport(app=app)


# ===========================================================================
# Health & Root
# ===========================================================================


@pytest.mark.asyncio
class TestHealthAndRoot:

    async def test_health_returns_200(self):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "timestamp" in body

    async def test_root_returns_api_info(self):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Face Recognition API"
        assert "version" in body
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"


# ===========================================================================
# Authentication
# ===========================================================================


@pytest.mark.asyncio
class TestAuthentication:

    async def test_enroll_without_token_returns_401(self, test_image_bytes):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                data={"user_name": "alice"},
            )
        assert resp.status_code == 401
        body = resp.json()
        assert body["success"] is False

    async def test_recognize_without_token_returns_401(self, test_image_bytes):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/recognize",
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
            )
        assert resp.status_code == 401

    async def test_invalid_token_returns_401(self, test_image_bytes):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                headers={"x-face-token": "wrong-token-value"},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                data={"user_name": "alice"},
            )
        assert resp.status_code == 401
        body = resp.json()
        assert body["success"] is False
        assert "token" in body["error"].lower() or "token" in body["detail"].lower()

    async def test_list_faces_without_token_returns_401(self):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/api/v1/faces")
        assert resp.status_code == 401

    async def test_delete_face_without_token_returns_401(self):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.delete("/api/v1/faces/1")
        assert resp.status_code == 401


# ===========================================================================
# Enroll endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestEnrollEndpoint:

    async def test_enroll_success(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        face = _make_face()
        mock_face_service.enroll_face.return_value = face

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                data={"user_name": "alice", "user_email": "alice@example.com"},
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["success"] is True
        assert body["message"] == "Face enrolled successfully"
        assert body["face"]["user_name"] == "alice"
        assert body["face"]["id"] == 1
        mock_face_service.enroll_face.assert_awaited_once()

    async def test_enroll_without_image_returns_422(self, _override_service):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                headers={"x-face-token": TEST_TOKEN},
                data={"user_name": "alice"},
            )
        assert resp.status_code == 422

    async def test_enroll_without_user_name_returns_422(
        self, test_image_bytes, _override_service
    ):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                # no user_name
            )
        assert resp.status_code == 422

    async def test_enroll_face_recognition_error(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        """FaceRecognitionError raised by service propagates through exception handler."""
        mock_face_service.enroll_face.side_effect = NoFaceDetectedError()

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                data={"user_name": "alice"},
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "NoFaceDetectedError"

    async def test_enroll_unexpected_error_returns_500(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        mock_face_service.enroll_face.side_effect = RuntimeError("boom")

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                data={"user_name": "alice"},
            )

        assert resp.status_code == 500


# ===========================================================================
# Recognize endpoint
# ===========================================================================


@pytest.mark.asyncio
class TestRecognizeEndpoint:

    async def test_recognize_success(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        face = _make_face()
        # recognize_face returns (matches_list, processor_string)
        mock_face_service.recognize_face.return_value = (
            [(face, 0.95, False, "antelopev2")],
            "antelopev2",
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/recognize",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results": "5", "confidence_threshold": "0.7"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total_matches"] == 1
        assert body["processor"] == "antelopev2"
        assert len(body["matches"]) == 1
        match = body["matches"][0]
        assert match["face"]["user_name"] == "alice"
        assert match["similarity"] == 0.95
        assert match["processor"] == "antelopev2"
        assert isinstance(body["execution_time"], float)
        assert isinstance(body["detection_time"], float)
        assert isinstance(body["recognition_time"], float)

    async def test_recognize_no_matches(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        mock_face_service.recognize_face.return_value = ([], "antelopev2")

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/recognize",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total_matches"] == 0
        assert body["matches"] == []

    async def test_recognize_response_format(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        """Verify response matches RecognizeFaceResponse schema exactly."""
        face = _make_face()
        mock_face_service.recognize_face.return_value = (
            [(face, 0.88, True, "antelopev2+aws")],
            "antelopev2+aws",
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/recognize",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
            )

        body = resp.json()
        # Top-level keys from RecognizeFaceResponse
        expected_keys = {
            "success", "message", "matches", "total_matches",
            "processor", "execution_time", "detection_time", "recognition_time",
        }
        assert expected_keys.issubset(set(body.keys()))

        # Match-level keys from FaceMatchResponse
        match = body["matches"][0]
        assert "face" in match
        assert "similarity" in match
        assert "photo_captured" in match
        assert "processor" in match

        # Face-level keys from FaceResponse
        face_data = match["face"]
        face_keys = {
            "id", "user_name", "user_email", "provider_name",
            "provider_face_id", "image_path", "image_storage",
            "quality_score", "confidence_score", "photo_type",
            "verified_at", "verified_confidence", "verified_by_processor",
            "created_at", "updated_at",
        }
        assert face_keys.issubset(set(face_data.keys()))

    async def test_recognize_no_face_detected_error(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        mock_face_service.recognize_face.side_effect = NoFaceDetectedError()

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/recognize",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "NoFaceDetectedError"

    async def test_recognize_invalid_image_error(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        mock_face_service.recognize_face.side_effect = InvalidImageError(
            "Cannot decode image"
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/recognize",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body["success"] is False

    async def test_recognize_unexpected_error_returns_500(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        mock_face_service.recognize_face.side_effect = ValueError("unexpected")

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/recognize",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
            )

        assert resp.status_code == 500


# ===========================================================================
# List faces
# ===========================================================================


@pytest.mark.asyncio
class TestListFaces:

    async def test_list_faces_returns_paginated(
        self, mock_face_service, _override_service
    ):
        faces = [_make_face(id=i, user_name=f"user_{i}") for i in range(1, 4)]
        mock_face_service.list_faces.return_value = (faces, 3)

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces",
                headers={"x-face-token": TEST_TOKEN},
                params={"limit": 10, "offset": 0},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] == 3
        assert body["limit"] == 10
        assert body["offset"] == 0
        assert len(body["faces"]) == 3

    async def test_list_faces_pagination_params(
        self, mock_face_service, _override_service
    ):
        mock_face_service.list_faces.return_value = ([], 50)

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces",
                headers={"x-face-token": TEST_TOKEN},
                params={"limit": 5, "offset": 20},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 5
        assert body["offset"] == 20
        assert body["total"] == 50
        mock_face_service.list_faces.assert_awaited_once_with(limit=5, offset=20)

    async def test_list_faces_empty(self, mock_face_service, _override_service):
        mock_face_service.list_faces.return_value = ([], 0)

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["faces"] == []
        assert body["total"] == 0


# ===========================================================================
# Get face by ID
# ===========================================================================


@pytest.mark.asyncio
class TestGetFace:

    async def test_get_face_success(self, mock_face_service, _override_service):
        face = _make_face(id=42, user_name="bob")
        mock_face_service.get_face_by_id.return_value = face

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces/42",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 42
        assert body["user_name"] == "bob"

    async def test_get_face_not_found(self, mock_face_service, _override_service):
        mock_face_service.get_face_by_id.return_value = None

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces/999999",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 404


# ===========================================================================
# Delete face
# ===========================================================================


@pytest.mark.asyncio
class TestDeleteFace:

    async def test_delete_face_success(self, mock_face_service, _override_service):
        mock_face_service.delete_face.return_value = True

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.delete(
                "/api/v1/faces/1",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "deleted" in body["message"].lower()
        mock_face_service.delete_face.assert_awaited_once_with(1)

    async def test_delete_face_not_found(self, mock_face_service, _override_service):
        mock_face_service.delete_face.return_value = False

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.delete(
                "/api/v1/faces/999999",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 404

    async def test_delete_face_service_error(
        self, mock_face_service, _override_service
    ):
        mock_face_service.delete_face.side_effect = FaceRecognitionError(
            "Provider unavailable", status_code=502
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.delete(
                "/api/v1/faces/1",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 502
        body = resp.json()
        assert body["success"] is False


# ===========================================================================
# Error handling
# ===========================================================================


@pytest.mark.asyncio
class TestErrorHandling:

    async def test_face_recognition_error_returns_json(
        self, mock_face_service, _override_service
    ):
        """FaceRecognitionError subclasses produce structured JSON with success=false."""
        mock_face_service.list_faces.side_effect = FaceRecognitionError(
            "Something went wrong", status_code=400
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body["success"] is False
        assert "error" in body
        assert "detail" in body

    async def test_face_not_found_error(
        self, mock_face_service, _override_service
    ):
        mock_face_service.get_face_by_id.side_effect = FaceNotFoundError(face_id=7)

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces/7",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 404
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "FaceNotFoundError"

    async def test_generic_exception_returns_500(
        self, mock_face_service, _override_service
    ):
        """Unhandled exceptions in the service are caught by the route handler."""
        mock_face_service.list_faces.side_effect = RuntimeError("database down")

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/faces",
                headers={"x-face-token": TEST_TOKEN},
            )

        assert resp.status_code == 500

    async def test_invalid_image_error(
        self, mock_face_service, test_image_bytes, _override_service
    ):
        mock_face_service.enroll_face.side_effect = InvalidImageError(
            "Corrupt JPEG data"
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/faces/enroll",
                headers={"x-face-token": TEST_TOKEN},
                files={"image": ("face.jpg", test_image_bytes, "image/jpeg")},
                data={"user_name": "alice"},
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "InvalidImageError"
        assert "Corrupt JPEG data" in body["detail"]
