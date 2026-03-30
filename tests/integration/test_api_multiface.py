"""Integration tests for multi-face recognition API.

Previously skipped tests now use mocked FaceService so they run without
Docker, InsightFace models, or a live database.
"""

import io
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from src.exceptions import FaceRecognitionError, InvalidImageError
from src.main import app


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_TOKEN = "test-api-token-for-testing"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg_bytes() -> bytes:
    """Create a simple 640x480 JPEG image."""
    image = Image.new("RGB", (640, 480), color=(73, 109, 137))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


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
    face.__dict__.update(defaults)
    return face


def _make_bbox():
    """Build a mock BoundingBox-like object with the attributes the route expects."""
    bbox = MagicMock()
    bbox.x1 = 100
    bbox.y1 = 50
    bbox.x2 = 250
    bbox.y2 = 300
    bbox.width = 150
    bbox.height = 250
    bbox.area = 37500
    bbox.center = (175, 175)
    return bbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_image_bytes():
    """Create a test image."""
    return _make_jpeg_bytes()


@pytest.fixture
def api_token():
    """Return test API token."""
    return TEST_TOKEN


@pytest.fixture(autouse=True)
def _disable_lifespan():
    """Disable the lifespan for every test so startup validation is skipped."""
    original = app.router.lifespan_context
    app.router.lifespan_context = None  # type: ignore[assignment]
    yield
    app.router.lifespan_context = original


@pytest.fixture()
def _patch_settings():
    """Patch settings.secret_key so the test token is accepted."""
    with patch("src.middleware.auth.settings") as mock_settings:
        mock_settings.secret_key = TEST_TOKEN
        yield mock_settings


@pytest.fixture()
def mock_face_service():
    """Return a fully-mocked FaceService."""
    return AsyncMock()


@pytest.fixture()
def _override_service(mock_face_service, _patch_settings):
    """Wire the mock FaceService into the app's DI container."""
    from src.api.routes import get_face_service

    app.dependency_overrides[get_face_service] = lambda: mock_face_service
    yield
    app.dependency_overrides.clear()


def _transport():
    return ASGITransport(app=app)


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.asyncio
class TestMultiFaceRecognitionAPI:
    """Test suite for multi-face recognition API endpoint."""

    # ------------------------------------------------------------------
    # Health & root (no auth required)
    # ------------------------------------------------------------------

    async def test_health_check(self):
        """Test health check endpoint."""
        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data

    async def test_root_endpoint(self):
        """Test root endpoint."""
        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Face Recognition API"
            assert "version" in data
            assert "docs" in data

    # ------------------------------------------------------------------
    # Auth enforcement
    # ------------------------------------------------------------------

    async def test_recognize_multiple_without_token(self, test_image_bytes):
        """Test that API requires authentication."""
        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7},
            )
            assert response.status_code in [401, 403]

    # ------------------------------------------------------------------
    # Mocked multi-face tests (formerly Docker-only)
    # ------------------------------------------------------------------

    async def test_recognize_multiple_with_token(
        self, test_image_bytes, api_token, mock_face_service, _override_service
    ):
        """Test multi-face recognition with valid token and mocked service."""
        face = _make_face()
        bbox = _make_bbox()

        mock_face_service.recognize_multiple_faces.return_value = (
            [
                {
                    "face_id": "face_0",
                    "bbox": bbox,
                    "det_confidence": 0.99,
                    "matches": [(face, 0.92, False, "antelopev2")],
                },
            ],
            "antelopev2",
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "detected_faces" in data
        assert data["total_faces_detected"] == 1
        assert data["total_faces_recognized"] == 1
        assert "processor" in data
        assert "execution_time" in data

    async def test_recognize_multiple_invalid_image(
        self, api_token, mock_face_service, _override_service
    ):
        """Test multi-face recognition with invalid image data."""
        mock_face_service.recognize_multiple_faces.side_effect = InvalidImageError(
            "Cannot decode image"
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            invalid_data = b"not an image"
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", invalid_data, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7},
            )
        assert response.status_code == 400

    async def test_recognize_multiple_parameters(
        self, test_image_bytes, api_token, mock_face_service, _override_service
    ):
        """Test multi-face recognition with different parameters."""
        face = _make_face()
        bbox = _make_bbox()

        # Return a single match per face (max_results_per_face=1)
        mock_face_service.recognize_multiple_faces.return_value = (
            [
                {
                    "face_id": "face_0",
                    "bbox": bbox,
                    "det_confidence": 0.95,
                    "matches": [(face, 0.88, False, "antelopev2")],
                },
                {
                    "face_id": "face_1",
                    "bbox": _make_bbox(),
                    "det_confidence": 0.91,
                    "matches": [],
                },
            ],
            "antelopev2",
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 1, "confidence_threshold": 0.8},
            )

        assert response.status_code == 200
        data = response.json()
        for face_result in data.get("detected_faces", []):
            assert len(face_result.get("matches", [])) <= 1

    async def test_recognize_multiple_response_structure(
        self, test_image_bytes, api_token, mock_face_service, _override_service
    ):
        """Test that response has correct structure."""
        face = _make_face()
        bbox = _make_bbox()

        mock_face_service.recognize_multiple_faces.return_value = (
            [
                {
                    "face_id": "face_0",
                    "bbox": bbox,
                    "det_confidence": 0.97,
                    "matches": [(face, 0.90, True, "antelopev2")],
                },
            ],
            "antelopev2",
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7},
            )

        assert response.status_code == 200
        data = response.json()

        # Verify top-level structure
        assert isinstance(data["success"], bool)
        assert isinstance(data["detected_faces"], list)
        assert isinstance(data["total_faces_detected"], int)
        assert isinstance(data["total_faces_recognized"], int)
        assert isinstance(data["processor"], str)
        assert isinstance(data["execution_time"], float)

        # Verify detected_faces structure
        for face_data in data["detected_faces"]:
            assert "face_id" in face_data
            assert "bbox" in face_data
            assert "det_confidence" in face_data
            assert "matches" in face_data
            assert "total_matches" in face_data

            # Verify bbox structure
            bbox_data = face_data["bbox"]
            assert all(
                k in bbox_data
                for k in ["x1", "y1", "x2", "y2", "width", "height", "area", "center_x", "center_y"]
            )

            # Verify matches structure
            for match in face_data["matches"]:
                assert "face" in match
                assert "similarity" in match
                assert "photo_captured" in match
                assert "processor" in match

    async def test_recognize_multiple_no_faces_detected(
        self, test_image_bytes, api_token, mock_face_service, _override_service
    ):
        """Test response when no faces are detected in the image."""
        mock_face_service.recognize_multiple_faces.return_value = ([], "antelopev2")

        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_faces_detected"] == 0
        assert data["total_faces_recognized"] == 0
        assert data["detected_faces"] == []

    async def test_recognize_multiple_service_error(
        self, test_image_bytes, api_token, mock_face_service, _override_service
    ):
        """Test that FaceRecognitionError is handled properly."""
        mock_face_service.recognize_multiple_faces.side_effect = FaceRecognitionError(
            "Provider timeout", status_code=502
        )

        async with AsyncClient(transport=_transport(), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7},
            )

        assert response.status_code == 502
        data = response.json()
        assert data["success"] is False
