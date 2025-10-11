"""Integration tests for multi-face recognition API."""
import pytest
import io
from PIL import Image
import numpy as np
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.fixture
def test_image_bytes():
    """Create a test image."""
    # Create a simple 640x480 RGB image
    image = Image.new("RGB", (640, 480), color=(73, 109, 137))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def api_token():
    """Return test API token."""
    return "test-api-token-for-testing"


@pytest.mark.asyncio
class TestMultiFaceRecognitionAPI:
    """Test suite for multi-face recognition API endpoint."""

    async def test_health_check(self):
        """Test health check endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data

    async def test_root_endpoint(self):
        """Test root endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Face Recognition API"
            assert "version" in data
            assert "docs" in data

    async def test_recognize_multiple_without_token(self, test_image_bytes):
        """Test that API requires authentication."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7}
            )
            # Should require authentication
            assert response.status_code in [401, 403]

    @pytest.mark.skip(reason="Requires full Docker environment with InsightFace models")
    async def test_recognize_multiple_with_token(self, test_image_bytes, api_token):
        """Test multi-face recognition with valid token."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7}
            )
            assert response.status_code == 200
            data = response.json()
            # Response should have expected structure
            assert "success" in data
            assert "detected_faces" in data
            assert "total_faces_detected" in data
            assert "total_faces_recognized" in data
            assert "processor" in data
            assert "execution_time" in data

    @pytest.mark.skip(reason="Requires full Docker environment with InsightFace models")
    async def test_recognize_multiple_invalid_image(self, api_token):
        """Test multi-face recognition with invalid image data."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Send invalid image data
            invalid_data = b"not an image"
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", invalid_data, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7}
            )
            # Should return error for invalid image
            assert response.status_code == 400

    @pytest.mark.skip(reason="Requires full Docker environment with InsightFace models")
    async def test_recognize_multiple_parameters(self, test_image_bytes, api_token):
        """Test multi-face recognition with different parameters."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Test with different max_results_per_face
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 1, "confidence_threshold": 0.8}
            )
            assert response.status_code == 200
            data = response.json()
            # Each detected face should have at most 1 match
            for face in data.get("detected_faces", []):
                assert len(face.get("matches", [])) <= 1

    @pytest.mark.skip(reason="Requires full Docker environment with InsightFace models")
    async def test_recognize_multiple_response_structure(self, test_image_bytes, api_token):
        """Test that response has correct structure."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/faces/recognize-multiple",
                headers={"x-face-token": api_token},
                files={"image": ("test.jpg", test_image_bytes, "image/jpeg")},
                data={"max_results_per_face": 5, "confidence_threshold": 0.7}
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
            for face in data["detected_faces"]:
                assert "face_id" in face
                assert "bbox" in face
                assert "det_confidence" in face
                assert "matches" in face
                assert "total_matches" in face

                # Verify bbox structure
                bbox = face["bbox"]
                assert all(k in bbox for k in ["x1", "y1", "x2", "y2", "width", "height"])

                # Verify matches structure
                for match in face["matches"]:
                    assert "face" in match
                    assert "similarity" in match
                    assert "photo_captured" in match
                    assert "processor" in match
