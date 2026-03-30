"""Tests for APITokenMiddleware (src/middleware/auth.py)."""
import hmac
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from src.middleware.auth import APITokenMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VALID_TOKEN = "super-secret-test-token"


def _make_app(secret_key: str = VALID_TOKEN) -> Starlette:
    """Build a tiny Starlette app wrapped with APITokenMiddleware."""

    async def _index(request):
        return PlainTextResponse("ok-index")

    async def _health(request):
        return PlainTextResponse("ok-health")

    async def _docs(request):
        return PlainTextResponse("ok-docs")

    async def _redoc(request):
        return PlainTextResponse("ok-redoc")

    async def _openapi(request):
        return PlainTextResponse("ok-openapi")

    async def _protected(request):
        return PlainTextResponse("ok-protected")

    async def _image_endpoint(request):
        return PlainTextResponse("ok-image")

    async def _webcam_stream(request):
        return PlainTextResponse("ok-webcam")

    app = Starlette(
        routes=[
            Route("/", _index),
            Route("/health", _health),
            Route("/docs", _docs),
            Route("/redoc", _redoc),
            Route("/openapi.json", _openapi),
            Route("/api/v1/faces", _protected),
            Route("/api/v1/faces/{face_id}/image", _image_endpoint),
            Route("/api/v1/webcam/stream", _webcam_stream),
        ],
    )
    app.add_middleware(APITokenMiddleware)
    return app


# ---------------------------------------------------------------------------
# Tests: timing-safe comparison
# ---------------------------------------------------------------------------
class TestHmacUsage:
    """Verify that hmac.compare_digest is used for token comparison."""

    def test_compare_digest_called_for_valid_token(self):
        """hmac.compare_digest must be used (not ==) to prevent timing attacks."""
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            app = _make_app()
            client = TestClient(app, raise_server_exceptions=False)

            with patch("src.middleware.auth.hmac.compare_digest", wraps=hmac.compare_digest) as spy:
                resp = client.get("/api/v1/faces", headers={"x-face-token": VALID_TOKEN})
                spy.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: missing / invalid tokens
# ---------------------------------------------------------------------------
class TestAuthRejection:
    """Requests without a valid token must receive 401."""

    def test_missing_token_returns_401(self):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get("/api/v1/faces")

            assert resp.status_code == 401
            body = resp.json()
            assert body["success"] is False
            assert "Invalid or missing API token" in body["error"]

    def test_invalid_token_returns_401(self):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get(
                "/api/v1/faces",
                headers={"x-face-token": "wrong-token"},
            )

            assert resp.status_code == 401
            body = resp.json()
            assert body["success"] is False

    def test_401_includes_www_authenticate_header(self):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get("/api/v1/faces")

            assert resp.headers.get("www-authenticate") == "Token"

    def test_empty_string_token_returns_401(self):
        """An empty x-face-token header should still be rejected."""
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get(
                "/api/v1/faces",
                headers={"x-face-token": ""},
            )

            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: valid token passes through
# ---------------------------------------------------------------------------
class TestAuthAccepted:
    """Requests with the correct token should reach the handler."""

    def test_valid_token_passes_through(self):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get(
                "/api/v1/faces",
                headers={"x-face-token": VALID_TOKEN},
            )

            assert resp.status_code == 200
            assert resp.text == "ok-protected"


# ---------------------------------------------------------------------------
# Tests: excluded paths bypass auth
# ---------------------------------------------------------------------------
class TestExcludedPaths:
    """Paths listed in EXCLUDED_PATHS must be accessible without a token."""

    @pytest.mark.parametrize(
        "path,expected_body",
        [
            ("/", "ok-index"),
            ("/health", "ok-health"),
            ("/docs", "ok-docs"),
            ("/redoc", "ok-redoc"),
            ("/openapi.json", "ok-openapi"),
        ],
    )
    def test_excluded_path_bypasses_auth(self, path, expected_body):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get(path)

            assert resp.status_code == 200
            assert resp.text == expected_body


# ---------------------------------------------------------------------------
# Tests: image endpoints bypass auth
# ---------------------------------------------------------------------------
class TestImageEndpointBypass:
    """Image-serving paths (containing '/image') should bypass auth."""

    def test_image_path_bypasses_auth(self):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get("/api/v1/faces/42/image")

            assert resp.status_code == 200
            assert resp.text == "ok-image"


# ---------------------------------------------------------------------------
# Tests: webcam stream endpoint bypass
# ---------------------------------------------------------------------------
class TestWebcamStreamBypass:
    """Webcam stream endpoint (containing '/webcam/stream') should bypass auth."""

    def test_webcam_stream_bypasses_auth(self):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = VALID_TOKEN
            client = TestClient(_make_app(), raise_server_exceptions=False)
            resp = client.get("/api/v1/webcam/stream")

            assert resp.status_code == 200
            assert resp.text == "ok-webcam"
