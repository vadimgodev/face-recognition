"""Authentication middleware for API token validation."""

import hmac
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config.settings import settings

logger = logging.getLogger(__name__)


class APITokenMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API token in x-face-token header.
    Excludes /health, /docs, image endpoints, and webcam stream.
    """

    EXCLUDED_PATHS = ["/health", "/docs", "/redoc", "/openapi.json", "/"]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip authentication for excluded paths
        if path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Skip authentication for the face-image endpoint only
        # (GET /api/v1/faces/{id}/image). Images are loaded via <img> tags
        # which can't send custom headers; Basic Auth at Traefik still applies.
        if path.startswith("/api/v1/faces/") and path.endswith("/image"):
            return await call_next(request)

        # Skip authentication for the webcam SSE stream only.
        # EventSource doesn't support custom headers; Basic Auth still applies.
        if path == "/api/v1/webcam/stream":
            return await call_next(request)

        # Get token from header
        token = request.headers.get("x-face-token")

        # Validate token (constant-time comparison to prevent timing attacks).
        # Compare as bytes: compare_digest raises TypeError on non-ASCII str input.
        valid = bool(
            token
            and hmac.compare_digest(token.encode("utf-8"), settings.secret_key.encode("utf-8"))
        )
        if not valid:
            logger.warning(
                f"Invalid auth attempt from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Invalid or missing API token",
                    "detail": "Please provide a valid x-face-token header",
                },
                headers={"WWW-Authenticate": "Token"},
            )

        # Token is valid, proceed with request
        response = await call_next(request)
        return response
