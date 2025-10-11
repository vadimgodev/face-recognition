"""Authentication middleware for API token validation."""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from src.config.settings import settings


class APITokenMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API token in x-face-token header.
    Excludes /health, /docs, image endpoints, and webcam stream.
    """

    EXCLUDED_PATHS = ["/health", "/docs", "/redoc", "/openapi.json", "/"]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Skip authentication for image endpoints
        # Images are loaded via <img> tags which can't send custom headers
        # They're still protected by Basic Auth at Traefik level
        if "/image" in request.url.path:
            return await call_next(request)

        # Skip authentication for webcam stream endpoint
        # EventSource (SSE) doesn't support custom headers
        # Still protected by Basic Auth at Traefik level
        if "/webcam/stream" in request.url.path:
            return await call_next(request)

        # Get token from header
        token = request.headers.get("x-face-token")

        # Validate token
        if not token or token != settings.secret_key:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Invalid or missing API token",
                    "detail": "Please provide a valid x-face-token header"
                },
                headers={"WWW-Authenticate": "Token"},
            )

        # Token is valid, proceed with request
        response = await call_next(request)
        return response
