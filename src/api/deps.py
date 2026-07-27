from __future__ import annotations

from fastapi import Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.database.base import get_db
from src.exceptions import InvalidImageError
from src.services.face_service import FaceService


async def read_image_upload(image: UploadFile) -> bytes:
    """Read an uploaded image with content-type and size validation."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise InvalidImageError(f"Unsupported content type: {image.content_type}")
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    data = await image.read(max_bytes + 1)
    if not data:
        raise InvalidImageError("Empty image upload")
    if len(data) > max_bytes:
        raise InvalidImageError(f"Image exceeds maximum size of {settings.max_upload_size_mb} MB")
    return data


async def get_face_service(db: AsyncSession = Depends(get_db)) -> FaceService:
    """Dependency injection for FaceService."""
    return FaceService(db)
