"""
Auto-capture service for verified face photos.

Automatically captures high-confidence recognition photos and stores them
as "verified" photos using a FIFO stack per user.
"""

import hashlib
from datetime import datetime
from typing import Optional

from src.config.settings import settings
from src.database.models import Face
from src.database.repository import FaceRepository
import logging

logger = logging.getLogger(__name__)


class AutoCaptureService:
    """Manages automatic capture of verified face photos (FIFO stack)."""

    def __init__(self, repository: FaceRepository, storage,
                 insightface_provider=None):
        """
        Args:
            repository: FaceRepository for DB access
            storage: Storage backend for saving images
            insightface_provider: Optional provider for embedding extraction
        """
        self.repository = repository
        self.storage = storage
        self.insightface_provider = insightface_provider

    async def capture_if_eligible(
        self,
        image_data: bytes,
        matched_face: Face,
        confidence: float,
        processor: str,
    ) -> bool:
        """
        Capture a verified photo if confidence exceeds threshold.

        Implements FIFO stack behavior:
        - Keep max N verified photos per person
        - Delete oldest when limit reached
        - Save new photo with metadata

        Args:
            image_data: Image bytes from recognition
            matched_face: The matched Face record
            confidence: Recognition confidence score
            processor: Recognition processor used

        Returns:
            True if photo was captured, False otherwise
        """
        if not settings.auto_capture_enabled:
            return False

        if confidence < settings.auto_capture_confidence_threshold:
            return False

        try:
            # Check verified photos count
            verified_count = await self.repository.get_verified_photos_count(
                matched_face.user_name
            )

            # If at max, delete oldest (FIFO)
            if verified_count >= settings.auto_capture_max_verified_photos:
                oldest = await self.repository.get_oldest_verified_photo(
                    matched_face.user_name
                )
                if oldest:
                    try:
                        await self.storage.delete(oldest.image_path)
                    except Exception:
                        pass  # Continue even if storage deletion fails
                    await self.repository.delete(oldest.id)

            # Extract embedding from verified photo
            embedding = None
            if self.insightface_provider:
                embedding = await self.insightface_provider.extract_embedding(
                    image_data
                )

            # Generate unique image path
            image_hash = hashlib.sha256(image_data).hexdigest()[:16]
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            image_filename = (
                f"{matched_face.user_name}_verified_{timestamp}_{image_hash}.jpg"
            )
            image_path = f"faces/{matched_face.user_name}/{image_filename}"

            # Save image to storage
            await self.storage.save(image_path, image_data)

            # Create verified photo record
            verified_face = Face(
                user_name=matched_face.user_name,
                user_email=matched_face.user_email,
                user_metadata=matched_face.user_metadata,
                provider_name=matched_face.provider_name,
                provider_face_id=f"verified_{timestamp}",
                provider_collection_id=matched_face.provider_collection_id,
                embedding_insightface=embedding,
                embedding_model=settings.insightface_model if embedding else None,
                image_path=image_path,
                image_storage=settings.storage_backend,
                quality_score=None,
                confidence_score=confidence,
                photo_type="verified",
                verified_at=datetime.utcnow(),
                verified_confidence=confidence,
                verified_by_processor=processor,
            )

            await self.repository.create(verified_face)

            return True

        except Exception:
            # Don't fail recognition if auto-capture fails
            return False
