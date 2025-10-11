import hashlib
from datetime import datetime
from typing import List, Optional
import logging


from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from src.database.models import Face
from src.database.repository import FaceRepository
from src.providers.base import FaceProvider, FaceMetadata
from src.providers.factory import get_face_provider
from src.providers.collection_manager import get_collection_manager
from src.storage.base import StorageBackend
from src.storage.factory import get_storage
from src.config.settings import settings
from src.cache.redis_client import get_redis_client


class FaceService:
    """
    Service layer for face recognition operations.

    Automatically delegates to HybridFaceService if hybrid mode is enabled.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        provider: FaceProvider = None,
        storage: StorageBackend = None,
    ):
        """
        Initialize face service.

        Args:
            db_session: Database session
            provider: Face recognition provider (uses default if None)
            storage: Storage backend (uses default if None)
        """
        self.db_session = db_session
        self.repository = FaceRepository(db_session)
        self.provider = provider or get_face_provider()
        self.storage = storage or get_storage()

        # Lazy-load hybrid service if needed
        self._hybrid_service = None
        self._liveness_provider = None

    def _get_hybrid_service(self):
        """Get or create hybrid service instance."""
        if settings.use_hybrid_recognition:
            if self._hybrid_service is None:
                from src.services.hybrid_face_service import HybridFaceService
                self._hybrid_service = HybridFaceService(self.db_session)
            return self._hybrid_service
        return None

    def _get_liveness_provider(self):
        """Get or create liveness provider instance."""
        if not settings.liveness_enabled:
            return None

        if self._liveness_provider is None:
            from src.providers.silent_face_liveness import get_liveness_provider
            self._liveness_provider = get_liveness_provider(
                device_id=settings.liveness_device_id,
                model_dir=settings.liveness_model_dir,
                detector_path=settings.liveness_detector_path,
            )
        return self._liveness_provider

    async def _check_liveness(self, image_data: bytes, threshold: Optional[float] = None) -> None:
        """
        Check if image contains a real live person.

        Args:
            image_data: Image bytes
            threshold: Liveness threshold (uses config default if None)

        Raises:
            ValueError: If liveness check fails or image is determined to be fake
        """
        liveness_provider = self._get_liveness_provider()

        # SECURITY: Enforce hard requirement when liveness is enabled
        if settings.liveness_enabled and liveness_provider is None:
            raise RuntimeError(
                "SECURITY CRITICAL: Liveness detection is ENABLED in settings but provider is not available. "
                "This is a critical security issue. Check model files and initialization logs."
            )

        if liveness_provider is None:
            return  # Liveness explicitly disabled, skip check

        detection_threshold = threshold if threshold is not None else settings.liveness_threshold

        # Check cache first (results are deterministic for same image + threshold)
        cache = get_redis_client()
        image_hash = hashlib.sha256(image_data).hexdigest()
        cache_key = f"liveness:{image_hash}:{detection_threshold}"

        cached_result = await cache.get_json(cache_key)
        if cached_result is not None:
            logger.debug(f"Liveness cache HIT for image hash {image_hash[:16]}")
            # Cached result format: {"is_real": bool, "error": str or None}
            if not cached_result["is_real"]:
                raise ValueError(cached_result["error"])
            return

        logger.debug(f"Liveness cache MISS for image hash {image_hash[:16]}")

        try:
            result = await liveness_provider.check_liveness(image_data, detection_threshold)

            if not result.is_real:
                error_msg = (
                    f"Liveness check failed: Image appears to be fake "
                    f"(spoofing type: {result.spoofing_type.value}, "
                    f"confidence: {result.confidence:.3f}, "
                    f"threshold: {detection_threshold})"
                )
                logger.warning(
                    f"Liveness check failed: spoofing detected "
                    f"(confidence: {result.confidence:.3f}, threshold: {detection_threshold})"
                )
                # Cache the failure (TTL: 60 seconds)
                await cache.set_json(
                    cache_key,
                    {"is_real": False, "error": error_msg},
                    ex=60
                )
                raise ValueError(error_msg)

            logger.info(f"Liveness check passed (confidence: {result.confidence:.3f})")
            # Cache the success (TTL: 60 seconds)
            await cache.set_json(
                cache_key,
                {"is_real": True, "error": None},
                ex=60
            )

        except ValueError:
            # Re-raise liveness failures
            raise
        except Exception as e:
            logger.error(f"Liveness check error: {e}", exc_info=True)
            raise ValueError(f"Liveness detection failed: {e}") from e

    async def enroll_face(
        self,
        image_data: bytes,
        user_name: str,
        user_email: Optional[str] = None,
        additional_metadata: Optional[dict] = None,
    ) -> Face:
        """
        Enroll a new face.

        Delegates to HybridFaceService if hybrid mode is enabled.

        Args:
            image_data: Image bytes
            user_name: User's display name
            user_email: User's email (optional)
            additional_metadata: Additional metadata as dict

        Returns:
            Face model instance

        Raises:
            ValueError: If face enrollment fails or liveness check fails
            Exception: For other errors
        """
        # Check liveness before enrollment if enabled
        if settings.liveness_enabled and settings.liveness_on_enrollment:
            logger.info("Checking liveness before enrollment")
            await self._check_liveness(image_data)

        # Use hybrid service if enabled
        hybrid_service = self._get_hybrid_service()
        if hybrid_service:
            return await hybrid_service.enroll_face(
                image_data, user_name, user_email, additional_metadata
            )
        # Prepare metadata for provider
        # Generate user_id from name for provider metadata
        user_id_for_provider = user_name.lower().replace(" ", "_")
        metadata = FaceMetadata(
            user_id=user_id_for_provider,
            user_name=user_name,
            user_email=user_email,
            additional_data=additional_metadata,
        )

        # Enroll face with provider
        enrollment_result = await self.provider.enroll_face(image_data, metadata)

        # Generate unique image path
        image_hash = hashlib.sha256(image_data).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{user_name}_{timestamp}_{image_hash}.jpg"
        image_path = f"faces/{user_name}/{image_filename}"

        # Save image to storage
        await self.storage.save(image_path, image_data)

        # Get the collection ID used for this user (sharded or single)
        collection_manager = get_collection_manager()
        collection_id = collection_manager.get_collection_for_user(user_id_for_provider)

        # Create database record
        face = Face(
            user_name=user_name,
            user_email=user_email,
            user_metadata=str(additional_metadata) if additional_metadata else None,
            provider_name=self.provider.provider_name,
            provider_face_id=enrollment_result.face_id,
            provider_collection_id=collection_id,  # Store the actual sharded collection
            embedding=enrollment_result.embedding,
            embedding_model=None,  # Not all providers expose model name
            image_path=image_path,
            image_storage=settings.storage_backend,
            quality_score=enrollment_result.quality_score,
            confidence_score=enrollment_result.confidence,
        )

        # Save to database
        face = await self.repository.create(face)

        return face

    async def recognize_face(
        self,
        image_data: bytes,
        max_results: int = 10,
        confidence_threshold: float = 0.8,
    ) -> List[tuple[Face, float]]:
        """
        Recognize face(s) from image.

        Delegates to HybridFaceService if hybrid mode is enabled.

        Args:
            image_data: Image bytes
            max_results: Maximum number of matches
            confidence_threshold: Minimum confidence (0-1)

        Returns:
            List of tuples (Face, similarity_score)

        Raises:
            ValueError: If no face detected or liveness check fails
            Exception: For other errors

        Performance Note:
            - Hybrid mode (insightface_only): ~100-200ms for millions of faces
            - Hybrid mode (insightface_aws): ~500ms-1s
            - AWS only mode: ~5s for 20M faces (searches all collections)
        """
        # Check liveness before recognition if enabled
        if settings.liveness_enabled and settings.liveness_on_recognition:
            logger.info("Checking liveness before recognition")
            await self._check_liveness(image_data)

        # Use hybrid service if enabled
        hybrid_service = self._get_hybrid_service()
        if hybrid_service:
            return await hybrid_service.recognize_face(
                image_data, max_results, confidence_threshold
            )
        # Search for matches using provider
        matches = await self.provider.recognize_face(
            image_data, max_results, confidence_threshold
        )

        # Fetch Face records from database
        results = []
        for match in matches:
            # Get face from database by provider_face_id
            face = await self.repository.get_by_provider_face_id(
                match.face_id, self.provider.provider_name
            )
            if face:
                results.append((face, match.similarity))

        return results

    async def get_face_by_id(self, face_id: int) -> Optional[Face]:
        """
        Get face by ID.

        Args:
            face_id: Face ID

        Returns:
            Face model or None
        """
        return await self.repository.get_by_id(face_id)

    async def list_faces(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[List[Face], int]:
        """
        List all faces with pagination.

        Args:
            limit: Number of results per page
            offset: Offset for pagination

        Returns:
            Tuple of (faces, total_count)
        """
        return await self.repository.list_all(limit, offset)

    async def delete_face(self, face_id: int) -> bool:
        """
        Delete a face.

        Args:
            face_id: Face UUID

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If face not found
        """
        # Get face from database
        face = await self.repository.get_by_id(face_id)
        if not face:
            raise ValueError(f"Face not found: {face_id}")

        # Delete from provider
        await self.provider.delete_face(face.provider_face_id)

        # Delete image from storage
        try:
            await self.storage.delete(face.image_path)
        except Exception:
            # Log error but don't fail if image deletion fails
            pass

        # Delete from database
        deleted = await self.repository.delete(face_id)

        return deleted

    async def get_face_image(self, face_id: int) -> bytes:
        """
        Get face image data.

        Args:
            face_id: Face UUID

        Returns:
            Image bytes

        Raises:
            ValueError: If face not found
            FileNotFoundError: If image file not found
        """
        face = await self.repository.get_by_id(face_id)
        if not face:
            raise ValueError(f"Face not found: {face_id}")

        image_data = await self.storage.read(face.image_path)
        return image_data

    async def get_user_photos(self, user_name: str) -> List[Face]:
        """
        Get all photos (enrolled + verified) for a user.

        Delegates to HybridFaceService if hybrid mode is enabled.

        Args:
            user_name: User's name

        Returns:
            List of Face records
        """
        # Use hybrid service if enabled
        hybrid_service = self._get_hybrid_service()
        if hybrid_service:
            return await hybrid_service.get_user_photos(user_name)

        # Fallback: use repository directly
        return await self.repository.get_photos_by_user_name(user_name)

    async def recognize_multiple_faces(
        self,
        image_data: bytes,
        max_results_per_face: int = 5,
        confidence_threshold: float = 0.8,
    ):
        """
        Recognize multiple faces from an image.

        Delegates to HybridFaceService (required for multi-face support).

        Args:
            image_data: Image bytes containing multiple faces
            max_results_per_face: Maximum matches per detected face
            confidence_threshold: Minimum confidence threshold (0-1)

        Returns:
            Tuple of (face_results, processor_name)
            - face_results: List of dicts with face detection and recognition data
            - processor_name: Recognition processor used

        Raises:
            ValueError: If hybrid mode is not enabled or liveness check fails
        """
        # Check liveness before recognition if enabled
        if settings.liveness_enabled and settings.liveness_on_recognition:
            logger.info("Checking liveness before multi-face recognition")
            await self._check_liveness(image_data)

        # Multi-face recognition requires hybrid service (InsightFace provider)
        hybrid_service = self._get_hybrid_service()
        if not hybrid_service:
            raise ValueError(
                "Multi-face recognition requires hybrid mode. "
                "Set USE_HYBRID_RECOGNITION=true and HYBRID_MODE to "
                "'insightface_only', 'insightface_aws', or 'smart_hybrid'."
            )

        return await hybrid_service.recognize_multiple_faces(
            image_data=image_data,
            max_results_per_face=max_results_per_face,
            confidence_threshold=confidence_threshold,
        )
