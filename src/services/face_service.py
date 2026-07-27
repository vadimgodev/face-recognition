"""
Face recognition service.

Single service layer over the strategy engine, composing focused sub-services:
- TemplateService: Template averaging and cosine similarity
- RecognitionStrategy: Pluggable recognition strategies (local/hybrid/cloud)
- AutoCaptureService: Verified photo auto-capture (FIFO)
- MultiFaceService: Multi-face detection and recognition
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from src.cache.redis_client import get_redis_client
from src.config.settings import settings
from src.database.models import Face
from src.database.repository import FaceRepository
from src.exceptions import (
    InvalidImageError,
    LivenessCheckFailedError,
    NoFaceDetectedError,
)
from src.providers.base import FaceMetadata, FaceProvider
from src.providers.collection_manager import get_collection_manager
from src.providers.factory import get_aws_provider, get_insightface_provider
from src.providers.liveness_base import LivenessProvider
from src.services.auto_capture_service import AutoCaptureService
from src.services.multiface_service import MultiFaceService
from src.services.recognition_strategies import create_strategy
from src.services.template_service import TemplateService
from src.storage.factory import get_storage
from src.utils.face_detector import create_face_detector

logger = logging.getLogger(__name__)


class FaceService:
    """
    Service layer for face recognition operations.

    Recognition modes (RECOGNITION_MODE):
    1. local  - Fast vector search only (~100-200ms for 20M faces)
    2. cloud  - Full AWS search (~5s for 20M faces)
    3. hybrid - Adaptive: use AWS only for low-confidence matches
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize face service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.repository = FaceRepository(db_session)
        self.storage = get_storage()

        self.insightface_provider = None
        self.aws_provider = None

        if settings.recognition_mode in ["local", "hybrid"]:
            self.insightface_provider = get_insightface_provider()

        if settings.recognition_mode == "cloud":
            self.aws_provider = get_aws_provider()
        elif settings.recognition_mode == "hybrid":
            try:
                self.aws_provider = get_aws_provider()
            except Exception:
                self.aws_provider = None

        self.provider = (
            self.aws_provider if settings.recognition_mode == "cloud" else self.insightface_provider
        )

        self.template_service = TemplateService(self.repository)

        self.strategy = create_strategy(
            mode=settings.recognition_mode,
            insightface_provider=self.insightface_provider,
            aws_provider=self.aws_provider,
            repository=self.repository,
            template_service=self.template_service,
            storage=self.storage,
        )

        self.auto_capture = AutoCaptureService(
            repository=self.repository,
            storage=self.storage,
            insightface_provider=self.insightface_provider,
        )

        self.face_detector = None
        if settings.multiface_enabled:
            self.face_detector = create_face_detector(
                method=settings.face_detection_method,
                min_face_size=settings.min_face_size,
                confidence_threshold=settings.detection_confidence_threshold,
            )
            logger.info(
                f"Initialized {settings.face_detection_method} face detector "
                f"(min_size: {settings.min_face_size}px)"
            )

        self.multiface_service = MultiFaceService(
            insightface_provider=self.insightface_provider,
            face_detector=self.face_detector,
            strategy=self.strategy,
            auto_capture=self.auto_capture,
        )

        self._liveness_provider: LivenessProvider | None = None

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

    async def _check_liveness(self, image_data: bytes, threshold: float | None = None) -> None:
        """
        Check if image contains a real live person.

        Args:
            image_data: Image bytes
            threshold: Liveness threshold (uses config default if None)

        Raises:
            LivenessCheckFailedError: If the image is determined to be fake
        """
        liveness_provider = self._get_liveness_provider()

        if settings.liveness_enabled and liveness_provider is None:
            raise RuntimeError(
                "SECURITY CRITICAL: Liveness detection is ENABLED in settings but provider is not available. "
                "This is a critical security issue. Check model files and initialization logs."
            )

        if liveness_provider is None:
            return

        detection_threshold = threshold if threshold is not None else settings.liveness_threshold

        cache = get_redis_client()
        image_hash = hashlib.sha256(image_data).hexdigest()
        cache_key = f"liveness:{image_hash}:{detection_threshold}"

        cached_result = await cache.get_json(cache_key)
        if cached_result is not None:
            logger.debug(f"Liveness cache HIT for image hash {image_hash[:16]}")
            if not cached_result["is_real"]:
                # Cache only stores spoof failures (provider errors aren't cached)
                raise LivenessCheckFailedError(
                    confidence=0.0,
                    spoofing_type="cached",
                    threshold=detection_threshold,
                )
            return

        logger.debug(f"Liveness cache MISS for image hash {image_hash[:16]}")

        try:
            result = await liveness_provider.check_liveness(image_data, detection_threshold)

            if not result.is_real:
                logger.warning(
                    f"Liveness check failed: spoofing detected "
                    f"(confidence: {result.confidence:.3f}, threshold: {detection_threshold})"
                )
                exc = LivenessCheckFailedError(
                    confidence=result.confidence,
                    spoofing_type=result.spoofing_type.value,
                    threshold=detection_threshold,
                )
                await cache.set_json(cache_key, {"is_real": False, "error": exc.message}, ex=60)
                raise exc

            logger.info(f"Liveness check passed (confidence: {result.confidence:.3f})")
            await cache.set_json(cache_key, {"is_real": True, "error": None}, ex=60)

        except LivenessCheckFailedError:
            raise
        except ValueError as e:
            # Inner provider raises ValueError for "No face detected" / invalid image.
            # Map to typed exceptions so the global handler returns 400 with a usable detail.
            err = str(e)
            prefix = "Liveness check failed: "
            detail = err if err.startswith(prefix) else f"{prefix}{err}"
            logger.warning(f"Liveness provider validation error: {err}")
            if "No face detected" in err:
                raise NoFaceDetectedError(detail) from e
            raise InvalidImageError(detail) from e
        except Exception as e:
            logger.error(f"Liveness check error: {e}", exc_info=True)
            raise LivenessCheckFailedError(
                confidence=0.0, spoofing_type="error", threshold=detection_threshold
            ) from e

    def _new_provider_face_id(self, prefix: str) -> str:
        """Generate a collision-safe provider face id."""
        return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    async def _create_face_record(self, **fields) -> Face:
        """Create a Face row, rolling the session back if the insert fails."""
        try:
            return await self.repository.create(Face(**fields))
        except Exception:
            try:
                await self.repository.session.rollback()
            except Exception:
                pass
            raise

    async def enroll_face(
        self,
        image_data: bytes,
        user_name: str,
        user_email: str | None = None,
        additional_metadata: dict | None = None,
    ) -> Face:
        """
        Enroll a new face.

        Extracts the local embedding (local/hybrid modes), indexes into the
        cloud provider (cloud mode only), saves the image and creates the
        database record. Cleans up the stored image if the insert fails.

        Args:
            image_data: Image bytes
            user_name: User's display name
            user_email: User's email (optional)
            additional_metadata: Additional metadata as dict

        Returns:
            Face model instance

        Raises:
            LivenessCheckFailedError: If the liveness gate is enabled and fails
            ValueError: If face enrollment fails
        """
        if settings.liveness_enabled and settings.liveness_on_enrollment:
            logger.info("Checking liveness before enrollment")
            await self._check_liveness(image_data)

        insightface_embedding = None
        if self.insightface_provider:
            insightface_embedding = await self.insightface_provider.extract_embedding(image_data)

        aws_face_id = None
        aws_collection_id = None
        if self.aws_provider and settings.recognition_mode != "hybrid":
            user_id_for_provider = user_name.lower().replace(" ", "_")
            metadata = FaceMetadata(
                user_id=user_id_for_provider,
                user_name=user_name,
                user_email=user_email,
                additional_data=additional_metadata,
            )

            enrollment_result = await self.aws_provider.enroll_face(
                image_bytes=image_data,
                metadata=metadata,
            )

            aws_face_id = enrollment_result.face_id

            collection_manager = get_collection_manager()
            aws_collection_id = collection_manager.get_collection_for_user(user_id_for_provider)

        image_hash = hashlib.sha256(image_data).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{user_name}_{timestamp}_{image_hash}.jpg"
        image_path = f"faces/{user_name}/{image_filename}"

        await self.storage.save(image_path, image_data)

        provider_name = (
            "hybrid"
            if insightface_embedding and aws_face_id
            else ("insightface" if insightface_embedding else "aws_rekognition")
        )

        try:
            return await self._create_face_record(
                user_name=user_name,
                user_email=user_email,
                user_metadata=str(additional_metadata) if additional_metadata else None,
                provider_name=provider_name,
                provider_face_id=aws_face_id or self._new_provider_face_id("insightface"),
                provider_collection_id=aws_collection_id,
                embedding_local=insightface_embedding,
                embedding_model=settings.insightface_model if insightface_embedding else None,
                image_path=image_path,
                image_storage=settings.storage_backend,
                quality_score=None,
                confidence_score=None,
                photo_type="enrolled",
                verified_at=None,
                verified_confidence=None,
            )
        except Exception:
            try:
                await self.storage.delete(image_path)
            except Exception:
                logger.warning(f"Could not clean up orphaned image {image_path}")
            raise

    async def recognize_face(
        self,
        image_data: bytes,
        max_results: int = 10,
        confidence_threshold: float = 0.8,
    ) -> tuple[list[tuple[Face, float, bool, str]], str]:
        """
        Recognize face(s) from image using the configured strategy.

        Args:
            image_data: Image bytes
            max_results: Maximum number of matches
            confidence_threshold: Minimum confidence (0-1)

        Returns:
            Tuple of (results, processor_name):
            - results: List of tuples (Face, similarity_score, photo_captured, processor)
            - processor_name: Overall processor used for this recognition
        """
        if settings.liveness_enabled and settings.liveness_on_recognition:
            logger.info("Checking liveness before recognition")
            await self._check_liveness(image_data)

        results_with_aws_flag = await self.strategy.recognize(
            image_data, max_results, confidence_threshold
        )

        base_processor = _base_processor_name()

        photo_captured = False
        if results_with_aws_flag:
            best_face, best_similarity, best_aws_used = results_with_aws_flag[0]
            photo_captured = await self.auto_capture.capture_if_eligible(
                image_data=image_data,
                matched_face=best_face,
                confidence=best_similarity,
                processor=_match_processor_name(best_aws_used),
            )

        results_with_metadata = []
        for i, (face, score, aws_used) in enumerate(results_with_aws_flag):
            results_with_metadata.append(
                (
                    face,
                    score,
                    photo_captured if i == 0 else False,
                    _match_processor_name(aws_used),
                )
            )

        return results_with_metadata, base_processor

    async def recognize_multiple_faces(
        self,
        image_data: bytes,
        max_results_per_face: int = 5,
        confidence_threshold: float = 0.8,
    ) -> tuple[list[dict], str, float, float]:
        """
        Recognize multiple faces in a single image.

        Args:
            image_data: Image bytes containing multiple faces
            max_results_per_face: Maximum matches per detected face
            confidence_threshold: Minimum confidence threshold (0-1)

        Returns:
            Tuple of (face_results, processor_name, detection_time, recognition_time)
        """
        if settings.liveness_enabled and settings.liveness_on_recognition:
            logger.info("Checking liveness before multi-face recognition")
            await self._check_liveness(image_data)

        return await self.multiface_service.recognize_multiple(
            image_data, max_results_per_face, confidence_threshold
        )

    async def get_face_by_id(self, face_id: int) -> Face | None:
        """Get face by ID."""
        return await self.repository.get_by_id(face_id)

    async def list_faces(self, limit: int = 100, offset: int = 0) -> tuple[list[Face], int]:
        """List all faces with pagination."""
        return await self.repository.list_all(limit, offset)

    async def delete_face(self, face_id: int) -> bool:
        """
        Delete a face from provider, storage and database.

        Args:
            face_id: Face ID

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If face not found
        """
        face = await self.repository.get_by_id(face_id)
        if not face:
            raise ValueError(f"Face not found: {face_id}")

        await cast(FaceProvider, self.provider).delete_face(
            face.provider_face_id, collection_id=face.provider_collection_id
        )

        try:
            await self.storage.delete(face.image_path)
        except Exception as e:
            logger.warning(f"Could not delete image {face.image_path}: {e}")

        return await self.repository.delete(face_id)

    async def get_face_image(self, face_id: int) -> bytes:
        """
        Get face image data.

        Raises:
            ValueError: If face not found
        """
        face = await self.repository.get_by_id(face_id)
        if not face:
            raise ValueError(f"Face not found: {face_id}")

        return await self.storage.read(face.image_path)

    async def get_user_photos(self, user_name: str) -> list[Face]:
        """Get all photos (enrolled + verified) for a user."""
        return await self.repository.get_photos_by_user_name(user_name)


def _base_processor_name() -> str:
    """Return the base processor name for the current recognition mode."""
    if settings.recognition_mode == "local":
        return f"insightface_{settings.insightface_model}"
    elif settings.recognition_mode == "hybrid":
        return f"hybrid_{settings.insightface_model}"
    else:
        return "aws_rekognition"


def _match_processor_name(aws_used: bool) -> str:
    """Return the per-match processor name based on mode and AWS usage."""
    if settings.recognition_mode == "hybrid":
        if aws_used:
            return f"{settings.insightface_model}+aws"
        return f"{settings.insightface_model}"
    elif settings.recognition_mode == "local":
        return f"{settings.insightface_model}"
    else:
        return "aws_rekognition"
