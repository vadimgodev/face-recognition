"""
Hybrid Face Recognition Service.

Thin orchestration layer that composes focused sub-services:
- TemplateService: Template averaging and cosine similarity
- RecognitionStrategy: Pluggable recognition strategies
- AutoCaptureService: Verified photo auto-capture (FIFO)
- MultiFaceService: Multi-face detection and recognition
"""

import hashlib
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Face
from src.database.repository import FaceRepository
from src.providers.factory import get_insightface_provider, get_aws_provider
from src.providers.collection_manager import get_collection_manager
from src.storage.factory import get_storage
from src.config.settings import settings
from src.utils.face_detector import create_face_detector
from src.cache.redis_client import get_redis_client
from src.services.template_service import TemplateService
from src.services.recognition_strategies import create_strategy
from src.services.auto_capture_service import AutoCaptureService
from src.services.multiface_service import MultiFaceService
import logging

logger = logging.getLogger(__name__)


class HybridFaceService:
    """
    Hybrid face service combining InsightFace and AWS Rekognition.

    Search Modes:
    1. insightface_only - Fast vector search only (~100-200ms for 20M faces)
    2. insightface_aws - Vector search + AWS verification (~500ms-1s)
    3. aws_only - Full AWS search (fallback, ~5s for 20M faces)
    4. smart_hybrid - Adaptive: use AWS only for low-confidence matches
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize hybrid face service.

        Args:
            db_session: Database session
        """
        self.repository = FaceRepository(db_session)
        self.storage = get_storage()
        self.cache = get_redis_client()

        # Initialize providers based on hybrid mode
        self.insightface_provider = None
        self.aws_provider = None

        if settings.hybrid_mode in ["insightface_only", "insightface_aws", "smart_hybrid"]:
            self.insightface_provider = get_insightface_provider()

        # AWS provider is optional for smart_hybrid (only used for low-confidence verification)
        # Required for insightface_aws and aws_only modes
        if settings.hybrid_mode in ["insightface_aws", "aws_only"]:
            self.aws_provider = get_aws_provider()
        elif settings.hybrid_mode == "smart_hybrid":
            # Try to initialize AWS provider, but don't fail if credentials missing
            try:
                self.aws_provider = get_aws_provider()
            except Exception:
                # AWS provider is optional in smart_hybrid mode
                # System will work fine with just InsightFace
                self.aws_provider = None

        # Compose sub-services
        self.template_service = TemplateService(self.repository)

        self.strategy = create_strategy(
            mode=settings.hybrid_mode,
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

        # Initialize fast face detector for multi-face scenarios
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

    async def enroll_face(
        self,
        image_data: bytes,
        user_name: str,
        user_email: Optional[str] = None,
        additional_metadata: Optional[dict] = None,
    ) -> Face:
        """
        Enroll a new face with hybrid providers.

        Depending on hybrid_mode:
        - insightface_only: Extract InsightFace embedding only
        - insightface_aws: Extract both InsightFace + AWS indexing
        - aws_only: AWS indexing only (fallback to old behavior)

        Args:
            image_data: Image bytes
            user_name: User's display name
            user_email: User's email (optional)
            additional_metadata: Additional metadata as dict

        Returns:
            Face model instance

        Raises:
            ValueError: If face enrollment fails
        """
        # Extract InsightFace embedding
        insightface_embedding = None
        if self.insightface_provider:
            insightface_embedding = await self.insightface_provider.extract_embedding(
                image_data
            )

        # Index in AWS Rekognition (if needed and if not smart_hybrid mode)
        aws_face_id = None
        aws_collection_id = None
        if self.aws_provider and settings.hybrid_mode != "smart_hybrid":
            # For smart_hybrid, we don't enroll in AWS upfront (collection-free approach)
            # AWS is only used for on-demand verification of low-confidence matches
            from src.providers.base import FaceMetadata as ProviderMetadata

            # Generate user_id from name for provider metadata
            user_id_for_provider = user_name.lower().replace(" ", "_")
            metadata = ProviderMetadata(
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

            # Get collection ID from collection manager
            collection_manager = get_collection_manager()
            aws_collection_id = collection_manager.get_collection_for_user(
                user_id_for_provider
            )

        # Generate unique image path
        image_hash = hashlib.sha256(image_data).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{user_name}_{timestamp}_{image_hash}.jpg"
        image_path = f"faces/{user_name}/{image_filename}"

        # Save image to storage
        await self.storage.save(image_path, image_data)

        # Create database record
        provider_name = (
            "hybrid"
            if insightface_embedding and aws_face_id
            else ("insightface" if insightface_embedding else "aws_rekognition")
        )

        face = Face(
            user_name=user_name,
            user_email=user_email,
            user_metadata=str(additional_metadata) if additional_metadata else None,
            provider_name=provider_name,
            provider_face_id=aws_face_id or f"insightface_{timestamp}",
            provider_collection_id=aws_collection_id,
            embedding_insightface=insightface_embedding,
            embedding_model=settings.insightface_model if insightface_embedding else None,
            image_path=image_path,
            image_storage=settings.storage_backend,
            quality_score=None,
            confidence_score=None,
            photo_type="enrolled",
            verified_at=None,
            verified_confidence=None,
        )

        # Save to database
        face = await self.repository.create(face)

        return face

    async def recognize_face(
        self,
        image_data: bytes,
        max_results: int = 10,
        confidence_threshold: float = 0.8,
    ) -> Tuple[List[Tuple[Face, float, bool, str]], str]:
        """
        Recognize face using hybrid approach.

        Routes to appropriate search method based on hybrid_mode:
        - insightface_only: Fast vector search
        - insightface_aws: Vector search + AWS verification
        - smart_hybrid: Adaptive AWS usage based on confidence
        - aws_only: Full AWS search (fallback)

        Args:
            image_data: Image bytes
            max_results: Maximum number of matches
            confidence_threshold: Minimum confidence (0-1)

        Returns:
            Tuple of (results, processor_name):
            - results: List of tuples (Face, similarity_score, photo_captured, processor)
            - processor_name: Overall processor used for this recognition
        """
        # Delegate recognition to strategy
        results_with_aws_flag = await self.strategy.recognize(
            image_data, max_results, confidence_threshold
        )

        # Determine base processor name
        base_processor = _base_processor_name()

        # Auto-capture high-confidence matches if enabled
        photo_captured = False
        if results_with_aws_flag:
            best_face, best_similarity, best_aws_used = results_with_aws_flag[0]
            processor_for_capture = _match_processor_name(best_aws_used)

            photo_captured = await self.auto_capture.capture_if_eligible(
                image_data=image_data,
                matched_face=best_face,
                confidence=best_similarity,
                processor=processor_for_capture,
            )

        # Add photo_captured flag and per-match processor to results
        results_with_metadata = []
        for i, (face, score, aws_used) in enumerate(results_with_aws_flag):
            match_processor = _match_processor_name(aws_used)
            results_with_metadata.append((
                face,
                score,
                photo_captured if i == 0 else False,
                match_processor,
            ))

        return results_with_metadata, base_processor

    async def recognize_multiple_faces(
        self,
        image_data: bytes,
        max_results_per_face: int = 5,
        confidence_threshold: float = 0.8,
    ) -> Tuple[List[dict], str]:
        """
        Recognize multiple faces in a single image.

        TWO-STAGE PIPELINE:
        1. Fast Detection: OpenCV/DNN finds all faces (~20-100ms)
        2. Accurate Recognition: InsightFace + smart_hybrid for each face (~100-500ms/face)

        Args:
            image_data: Image bytes containing multiple faces
            max_results_per_face: Maximum matches to return per detected face
            confidence_threshold: Minimum confidence threshold (0-1)

        Returns:
            Tuple of (face_results, processor_name):
            - face_results: List of dicts, each containing:
                - face_id: Sequential ID (face_0, face_1, ...)
                - bbox: BoundingBox object with coordinates
                - confidence: Detection confidence
                - matches: List of recognition results [(Face, similarity, photo_captured, processor)]
            - processor_name: Overall processor used for recognition
        """
        return await self.multiface_service.recognize_multiple(
            image_data, max_results_per_face, confidence_threshold
        )

    # ------------------------------------------------------------------
    # CRUD / utility methods (kept here as they are already concise)
    # ------------------------------------------------------------------

    async def get_face_by_id(self, face_id: int) -> Optional[Face]:
        """Get face by ID."""
        return await self.repository.get_by_id(face_id)

    async def list_faces(
        self, limit: int = 100, offset: int = 0
    ) -> Tuple[List[Face], int]:
        """List all faces with pagination."""
        return await self.repository.list_all(limit, offset)

    async def delete_face(self, face_id: int) -> bool:
        """
        Delete a face from all providers and storage.

        Args:
            face_id: Face UUID

        Returns:
            True if deleted successfully
        """
        face = await self.repository.get_by_id(face_id)
        if not face:
            raise ValueError(f"Face not found: {face_id}")

        # Delete from AWS if indexed there
        if self.aws_provider and face.provider_face_id and face.provider_collection_id:
            try:
                await self.aws_provider.delete_face(
                    face.provider_face_id, face.provider_collection_id
                )
            except Exception:
                # Log error but continue with deletion
                pass

        # Delete image from storage
        try:
            await self.storage.delete(face.image_path)
        except Exception:
            # Log error but continue
            pass

        # Delete from database (this removes InsightFace embedding too)
        deleted = await self.repository.delete(face_id)

        return deleted

    async def get_face_image(self, face_id: int) -> bytes:
        """Get face image data."""
        face = await self.repository.get_by_id(face_id)
        if not face:
            raise ValueError(f"Face not found: {face_id}")

        image_data = await self.storage.read(face.image_path)
        return image_data

    async def get_user_photos(self, user_name: str) -> List[Face]:
        """
        Get all photos (enrolled + verified) for a user.

        Args:
            user_name: User's name

        Returns:
            List of Face records
        """
        return await self.repository.get_photos_by_user_name(user_name)


# ------------------------------------------------------------------
# Module-level helpers for processor naming
# ------------------------------------------------------------------


def _base_processor_name() -> str:
    """Return the base processor name for the current hybrid mode."""
    if settings.hybrid_mode == "insightface_only":
        return f"insightface_{settings.insightface_model}"
    elif settings.hybrid_mode == "smart_hybrid":
        return f"smart_hybrid_{settings.insightface_model}"
    elif settings.hybrid_mode == "insightface_aws":
        return f"hybrid_{settings.insightface_model}+aws"
    else:
        return "aws_rekognition"


def _match_processor_name(aws_used: bool) -> str:
    """Return the per-match processor name based on mode and AWS usage."""
    if settings.hybrid_mode == "smart_hybrid":
        if aws_used:
            return f"{settings.insightface_model}+aws"
        return f"{settings.insightface_model}"
    elif settings.hybrid_mode == "insightface_aws":
        return f"{settings.insightface_model}+aws"
    elif settings.hybrid_mode == "insightface_only":
        return f"{settings.insightface_model}"
    else:
        return "aws_rekognition"
