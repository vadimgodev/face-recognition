"""
Recognition strategy implementations.

Each strategy encapsulates a different approach to face recognition:
- Local: Vector search with pgvector + template averaging
- Hybrid: Three-tier confidence with adaptive AWS verification
- Cloud: AWS Rekognition search only
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.config.settings import settings
from src.database.models import Face
from src.database.repository import FaceRepository
from src.services.template_service import TemplateService

logger = logging.getLogger(__name__)


class RecognitionResult:
    """Standardized recognition result."""

    def __init__(self, face: Face, similarity: float, aws_verified: bool = False):
        self.face = face
        self.similarity = similarity
        self.aws_verified = aws_verified


class RecognitionStrategy(ABC):
    """Abstract base for recognition strategies."""

    @abstractmethod
    async def recognize(
        self,
        image_data: bytes,
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple[Face, float, bool]]:
        """
        Recognize faces from raw image data.

        Returns:
            List of (Face, similarity, aws_used) tuples
        """
        pass

    @abstractmethod
    async def recognize_from_embedding(
        self,
        embedding: list[float],
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple]:
        """
        Recognize faces from a pre-extracted embedding.

        Returns:
            List of tuples -- shape depends on strategy.
        """
        pass


class LocalStrategy(RecognitionStrategy):
    """
    Vector search with pgvector + template averaging.

    Performance: ~100-200ms for 20M faces
    Cost: Free (only compute)
    """

    def __init__(
        self, insightface_provider, repository: FaceRepository, template_service: TemplateService
    ):
        self.provider = insightface_provider
        self.repository = repository
        self.template = template_service

    async def recognize(
        self,
        image_data: bytes,
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple[Face, float, bool]]:
        """
        Fast recognition using InsightFace embeddings + pgvector search
        with template averaging.

        Returns:
            List of (Face, template_similarity, aws_used=False) tuples
        """
        query_embedding = await self.provider.extract_embedding(image_data)

        results = await self.repository.search_by_embedding(
            embedding=query_embedding,
            threshold=confidence_threshold,
            limit=max_results * 5,  # Extra candidates for template computation
        )

        template_results = await self.template.compute_template_results(
            query_embedding=query_embedding,
            candidates=results,
            confidence_threshold=confidence_threshold,
            max_results=max_results,
        )

        # Return with aws_used=False
        return [(face, score, False) for face, score in template_results]

    async def recognize_from_embedding(
        self,
        embedding: list[float],
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple[Face, float]]:
        """Recognize from pre-extracted embedding with template averaging."""
        results = await self.repository.search_by_embedding(
            embedding=embedding,
            threshold=confidence_threshold,
            limit=max_results * 5,
        )

        return await self.template.compute_template_results(
            query_embedding=embedding,
            candidates=results,
            confidence_threshold=confidence_threshold,
            max_results=max_results,
        )


class HybridStrategy(RecognitionStrategy):
    """
    Three-tier confidence: high=accept, medium=verify with AWS, low=reject.

    Performance:
    - High confidence (80%): ~100-200ms, no AWS call
    - Medium confidence (10-20%): ~500ms-1s, AWS verification (max 3 calls)
    - Low confidence: Rejected immediately

    Cost: 80-90% cheaper than always-AWS approach
    """

    def __init__(
        self,
        insightface_provider,
        aws_provider,
        repository: FaceRepository,
        template_service: TemplateService,
        storage,
    ):
        self.insightface = insightface_provider
        self.aws_provider = aws_provider
        self.repository = repository
        self.template = template_service
        self.storage = storage

    async def recognize(
        self,
        image_data: bytes,
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple[Face, float, bool]]:
        """
        Smart hybrid recognition with adaptive AWS verification.

        Three-tier strategy:
        1. High confidence (>=high_threshold): Trust InsightFace immediately
        2. Medium confidence (medium..high): Use AWS CompareFaces for verification
        3. Low confidence (<medium): Reject
        """
        query_embedding = await self.insightface.extract_embedding(image_data)

        candidates = await self.repository.search_by_embedding(
            embedding=query_embedding,
            threshold=settings.insightface_medium_confidence,
            limit=3,  # Cost control
        )

        if not candidates:
            return []

        verified_results = []
        high_threshold = settings.insightface_high_confidence
        medium_threshold = settings.insightface_medium_confidence

        for face, similarity in candidates:
            # TIER 1: High confidence -- trust InsightFace immediately
            if similarity >= high_threshold:
                verified_results.append((face, similarity, False))

            # TIER 2: Medium confidence -- AWS verification
            elif similarity >= medium_threshold:
                aws_result = await self._verify_with_aws(image_data, face)
                if aws_result is not None:
                    verified_results.append((face, aws_result, True))

            # TIER 3: Low confidence -- rejected

        # Sort by similarity, filter by threshold
        verified_results.sort(key=lambda x: x[1], reverse=True)
        final_results = [
            (face, score, aws_used)
            for face, score, aws_used in verified_results
            if score >= confidence_threshold
        ]

        # Group by user_name and compute template similarity or keep AWS score
        user_groups: dict = {}
        for face, score, aws_used in final_results:
            if face.user_name not in user_groups:
                user_groups[face.user_name] = {
                    "faces": [],
                    "aws_used": aws_used,
                    "aws_score": score if aws_used else None,
                }
            user_groups[face.user_name]["faces"].append(face)

        final_user_results = []
        for user_name, group_data in user_groups.items():
            faces = group_data["faces"]
            aws_used = group_data["aws_used"]
            aws_score = group_data["aws_score"]

            all_user_faces = await self.repository.get_photos_by_user_name(user_name)
            representative_face = TemplateService.get_representative_face(
                all_user_faces, fallback=faces[0]
            )

            if aws_used and aws_score is not None:
                # AWS is authoritative when used
                final_user_results.append((representative_face, aws_score, True))
            else:
                # InsightFace only -- compute template similarity
                result = await self.template.compute_template_results_single_user(
                    query_embedding, user_name, fallback_face=faces[0]
                )
                if result is not None:
                    rep_face, template_sim = result
                    final_user_results.append((rep_face, template_sim, False))

        final_user_results.sort(key=lambda x: x[1], reverse=True)
        return final_user_results[:max_results]

    async def recognize_from_embedding(
        self,
        embedding: list[float],
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple[Face, float, bool]]:
        """
        Smart hybrid from embedding. For multi-face use, AWS verification is
        not used (we lack the original face crop for CompareFaces).
        """
        candidates = await self.repository.search_by_embedding(
            embedding=embedding,
            threshold=settings.insightface_medium_confidence,
            limit=3,
        )

        if not candidates:
            return []

        high_threshold = settings.insightface_high_confidence
        verified_results = [
            (face, similarity, False)
            for face, similarity in candidates
            if similarity >= high_threshold
        ]

        verified_results.sort(key=lambda x: x[1], reverse=True)
        final_results = [
            (face, score, aws_used)
            for face, score, aws_used in verified_results
            if score >= confidence_threshold
        ]

        # Group by user and compute template similarity
        user_groups: dict = {}
        for face, _score, aws_used in final_results:
            if face.user_name not in user_groups:
                user_groups[face.user_name] = {
                    "faces": [],
                    "aws_used": aws_used,
                }
            user_groups[face.user_name]["faces"].append(face)

        final_user_results = []
        for user_name, group_data in user_groups.items():
            faces = group_data["faces"]
            aws_used = group_data["aws_used"]

            result = await self.template.compute_template_results_single_user(
                embedding, user_name, fallback_face=faces[0]
            )
            if result is not None:
                rep_face, template_sim = result
                final_user_results.append((rep_face, template_sim, aws_used))

        final_user_results.sort(key=lambda x: x[1], reverse=True)
        return final_user_results[:max_results]

    async def _verify_with_aws(self, query_image: bytes, face: Face) -> float | None:
        """
        Verify a medium-confidence match via the cloud provider's CompareFaces.

        Returns the similarity score (0-1) if confirmed, or None.
        """
        if not self.aws_provider:
            return None

        try:
            stored_image = await self.storage.read(face.image_path)
            score: float | None = await self.aws_provider.compare_faces(query_image, stored_image)
        except Exception:
            # Cloud failure -- reject medium-confidence to be safe
            return None

        if score is not None and score >= 0.6:
            return score
        return None


class CloudStrategy(RecognitionStrategy):
    """
    AWS Rekognition search only.

    Performance: ~5s for 20M faces
    Cost: Expensive ($10 per 1K searches)
    """

    def __init__(self, aws_provider, repository: FaceRepository):
        self.aws_provider = aws_provider
        self.repository = repository

    async def recognize(
        self,
        image_data: bytes,
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple[Face, float, bool]]:
        """Full AWS Rekognition search."""
        matches = await self.aws_provider.recognize_face(
            image_data, max_results, confidence_threshold
        )

        results = []
        for match in matches:
            face = await self.repository.get_by_provider_face_id(match.face_id, "aws_rekognition")
            if face:
                results.append((face, match.similarity, False))

        return results

    async def recognize_from_embedding(
        self,
        embedding: list[float],
        max_results: int,
        confidence_threshold: float,
    ) -> list[tuple]:
        """AWS-only does not support embedding-based recognition."""
        raise ValueError("cloud mode does not support multi-face recognition. Use local or hybrid.")


_MODE_STRATEGIES = {
    "local": "local",
    "cloud": "cloud",
    "hybrid": "hybrid",
    "insightface_only": "local",
    "smart_hybrid": "hybrid",
    "insightface_aws": "hybrid",
    "aws_only": "cloud",
}


def create_strategy(
    mode: str,
    insightface_provider=None,
    aws_provider=None,
    repository: FaceRepository | None = None,
    template_service: TemplateService | None = None,
    storage=None,
) -> RecognitionStrategy:
    """
    Factory to create the appropriate strategy for a recognition mode.

    Accepts the canonical modes (local, cloud, hybrid) and their deprecated
    legacy aliases.
    """
    resolved = _MODE_STRATEGIES.get(mode)
    if resolved == "local":
        assert repository is not None and template_service is not None
        return LocalStrategy(insightface_provider, repository, template_service)
    if resolved == "hybrid":
        assert repository is not None and template_service is not None
        return HybridStrategy(
            insightface_provider, aws_provider, repository, template_service, storage
        )
    if resolved == "cloud":
        assert repository is not None
        return CloudStrategy(aws_provider, repository)
    raise ValueError(f"Unknown recognition mode: {mode!r}")
