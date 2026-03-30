from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FaceMatch:
    """Result of a face recognition match."""

    face_id: str  # Provider's face ID
    confidence: float  # Confidence score (0-1)
    similarity: float  # Similarity score (0-1)
    user_id: str | None = None  # User ID (if available from metadata)
    bounding_box: dict | None = None  # Face bounding box coordinates


@dataclass
class EnrollmentResult:
    """Result of face enrollment."""

    face_id: str  # Provider's face ID
    confidence: float  # Quality/confidence score
    bounding_box: dict | None = None  # Face bounding box coordinates
    quality_score: float | None = None  # Face quality score
    embedding: list[float] | None = None  # Face embedding (if available)


@dataclass
class FaceMetadata:
    """Metadata for face storage in provider."""

    user_id: str
    user_name: str
    user_email: str | None = None
    additional_data: dict | None = None


class FaceProvider(ABC):
    """
    Abstract base class for face recognition providers.

    Implementations should provide concrete methods for:
    - AWS Rekognition
    - Azure Face API
    - Google Cloud Vision
    - etc.
    """

    @abstractmethod
    async def enroll_face(self, image_bytes: bytes, metadata: FaceMetadata) -> EnrollmentResult:
        """
        Enroll a face in the provider's database.

        Args:
            image_bytes: Image data as bytes
            metadata: User metadata to associate with the face

        Returns:
            EnrollmentResult with face_id and other details

        Raises:
            ValueError: If no face found or multiple faces detected
            Exception: For provider-specific errors
        """
        pass

    @abstractmethod
    async def recognize_face(
        self, image_bytes: bytes, max_results: int = 10, confidence_threshold: float = 0.8
    ) -> list[FaceMatch]:
        """
        Recognize faces in an image.

        Args:
            image_bytes: Image data as bytes
            max_results: Maximum number of matches to return
            confidence_threshold: Minimum confidence threshold (0-1)

        Returns:
            List of FaceMatch objects sorted by confidence (descending)

        Raises:
            ValueError: If no face found in image
            Exception: For provider-specific errors
        """
        pass

    @abstractmethod
    async def delete_face(self, face_id: str, collection_id: str = None) -> bool:
        """
        Delete a face from the provider's database.

        Args:
            face_id: Provider's face ID
            collection_id: Optional collection ID override

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            Exception: For provider-specific errors
        """
        pass

    @abstractmethod
    async def get_face_details(self, face_id: str) -> dict | None:
        """
        Get face details from provider.

        Args:
            face_id: Provider's face ID

        Returns:
            Dictionary with face details or None if not found

        Raises:
            Exception: For provider-specific errors
        """
        pass

    @abstractmethod
    async def initialize_collection(self, collection_id: str) -> bool:
        """
        Initialize/create a face collection if it doesn't exist.

        Args:
            collection_id: Collection identifier

        Returns:
            True if created or already exists

        Raises:
            Exception: For provider-specific errors
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider (e.g., 'aws_rekognition')."""
        pass

    @property
    @abstractmethod
    def supports_embeddings(self) -> bool:
        """Return whether this provider exposes face embeddings."""
        pass
