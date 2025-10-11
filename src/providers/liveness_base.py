"""
Base classes for liveness detection providers.

Liveness detection helps prevent spoofing attacks such as:
- Photo attacks (printed photos)
- Video replay attacks
- Mask attacks (2D/3D masks)
- Deep fake attacks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SpoofingType(str, Enum):
    """Types of spoofing attacks that can be detected."""

    REAL = "real"  # Real live person
    PRINT = "print"  # Printed photo attack
    VIDEO = "video"  # Video replay attack
    MASK = "mask"  # 2D/3D mask attack
    UNKNOWN = "unknown"  # Unknown/unclassified attack


@dataclass
class LivenessResult:
    """Result of liveness detection check."""

    is_real: bool  # True if real person detected
    confidence: float  # Confidence score (0-1)
    spoofing_type: SpoofingType  # Type of attack detected (if any)
    details: Optional[dict] = None  # Additional provider-specific details


class LivenessProvider(ABC):
    """
    Abstract base class for liveness detection providers.

    Implementations should provide concrete methods for:
    - Passive liveness (single image analysis)
    - Active liveness (challenge-response, movement detection)
    """

    @abstractmethod
    async def check_liveness(
        self, image_bytes: bytes, threshold: float = 0.5
    ) -> LivenessResult:
        """
        Check if the image contains a real live person.

        Args:
            image_bytes: Image data as bytes
            threshold: Liveness threshold (0-1). Images with scores above
                      this threshold are considered real.

        Returns:
            LivenessResult with detection details

        Raises:
            ValueError: If image is invalid or cannot be processed
            Exception: For provider-specific errors
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the liveness provider."""
        pass

    @property
    @abstractmethod
    def is_passive(self) -> bool:
        """
        Return whether this is a passive liveness detector.

        Passive: Analyzes single image, no user interaction
        Active: Requires user interaction (head movement, blinking, etc.)
        """
        pass
