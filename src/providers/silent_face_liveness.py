"""Silent-Face anti-spoofing liveness detection provider."""

from __future__ import annotations

import asyncio
import logging
import threading

from src.antispoof.predictor import AntiSpoofPredictor
from src.providers.liveness_base import LivenessProvider, LivenessResult, SpoofingType

logger = logging.getLogger(__name__)


class SilentFaceLivenessProvider(LivenessProvider):
    """
    Passive liveness detection using Silent-Face-Anti-Spoofing.

    This provider uses MiniFASNet models to detect spoofing attacks from
    a single image without requiring user interaction.

    Detection capabilities:
    - Printed photo attacks
    - Video replay attacks
    - Basic mask attacks

    Limitations:
    - Less robust than active liveness (challenge-response)
    - May struggle with high-quality 3D masks or sophisticated deepfakes
    - Performance depends on image quality and lighting
    """

    def __init__(
        self,
        device_id: int = -1,
        model_dir: str = "./models/anti_spoof",
        detector_path: str = "./models",
    ):
        """
        Initialize Silent-Face liveness provider.

        Args:
            device_id: GPU device ID (-1 for CPU, 0+ for GPU)
            model_dir: Directory containing anti-spoofing model weights
            detector_path: Path to face detector models
        """
        self.device_id = device_id
        self.model_dir = model_dir
        self.detector_path = detector_path
        self._predictor: AntiSpoofPredictor | None = None
        self._lock = threading.Lock()  # Prevent concurrent model loading

    def _get_predictor(self) -> AntiSpoofPredictor:
        """
        Lazy load the predictor with thread safety.

        Returns:
            Initialized AntiSpoofPredictor instance

        Raises:
            RuntimeError: If predictor initialization fails
        """
        # Double-checked locking pattern
        if self._predictor is None:
            with self._lock:
                # Check again inside lock to prevent race condition
                if self._predictor is None:
                    try:
                        logger.info(
                            f"Loading Silent-Face anti-spoofing models from {self.model_dir}"
                        )
                        self._predictor = AntiSpoofPredictor(
                            device_id=self.device_id,
                            model_dir=self.model_dir,
                            detector_path=self.detector_path,
                        )
                        logger.info("✅ Silent-Face anti-spoofing models loaded successfully")
                    except Exception as e:
                        logger.error(
                            f"Failed to load Silent-Face anti-spoofing models: {e}", exc_info=True
                        )
                        # Don't set _predictor on failure - allow retry on next call
                        # Propagate the error to caller
                        raise RuntimeError(
                            f"Failed to initialize liveness detection models. "
                            f"Check that model files exist in {self.model_dir} and {self.detector_path}. "
                            f"Error: {e}"
                        ) from e

        return self._predictor

    async def check_liveness(self, image_bytes: bytes, threshold: float = 0.5) -> LivenessResult:
        """
        Check if image contains a real live person (passive detection).

        Args:
            image_bytes: Image data as bytes
            threshold: Liveness threshold (0-1). Scores above this are considered real.
                      Default 0.5. Higher threshold = stricter (fewer false accepts).

        Returns:
            LivenessResult with detection details

        Raises:
            ValueError: If no face detected or image is invalid
            Exception: For other processing errors
        """
        try:
            # Run prediction in thread pool with global lock
            # OpenCV DNN has global state that's not thread-safe
            loop = asyncio.get_event_loop()
            predictor = self._get_predictor()

            def _predict_with_lock():
                with _inference_lock:
                    return predictor.predict(image_bytes, return_bbox=True)

            real_score, bbox = await loop.run_in_executor(None, _predict_with_lock)

            # Determine if real or fake
            is_real = real_score >= threshold

            # Classify spoofing type (simple heuristic based on confidence)
            if is_real:
                spoofing_type = SpoofingType.REAL
            else:
                # Without more sophisticated analysis, we can't distinguish
                # between print, video, and mask attacks - classify as unknown
                spoofing_type = SpoofingType.UNKNOWN

            logger.info(
                f"Liveness check: {'REAL' if is_real else 'FAKE'} "
                f"(score: {real_score:.3f}, threshold: {threshold})"
            )

            return LivenessResult(
                is_real=is_real,
                confidence=real_score,
                spoofing_type=spoofing_type,
                details={
                    "real_score": real_score,
                    "fake_score": 1.0 - real_score,
                    "threshold": threshold,
                    "bbox": bbox,
                    "model": "MiniFASNet",
                },
            )

        except ValueError as e:
            # Re-raise validation errors (no face, invalid image, etc.)
            raise ValueError(f"Liveness check failed: {e}") from e

        except Exception as e:
            logger.error(f"Error during liveness check: {e}", exc_info=True)
            raise Exception(f"Liveness detection error: {e}") from e

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "silent_face_antispoof"

    @property
    def is_passive(self) -> bool:
        """Silent-Face is a passive liveness detector."""
        return True


# Global singleton instance and locks
_liveness_provider_instance: SilentFaceLivenessProvider | None = None
_instance_lock = threading.Lock()

# CRITICAL: Global lock for all OpenCV DNN operations
# OpenCV DNN has global state and is NOT thread-safe. This lock must be used
# whenever calling predictor.predict() or any other OpenCV DNN inference.
# The predictor has its own internal lock for defense in depth, but this
# global lock ensures no concurrent OpenCV operations across different predictors.
_inference_lock = threading.Lock()


def get_liveness_provider(
    device_id: int = -1,
    model_dir: str = "./models/anti_spoof",
    detector_path: str = "./models",
) -> LivenessProvider:
    """
    Factory function to get liveness provider instance (singleton).

    Args:
        device_id: GPU device ID (-1 for CPU)
        model_dir: Anti-spoofing models directory
        detector_path: Face detector models path

    Returns:
        LivenessProvider instance (singleton)
    """
    global _liveness_provider_instance

    if _liveness_provider_instance is None:
        with _instance_lock:
            if _liveness_provider_instance is None:
                _liveness_provider_instance = SilentFaceLivenessProvider(
                    device_id=device_id,
                    model_dir=model_dir,
                    detector_path=detector_path,
                )

    return _liveness_provider_instance
