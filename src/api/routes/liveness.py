from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.deps import read_image_upload
from src.api.schemas import LivenessCheckResponse
from src.config.settings import settings
from src.exceptions import FaceRecognitionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/faces", tags=["faces"])


@router.post(
    "/liveness/check",
    response_model=LivenessCheckResponse,
    summary="Check liveness/anti-spoofing",
    description="Verify if an image contains a real live person (detect spoofing attacks)",
)
async def check_liveness(
    image: UploadFile = File(..., description="Face image file to check"),
    threshold: float = Form(
        None, description="Liveness threshold (0.0-1.0, uses config default if not provided)"
    ),
):
    """
    Check if image contains a real live person using passive liveness detection.

    Detects spoofing attacks including:
    - Printed photo attacks
    - Video replay attacks
    - Basic mask attacks

    Note: This is passive detection (single image analysis, no user interaction required).
    For maximum security, consider active liveness detection methods.
    """
    if not settings.liveness_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Liveness detection is not enabled in settings (LIVENESS_ENABLED=false)",
        )

    try:
        start_time = time.time()

        # Read image data
        image_data = await read_image_upload(image)

        # Get liveness provider
        from src.providers.silent_face_liveness import get_liveness_provider

        liveness_provider = get_liveness_provider(
            device_id=settings.liveness_device_id,
            model_dir=settings.liveness_model_dir,
            detector_path=settings.liveness_detector_path,
        )

        # Use threshold from request or config
        detection_threshold = threshold if threshold is not None else settings.liveness_threshold

        # Validate threshold
        if detection_threshold < 0.0 or detection_threshold > 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Threshold must be between 0.0 and 1.0",
            )

        # Check liveness
        result = await liveness_provider.check_liveness(
            image_bytes=image_data,
            threshold=detection_threshold,
        )

        execution_time = time.time() - start_time

        return LivenessCheckResponse(
            success=True,
            is_real=result.is_real,
            confidence=result.confidence,
            spoofing_type=result.spoofing_type.value,
            threshold=detection_threshold,
            provider=liveness_provider.provider_name,
            details=result.details,
            execution_time=execution_time,
        )

    except HTTPException:
        raise
    except FaceRecognitionError:
        raise  # Handled by global exception handler in main.py
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e
