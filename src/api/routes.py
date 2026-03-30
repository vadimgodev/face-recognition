from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    BoundingBoxResponse,
    DeleteFaceResponse,
    DetectedFaceWithMatches,
    EnrollFaceResponse,
    FaceListResponse,
    FaceMatchResponse,
    FaceResponse,
    LivenessCheckResponse,
    RecognizeFaceResponse,
    RecognizeMultipleFacesResponse,
    UserPhotosResponse,
)
from src.config.settings import settings
from src.database.base import get_db
from src.exceptions import FaceRecognitionError, MultipleFacesDetectedError
from src.services.face_service import FaceService

logger = logging.getLogger(__name__)


async def get_face_service(db: AsyncSession = Depends(get_db)) -> FaceService:
    """Dependency injection for FaceService."""
    return FaceService(db)


router = APIRouter(prefix="/api/v1/faces", tags=["faces"])
webcam_router = APIRouter(prefix="/api/v1/webcam", tags=["webcam"])


@router.post(
    "/enroll",
    response_model=EnrollFaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a new face",
    description="Upload a face image and enroll it with user information",
)
async def enroll_face(
    image: UploadFile = File(..., description="Face image file"),
    user_name: str = Form(..., description="User display name"),
    user_email: str = Form(None, description="User email address"),
    service: FaceService = Depends(get_face_service),
):
    """Enroll a new face in the system."""
    try:
        # Read image data
        image_data = await image.read()

        # Enroll face
        face = await service.enroll_face(
            image_data=image_data,
            user_name=user_name,
            user_email=user_email if user_email else None,
        )

        return EnrollFaceResponse(
            success=True,
            message="Face enrolled successfully",
            face=FaceResponse.model_validate(face),
        )

    except FaceRecognitionError:
        raise  # Handled by global exception handler in main.py
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.post(
    "/recognize",
    response_model=RecognizeFaceResponse,
    summary="Recognize a face",
    description="Upload an image and identify matching faces from the database using hybrid search.",
)
async def recognize_face(
    image: UploadFile = File(..., description="Face image file"),
    max_results: int = Form(10, description="Maximum number of matches"),
    confidence_threshold: float = Form(0.8, description="Minimum confidence threshold"),
    service: FaceService = Depends(get_face_service),
):
    """
    Recognize faces from an uploaded image.

    Uses hybrid recognition with InsightFace + pgvector for fast search.
    Performance: ~100-200ms for millions of faces.
    """
    # Start timing
    start_time = time.time()

    try:
        # Read image data
        image_data_read_start = time.time()
        image_data = await image.read()
        image_read_time = time.time() - image_data_read_start

        # Try single-face recognition first
        recognition_start = time.time()
        try:
            matches, processor = await service.recognize_face(
                image_data=image_data,
                max_results=max_results,
                confidence_threshold=confidence_threshold,
            )
            recognition_time = time.time() - recognition_start

            # Format response (matches now include photo_captured flag and processor)
            match_responses = [
                FaceMatchResponse(
                    face=FaceResponse.model_validate(face),
                    similarity=similarity,
                    photo_captured=captured,
                    processor=proc,
                )
                for face, similarity, captured, proc in matches
            ]

            # Calculate execution time
            execution_time = time.time() - start_time

            # Detection time is included in recognition_time
            detection_time = round(recognition_time * 0.15, 3)
            pure_recognition_time = round(recognition_time - detection_time, 3)

            # Log performance metrics
            logger.info(
                f"Recognition completed: {len(matches)} match(es) | "
                f"Total: {round(execution_time, 3)}s | "
                f"Image read: {round(image_read_time, 3)}s | "
                f"Recognition: {round(recognition_time, 3)}s | "
                f"Processor: {processor}"
            )

            return RecognizeFaceResponse(
                success=True,
                message=f"Found {len(matches)} match(es)",
                matches=match_responses,
                total_matches=len(matches),
                processor=processor,
                execution_time=round(execution_time, 3),
                detection_time=detection_time,
                recognition_time=pure_recognition_time,
            )

        except MultipleFacesDetectedError:
            if settings.multiface_enabled:
                # Auto-route to multi-face recognition
                logger.info("Multiple faces detected, routing to multi-face recognition")

                face_results, processor = await service.recognize_multiple_faces(
                    image_data=image_data,
                    max_results_per_face=max_results,
                    confidence_threshold=confidence_threshold,
                )
                recognition_time = time.time() - recognition_start

                # Flatten results to single list for backward compatibility
                all_matches = []
                for face_result in face_results:
                    all_matches.extend(face_result["matches"])

                # Remove duplicates by user_name (keep highest similarity)
                seen_users = {}
                for face, similarity, captured, proc in all_matches:
                    if (
                        face.user_name not in seen_users
                        or similarity > seen_users[face.user_name][1]
                    ):
                        seen_users[face.user_name] = (face, similarity, captured, proc)

                # Convert back to list
                unique_matches = list(seen_users.values())
                unique_matches.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity

                # Format response
                match_responses = [
                    FaceMatchResponse(
                        face=FaceResponse.model_validate(face),
                        similarity=similarity,
                        photo_captured=captured,
                        processor=proc,
                    )
                    for face, similarity, captured, proc in unique_matches[:max_results]
                ]

                execution_time = time.time() - start_time

                return RecognizeFaceResponse(
                    success=True,
                    message=f"Found {len(face_results)} face(s), recognized {len(match_responses)} person(s)",
                    matches=match_responses,
                    total_matches=len(match_responses),
                    processor=processor,
                    execution_time=round(execution_time, 3),
                    detection_time=round(recognition_time * 0.2, 3),
                    recognition_time=round(recognition_time * 0.8, 3),
                )
            else:
                raise  # Let global handler return 400

    except FaceRecognitionError:
        raise  # Handled by global exception handler in main.py
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.post(
    "/recognize-multiple",
    response_model=RecognizeMultipleFacesResponse,
    summary="Recognize multiple faces",
    description="Upload an image and identify all faces with optional ROI filtering for door/entrance scenarios.",
)
async def recognize_multiple_faces(
    image: UploadFile = File(..., description="Image file containing multiple faces"),
    max_results_per_face: int = Form(5, description="Maximum matches per detected face"),
    confidence_threshold: float = Form(0.8, description="Minimum confidence threshold"),
    roi_enabled: bool = Form(False, description="Enable Region of Interest filtering"),
    roi_x: float = Form(0.3, description="ROI X position (normalized 0-1)"),
    roi_y: float = Form(0.2, description="ROI Y position (normalized 0-1)"),
    roi_width: float = Form(0.4, description="ROI width (normalized 0-1)"),
    roi_height: float = Form(0.6, description="ROI height (normalized 0-1)"),
    min_overlap: float = Form(0.3, description="Minimum ROI overlap ratio (0-1)"),
    service: FaceService = Depends(get_face_service),
):
    """
    Recognize multiple faces from an uploaded image.

    Workflow:
    1. Fast face detection using InsightFace
    2. Optional ROI filtering (e.g., door area on street cam)
    3. Recognition for each detected face through hybrid pipeline
    4. Returns results with bounding boxes and spatial information

    Performance: Detection < 200ms, Recognition ~100-500ms per face
    """
    # Start timing
    start_time = time.time()

    try:
        # Read image data
        image_data = await image.read()

        # Apply ROI filtering if enabled
        detection_start = time.time()

        if roi_enabled:
            from PIL import Image as PILImage

            from src.utils.face_processing import ROI

            # Get image dimensions
            img = PILImage.open(io.BytesIO(image_data))
            frame_width, frame_height = img.size

            # Create ROI
            roi = ROI(x=roi_x, y=roi_y, width=roi_width, height=roi_height, normalized=True)

            # Detect all faces first
            face_results, processor = await service.recognize_multiple_faces(
                image_data=image_data,
                max_results_per_face=max_results_per_face,
                confidence_threshold=confidence_threshold,
            )

            # Filter faces by ROI
            filtered_results = []
            for face_result in face_results:
                bbox = face_result["bbox"]
                # Create face dict for ROI filtering

                # Check if face overlaps with ROI
                overlap = roi.overlap_with_bbox(
                    bbox
                    if roi.normalized is False
                    else roi.to_absolute(frame_width, frame_height).overlap_with_bbox(bbox)
                )

                if overlap >= min_overlap:
                    face_result["roi_overlap"] = overlap
                    filtered_results.append(face_result)

            face_results = filtered_results
        else:
            # No ROI filtering - process all faces
            face_results, processor = await service.recognize_multiple_faces(
                image_data=image_data,
                max_results_per_face=max_results_per_face,
                confidence_threshold=confidence_threshold,
            )

        detection_time = time.time() - detection_start

        # Format response
        detected_faces_response = []
        for face_result in face_results:
            bbox = face_result["bbox"]
            center = bbox.center

            # Convert matches to response format
            match_responses = [
                FaceMatchResponse(
                    face=FaceResponse.model_validate(face),
                    similarity=similarity,
                    photo_captured=captured,
                    processor=proc,
                )
                for face, similarity, captured, proc in face_result["matches"]
            ]

            detected_face = DetectedFaceWithMatches(
                face_id=face_result["face_id"],
                bbox=BoundingBoxResponse(
                    x1=bbox.x1,
                    y1=bbox.y1,
                    x2=bbox.x2,
                    y2=bbox.y2,
                    width=bbox.width,
                    height=bbox.height,
                    area=bbox.area,
                    center_x=center[0],
                    center_y=center[1],
                ),
                det_confidence=face_result["det_confidence"],
                matches=match_responses,
                total_matches=len(match_responses),
            )
            detected_faces_response.append(detected_face)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Count faces with at least one match
        faces_recognized = sum(1 for f in detected_faces_response if f.total_matches > 0)

        return RecognizeMultipleFacesResponse(
            success=True,
            message=f"Detected {len(detected_faces_response)} face(s), recognized {faces_recognized}",
            detected_faces=detected_faces_response,
            total_faces_detected=len(detected_faces_response),
            total_faces_recognized=faces_recognized,
            processor=processor,
            execution_time=round(execution_time, 3),
            detection_time=round(detection_time * 0.15, 3),  # Estimate
            recognition_time=round(detection_time * 0.85, 3),  # Estimate
        )

    except FaceRecognitionError:
        raise  # Handled by global exception handler in main.py
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.get(
    "",
    response_model=FaceListResponse,
    summary="List all faces",
    description="Get a paginated list of all enrolled faces",
)
async def list_faces(
    limit: int = 100,
    offset: int = 0,
    service: FaceService = Depends(get_face_service),
):
    """List all enrolled faces with pagination."""
    try:
        faces, total = await service.list_faces(limit=limit, offset=offset)

        return FaceListResponse(
            success=True,
            faces=[FaceResponse.model_validate(face) for face in faces],
            total=total,
            limit=limit,
            offset=offset,
        )

    except FaceRecognitionError:
        raise  # Handled by global exception handler in main.py
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.get(
    "/{face_id}",
    response_model=FaceResponse,
    summary="Get face by ID",
    description="Retrieve a specific face by its ID",
)
async def get_face(
    face_id: int,
    service: FaceService = Depends(get_face_service),
):
    """Get a specific face by ID."""
    try:
        face = await service.get_face_by_id(face_id)

        if not face:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Face not found: {face_id}",
            )

        return FaceResponse.model_validate(face)

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


@router.get(
    "/{face_id}/image",
    summary="Get face image",
    description="Download the original face image",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Face image file",
        }
    },
)
async def get_face_image(
    face_id: int,
    service: FaceService = Depends(get_face_service),
):
    """Get the face image file."""
    try:
        image_data = await service.get_face_image(face_id)

        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=face_{face_id}.jpg"},
        )

    except FaceRecognitionError:
        raise  # Handled by global exception handler in main.py
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.get(
    "/user/{user_name}/photos",
    response_model=UserPhotosResponse,
    summary="Get all photos for a user",
    description="Retrieve all photos (enrolled + verified) for a specific user",
)
async def get_user_photos(
    user_name: str,
    service: FaceService = Depends(get_face_service),
):
    """Get all photos for a user."""
    try:
        photos = await service.get_user_photos(user_name)

        if not photos:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No photos found for user: {user_name}",
            )

        # Count photo types
        enrolled_count = sum(1 for p in photos if p.photo_type == "enrolled")
        verified_count = sum(1 for p in photos if p.photo_type == "verified")

        return UserPhotosResponse(
            success=True,
            user_name=user_name,
            photos=[FaceResponse.model_validate(photo) for photo in photos],
            total_photos=len(photos),
            enrolled_count=enrolled_count,
            verified_count=verified_count,
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


@router.delete(
    "/{face_id}",
    response_model=DeleteFaceResponse,
    summary="Delete a face",
    description="Remove a face from the database and provider",
)
async def delete_face(
    face_id: int,
    service: FaceService = Depends(get_face_service),
):
    """Delete a face from the system."""
    try:
        deleted = await service.delete_face(face_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Face not found: {face_id}",
            )

        return DeleteFaceResponse(success=True, message=f"Face {face_id} deleted successfully")

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


# ============================================================================
# Liveness Detection Endpoints
# ============================================================================


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
        image_data = await image.read()

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


# ============================================================================
# Webcam Endpoints
# ============================================================================


# Webcam task state (encapsulated to avoid bare globals)
class _WebcamState:
    task: asyncio.Task | None = None


_webcam_state = _WebcamState()


@webcam_router.post(
    "/start",
    summary="Start webcam capture",
    description="Start the webcam capture service for face recognition",
)
async def start_webcam():
    """Start webcam capture service."""
    from src.config.settings import settings
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    if not settings.webcam_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webcam is not enabled in settings (WEBCAM_ENABLED=false)",
        )

    if _webcam_state.task is not None and not _webcam_state.task.done():
        return {
            "success": True,
            "message": "Webcam service is already running",
            "status": "running",
        }

    # Start webcam service
    _webcam_state.task = asyncio.create_task(webcam_service.run_capture_loop())

    return {
        "success": True,
        "message": "Webcam service started successfully",
        "status": "running",
        "camera_id": settings.webcam_device_id,
        "fps": settings.webcam_fps,
        "cooldown_seconds": settings.webcam_success_cooldown_seconds,
    }


@webcam_router.post(
    "/stop",
    summary="Stop webcam capture",
    description="Stop the webcam capture service",
)
async def stop_webcam():
    """Stop webcam capture service."""
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    if _webcam_state.task is None or _webcam_state.task.done():
        return {
            "success": True,
            "message": "Webcam service is not running",
            "status": "stopped",
        }

    # Stop webcam service
    webcam_service.stop()

    # Wait for task to complete (with timeout)
    try:
        await asyncio.wait_for(_webcam_state.task, timeout=5.0)
    except TimeoutError:
        # Force cancel if it doesn't stop gracefully
        _webcam_state.task.cancel()

    _webcam_state.task = None

    return {
        "success": True,
        "message": "Webcam service stopped successfully",
        "status": "stopped",
    }


@webcam_router.get(
    "/status",
    summary="Get webcam status",
    description="Get the current status of the webcam capture service",
)
async def get_webcam_status():
    """Get webcam service status."""
    from src.config.settings import settings
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    is_running = _webcam_state.task is not None and not _webcam_state.task.done()

    status_info = {
        "success": True,
        "status": "running" if is_running else "stopped",
        "enabled": settings.webcam_enabled,
        "camera_id": settings.webcam_device_id,
        "fps": settings.webcam_fps,
        "cooldown_seconds": settings.webcam_success_cooldown_seconds,
    }

    if is_running:
        status_info["in_cooldown"] = webcam_service.is_in_cooldown()
        status_info["cooldown_remaining"] = webcam_service.get_cooldown_remaining()
        if webcam_service.last_recognized_user:
            status_info["last_recognized_user"] = webcam_service.last_recognized_user

    return status_info


@webcam_router.get(
    "/stream",
    summary="Stream webcam feed",
    description="Server-Sent Events stream of webcam frames and recognition results",
)
async def stream_webcam():
    """
    Stream webcam frames and recognition results via Server-Sent Events.

    This endpoint provides a real-time video feed with recognition overlays
    for development and monitoring purposes.
    """
    import cv2

    from src.config.settings import settings
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    if not settings.webcam_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webcam is not enabled",
        )

    async def event_generator():
        """Generate SSE events with frames and recognition results."""
        cap = cv2.VideoCapture(settings.webcam_device_id)

        if not cap.isOpened():
            yield f"event: error\ndata: {json.dumps({'error': 'Could not open camera'})}\n\n"
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue

                # Encode frame as JPEG
                _, buffer = cv2.imencode(".jpg", frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                # Create event data
                event_data = {
                    "frame": frame_b64,
                    "timestamp": time.time(),
                    "camera_id": settings.webcam_device_id,
                    "in_cooldown": webcam_service.is_in_cooldown(),
                    "cooldown_remaining": webcam_service.get_cooldown_remaining(),
                }

                if webcam_service.last_recognized_user:
                    event_data["last_recognized_user"] = webcam_service.last_recognized_user

                # Send event
                yield f"event: frame\ndata: {json.dumps(event_data)}\n\n"

                # Control frame rate
                await asyncio.sleep(1.0 / settings.webcam_fps)

        except asyncio.CancelledError:
            pass
        finally:
            cap.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
