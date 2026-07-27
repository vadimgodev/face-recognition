from __future__ import annotations

import io
import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.api.deps import get_face_service, read_image_upload
from src.api.schemas import (
    BoundingBoxResponse,
    DetectedFaceWithMatches,
    EnrollFaceResponse,
    FaceMatchResponse,
    FaceResponse,
    RecognizeFaceResponse,
    RecognizeMultipleFacesResponse,
)
from src.config.settings import settings
from src.exceptions import FaceRecognitionError, MultipleFacesDetectedError
from src.services.face_service import FaceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/faces", tags=["faces"])


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
        image_data = await read_image_upload(image)

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
        image_data = await read_image_upload(image)
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
                detection_time=None,
                recognition_time=None,
            )

        except MultipleFacesDetectedError:
            if settings.multiface_enabled:
                # Auto-route to multi-face recognition
                logger.info("Multiple faces detected, routing to multi-face recognition")

                face_results, processor, detection_time, recognition_time = (
                    await service.recognize_multiple_faces(
                        image_data=image_data,
                        max_results_per_face=max_results,
                        confidence_threshold=confidence_threshold,
                    )
                )

                # Flatten results to single list for backward compatibility
                all_matches = []
                for face_result in face_results:
                    all_matches.extend(face_result["matches"])

                # Remove duplicates by user_name (keep highest similarity)
                seen_users: dict[str, tuple] = {}
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
                    detection_time=round(detection_time, 3),
                    recognition_time=round(recognition_time, 3),
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
        image_data = await read_image_upload(image)

        if roi_enabled:
            from PIL import Image as PILImage

            from src.utils.face_processing import ROI

            # Get image dimensions
            img = PILImage.open(io.BytesIO(image_data))
            frame_width, frame_height = img.size

            # Create ROI
            roi = ROI(x=roi_x, y=roi_y, width=roi_width, height=roi_height, normalized=True)

            # Detect all faces first
            face_results, processor, detection_time, recognition_time = (
                await service.recognize_multiple_faces(
                    image_data=image_data,
                    max_results_per_face=max_results_per_face,
                    confidence_threshold=confidence_threshold,
                )
            )

            # Filter faces by ROI (convert to absolute pixel coordinates first)
            abs_roi = roi.to_absolute(frame_width, frame_height) if roi.normalized else roi
            filtered_results = []
            for face_result in face_results:
                bbox = face_result["bbox"]

                # Check if face overlaps with ROI
                overlap = abs_roi.overlap_with_bbox(bbox)

                if overlap >= min_overlap:
                    face_result["roi_overlap"] = overlap
                    filtered_results.append(face_result)

            face_results = filtered_results
        else:
            # No ROI filtering - process all faces
            face_results, processor, detection_time, recognition_time = (
                await service.recognize_multiple_faces(
                    image_data=image_data,
                    max_results_per_face=max_results_per_face,
                    confidence_threshold=confidence_threshold,
                )
            )

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
            detection_time=round(detection_time, 3),
            recognition_time=round(recognition_time, 3),
        )

    except FaceRecognitionError:
        raise  # Handled by global exception handler in main.py
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e
