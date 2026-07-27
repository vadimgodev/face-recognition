from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from src.api.deps import get_face_service
from src.api.schemas import (
    DeleteFaceResponse,
    FaceListResponse,
    FaceResponse,
    UserPhotosResponse,
)
from src.exceptions import FaceRecognitionError
from src.services.face_service import FaceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/faces", tags=["faces"])


@router.get(
    "",
    response_model=FaceListResponse,
    summary="List all faces",
    description="Get a paginated list of all enrolled faces",
)
async def list_faces(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
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
