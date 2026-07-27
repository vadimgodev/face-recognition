from fastapi import APIRouter

from src.api.deps import get_face_service, read_image_upload
from src.api.routes.faces import router as faces_router
from src.api.routes.liveness import router as liveness_router
from src.api.routes.recognition import router as recognition_router
from src.api.routes.webcam import router as webcam_router

router = APIRouter()
router.include_router(recognition_router)
router.include_router(faces_router)
router.include_router(liveness_router)

__all__ = ["get_face_service", "read_image_upload", "router", "webcam_router"]
