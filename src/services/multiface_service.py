"""
Multi-face detection and recognition service.

Two-stage pipeline:
1. Fast detection (OpenCV/DNN) finds all faces in an image
2. Accurate recognition (InsightFace + strategy) identifies each face
"""

from io import BytesIO
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.config.settings import settings
from src.database.models import Face
from src.services.auto_capture_service import AutoCaptureService
from src.services.recognition_strategies import RecognitionStrategy
from src.utils.face_processing import crop_face_from_bbox
import logging

logger = logging.getLogger(__name__)


class MultiFaceService:
    """Detection and recognition of multiple faces in a single image."""

    def __init__(
        self,
        insightface_provider,
        face_detector,
        strategy: RecognitionStrategy,
        auto_capture: AutoCaptureService,
    ):
        """
        Args:
            insightface_provider: InsightFace provider for embedding extraction
            face_detector: OpenCV/DNN fast face detector (may be None)
            strategy: Recognition strategy to use per-face
            auto_capture: Auto-capture service for verified photos
        """
        self.insightface_provider = insightface_provider
        self.face_detector = face_detector
        self.strategy = strategy
        self.auto_capture = auto_capture

    async def recognize_multiple(
        self,
        image_data: bytes,
        max_results_per_face: int = 5,
        confidence_threshold: float = 0.8,
    ) -> Tuple[List[dict], str]:
        """
        Recognize multiple faces in a single image.

        TWO-STAGE PIPELINE:
        1. Fast Detection: OpenCV/DNN finds all faces (~20-100ms)
        2. Accurate Recognition: InsightFace + strategy for each face (~100-500ms/face)

        Args:
            image_data: Image bytes containing multiple faces
            max_results_per_face: Maximum matches to return per detected face
            confidence_threshold: Minimum confidence threshold (0-1)

        Returns:
            Tuple of (face_results, processor_name):
            - face_results: List of dicts with face_id, bbox, confidence, matches
            - processor_name: Overall processor used
        """
        if not self.insightface_provider:
            raise ValueError(
                "Multi-face recognition requires InsightFace provider. "
                "Set HYBRID_MODE to 'insightface_only', 'insightface_aws', or 'smart_hybrid'."
            )

        # Step 1: FAST DETECTION
        image_pil = Image.open(BytesIO(image_data))
        image_np = np.array(image_pil.convert("RGB"))
        image_bgr = image_np[:, :, ::-1].copy()  # RGB -> BGR for OpenCV

        if self.face_detector:
            detected_bboxes = self.face_detector.detect_faces(
                image_bgr,
                confidence_threshold=settings.detection_confidence_threshold,
            )
            logger.info(
                f"Fast detection ({settings.face_detection_method}): "
                f"Found {len(detected_bboxes)} faces"
            )
        else:
            # Fallback to InsightFace if face_detector not initialized
            logger.warning(
                "Face detector not initialized, using InsightFace for detection"
            )
            detected_faces_insightface = (
                await self.insightface_provider.detect_multiple_faces(image_data)
            )
            detected_bboxes = [f["bbox"] for f in detected_faces_insightface]

        if not detected_bboxes:
            return (
                [],
                f"detection:{settings.face_detection_method}"
                f"+recognition:{settings.hybrid_mode}",
            )

        # Limit number of faces
        if len(detected_bboxes) > settings.max_faces_per_frame:
            logger.warning(
                f"Detected {len(detected_bboxes)} faces, "
                f"limiting to {settings.max_faces_per_frame}"
            )
            detected_bboxes.sort(key=lambda b: b.area, reverse=True)
            detected_bboxes = detected_bboxes[: settings.max_faces_per_frame]

        # Step 2: ACCURATE RECOGNITION for each detected face
        processor_name = (
            f"detection:{settings.face_detection_method}"
            f"+recognition:{settings.hybrid_mode}"
        )
        face_results = []

        for bbox in detected_bboxes:
            face_result = await self._process_single_detected_face(
                image_np=image_np,
                bbox=bbox,
                max_results=max_results_per_face,
                confidence_threshold=confidence_threshold,
            )
            face_results.append(face_result)

        logger.info(
            f"Processed {len(face_results)} faces, "
            f"found matches for "
            f"{sum(1 for f in face_results if f['matches'])} faces"
        )

        return face_results, processor_name

    async def _process_single_detected_face(
        self,
        image_np: np.ndarray,
        bbox,
        max_results: int,
        confidence_threshold: float,
    ) -> dict:
        """
        Process a single detected face: crop, extract embedding, recognize,
        and optionally auto-capture.

        Returns:
            Dict with face_id, bbox, det_confidence, and matches.
        """
        face_id = bbox.face_id
        det_confidence = bbox.confidence

        # Crop face from image
        face_crop = crop_face_from_bbox(
            image_np,
            bbox,
            padding=settings.face_crop_padding,
        )

        # Convert to bytes for InsightFace
        face_pil = Image.fromarray(face_crop)
        face_bytes_io = BytesIO()
        face_pil.save(face_bytes_io, format="JPEG")
        face_bytes = face_bytes_io.getvalue()

        # Extract embedding
        try:
            embedding = await self.insightface_provider.extract_embedding(
                face_bytes,
                allow_multiple=False,
            )
        except ValueError as e:
            logger.warning(f"Failed to extract embedding for {face_id}: {e}")
            return {
                "face_id": face_id,
                "bbox": bbox,
                "det_confidence": det_confidence,
                "matches": [],
            }

        # Recognize using strategy's embedding path
        matches = await self.strategy.recognize_from_embedding(
            embedding=embedding,
            max_results=max_results,
            confidence_threshold=confidence_threshold,
        )

        # Format matches and auto-capture for best match
        formatted_matches = []
        best_match_face = None
        best_match_processor = None
        best_match_similarity = 0.0

        for match_data in matches:
            if len(match_data) == 2:
                face, similarity = match_data
                aws_used = False
            elif len(match_data) == 3:
                face, similarity, aws_used = match_data
            else:
                continue

            match_processor = _compute_match_processor(aws_used)

            if best_match_face is None:
                best_match_face = face
                best_match_processor = match_processor
                best_match_similarity = similarity

            formatted_matches.append((
                face,
                similarity,
                False,  # photo_captured placeholder
                match_processor,
            ))

        # Auto-capture for best match
        photo_captured = False
        if best_match_face is not None:
            photo_captured = await self.auto_capture.capture_if_eligible(
                image_data=face_bytes,
                matched_face=best_match_face,
                confidence=best_match_similarity,
                processor=best_match_processor,
            )
            if photo_captured and formatted_matches:
                face, similarity, _, processor = formatted_matches[0]
                formatted_matches[0] = (face, similarity, True, processor)

        return {
            "face_id": face_id,
            "bbox": bbox,
            "det_confidence": det_confidence,
            "matches": formatted_matches,
        }


def _compute_match_processor(aws_used: bool) -> str:
    """Determine the processor name for a match based on mode and AWS usage."""
    if settings.hybrid_mode == "smart_hybrid":
        if aws_used:
            return f"{settings.insightface_model}+aws"
        return settings.insightface_model
    elif settings.hybrid_mode == "insightface_aws":
        return f"{settings.insightface_model}+aws"
    elif settings.hybrid_mode == "insightface_only":
        return settings.insightface_model
    else:
        return "aws_rekognition"
