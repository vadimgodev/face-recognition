"""
Fast face detection using OpenCV.

This module provides lightweight face detection optimized for speed.
Separates detection from recognition for multi-face scenarios.
"""

import logging
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np
import cv2
from pathlib import Path

from src.utils.face_processing import BoundingBox

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Face detection methods."""
    HAAR_CASCADE = "haar"
    DNN = "dnn"
    INSIGHTFACE = "insightface"


class FastFaceDetector:
    """
    Fast face detector using OpenCV.

    Supports multiple detection methods:
    - Haar Cascade: Fastest, good for frontal faces (~20-50ms)
    - DNN: More accurate, handles various angles (~50-100ms)
    - InsightFace: Fallback to InsightFace detector (~100-200ms)
    """

    def __init__(
        self,
        method: DetectionMethod = DetectionMethod.DNN,
        min_face_size: Tuple[int, int] = (80, 80),
        scale_factor: float = 1.1,
        min_neighbors: int = 4,
    ):
        """
        Initialize face detector.

        Args:
            method: Detection method to use
            min_face_size: Minimum face size (width, height) in pixels
            scale_factor: Scale factor for Haar cascade (1.1 = 10% increase per scale)
            min_neighbors: Minimum neighbors for Haar cascade (higher = fewer false positives)
        """
        self.method = method
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        # Lazy-loaded detectors
        self._haar_cascade = None
        self._dnn_net = None
        self._insightface_app = None

    def _get_haar_cascade(self):
        """Lazy-load Haar Cascade classifier."""
        if self._haar_cascade is None:
            # Use OpenCV's pre-trained frontal face cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)

            if self._haar_cascade.empty():
                raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")

            logger.info("Loaded Haar Cascade face detector")

        return self._haar_cascade

    def _get_dnn_net(self):
        """Lazy-load DNN face detector."""
        if self._dnn_net is None:
            # Use OpenCV's DNN face detector (ResNet-based)
            # Download these files if not present:
            # https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
            # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

            # Try to load from common paths
            possible_paths = [
                ("./models/deploy.prototxt", "./models/res10_300x300_ssd_iter_140000.caffemodel"),
                ("/app/models/deploy.prototxt", "/app/models/res10_300x300_ssd_iter_140000.caffemodel"),
                ("~/.opencv/face_detector/deploy.prototxt", "~/.opencv/face_detector/res10_300x300_ssd_iter_140000.caffemodel"),
            ]

            for prototxt_path, model_path in possible_paths:
                prototxt_path = Path(prototxt_path).expanduser()
                model_path = Path(model_path).expanduser()

                if prototxt_path.exists() and model_path.exists():
                    self._dnn_net = cv2.dnn.readNetFromCaffe(
                        str(prototxt_path),
                        str(model_path)
                    )
                    logger.info(f"Loaded DNN face detector from {model_path}")
                    break

            if self._dnn_net is None:
                # Fallback to Haar Cascade if DNN models not found
                logger.warning(
                    "DNN face detector models not found. "
                    "Falling back to Haar Cascade. "
                    "For better accuracy, download DNN models from: "
                    "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
                )
                self.method = DetectionMethod.HAAR_CASCADE
                return None

        return self._dnn_net

    def _get_insightface_app(self):
        """Lazy-load InsightFace detector as fallback."""
        if self._insightface_app is None:
            from insightface.app import FaceAnalysis
            from src.config.settings import settings

            # Use the configured detection model (separate from recognition model)
            detection_model = settings.insightface_detection_model

            self._insightface_app = FaceAnalysis(
                name=detection_model,
                providers=["CPUExecutionProvider"]
            )
            self._insightface_app.prepare(ctx_id=-1, det_size=(320, 320))
            logger.info(f"Loaded InsightFace detector ({detection_model}) for DETECTION only")

        return self._insightface_app

    def detect_faces(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[BoundingBox]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for DNN/InsightFace methods

        Returns:
            List of BoundingBox objects for detected faces
        """
        if self.method == DetectionMethod.HAAR_CASCADE:
            return self._detect_haar(image)
        elif self.method == DetectionMethod.DNN:
            return self._detect_dnn(image, confidence_threshold)
        elif self.method == DetectionMethod.INSIGHTFACE:
            return self._detect_insightface(image, confidence_threshold)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    def _detect_haar(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect faces using Haar Cascade."""
        cascade = self._get_haar_cascade()

        # Convert to grayscale (Haar works on grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
        )

        # Convert to BoundingBox objects
        bboxes = []
        for idx, (x, y, w, h) in enumerate(faces):
            bbox = BoundingBox(
                x1=int(x),
                y1=int(y),
                x2=int(x + w),
                y2=int(y + h),
                confidence=1.0,  # Haar doesn't provide confidence
                face_id=f"face_{idx}",
            )
            bboxes.append(bbox)

        logger.debug(f"Haar Cascade detected {len(bboxes)} faces")
        return bboxes

    def _detect_dnn(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[BoundingBox]:
        """Detect faces using DNN."""
        net = self._get_dnn_net()

        # Fallback to Haar if DNN not available
        if net is None:
            return self._detect_haar(image)

        height, width = image.shape[:2]

        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )

        # Run detection
        net.setInput(blob)
        detections = net.forward()

        # Parse detections
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < confidence_threshold:
                continue

            # Get bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Filter by minimum size
            if (x2 - x1) < self.min_face_size[0] or (y2 - y1) < self.min_face_size[1]:
                continue

            bbox = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(confidence),
                face_id=f"face_{len(bboxes)}",
            )
            bboxes.append(bbox)

        logger.debug(f"DNN detected {len(bboxes)} faces")
        return bboxes

    def _detect_insightface(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[BoundingBox]:
        """Detect faces using InsightFace."""
        app = self._get_insightface_app()

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = app.get(image_rgb)

        # Convert to BoundingBox objects
        bboxes = []
        for idx, face in enumerate(faces):
            if face.det_score < confidence_threshold:
                continue

            # InsightFace bbox format: [x1, y1, x2, y2]
            bbox_array = face.bbox
            bbox = BoundingBox(
                x1=int(bbox_array[0]),
                y1=int(bbox_array[1]),
                x2=int(bbox_array[2]),
                y2=int(bbox_array[3]),
                confidence=float(face.det_score),
                face_id=f"face_{idx}",
            )

            # Filter by minimum size
            if bbox.width < self.min_face_size[0] or bbox.height < self.min_face_size[1]:
                continue

            bboxes.append(bbox)

        logger.debug(f"InsightFace detected {len(bboxes)} faces")
        return bboxes


def create_face_detector(
    method: str = "dnn",
    min_face_size: int = 80,
    confidence_threshold: float = 0.5,
) -> FastFaceDetector:
    """
    Factory function to create a face detector.

    Args:
        method: Detection method ('haar', 'dnn', or 'insightface')
        min_face_size: Minimum face size in pixels
        confidence_threshold: Minimum confidence (for DNN/InsightFace)

    Returns:
        FastFaceDetector instance
    """
    method_enum = DetectionMethod(method.lower())

    detector = FastFaceDetector(
        method=method_enum,
        min_face_size=(min_face_size, min_face_size),
    )

    return detector
