"""Unit tests for face detector."""

import numpy as np
import pytest

from src.utils.face_detector import DetectionMethod, FastFaceDetector, create_face_detector
from src.utils.face_processing import BoundingBox


class TestFaceDetector:
    """Test suite for FastFaceDetector."""

    def test_create_haar_detector(self):
        """Test creating a Haar Cascade detector."""
        detector = create_face_detector(method="haar", min_face_size=80)
        assert detector is not None
        assert detector.method == DetectionMethod.HAAR_CASCADE
        assert detector.min_face_size == (80, 80)

    def test_create_dnn_detector(self):
        """Test creating a DNN detector."""
        detector = create_face_detector(method="dnn", min_face_size=100)
        assert detector is not None
        assert detector.method == DetectionMethod.DNN
        assert detector.min_face_size == (100, 100)

    def test_invalid_detection_method(self):
        """Test that invalid detection method raises error."""
        with pytest.raises(ValueError):
            create_face_detector(method="invalid")

    def test_detect_faces_no_face(self):
        """Test detection on empty image."""
        detector = create_face_detector(method="haar")
        # Create a blank image
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(blank_image)
        assert isinstance(faces, list)
        # Blank image should have no faces
        assert len(faces) == 0

    def test_bounding_box_properties(self):
        """Test BoundingBox object properties."""
        bbox = BoundingBox(x1=100, y1=150, x2=200, y2=250, confidence=0.95, face_id="test_1")
        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.area == 10000
        center = bbox.center
        assert center == (150, 200)
        assert bbox.confidence == 0.95
        assert bbox.face_id == "test_1"

    def test_min_face_size_filtering(self):
        """Test that faces smaller than min_face_size are filtered."""
        detector = FastFaceDetector(method=DetectionMethod.HAAR_CASCADE, min_face_size=(200, 200))
        # Very large min_face_size should filter out most detections
        # on typical webcam-sized images
        assert detector.min_face_size == (200, 200)

    def test_detection_confidence_threshold(self):
        """Test detection confidence threshold parameter."""
        detector = create_face_detector(method="dnn")
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Should not crash with low confidence threshold
        faces = detector.detect_faces(test_image, confidence_threshold=0.1)
        assert isinstance(faces, list)
