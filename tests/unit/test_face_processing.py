"""Unit tests for face processing utilities."""
import pytest
import numpy as np
from src.utils.face_processing import BoundingBox, crop_face_from_bbox


class TestBoundingBox:
    """Test suite for BoundingBox class."""

    def test_bounding_box_creation(self):
        """Test creating a BoundingBox instance."""
        bbox = BoundingBox(
            x1=10, y1=20, x2=110, y2=120,
            confidence=0.95, face_id="test_1"
        )
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 120
        assert bbox.confidence == 0.95
        assert bbox.face_id == "test_1"

    def test_bounding_box_width_height(self):
        """Test width and height calculations."""
        bbox = BoundingBox(x1=50, y1=100, x2=150, y2=250, confidence=0.9, face_id="test")
        assert bbox.width == 100
        assert bbox.height == 150

    def test_bounding_box_area(self):
        """Test area calculation."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=200, confidence=0.85, face_id="test")
        assert bbox.area == 20000

    def test_bounding_box_center(self):
        """Test center point calculation."""
        bbox = BoundingBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9, face_id="test")
        center = bbox.center
        assert center == (150, 300)

    def test_bounding_box_negative_dimensions(self):
        """Test that invalid coordinates are handled."""
        # x2 < x1 should still work but give negative width
        bbox = BoundingBox(x1=100, y1=100, x2=50, y2=200, confidence=0.8, face_id="test")
        assert bbox.width == -50

    def test_crop_face_from_bbox(self):
        """Test cropping face from image using bounding box."""
        # Create a test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, face_id="test")

        # Crop face
        cropped = crop_face_from_bbox(image, bbox, padding=0.0)

        # Should be roughly 100x100 (bbox dimensions)
        assert cropped.shape[0] == 100  # height
        assert cropped.shape[1] == 100  # width
        assert cropped.shape[2] == 3    # RGB channels

    def test_crop_face_with_padding(self):
        """Test cropping with padding."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=200, y1=200, x2=300, y2=300, confidence=0.9, face_id="test")

        # Crop with 20% padding
        cropped = crop_face_from_bbox(image, bbox, padding=0.2)

        # Should be larger than bbox due to padding
        assert cropped.shape[0] > 100
        assert cropped.shape[1] > 100

    def test_crop_face_at_image_edge(self):
        """Test cropping near image boundaries."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # BBox near top-left corner
        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, face_id="test")

        # Should not crash even at edge
        cropped = crop_face_from_bbox(image, bbox, padding=0.1)
        assert cropped.shape[2] == 3  # Should still have 3 channels


class TestFaceProcessingHelpers:
    """Test suite for face processing helper functions."""

    def test_bounding_box_comparison(self):
        """Test comparing bounding boxes by area."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9, face_id="1")
        bbox2 = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.8, face_id="2")

        assert bbox1.area > bbox2.area

    def test_bounding_box_sorting(self):
        """Test sorting bounding boxes by area."""
        bboxes = [
            BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, face_id="small"),
            BoundingBox(x1=0, y1=0, x2=200, y2=200, confidence=0.95, face_id="large"),
            BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.85, face_id="medium"),
        ]

        # Sort by area (descending)
        sorted_bboxes = sorted(bboxes, key=lambda b: b.area, reverse=True)

        assert sorted_bboxes[0].face_id == "large"
        assert sorted_bboxes[1].face_id == "medium"
        assert sorted_bboxes[2].face_id == "small"
