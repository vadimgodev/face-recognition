"""
Face processing utilities for multi-face detection and ROI filtering.

This module provides utilities for:
- Face cropping from bounding boxes
- Region of Interest (ROI) filtering
- Face quality assessment
- Spatial distance calculations
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a face bounding box with coordinates and metadata."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    face_id: Optional[str] = None

    @property
    def width(self) -> int:
        """Get bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Get bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Get bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center": self.center,
            "confidence": self.confidence,
            "face_id": self.face_id,
        }


@dataclass
class ROI:
    """Represents a Region of Interest."""

    x: float  # Normalized (0-1) or absolute pixel coordinates
    y: float
    width: float
    height: float
    normalized: bool = True

    def to_absolute(self, frame_width: int, frame_height: int) -> "ROI":
        """Convert normalized ROI to absolute pixel coordinates."""
        if not self.normalized:
            return self

        return ROI(
            x=int(self.x * frame_width),
            y=int(self.y * frame_height),
            width=int(self.width * frame_width),
            height=int(self.height * frame_height),
            normalized=False,
        )

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of ROI."""
        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the ROI."""
        return (
            self.x <= x <= self.x + self.width
            and self.y <= y <= self.y + self.height
        )

    def overlap_with_bbox(self, bbox: BoundingBox) -> float:
        """
        Calculate overlap between ROI and bounding box.

        Returns:
            Float between 0-1 representing overlap ratio (intersection / bbox_area)
        """
        # Calculate intersection
        x1_intersect = max(self.x, bbox.x1)
        y1_intersect = max(self.y, bbox.y1)
        x2_intersect = min(self.x + self.width, bbox.x2)
        y2_intersect = min(self.y + self.height, bbox.y2)

        # Check if there's any overlap
        if x1_intersect >= x2_intersect or y1_intersect >= y2_intersect:
            return 0.0

        # Calculate intersection area
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)

        # Return ratio of intersection to bbox area
        return intersection_area / bbox.area if bbox.area > 0 else 0.0


def ensure_bounding_box(bbox, confidence: float = 1.0) -> BoundingBox:
    """Convert array-like bbox to BoundingBox if needed."""
    if isinstance(bbox, BoundingBox):
        return bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return BoundingBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3]),
            confidence=float(bbox[4]) if len(bbox) > 4 else confidence,
        )
    if isinstance(bbox, np.ndarray) and len(bbox) >= 4:
        return BoundingBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3]),
            confidence=float(bbox[4]) if len(bbox) > 4 else confidence,
        )
    raise ValueError(f"Cannot convert to BoundingBox: {type(bbox)}")


def crop_face_from_bbox(
    image: np.ndarray,
    bbox: BoundingBox,
    padding: float = 0.2,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Crop face region from image using bounding box with padding.

    Args:
        image: Input image (numpy array)
        bbox: Bounding box coordinates
        padding: Padding ratio to add around bbox (0.2 = 20% padding)
        target_size: Optional (width, height) to resize cropped face

    Returns:
        Cropped face image as numpy array
    """
    height, width = image.shape[:2]

    # Calculate padding in pixels
    pad_w = int(bbox.width * padding)
    pad_h = int(bbox.height * padding)

    # Apply padding and ensure within image bounds
    x1 = max(0, bbox.x1 - pad_w)
    y1 = max(0, bbox.y1 - pad_h)
    x2 = min(width, bbox.x2 + pad_w)
    y2 = min(height, bbox.y2 + pad_h)

    # Crop face region
    face_crop = image[y1:y2, x1:x2]

    # Resize if target size specified
    if target_size is not None:
        face_crop = cv2.resize(face_crop, target_size)

    logger.debug(
        f"Cropped face from bbox ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}) "
        f"with padding {padding} to size {face_crop.shape}"
    )

    return face_crop


def filter_faces_by_roi(
    faces: List[Dict[str, Any]],
    roi: ROI,
    frame_width: int,
    frame_height: int,
    min_overlap: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Filter faces that overlap with Region of Interest.

    Args:
        faces: List of face dictionaries with 'bbox' key
        roi: Region of Interest (normalized or absolute coordinates)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        min_overlap: Minimum overlap ratio (0-1) to include face

    Returns:
        Filtered list of faces that overlap with ROI
    """
    # Convert ROI to absolute coordinates if needed
    if roi.normalized:
        roi = roi.to_absolute(frame_width, frame_height)

    filtered_faces = []

    for face in faces:
        bbox = face.get("bbox")
        if bbox is None:
            continue

        # Create BoundingBox object
        bbox_obj = ensure_bounding_box(bbox, confidence=face.get("confidence", 1.0))

        # Calculate overlap
        overlap = roi.overlap_with_bbox(bbox_obj)

        if overlap >= min_overlap:
            face["roi_overlap"] = overlap
            face["bbox"] = bbox_obj  # Store BoundingBox object
            filtered_faces.append(face)
            logger.debug(
                f"Face at {bbox_obj.center} has {overlap:.2%} overlap with ROI"
            )

    # Sort by overlap (highest first)
    filtered_faces.sort(key=lambda f: f["roi_overlap"], reverse=True)

    logger.info(
        f"Filtered {len(filtered_faces)} faces from {len(faces)} "
        f"with ROI overlap >= {min_overlap:.0%}"
    )

    return filtered_faces


def calculate_roi_distance(
    bbox: BoundingBox,
    roi: ROI,
) -> float:
    """
    Calculate Euclidean distance from bbox center to ROI center.

    Args:
        bbox: Face bounding box
        roi: Region of Interest

    Returns:
        Distance in pixels
    """
    bbox_center = bbox.center
    roi_center = roi.center

    distance = np.sqrt(
        (bbox_center[0] - roi_center[0]) ** 2
        + (bbox_center[1] - roi_center[1]) ** 2
    )

    return float(distance)


def sort_faces_by_roi_proximity(
    faces: List[Dict[str, Any]],
    roi: ROI,
    frame_width: int,
    frame_height: int,
) -> List[Dict[str, Any]]:
    """
    Sort faces by proximity to ROI center (closest first).

    Args:
        faces: List of face dictionaries with 'bbox' key
        roi: Region of Interest
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Sorted list of faces by distance to ROI center
    """
    # Convert ROI to absolute coordinates if needed
    if roi.normalized:
        roi = roi.to_absolute(frame_width, frame_height)

    # Calculate distance for each face
    for face in faces:
        bbox = face.get("bbox")
        bbox_obj = ensure_bounding_box(bbox)

        face["roi_distance"] = calculate_roi_distance(bbox_obj, roi)

    # Sort by distance (closest first)
    faces.sort(key=lambda f: f.get("roi_distance", float("inf")))

    return faces


def check_face_quality(
    face_image: np.ndarray,
    min_size: Optional[int] = None,
    max_blur_variance: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if face image meets quality requirements.

    Args:
        face_image: Face image as numpy array
        min_size: Minimum face size (width or height) in pixels
        max_blur_variance: Maximum blur variance (lower = more blurry)

    Returns:
        Tuple of (is_valid, quality_metrics)
    """
    if min_size is None:
        min_size = settings.face_quality_min_size
    if max_blur_variance is None:
        max_blur_variance = settings.face_quality_max_blur

    height, width = face_image.shape[:2]

    quality_metrics = {
        "width": width,
        "height": height,
        "is_valid": True,
        "reasons": [],
    }

    # Check minimum size
    if width < min_size or height < min_size:
        quality_metrics["is_valid"] = False
        quality_metrics["reasons"].append(
            f"Face too small ({width}x{height}, min: {min_size})"
        )

    # Check blur using Laplacian variance
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_metrics["blur_variance"] = laplacian_var

    if laplacian_var < max_blur_variance:
        quality_metrics["is_valid"] = False
        quality_metrics["reasons"].append(
            f"Face too blurry (variance: {laplacian_var:.2f}, max: {max_blur_variance})"
        )

    # Check brightness
    mean_brightness = np.mean(gray)
    quality_metrics["brightness"] = mean_brightness

    min_brightness = settings.face_quality_min_brightness
    max_brightness = settings.face_quality_max_brightness

    if mean_brightness < min_brightness:
        quality_metrics["is_valid"] = False
        quality_metrics["reasons"].append(
            f"Face too dark (brightness: {mean_brightness:.2f})"
        )
    elif mean_brightness > max_brightness:
        quality_metrics["is_valid"] = False
        quality_metrics["reasons"].append(
            f"Face too bright (brightness: {mean_brightness:.2f})"
        )

    is_valid = quality_metrics["is_valid"]

    if not is_valid:
        logger.debug(f"Face quality check failed: {quality_metrics['reasons']}")

    return is_valid, quality_metrics


def convert_insightface_bbox(insightface_bbox: np.ndarray) -> BoundingBox:
    """
    Convert InsightFace bbox format to BoundingBox object.

    InsightFace bbox format: [x1, y1, x2, y2]

    Args:
        insightface_bbox: InsightFace bbox as numpy array

    Returns:
        BoundingBox object
    """
    return BoundingBox(
        x1=int(insightface_bbox[0]),
        y1=int(insightface_bbox[1]),
        x2=int(insightface_bbox[2]),
        y2=int(insightface_bbox[3]),
    )


def draw_roi_on_frame(
    frame: np.ndarray,
    roi: ROI,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw ROI rectangle on frame for visualization.

    Args:
        frame: Input frame
        roi: Region of Interest
        color: Rectangle color in BGR
        thickness: Rectangle line thickness

    Returns:
        Frame with ROI drawn
    """
    height, width = frame.shape[:2]

    # Convert to absolute coordinates if needed
    if roi.normalized:
        roi = roi.to_absolute(width, height)

    # Draw rectangle
    cv2.rectangle(
        frame,
        (int(roi.x), int(roi.y)),
        (int(roi.x + roi.width), int(roi.y + roi.height)),
        color,
        thickness,
    )

    return frame


def draw_faces_on_frame(
    frame: np.ndarray,
    faces: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw face bounding boxes on frame for visualization.

    Args:
        frame: Input frame
        faces: List of face dictionaries with 'bbox' key
        color: Rectangle color in BGR
        thickness: Rectangle line thickness
        show_labels: Whether to show labels with face info

    Returns:
        Frame with faces drawn
    """
    for idx, face in enumerate(faces):
        bbox = face.get("bbox")
        if bbox is None:
            continue

        bbox_obj = ensure_bounding_box(bbox)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (bbox_obj.x1, bbox_obj.y1),
            (bbox_obj.x2, bbox_obj.y2),
            color,
            thickness,
        )

        # Draw label
        if show_labels:
            label = f"Face {idx + 1}"
            if "user_name" in face:
                label = face["user_name"]
            if "confidence" in face:
                label += f" ({face['confidence']:.2%})"

            # Put text above bbox
            cv2.putText(
                frame,
                label,
                (bbox_obj.x1, bbox_obj.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return frame
