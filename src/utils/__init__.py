"""Utility modules for the face recognition system."""

from .face_detector import (
    DetectionMethod,
    FastFaceDetector,
    create_face_detector,
)
from .face_processing import (
    ROI,
    BoundingBox,
    calculate_roi_distance,
    check_face_quality,
    convert_insightface_bbox,
    crop_face_from_bbox,
    draw_faces_on_frame,
    draw_roi_on_frame,
    filter_faces_by_roi,
    sort_faces_by_roi_proximity,
)

__all__ = [
    "BoundingBox",
    "ROI",
    "crop_face_from_bbox",
    "filter_faces_by_roi",
    "calculate_roi_distance",
    "sort_faces_by_roi_proximity",
    "check_face_quality",
    "convert_insightface_bbox",
    "draw_roi_on_frame",
    "draw_faces_on_frame",
    "FastFaceDetector",
    "DetectionMethod",
    "create_face_detector",
]
