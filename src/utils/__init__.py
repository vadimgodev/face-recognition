"""Utility modules for the face recognition system."""

from .face_processing import (
    BoundingBox,
    ROI,
    crop_face_from_bbox,
    filter_faces_by_roi,
    calculate_roi_distance,
    sort_faces_by_roi_proximity,
    check_face_quality,
    convert_insightface_bbox,
    draw_roi_on_frame,
    draw_faces_on_frame,
)
from .face_detector import (
    FastFaceDetector,
    DetectionMethod,
    create_face_detector,
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
