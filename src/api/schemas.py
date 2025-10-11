from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, EmailStr


# Request schemas
class EnrollFaceRequest(BaseModel):
    """Request schema for face enrollment."""

    user_name: str = Field(..., description="User display name", min_length=1)
    user_email: Optional[EmailStr] = Field(None, description="User email address")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class RecognizeFaceRequest(BaseModel):
    """Request schema for face recognition."""

    max_results: int = Field(10, description="Maximum number of matches", ge=1, le=100)
    confidence_threshold: float = Field(
        0.8, description="Minimum confidence threshold", ge=0.0, le=1.0
    )


# Response schemas
class FaceResponse(BaseModel):
    """Response schema for a single face."""

    id: int = Field(..., description="Face ID (auto-increment)")
    user_name: str = Field(..., description="User display name")
    user_email: Optional[str] = Field(None, description="User email")
    provider_name: str = Field(..., description="Face recognition provider")
    provider_face_id: str = Field(..., description="Provider's face ID")
    image_path: str = Field(..., description="Image storage path")
    image_storage: str = Field(..., description="Storage backend type")
    quality_score: Optional[float] = Field(None, description="Face quality score")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    photo_type: str = Field(..., description="Photo type: 'enrolled' or 'verified'")
    verified_at: Optional[datetime] = Field(None, description="When photo was verified")
    verified_confidence: Optional[float] = Field(None, description="Verification confidence score")
    verified_by_processor: Optional[str] = Field(None, description="Recognition processor used (e.g., 'antelopev2', 'aws_rekognition')")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class EnrollFaceResponse(BaseModel):
    """Response schema for face enrollment."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    face: FaceResponse = Field(..., description="Enrolled face data")


class FaceMatchResponse(BaseModel):
    """Response schema for a face match."""

    face: FaceResponse = Field(..., description="Matched face data")
    similarity: float = Field(..., description="Similarity score (0-1)")
    photo_captured: bool = Field(False, description="Whether photo was auto-captured")
    processor: str = Field(..., description="Recognition processor used (e.g., 'antelopev2', 'antelopev2+aws', 'aws_rekognition')")


class RecognizeFaceResponse(BaseModel):
    """Response schema for face recognition."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    matches: List[FaceMatchResponse] = Field(..., description="List of face matches")
    total_matches: int = Field(..., description="Total number of matches found")
    processor: str = Field(..., description="Recognition processor used for this request")
    execution_time: float = Field(..., description="Execution time in seconds")
    detection_time: float = Field(default=0.0, description="Face detection time in seconds")
    recognition_time: float = Field(default=0.0, description="Face recognition time in seconds")


class FaceListResponse(BaseModel):
    """Response schema for listing faces."""

    success: bool = Field(..., description="Operation success status")
    faces: List[FaceResponse] = Field(..., description="List of faces")
    total: int = Field(..., description="Total number of faces")
    limit: int = Field(..., description="Results per page")
    offset: int = Field(..., description="Pagination offset")


class DeleteFaceResponse(BaseModel):
    """Response schema for face deletion."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    success: bool = Field(False, description="Operation success status")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class UserPhotosResponse(BaseModel):
    """Response schema for user photos listing."""

    success: bool = Field(..., description="Operation success status")
    user_name: str = Field(..., description="User name")
    photos: List[FaceResponse] = Field(..., description="List of photos (enrolled + verified)")
    total_photos: int = Field(..., description="Total number of photos")
    enrolled_count: int = Field(..., description="Number of enrolled photos")
    verified_count: int = Field(..., description="Number of verified photos")


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")


# Multi-face recognition schemas
class BoundingBoxResponse(BaseModel):
    """Response schema for face bounding box."""

    x1: int = Field(..., description="Left X coordinate")
    y1: int = Field(..., description="Top Y coordinate")
    x2: int = Field(..., description="Right X coordinate")
    y2: int = Field(..., description="Bottom Y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")
    area: int = Field(..., description="Bounding box area in pixels")
    center_x: int = Field(..., description="Center X coordinate")
    center_y: int = Field(..., description="Center Y coordinate")


class DetectedFaceWithMatches(BaseModel):
    """Response schema for a detected face with recognition matches."""

    face_id: str = Field(..., description="Sequential face ID (face_0, face_1, ...)")
    bbox: BoundingBoxResponse = Field(..., description="Face bounding box coordinates")
    det_confidence: float = Field(..., description="Detection confidence (0-1)")
    matches: List[FaceMatchResponse] = Field(..., description="Recognition matches for this face")
    total_matches: int = Field(..., description="Number of matches found for this face")


class RecognizeMultipleFacesRequest(BaseModel):
    """Request schema for multi-face recognition."""

    max_results_per_face: int = Field(
        5, description="Maximum matches per detected face", ge=1, le=50
    )
    confidence_threshold: float = Field(
        0.8, description="Minimum confidence threshold", ge=0.0, le=1.0
    )
    roi_enabled: bool = Field(
        False, description="Enable Region of Interest filtering"
    )
    roi_x: Optional[float] = Field(
        None, description="ROI X position (normalized 0-1)", ge=0.0, le=1.0
    )
    roi_y: Optional[float] = Field(
        None, description="ROI Y position (normalized 0-1)", ge=0.0, le=1.0
    )
    roi_width: Optional[float] = Field(
        None, description="ROI width (normalized 0-1)", ge=0.0, le=1.0
    )
    roi_height: Optional[float] = Field(
        None, description="ROI height (normalized 0-1)", ge=0.0, le=1.0
    )
    min_overlap: float = Field(
        0.3, description="Minimum ROI overlap ratio (0-1)", ge=0.0, le=1.0
    )


class RecognizeMultipleFacesResponse(BaseModel):
    """Response schema for multi-face recognition."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    detected_faces: List[DetectedFaceWithMatches] = Field(
        ..., description="List of detected faces with their matches"
    )
    total_faces_detected: int = Field(..., description="Total number of faces detected")
    total_faces_recognized: int = Field(
        ..., description="Number of faces with at least one match"
    )
    processor: str = Field(..., description="Recognition processor used")
    execution_time: float = Field(..., description="Total execution time in seconds")
    detection_time: float = Field(..., description="Face detection time in seconds")
    recognition_time: float = Field(..., description="Face recognition time in seconds")


# Liveness detection schemas
class LivenessCheckRequest(BaseModel):
    """Request schema for liveness detection check."""

    threshold: Optional[float] = Field(
        None, description="Liveness threshold (0.0-1.0, uses config default if not provided)", ge=0.0, le=1.0
    )


class LivenessCheckResponse(BaseModel):
    """Response schema for liveness detection check."""

    success: bool = Field(..., description="Operation success status")
    is_real: bool = Field(..., description="Whether face is determined to be real")
    confidence: float = Field(..., description="Liveness confidence score (0-1)")
    spoofing_type: str = Field(..., description="Type of spoofing detected (real/print/video/mask/unknown)")
    threshold: float = Field(..., description="Threshold used for detection")
    provider: str = Field(..., description="Liveness detection provider used")
    details: Optional[dict] = Field(None, description="Additional provider-specific details")
    execution_time: float = Field(..., description="Execution time in seconds")
