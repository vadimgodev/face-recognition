from typing import Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="face-recognition-api", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Database
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_db: str = Field(default="facedb", alias="POSTGRES_DB")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")

    # Redis
    redis_enabled: bool = Field(default=True, alias="REDIS_ENABLED")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_max_connections: int = Field(default=50, alias="REDIS_MAX_CONNECTIONS")
    redis_cache_ttl: int = Field(default=3600, alias="REDIS_CACHE_TTL")

    # Storage
    storage_backend: str = Field(default="local", alias="STORAGE_BACKEND")
    storage_local_path: str = Field(default="./data/images", alias="STORAGE_LOCAL_PATH")
    storage_s3_bucket: str = Field(default="", alias="STORAGE_S3_BUCKET")
    storage_s3_region: str = Field(default="us-east-1", alias="STORAGE_S3_REGION")

    # AWS
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_rekognition_collection_id: str = Field(
        default="faces-collection", alias="AWS_REKOGNITION_COLLECTION_ID"
    )

    # Face Recognition Provider
    face_provider: str = Field(default="aws_rekognition", alias="FACE_PROVIDER")

    # Liveness Detection Settings
    liveness_enabled: bool = Field(
        default=False,
        alias="LIVENESS_ENABLED",
        description="Enable anti-spoofing/liveness detection",
    )
    liveness_provider: str = Field(
        default="silent_face",
        alias="LIVENESS_PROVIDER",
        description="Liveness detection provider: silent_face, aws_rekognition (future)",
    )
    liveness_threshold: float = Field(
        default=0.5,
        alias="LIVENESS_THRESHOLD",
        description="Liveness detection threshold (0.0-1.0, higher = stricter)",
    )
    liveness_device_id: int = Field(
        default=-1,
        alias="LIVENESS_DEVICE_ID",
        description="GPU device ID for liveness detection (-1 for CPU, 0+ for GPU)",
    )
    liveness_model_dir: str = Field(
        default="./models/anti_spoof",
        alias="LIVENESS_MODEL_DIR",
        description="Directory containing anti-spoofing model weights",
    )
    liveness_detector_path: str = Field(
        default="./models",
        alias="LIVENESS_DETECTOR_PATH",
        description="Path to face detector models for liveness",
    )
    liveness_on_enrollment: bool = Field(
        default=True,
        alias="LIVENESS_ON_ENROLLMENT",
        description="Require liveness check during face enrollment",
    )
    liveness_on_recognition: bool = Field(
        default=False,
        alias="LIVENESS_ON_RECOGNITION",
        description="Require liveness check during face recognition",
    )

    # Hybrid Recognition Settings
    use_hybrid_recognition: bool = Field(
        default=False,
        alias="USE_HYBRID_RECOGNITION",
        description="Enable hybrid InsightFace + AWS Rekognition approach",
    )
    hybrid_mode: str = Field(
        default="insightface_only",
        alias="HYBRID_MODE",
        description="Hybrid search strategy: insightface_only, insightface_aws, or aws_only",
    )
    insightface_model: str = Field(
        default="buffalo_l",
        alias="INSIGHTFACE_MODEL",
        description="InsightFace model name (buffalo_l, buffalo_s, antelopev2)",
    )
    insightface_det_size: int = Field(
        default=640,
        alias="INSIGHTFACE_DET_SIZE",
        description="InsightFace detection size (larger = more accurate but slower)",
    )
    insightface_ctx_id: int = Field(
        default=-1,
        alias="INSIGHTFACE_CTX_ID",
        description="InsightFace context ID (-1 for CPU, 0+ for GPU device ID)",
    )
    similarity_threshold: float = Field(
        default=0.6,
        alias="SIMILARITY_THRESHOLD",
        description="Minimum similarity for face matching with InsightFace (0.0-1.0)",
    )
    vector_search_candidates: int = Field(
        default=3,
        alias="VECTOR_SEARCH_CANDIDATES",
        description="Number of candidates from vector search before AWS verification",
    )
    aws_verification_count: int = Field(
        default=3,
        alias="AWS_VERIFICATION_COUNT",
        description="Number of top candidates to verify with AWS Rekognition",
    )

    # Smart Hybrid Confidence Thresholds
    insightface_high_confidence: float = Field(
        default=0.8,
        alias="INSIGHTFACE_HIGH_CONFIDENCE",
        description="High confidence threshold - trust InsightFace immediately (0.8-1.0)",
    )
    insightface_medium_confidence: float = Field(
        default=0.6,
        alias="INSIGHTFACE_MEDIUM_CONFIDENCE",
        description="Medium confidence threshold - use local re-verification (0.6-0.8)",
    )

    # Auto-Capture Settings
    auto_capture_enabled: bool = Field(
        default=True,
        alias="AUTO_CAPTURE_ENABLED",
        description="Enable automatic capture of high-confidence recognition photos",
    )
    auto_capture_confidence_threshold: float = Field(
        default=0.85,
        alias="AUTO_CAPTURE_CONFIDENCE_THRESHOLD",
        description="Minimum confidence to auto-capture photo (0.0-1.0)",
    )
    auto_capture_max_verified_photos: int = Field(
        default=4,
        alias="AUTO_CAPTURE_MAX_VERIFIED_PHOTOS",
        description="Maximum verified photos to keep per person (FIFO stack)",
    )

    # Webcam Settings
    webcam_enabled: bool = Field(
        default=False,
        alias="WEBCAM_ENABLED",
        description="Enable webcam capture for face recognition",
    )
    webcam_device_id: int = Field(
        default=0,
        alias="WEBCAM_DEVICE_ID",
        description="Webcam device ID (0 for default camera)",
    )
    webcam_fps: int = Field(
        default=2,
        alias="WEBCAM_FPS",
        description="Webcam capture rate (frames per second)",
    )
    webcam_success_cooldown_seconds: int = Field(
        default=5,
        alias="WEBCAM_SUCCESS_COOLDOWN_SECONDS",
        description="Seconds to wait after successful recognition before resuming",
    )
    webcam_api_url: str = Field(
        default="http://localhost:8000",
        alias="WEBCAM_API_URL",
        description="External API URL for webcam daemon (use http://face.test for Traefik)",
    )

    # Multi-Face Detection Settings
    multiface_enabled: bool = Field(
        default=False,
        alias="MULTIFACE_ENABLED",
        description="Enable multi-face detection and recognition",
    )
    face_detection_method: str = Field(
        default="dnn",
        alias="FACE_DETECTION_METHOD",
        description="Detection method: 'haar' (fastest), 'dnn' (balanced), 'insightface' (accurate)",
    )
    insightface_detection_model: str = Field(
        default="buffalo_s",
        alias="INSIGHTFACE_DETECTION_MODEL",
        description="InsightFace model for DETECTION only (buffalo_s recommended for speed)",
    )
    detection_confidence_threshold: float = Field(
        default=0.5,
        alias="DETECTION_CONFIDENCE_THRESHOLD",
        description="Minimum confidence for face detection (DNN/InsightFace only)",
    )
    max_faces_per_frame: int = Field(
        default=10,
        alias="MAX_FACES_PER_FRAME",
        description="Maximum number of faces to process per frame",
    )
    min_face_size: int = Field(
        default=80,
        alias="MIN_FACE_SIZE",
        description="Minimum face size in pixels (width or height)",
    )
    face_crop_padding: float = Field(
        default=0.2,
        alias="FACE_CROP_PADDING",
        description="Padding ratio around face bounding box (0.2 = 20%)",
    )
    save_all_detected_faces: bool = Field(
        default=True,
        alias="SAVE_ALL_DETECTED_FACES",
        description="Save cropped images of all detected faces (not just matched)",
    )

    # Region of Interest (ROI) Settings for Door/Entrance Scenarios
    roi_enabled: bool = Field(
        default=False,
        alias="ROI_ENABLED",
        description="Enable ROI filtering for multi-face detection",
    )
    roi_x: float = Field(
        default=0.3,
        alias="ROI_X",
        description="ROI X position (normalized 0-1, 0.3 = 30% from left)",
    )
    roi_y: float = Field(
        default=0.2,
        alias="ROI_Y",
        description="ROI Y position (normalized 0-1, 0.2 = 20% from top)",
    )
    roi_width: float = Field(
        default=0.4,
        alias="ROI_WIDTH",
        description="ROI width (normalized 0-1, 0.4 = 40% of frame width)",
    )
    roi_height: float = Field(
        default=0.6,
        alias="ROI_HEIGHT",
        description="ROI height (normalized 0-1, 0.6 = 60% of frame height)",
    )
    roi_min_overlap: float = Field(
        default=0.3,
        alias="ROI_MIN_OVERLAP",
        description="Minimum overlap ratio for face to be in ROI (0.3 = 30%)",
    )

    # Door Unlock Settings
    door_unlock_provider: str = Field(
        default="mock",
        alias="DOOR_UNLOCK_PROVIDER",
        description="Door unlock provider: mock, http, or gpio",
    )
    door_unlock_url: str = Field(
        default="http://door-controller/unlock",
        alias="DOOR_UNLOCK_URL",
        description="HTTP endpoint for door unlock (when provider=http)",
    )
    door_unlock_confidence_threshold: float = Field(
        default=0.85,
        alias="DOOR_UNLOCK_CONFIDENCE_THRESHOLD",
        description="Minimum confidence to trigger door unlock (0.0-1.0)",
    )

    # Access Logging Settings
    access_log_output: str = Field(
        default="stdout",
        alias="ACCESS_LOG_OUTPUT",
        description="Access log output: stdout, file, or both",
    )
    access_log_file_path: str = Field(
        default="/var/log/face-recognition/access.log",
        alias="ACCESS_LOG_FILE_PATH",
        description="File path for access logs (when output=file or both)",
    )
    access_log_format: str = Field(
        default="json",
        alias="ACCESS_LOG_FORMAT",
        description="Access log format: json or text",
    )
    access_log_include_cooldown_events: bool = Field(
        default=False,
        alias="ACCESS_LOG_INCLUDE_COOLDOWN_EVENTS",
        description="Include cooldown status events in logs",
    )

    # Security
    secret_key: str = Field(
        default="", alias="SECRET_KEY"
    )
    allowed_origins: Union[list[str], str] = Field(
        default="http://localhost:3000,http://localhost:8000",
        alias="ALLOWED_ORIGINS",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        """Parse comma-separated origins string into list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def database_url(self) -> str:
        """Build database URL dynamically from components."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Build Redis URL dynamically from components."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env.lower() == "development"


# Global settings instance
settings = Settings()
