"""Custom exception hierarchy for the Face Recognition API."""


class FaceRecognitionError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class FaceNotFoundError(FaceRecognitionError):
    """Raised when a face ID is not found."""

    def __init__(self, face_id: int):
        self.face_id = face_id
        super().__init__(f"Face not found: {face_id}", status_code=404)


class NoFaceDetectedError(FaceRecognitionError):
    """Raised when no face is detected in an image."""

    def __init__(self, detail: str = "No face detected in image"):
        super().__init__(detail, status_code=400)


class MultipleFacesDetectedError(FaceRecognitionError):
    """Raised when multiple faces found but only one expected."""

    def __init__(self, count: int):
        self.face_count = count
        super().__init__(
            f"Multiple faces detected ({count}). Use the multi-face endpoint or provide an image with a single face.",
            status_code=400,
        )


class LivenessCheckFailedError(FaceRecognitionError):
    """Raised when liveness/anti-spoof check fails."""

    def __init__(self, confidence: float, spoofing_type: str, threshold: float):
        self.confidence = confidence
        self.spoofing_type = spoofing_type
        self.threshold = threshold
        super().__init__(
            f"Liveness check failed: spoofing detected (type={spoofing_type}, "
            f"score={confidence:.3f}, threshold={threshold})",
            status_code=400,
        )


class ProviderError(FaceRecognitionError):
    """Raised for face recognition provider failures."""

    def __init__(self, provider: str, detail: str):
        self.provider = provider
        super().__init__(f"Provider '{provider}' error: {detail}", status_code=502)


class StorageError(FaceRecognitionError):
    """Raised for storage backend failures."""

    def __init__(self, detail: str):
        super().__init__(f"Storage error: {detail}", status_code=500)


class StoragePathError(StorageError):
    """Raised for invalid storage paths (e.g., path traversal attempts)."""

    def __init__(self, path: str):
        super().__init__(f"Invalid storage path: {path}")


class InvalidImageError(FaceRecognitionError):
    """Raised when image data is invalid or corrupt."""

    def __init__(self, detail: str = "Invalid image format"):
        super().__init__(detail, status_code=400)


class ConfigurationError(FaceRecognitionError):
    """Raised for configuration problems."""

    def __init__(self, detail: str):
        super().__init__(f"Configuration error: {detail}", status_code=500)
