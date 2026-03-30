"""Unit tests for the custom exception hierarchy."""
import pytest

from src.exceptions import (
    FaceRecognitionError,
    FaceNotFoundError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    LivenessCheckFailedError,
    ProviderError,
    StorageError,
    StoragePathError,
    InvalidImageError,
    ConfigurationError,
)


class TestFaceRecognitionError:
    """Tests for the base exception class."""

    def test_default_status_code(self):
        err = FaceRecognitionError("something broke")
        assert err.status_code == 500

    def test_custom_status_code(self):
        err = FaceRecognitionError("bad request", status_code=400)
        assert err.status_code == 400

    def test_message_attribute(self):
        err = FaceRecognitionError("detail here")
        assert err.message == "detail here"

    def test_str_representation(self):
        err = FaceRecognitionError("detail here")
        assert str(err) == "detail here"

    def test_is_exception(self):
        assert issubclass(FaceRecognitionError, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(FaceRecognitionError) as exc_info:
            raise FaceRecognitionError("boom")
        assert exc_info.value.message == "boom"


class TestFaceNotFoundError:
    """Tests for FaceNotFoundError."""

    def test_status_code(self):
        err = FaceNotFoundError(face_id=42)
        assert err.status_code == 404

    def test_message_format(self):
        err = FaceNotFoundError(face_id=99)
        assert err.message == "Face not found: 99"

    def test_face_id_stored(self):
        err = FaceNotFoundError(face_id=7)
        assert err.face_id == 7

    def test_inherits_base(self):
        assert issubclass(FaceNotFoundError, FaceRecognitionError)

    def test_catch_as_base(self):
        with pytest.raises(FaceRecognitionError):
            raise FaceNotFoundError(face_id=1)


class TestNoFaceDetectedError:
    """Tests for NoFaceDetectedError."""

    def test_status_code(self):
        err = NoFaceDetectedError()
        assert err.status_code == 400

    def test_default_message(self):
        err = NoFaceDetectedError()
        assert err.message == "No face detected in image"

    def test_custom_message(self):
        err = NoFaceDetectedError("custom detail")
        assert err.message == "custom detail"

    def test_inherits_base(self):
        assert issubclass(NoFaceDetectedError, FaceRecognitionError)


class TestMultipleFacesDetectedError:
    """Tests for MultipleFacesDetectedError."""

    def test_status_code(self):
        err = MultipleFacesDetectedError(count=3)
        assert err.status_code == 400

    def test_message_includes_count(self):
        err = MultipleFacesDetectedError(count=5)
        assert "5" in err.message
        assert "Multiple faces detected" in err.message

    def test_face_count_stored(self):
        err = MultipleFacesDetectedError(count=2)
        assert err.face_count == 2

    def test_inherits_base(self):
        assert issubclass(MultipleFacesDetectedError, FaceRecognitionError)


class TestLivenessCheckFailedError:
    """Tests for LivenessCheckFailedError."""

    def test_status_code(self):
        err = LivenessCheckFailedError(
            confidence=0.123, spoofing_type="print", threshold=0.5
        )
        assert err.status_code == 400

    def test_message_format(self):
        err = LivenessCheckFailedError(
            confidence=0.456, spoofing_type="replay", threshold=0.7
        )
        assert "replay" in err.message
        assert "0.456" in err.message
        assert "0.7" in err.message
        assert "Liveness check failed" in err.message

    def test_attributes_stored(self):
        err = LivenessCheckFailedError(
            confidence=0.3, spoofing_type="mask", threshold=0.5
        )
        assert err.confidence == 0.3
        assert err.spoofing_type == "mask"
        assert err.threshold == 0.5

    def test_inherits_base(self):
        assert issubclass(LivenessCheckFailedError, FaceRecognitionError)


class TestProviderError:
    """Tests for ProviderError."""

    def test_status_code(self):
        err = ProviderError(provider="aws", detail="timeout")
        assert err.status_code == 502

    def test_message_format(self):
        err = ProviderError(provider="aws_rekognition", detail="rate limited")
        assert "aws_rekognition" in err.message
        assert "rate limited" in err.message

    def test_provider_stored(self):
        err = ProviderError(provider="insightface", detail="load failed")
        assert err.provider == "insightface"

    def test_inherits_base(self):
        assert issubclass(ProviderError, FaceRecognitionError)


class TestStorageError:
    """Tests for StorageError."""

    def test_status_code(self):
        err = StorageError("disk full")
        assert err.status_code == 500

    def test_message_format(self):
        err = StorageError("disk full")
        assert err.message == "Storage error: disk full"

    def test_inherits_base(self):
        assert issubclass(StorageError, FaceRecognitionError)


class TestStoragePathError:
    """Tests for StoragePathError."""

    def test_status_code(self):
        err = StoragePathError("/etc/passwd")
        assert err.status_code == 500

    def test_message_format(self):
        err = StoragePathError("../../secret")
        assert "../../secret" in err.message
        assert "Invalid storage path" in err.message

    def test_inherits_storage_error(self):
        assert issubclass(StoragePathError, StorageError)

    def test_inherits_base(self):
        assert issubclass(StoragePathError, FaceRecognitionError)

    def test_catch_as_storage_error(self):
        with pytest.raises(StorageError):
            raise StoragePathError("/bad/path")

    def test_catch_as_base(self):
        with pytest.raises(FaceRecognitionError):
            raise StoragePathError("/bad/path")


class TestInvalidImageError:
    """Tests for InvalidImageError."""

    def test_status_code(self):
        err = InvalidImageError()
        assert err.status_code == 400

    def test_default_message(self):
        err = InvalidImageError()
        assert err.message == "Invalid image format"

    def test_custom_message(self):
        err = InvalidImageError("truncated JPEG")
        assert err.message == "truncated JPEG"

    def test_inherits_base(self):
        assert issubclass(InvalidImageError, FaceRecognitionError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_status_code(self):
        err = ConfigurationError("missing AWS key")
        assert err.status_code == 500

    def test_message_format(self):
        err = ConfigurationError("missing AWS key")
        assert err.message == "Configuration error: missing AWS key"

    def test_inherits_base(self):
        assert issubclass(ConfigurationError, FaceRecognitionError)


class TestInheritanceHierarchy:
    """Tests verifying the full inheritance tree."""

    def test_all_inherit_from_base(self):
        for cls in [
            FaceNotFoundError,
            NoFaceDetectedError,
            MultipleFacesDetectedError,
            LivenessCheckFailedError,
            ProviderError,
            StorageError,
            StoragePathError,
            InvalidImageError,
            ConfigurationError,
        ]:
            assert issubclass(cls, FaceRecognitionError), (
                f"{cls.__name__} does not inherit from FaceRecognitionError"
            )

    def test_storage_path_extends_storage(self):
        assert issubclass(StoragePathError, StorageError)

    def test_base_extends_exception(self):
        assert issubclass(FaceRecognitionError, Exception)

    def test_storage_path_not_directly_exception(self):
        """StoragePathError -> StorageError -> FaceRecognitionError -> Exception."""
        mro = StoragePathError.__mro__
        storage_idx = mro.index(StorageError)
        base_idx = mro.index(FaceRecognitionError)
        assert storage_idx < base_idx
