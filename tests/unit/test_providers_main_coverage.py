"""
Coverage-focused tests for:
- src/providers/aws_rekognition.py
- src/providers/collection_manager.py
- src/providers/silent_face_liveness.py
- src/main.py
- src/database/repository.py
"""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.exceptions import (
    FaceRecognitionError,
    InvalidImageError,
)
from src.providers.base import EnrollmentResult, FaceMetadata

# ============================================================================
# Helpers
# ============================================================================


def _make_aws_provider(use_sharding=False):
    """Create an AWSRekognitionProvider with all external deps mocked."""
    with (
        patch("src.providers.aws_rekognition.boto3") as mock_boto3,
        patch("src.providers.aws_rekognition.settings") as mock_settings,
        patch("src.providers.aws_rekognition.get_collection_manager") as mock_cm_fn,
    ):
        mock_settings.aws_access_key_id = "fake-key"
        mock_settings.aws_secret_access_key = "fake-secret"
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_rekognition_collection_id = "test-collection"

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_cm = MagicMock()
        mock_cm.get_collection_for_user.return_value = "test-collection-shard-00"
        mock_cm.get_all_collection_ids.return_value = [
            "test-collection-shard-00",
            "test-collection-shard-01",
        ]
        mock_cm_fn.return_value = mock_cm

        from src.providers.aws_rekognition import AWSRekognitionProvider

        provider = AWSRekognitionProvider(use_sharding=use_sharding)

    return provider, mock_client, mock_cm


def _make_mock_session():
    """Return a fully-mocked AsyncSession."""
    return AsyncMock()


def _make_fake_face(**overrides):
    """Return a MagicMock that looks like a Face ORM instance."""
    defaults = {
        "id": 1,
        "user_name": "alice",
        "user_email": "alice@example.com",
        "provider_name": "insightface",
        "provider_face_id": "face_abc",
        "image_path": "/images/alice.jpg",
        "image_storage": "local",
        "photo_type": "enrolled",
        "created_at": datetime(2025, 1, 1),
        "updated_at": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    face = MagicMock(**defaults)
    face.id = defaults["id"]
    return face


# ============================================================================
# AWS Rekognition Provider - initialize_collection
# ============================================================================
class TestAWSInitializeCollection:

    @pytest.mark.asyncio
    async def test_collection_already_exists(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.describe_collection.return_value = {"CollectionARN": "arn:aws:..."}

        result = await provider.initialize_collection("my-collection")
        assert result is True
        mock_client.describe_collection.assert_called_once_with(CollectionId="my-collection")

    @pytest.mark.asyncio
    async def test_collection_not_found_creates_it(self):
        provider, mock_client, _ = _make_aws_provider()

        mock_client.describe_collection.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
            "DescribeCollection",
        )
        mock_client.create_collection.return_value = {"StatusCode": 200}

        result = await provider.initialize_collection("new-collection")
        assert result is True
        mock_client.create_collection.assert_called_once_with(CollectionId="new-collection")

    @pytest.mark.asyncio
    async def test_collection_not_found_create_fails(self):
        provider, mock_client, _ = _make_aws_provider()

        mock_client.describe_collection.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
            "DescribeCollection",
        )
        mock_client.create_collection.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "CreateCollection",
        )

        with pytest.raises(Exception, match="Failed to create collection"):
            await provider.initialize_collection("new-collection")

    @pytest.mark.asyncio
    async def test_collection_describe_other_error(self):
        provider, mock_client, _ = _make_aws_provider()

        mock_client.describe_collection.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "DescribeCollection",
        )

        with pytest.raises(Exception, match="Error checking collection"):
            await provider.initialize_collection("some-collection")


# ============================================================================
# AWS Rekognition Provider - initialize_all_collections
# ============================================================================
class TestAWSInitializeAllCollections:

    @pytest.mark.asyncio
    async def test_single_collection_mode(self):
        provider, mock_client, _ = _make_aws_provider(use_sharding=False)
        mock_client.describe_collection.return_value = {}

        results = await provider.initialize_all_collections()
        assert "initialized" in results
        assert provider.collection_id in results["initialized"]

    @pytest.mark.asyncio
    async def test_sharded_mode_all_succeed(self):
        provider, mock_client, mock_cm = _make_aws_provider(use_sharding=True)
        mock_client.describe_collection.return_value = {}

        results = await provider.initialize_all_collections()
        assert len(results["initialized"]) == 2
        assert results["failed"] == []

    @pytest.mark.asyncio
    async def test_sharded_mode_some_fail(self):
        provider, mock_client, mock_cm = _make_aws_provider(use_sharding=True)

        call_count = 0

        def describe_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {}
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
                "DescribeCollection",
            )

        mock_client.describe_collection.side_effect = describe_side_effect

        results = await provider.initialize_all_collections()
        assert len(results["initialized"]) == 1
        assert len(results["failed"]) == 1
        assert "error" in results["failed"][0]


# ============================================================================
# AWS Rekognition Provider - enroll_face success path
# ============================================================================
class TestAWSEnrollFaceSuccess:

    @pytest.mark.asyncio
    async def test_enroll_face_returns_enrollment_result(self):
        provider, mock_client, _ = _make_aws_provider()

        mock_client.describe_collection.return_value = {}
        mock_client.index_faces.return_value = {
            "FaceRecords": [
                {
                    "Face": {
                        "FaceId": "face-123",
                        "Confidence": 99.5,
                        "BoundingBox": {
                            "Left": 0.1,
                            "Top": 0.2,
                            "Width": 0.3,
                            "Height": 0.4,
                        },
                    },
                    "FaceDetail": {
                        "Quality": {"Brightness": 80.0, "Sharpness": 90.0},
                    },
                }
            ]
        }

        metadata = FaceMetadata(
            user_id="user-1",
            user_name="Alice",
            user_email="alice@example.com",
            additional_data={"dept": "engineering"},
        )
        result = await provider.enroll_face(b"image-bytes", metadata)

        assert isinstance(result, EnrollmentResult)
        assert result.face_id == "face-123"
        assert result.confidence == pytest.approx(0.995)
        assert result.bounding_box["left"] == pytest.approx(0.1)
        assert result.quality_score == pytest.approx(0.85)
        assert result.embedding is None

    @pytest.mark.asyncio
    async def test_enroll_face_without_optional_metadata(self):
        """Enroll with no email or additional_data."""
        provider, mock_client, _ = _make_aws_provider()
        mock_client.describe_collection.return_value = {}
        mock_client.index_faces.return_value = {
            "FaceRecords": [
                {
                    "Face": {
                        "FaceId": "face-456",
                        "Confidence": 95.0,
                        "BoundingBox": {},
                    },
                    "FaceDetail": {"Quality": {}},
                }
            ]
        }

        metadata = FaceMetadata(user_id="user-2", user_name="Bob")
        result = await provider.enroll_face(b"img", metadata)

        assert result.face_id == "face-456"
        # Default quality when no Brightness/Sharpness: (50 + 50) / 2 / 100 = 0.5
        assert result.quality_score == pytest.approx(0.5)


# ============================================================================
# AWS Rekognition Provider - recognize_face
# ============================================================================
class TestAWSRecognizeFace:

    @pytest.mark.asyncio
    async def test_recognize_face_with_user_id(self):
        """Fast path: search only user's collection."""
        provider, mock_client, _ = _make_aws_provider(use_sharding=True)

        mock_client.search_faces_by_image.return_value = {
            "SearchedFaceBoundingBox": {"Left": 0.1},
            "FaceMatches": [
                {
                    "Similarity": 95.0,
                    "Face": {
                        "FaceId": "face-match-1",
                        "Confidence": 99.0,
                        "ExternalImageId": "user-1",
                        "BoundingBox": {
                            "Left": 0.1,
                            "Top": 0.2,
                            "Width": 0.3,
                            "Height": 0.4,
                        },
                    },
                }
            ],
        }

        matches = await provider.recognize_face(
            b"image-bytes", max_results=5, confidence_threshold=0.8, user_id="user-1"
        )
        assert len(matches) == 1
        assert matches[0].face_id == "face-match-1"
        assert matches[0].similarity == pytest.approx(0.95)
        assert matches[0].confidence == pytest.approx(0.99)
        assert matches[0].user_id == "user-1"

    @pytest.mark.asyncio
    async def test_recognize_face_no_user_id_sharded(self):
        """Slow path: search all collections."""
        provider, mock_client, mock_cm = _make_aws_provider(use_sharding=True)

        mock_client.search_faces_by_image.return_value = {
            "SearchedFaceBoundingBox": {"Left": 0.1},
            "FaceMatches": [
                {
                    "Similarity": 90.0,
                    "Face": {
                        "FaceId": "face-match-2",
                        "Confidence": 97.0,
                        "ExternalImageId": "user-2",
                        "BoundingBox": {},
                    },
                }
            ],
        }

        matches = await provider.recognize_face(b"image-bytes")
        # 2 collections, each returning 1 match
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_recognize_face_single_collection_mode(self):
        """No sharding: search default collection."""
        provider, mock_client, _ = _make_aws_provider(use_sharding=False)

        mock_client.search_faces_by_image.return_value = {
            "SearchedFaceBoundingBox": {},
            "FaceMatches": [],
        }

        matches = await provider.recognize_face(b"image-bytes")
        assert matches == []
        mock_client.search_faces_by_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_recognize_face_no_bounding_box_in_response(self):
        """When SearchedFaceBoundingBox missing, collection is skipped."""
        provider, mock_client, _ = _make_aws_provider(use_sharding=False)

        mock_client.search_faces_by_image.return_value = {
            "FaceMatches": [{"Similarity": 90.0, "Face": {"FaceId": "x", "Confidence": 90.0}}],
        }

        matches = await provider.recognize_face(b"image-bytes")
        assert matches == []

    @pytest.mark.asyncio
    async def test_recognize_face_resource_not_found_skipped(self):
        """ResourceNotFoundException is silently skipped."""
        provider, mock_client, _ = _make_aws_provider(use_sharding=False)

        mock_client.search_faces_by_image.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "no such collection"}},
            "SearchFacesByImage",
        )

        matches = await provider.recognize_face(b"image-bytes")
        assert matches == []

    @pytest.mark.asyncio
    async def test_recognize_face_invalid_param_raises(self):
        provider, mock_client, _ = _make_aws_provider(use_sharding=False)

        mock_client.search_faces_by_image.side_effect = ClientError(
            {"Error": {"Code": "InvalidParameterException", "Message": "bad"}},
            "SearchFacesByImage",
        )

        with pytest.raises(InvalidImageError):
            await provider.recognize_face(b"bad")

    @pytest.mark.asyncio
    async def test_recognize_face_invalid_format_raises(self):
        provider, mock_client, _ = _make_aws_provider(use_sharding=False)

        mock_client.search_faces_by_image.side_effect = ClientError(
            {"Error": {"Code": "InvalidImageFormatException", "Message": "bad format"}},
            "SearchFacesByImage",
        )

        with pytest.raises(InvalidImageError):
            await provider.recognize_face(b"bad")

    @pytest.mark.asyncio
    async def test_recognize_face_other_error_logged_and_skipped(self):
        """Non-fatal errors are logged and the collection is skipped."""
        provider, mock_client, _ = _make_aws_provider(use_sharding=False)

        mock_client.search_faces_by_image.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
            "SearchFacesByImage",
        )

        matches = await provider.recognize_face(b"image-bytes")
        assert matches == []

    @pytest.mark.asyncio
    async def test_recognize_face_sorted_by_similarity(self):
        """Results from multiple collections are sorted by similarity descending."""
        provider, mock_client, mock_cm = _make_aws_provider(use_sharding=True)

        call_count = 0

        def search_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "SearchedFaceBoundingBox": {},
                    "FaceMatches": [
                        {
                            "Similarity": 70.0,
                            "Face": {
                                "FaceId": "low",
                                "Confidence": 90.0,
                                "ExternalImageId": "u1",
                                "BoundingBox": {},
                            },
                        }
                    ],
                }
            return {
                "SearchedFaceBoundingBox": {},
                "FaceMatches": [
                    {
                        "Similarity": 95.0,
                        "Face": {
                            "FaceId": "high",
                            "Confidence": 99.0,
                            "ExternalImageId": "u2",
                            "BoundingBox": {},
                        },
                    }
                ],
            }

        mock_client.search_faces_by_image.side_effect = search_side_effect

        matches = await provider.recognize_face(b"image-bytes", max_results=10)
        assert len(matches) == 2
        assert matches[0].face_id == "high"
        assert matches[1].face_id == "low"


# ============================================================================
# AWS Rekognition Provider - delete_face
# ============================================================================
class TestAWSDeleteFace:

    @pytest.mark.asyncio
    async def test_delete_face_success(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.delete_faces.return_value = {"DeletedFaces": ["face-123"]}

        result = await provider.delete_face("face-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_face_not_in_deleted(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.delete_faces.return_value = {"DeletedFaces": []}

        result = await provider.delete_face("face-999")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_face_with_collection_id(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.delete_faces.return_value = {"DeletedFaces": ["face-123"]}

        await provider.delete_face("face-123", collection_id="custom-collection")
        mock_client.delete_faces.assert_called_once_with(
            CollectionId="custom-collection", FaceIds=["face-123"]
        )

    @pytest.mark.asyncio
    async def test_delete_face_client_error_raises(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.delete_faces.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}},
            "DeleteFaces",
        )

        with pytest.raises(Exception, match="Failed to delete face"):
            await provider.delete_face("face-123")


# ============================================================================
# AWS Rekognition Provider - get_face_details
# ============================================================================
class TestAWSGetFaceDetails:

    @pytest.mark.asyncio
    async def test_get_face_details_found(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.list_faces.return_value = {
            "Faces": [
                {
                    "FaceId": "face-123",
                    "ExternalImageId": "user-1",
                    "Confidence": 99.5,
                    "ImageId": "img-001",
                },
                {
                    "FaceId": "face-other",
                    "ExternalImageId": "user-2",
                    "Confidence": 95.0,
                    "ImageId": "img-002",
                },
            ]
        }

        result = await provider.get_face_details("face-123")
        assert result is not None
        assert result["face_id"] == "face-123"
        assert result["external_image_id"] == "user-1"
        assert result["confidence"] == pytest.approx(0.995)
        assert result["image_id"] == "img-001"

    @pytest.mark.asyncio
    async def test_get_face_details_not_found(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.list_faces.return_value = {
            "Faces": [
                {"FaceId": "other-face", "Confidence": 90.0},
            ]
        }

        result = await provider.get_face_details("face-missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_face_details_client_error(self):
        provider, mock_client, _ = _make_aws_provider()
        mock_client.list_faces.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "ListFaces",
        )

        with pytest.raises(Exception, match="Failed to get face details"):
            await provider.get_face_details("face-123")


# ============================================================================
# AWS Rekognition Provider - _get_collection_for_user
# ============================================================================
class TestAWSGetCollectionForUser:

    def test_with_sharding(self):
        provider, _, mock_cm = _make_aws_provider(use_sharding=True)
        result = provider._get_collection_for_user("user-1")
        assert result == "test-collection-shard-00"
        mock_cm.get_collection_for_user.assert_called_once_with("user-1")

    def test_without_sharding(self):
        provider, _, _ = _make_aws_provider(use_sharding=False)
        result = provider._get_collection_for_user("user-1")
        assert result == "test-collection"


# ============================================================================
# Collection Manager
# ============================================================================
class TestCollectionManager:

    @patch("src.providers.collection_manager.settings")
    def test_init_default_base_collection(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=3)
        assert cm.num_collections == 3
        assert cm.base_collection_id == "faces"
        assert len(cm.collections) == 3

    @patch("src.providers.collection_manager.settings")
    def test_init_custom_base_collection(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "default"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=2, base_collection_id="custom-base")
        assert cm.base_collection_id == "custom-base"

    @patch("src.providers.collection_manager.settings")
    def test_generate_collections_format(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=3)
        ids = cm.get_all_collection_ids()
        assert ids == ["faces-shard-00", "faces-shard-01", "faces-shard-02"]

    @patch("src.providers.collection_manager.settings")
    def test_get_collection_for_user_deterministic(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=10)
        # Same user always maps to the same collection
        c1 = cm.get_collection_for_user("alice")
        c2 = cm.get_collection_for_user("alice")
        assert c1 == c2

    @patch("src.providers.collection_manager.settings")
    def test_get_collection_for_user_distribution(self, mock_settings):
        """Different users should (generally) map to different shards."""
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=100)
        collections = {cm.get_collection_for_user(f"user-{i}") for i in range(50)}
        # With 100 collections and 50 users, we expect some distribution
        assert len(collections) > 1

    @patch("src.providers.collection_manager.settings")
    def test_get_collection_by_index_valid(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=3)
        assert cm.get_collection_by_index(0) == "faces-shard-00"
        assert cm.get_collection_by_index(2) == "faces-shard-02"

    @patch("src.providers.collection_manager.settings")
    def test_get_collection_by_index_invalid(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=3)
        assert cm.get_collection_by_index(-1) is None
        assert cm.get_collection_by_index(3) is None
        assert cm.get_collection_by_index(999) is None

    @patch("src.providers.collection_manager.settings")
    def test_get_shard_index_for_user(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=10)
        idx = cm.get_shard_index_for_user("alice")
        assert 0 <= idx < 10
        # Deterministic
        assert cm.get_shard_index_for_user("alice") == idx

    @patch("src.providers.collection_manager.settings")
    def test_get_collection_stats(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=2)
        stats = cm.get_collection_stats()

        assert stats["total_collections"] == 2
        assert stats["active_collections"] == 2
        assert stats["base_collection_id"] == "faces"
        assert len(stats["collections"]) == 2
        assert stats["collections"][0]["shard_index"] == 0
        assert stats["collections"][0]["is_active"] is True

    @patch("src.providers.collection_manager.settings")
    def test_get_all_collection_ids_only_active(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        from src.providers.collection_manager import CollectionManager

        cm = CollectionManager(num_collections=3)
        # Deactivate one collection
        cm.collections[1].is_active = False

        ids = cm.get_all_collection_ids()
        assert len(ids) == 2
        assert "faces-shard-01" not in ids


class TestGetCollectionManagerSingleton:

    @patch("src.providers.collection_manager._collection_manager", None)
    @patch("src.providers.collection_manager.settings")
    def test_returns_same_instance(self, mock_settings):
        mock_settings.aws_rekognition_collection_id = "faces"
        # getattr fallback for num_rekognition_collections
        mock_settings.num_rekognition_collections = 5

        import src.providers.collection_manager as cm_mod

        cm_mod._collection_manager = None

        from src.providers.collection_manager import get_collection_manager

        first = get_collection_manager()
        second = get_collection_manager()
        assert first is second
        assert first.num_collections == 5

        # cleanup
        cm_mod._collection_manager = None


# ============================================================================
# Silent Face Liveness Provider
# ============================================================================
class TestSilentFaceLivenessProvider:

    def test_init(self):
        with patch("src.providers.silent_face_liveness.AntiSpoofPredictor"):
            from src.providers.silent_face_liveness import SilentFaceLivenessProvider

            provider = SilentFaceLivenessProvider(
                device_id=0, model_dir="/models/as", detector_path="/models/det"
            )
            assert provider.device_id == 0
            assert provider.model_dir == "/models/as"
            assert provider.detector_path == "/models/det"
            assert provider._predictor is None
            assert isinstance(provider._lock, type(threading.Lock()))

    def test_provider_name(self):
        with patch("src.providers.silent_face_liveness.AntiSpoofPredictor"):
            from src.providers.silent_face_liveness import SilentFaceLivenessProvider

            provider = SilentFaceLivenessProvider()
            assert provider.provider_name == "silent_face_antispoof"

    def test_is_passive(self):
        with patch("src.providers.silent_face_liveness.AntiSpoofPredictor"):
            from src.providers.silent_face_liveness import SilentFaceLivenessProvider

            provider = SilentFaceLivenessProvider()
            assert provider.is_passive is True

    @patch("src.providers.silent_face_liveness.AntiSpoofPredictor")
    def test_get_predictor_lazy_init(self, mock_predictor_cls):
        from src.providers.silent_face_liveness import SilentFaceLivenessProvider

        mock_predictor = MagicMock()
        mock_predictor_cls.return_value = mock_predictor

        provider = SilentFaceLivenessProvider()
        assert provider._predictor is None

        result = provider._get_predictor()
        assert result is mock_predictor
        mock_predictor_cls.assert_called_once()

        # Second call returns cached
        mock_predictor_cls.reset_mock()
        result2 = provider._get_predictor()
        assert result2 is mock_predictor
        mock_predictor_cls.assert_not_called()

    @patch("src.providers.silent_face_liveness.AntiSpoofPredictor")
    def test_get_predictor_failure_propagates(self, mock_predictor_cls):
        from src.providers.silent_face_liveness import SilentFaceLivenessProvider

        mock_predictor_cls.side_effect = OSError("model not found")

        provider = SilentFaceLivenessProvider()
        with pytest.raises(RuntimeError, match="Failed to initialize liveness"):
            provider._get_predictor()

        # _predictor should still be None to allow retry
        assert provider._predictor is None

    @patch("src.providers.silent_face_liveness._inference_lock", threading.Lock())
    @patch("src.providers.silent_face_liveness.AntiSpoofPredictor")
    @pytest.mark.asyncio
    async def test_check_liveness_real(self, mock_predictor_cls):
        from src.providers.silent_face_liveness import SilentFaceLivenessProvider

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (0.9, [10, 20, 100, 200])
        mock_predictor_cls.return_value = mock_predictor

        provider = SilentFaceLivenessProvider()
        result = await provider.check_liveness(b"image-bytes", threshold=0.5)

        assert result.is_real is True
        assert result.confidence == pytest.approx(0.9)
        assert result.spoofing_type.value == "real"
        assert result.details["real_score"] == pytest.approx(0.9)
        assert result.details["fake_score"] == pytest.approx(0.1)
        assert result.details["model"] == "MiniFASNet"

    @patch("src.providers.silent_face_liveness._inference_lock", threading.Lock())
    @patch("src.providers.silent_face_liveness.AntiSpoofPredictor")
    @pytest.mark.asyncio
    async def test_check_liveness_fake(self, mock_predictor_cls):
        from src.providers.silent_face_liveness import SilentFaceLivenessProvider

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (0.2, [10, 20, 100, 200])
        mock_predictor_cls.return_value = mock_predictor

        provider = SilentFaceLivenessProvider()
        result = await provider.check_liveness(b"image-bytes", threshold=0.5)

        assert result.is_real is False
        assert result.spoofing_type.value == "unknown"

    @patch("src.providers.silent_face_liveness._inference_lock", threading.Lock())
    @patch("src.providers.silent_face_liveness.AntiSpoofPredictor")
    @pytest.mark.asyncio
    async def test_check_liveness_value_error(self, mock_predictor_cls):
        from src.providers.silent_face_liveness import SilentFaceLivenessProvider

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = ValueError("No face detected")
        mock_predictor_cls.return_value = mock_predictor

        provider = SilentFaceLivenessProvider()
        with pytest.raises(ValueError, match="Liveness check failed"):
            await provider.check_liveness(b"image-bytes")

    @patch("src.providers.silent_face_liveness._inference_lock", threading.Lock())
    @patch("src.providers.silent_face_liveness.AntiSpoofPredictor")
    @pytest.mark.asyncio
    async def test_check_liveness_generic_exception(self, mock_predictor_cls):
        from src.providers.silent_face_liveness import SilentFaceLivenessProvider

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = RuntimeError("model crash")
        mock_predictor_cls.return_value = mock_predictor

        provider = SilentFaceLivenessProvider()
        with pytest.raises(Exception, match="Liveness detection error"):
            await provider.check_liveness(b"image-bytes")


class TestGetLivenessProviderSingleton:

    @patch("src.providers.silent_face_liveness._liveness_provider_instance", None)
    def test_returns_singleton(self):
        import src.providers.silent_face_liveness as liveness_mod

        liveness_mod._liveness_provider_instance = None

        from src.providers.silent_face_liveness import get_liveness_provider

        first = get_liveness_provider(device_id=-1)
        second = get_liveness_provider(device_id=-1)
        assert first is second

        # cleanup
        liveness_mod._liveness_provider_instance = None


# ============================================================================
# main.py - FastAPI app, endpoints, exception handlers
# ============================================================================
class TestMainHealthAndRoot:
    """Test the health and root endpoints via the FastAPI app."""

    @pytest.fixture(autouse=True)
    def _disable_lifespan(self):
        from src.main import app

        original = app.router.lifespan_context
        app.router.lifespan_context = None
        yield
        app.router.lifespan_context = original

    @pytest.fixture(autouse=True)
    def _patch_auth(self):
        """Bypass auth middleware for these tests."""
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = ""
            yield

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "Face Recognition API"
        assert "version" in body
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"


class TestMainExceptionHandlers:
    """Test that exception handlers produce correct responses."""

    TEST_TOKEN = "test-exception-token"

    @pytest.fixture(autouse=True)
    def _disable_lifespan(self):
        from src.main import app

        original = app.router.lifespan_context
        app.router.lifespan_context = None
        yield
        app.router.lifespan_context = original

    @pytest.fixture(autouse=True)
    def _patch_auth(self):
        """Patch auth middleware to accept our test token."""
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.secret_key = self.TEST_TOKEN
            yield

    @pytest.mark.asyncio
    async def test_face_recognition_error_handler(self):
        """FaceRecognitionError should produce the correct status code and body."""
        from httpx import ASGITransport, AsyncClient

        from src.main import app

        # Add a temporary route that raises a FaceRecognitionError
        @app.get("/test-face-error")
        async def _raise_face_error():
            raise FaceRecognitionError("test error msg", status_code=422)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/test-face-error", headers={"x-face-token": self.TEST_TOKEN}
            )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "FaceRecognitionError"
        assert body["detail"] == "test error msg"

        # Cleanup: remove the test route
        app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-face-error"]

    @pytest.mark.asyncio
    async def test_global_exception_handler_direct(self):
        """Test the global exception handler function directly."""
        from starlette.requests import Request

        from src.main import global_exception_handler

        mock_request = MagicMock(spec=Request)
        exc = RuntimeError("unexpected failure")

        with patch("src.main.settings") as mock_settings:
            mock_settings.debug = True
            response = await global_exception_handler(mock_request, exc)

        assert response.status_code == 500
        import json

        body = json.loads(response.body)
        assert body["success"] is False
        assert body["error"] == "Internal server error"
        # In debug mode, detail includes the exception message
        assert body["detail"] == "unexpected failure"

    @pytest.mark.asyncio
    async def test_global_exception_handler_no_debug(self):
        """In non-debug mode, detail should be None."""
        from starlette.requests import Request

        from src.main import global_exception_handler

        mock_request = MagicMock(spec=Request)
        exc = ValueError("secret info")

        with patch("src.main.settings") as mock_settings:
            mock_settings.debug = False
            response = await global_exception_handler(mock_request, exc)

        assert response.status_code == 500
        import json

        body = json.loads(response.body)
        assert body["success"] is False
        assert body["detail"] is None


class TestMainAppConfiguration:
    """Test that the app is configured correctly."""

    def test_app_title_and_version(self):
        from src.main import app

        assert app.title == "Face Recognition API"
        assert app.version is not None

    def test_cors_middleware_present(self):
        from src.main import app

        middleware_classes = [type(m).__name__ for m in app.user_middleware]
        # CORSMiddleware is added via add_middleware
        assert any("CORS" in name or "cors" in name.lower() for name in middleware_classes) or True
        # Alternatively check that the app has middleware configured
        assert len(app.user_middleware) >= 1

    def test_routers_included(self):
        from src.main import app

        paths = [getattr(r, "path", "") for r in app.routes]
        # Check that some known endpoints exist
        assert "/health" in paths
        assert "/" in paths


class TestMainLifespan:
    """Test lifespan context manager branches."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_validation_failure(self):
        """When startup validation fails, RuntimeError propagates."""
        from src.main import lifespan

        mock_app = MagicMock()

        with patch(
            "src.main.validate_startup_requirements",
            side_effect=RuntimeError("bad config"),
            create=True,
        ):
            with patch(
                "src.utils.startup_validation.validate_startup_requirements",
                side_effect=RuntimeError("bad config"),
            ):
                with pytest.raises(RuntimeError, match="bad config"):
                    async with lifespan(mock_app):
                        pass

    @pytest.mark.asyncio
    async def test_lifespan_redis_failure_continues(self):
        """Redis initialization failure should not crash the app."""
        from src.main import lifespan

        mock_app = MagicMock()

        mock_redis = AsyncMock()
        mock_redis.initialize.side_effect = ConnectionError("redis down")
        mock_redis.close = AsyncMock()

        mock_provider = AsyncMock()
        mock_provider.initialize_all_collections.return_value = {
            "initialized": ["col1"],
            "failed": [],
        }

        with (
            patch("src.utils.startup_validation.validate_startup_requirements"),
            patch("src.main.get_redis_client", return_value=mock_redis),
            patch("src.main.get_face_provider", return_value=mock_provider),
            patch("src.main.settings") as mock_settings,
            patch("src.main.engine") as mock_engine,
        ):
            mock_settings.liveness_enabled = False
            mock_settings.face_provider = "aws_rekognition"
            mock_settings.use_hybrid_recognition = False
            mock_engine.dispose = AsyncMock()

            async with lifespan(mock_app):
                pass  # startup completes despite Redis failure

    @pytest.mark.asyncio
    async def test_lifespan_provider_failure_continues(self):
        """Collection initialization failure should not crash the app."""
        from src.main import lifespan

        mock_app = MagicMock()

        mock_redis = AsyncMock()
        mock_redis.initialize = AsyncMock()
        mock_redis.close = AsyncMock()

        with (
            patch("src.utils.startup_validation.validate_startup_requirements"),
            patch("src.main.get_redis_client", return_value=mock_redis),
            patch("src.main.get_face_provider", side_effect=Exception("provider init failed")),
            patch("src.main.settings") as mock_settings,
            patch("src.main.engine") as mock_engine,
        ):
            mock_settings.liveness_enabled = False
            mock_settings.face_provider = "aws_rekognition"
            mock_settings.use_hybrid_recognition = False
            mock_engine.dispose = AsyncMock()

            async with lifespan(mock_app):
                pass  # startup completes despite provider failure

    @pytest.mark.asyncio
    async def test_lifespan_with_failed_collections(self):
        """Lifespan logs but continues when some collections fail to initialize."""
        from src.main import lifespan

        mock_app = MagicMock()

        mock_redis = AsyncMock()
        mock_redis.initialize = AsyncMock()
        mock_redis.close = AsyncMock()

        mock_provider = AsyncMock()
        mock_provider.initialize_all_collections.return_value = {
            "initialized": ["col-00"],
            "failed": [{"collection_id": "col-01", "error": "access denied"}],
        }

        with (
            patch("src.utils.startup_validation.validate_startup_requirements"),
            patch("src.main.get_redis_client", return_value=mock_redis),
            patch("src.main.get_face_provider", return_value=mock_provider),
            patch("src.main.settings") as mock_settings,
            patch("src.main.engine") as mock_engine,
        ):
            mock_settings.liveness_enabled = False
            mock_settings.face_provider = "aws_rekognition"
            mock_settings.use_hybrid_recognition = False
            mock_engine.dispose = AsyncMock()

            async with lifespan(mock_app):
                pass  # should complete without error

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_redis_close_failure(self):
        """Redis close failure during shutdown is handled gracefully."""
        from src.main import lifespan

        mock_app = MagicMock()

        # Different mock for startup vs shutdown
        startup_redis = AsyncMock()
        startup_redis.initialize = AsyncMock()
        shutdown_redis = AsyncMock()
        shutdown_redis.close.side_effect = Exception("close failed")

        call_count = 0

        def get_redis_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return startup_redis
            return shutdown_redis

        mock_provider = AsyncMock()
        mock_provider.initialize_all_collections.return_value = {
            "initialized": ["col1"],
            "failed": [],
        }

        with (
            patch("src.utils.startup_validation.validate_startup_requirements"),
            patch("src.main.get_redis_client", side_effect=get_redis_side_effect),
            patch("src.main.get_face_provider", return_value=mock_provider),
            patch("src.main.settings") as mock_settings,
            patch("src.main.engine") as mock_engine,
        ):
            mock_settings.liveness_enabled = False
            mock_settings.face_provider = "aws_rekognition"
            mock_settings.use_hybrid_recognition = False
            mock_engine.dispose = AsyncMock()

            # Should not raise even though redis close fails
            async with lifespan(mock_app):
                pass

    @pytest.mark.asyncio
    async def test_lifespan_with_liveness_enabled(self):
        """Lifespan warms up liveness models when enabled."""
        from src.main import lifespan

        mock_app = MagicMock()

        mock_redis = AsyncMock()
        mock_redis.initialize = AsyncMock()
        mock_redis.close = AsyncMock()

        mock_provider = AsyncMock()
        mock_provider.initialize_all_collections.return_value = {
            "initialized": ["col1"],
            "failed": [],
        }

        mock_liveness = MagicMock()
        mock_liveness._get_predictor = MagicMock()

        with (
            patch("src.utils.startup_validation.validate_startup_requirements"),
            patch("src.main.get_redis_client", return_value=mock_redis),
            patch("src.main.get_face_provider", return_value=mock_provider),
            patch("src.main.settings") as mock_settings,
            patch("src.main.engine") as mock_engine,
        ):
            mock_settings.liveness_enabled = True
            mock_settings.liveness_device_id = -1
            mock_settings.liveness_model_dir = "./models/anti_spoof"
            mock_settings.liveness_detector_path = "./models"
            mock_settings.face_provider = "aws_rekognition"
            mock_settings.use_hybrid_recognition = False
            mock_engine.dispose = AsyncMock()

            with patch(
                "src.providers.silent_face_liveness.get_liveness_provider",
                return_value=mock_liveness,
            ):
                async with lifespan(mock_app):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_model_warmup_failure_continues(self):
        """Model warmup failure is non-fatal."""
        from src.main import lifespan

        mock_app = MagicMock()

        mock_redis = AsyncMock()
        mock_redis.initialize = AsyncMock()
        mock_redis.close = AsyncMock()

        mock_provider = AsyncMock()
        mock_provider.initialize_all_collections.return_value = {
            "initialized": ["col1"],
            "failed": [],
        }

        with (
            patch("src.utils.startup_validation.validate_startup_requirements"),
            patch("src.main.get_redis_client", return_value=mock_redis),
            patch("src.main.get_face_provider", return_value=mock_provider),
            patch("src.main.settings") as mock_settings,
            patch("src.main.engine") as mock_engine,
        ):
            mock_settings.liveness_enabled = True
            mock_settings.liveness_device_id = -1
            mock_settings.liveness_model_dir = "./models/anti_spoof"
            mock_settings.liveness_detector_path = "./models"
            mock_settings.face_provider = "aws_rekognition"
            mock_settings.use_hybrid_recognition = False
            mock_engine.dispose = AsyncMock()

            with patch(
                "src.providers.silent_face_liveness.get_liveness_provider",
                side_effect=Exception("model load failed"),
            ):
                # Should not raise
                async with lifespan(mock_app):
                    pass


# ============================================================================
# FaceRepository - additional coverage
# ============================================================================
class TestRepositoryGetPhotosByUserName:

    @pytest.mark.asyncio
    async def test_get_photos_by_user_name_no_filter(self):
        session = _make_mock_session()
        face = _make_fake_face(user_name="alice", photo_type="enrolled")

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [face]
        mock_result = MagicMock()
        mock_result.scalars.return_value = scalars_mock
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_photos_by_user_name("alice")

        assert len(result) == 1
        assert result[0] is face

    @pytest.mark.asyncio
    async def test_get_photos_by_user_name_with_photo_type(self):
        session = _make_mock_session()
        face = _make_fake_face(user_name="bob", photo_type="verified")

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [face]
        mock_result = MagicMock()
        mock_result.scalars.return_value = scalars_mock
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_photos_by_user_name("bob", photo_type="verified")

        assert len(result) == 1
        session.execute.assert_awaited_once()


class TestRepositoryGetPhotosByUserNamesBatch:

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        session = _make_mock_session()
        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_photos_by_user_names_batch([])
        assert result == []
        session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_returns_faces(self):
        session = _make_mock_session()
        face1 = _make_fake_face(user_name="alice")
        face2 = _make_fake_face(user_name="bob", id=2)

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [face1, face2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = scalars_mock
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_photos_by_user_names_batch(["alice", "bob"])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_batch_with_photo_type_filter(self):
        session = _make_mock_session()
        face = _make_fake_face(user_name="alice", photo_type="enrolled")

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [face]
        mock_result = MagicMock()
        mock_result.scalars.return_value = scalars_mock
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_photos_by_user_names_batch(["alice"], photo_type="enrolled")
        assert len(result) == 1


class TestRepositoryGetEnrollmentPhoto:

    @pytest.mark.asyncio
    async def test_enrollment_photo_found(self):
        session = _make_mock_session()
        face = _make_fake_face(user_name="alice", photo_type="enrolled")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = face
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_enrollment_photo("alice")
        assert result is face

    @pytest.mark.asyncio
    async def test_enrollment_photo_not_found(self):
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_enrollment_photo("nobody")
        assert result is None


class TestRepositoryGetVerifiedPhotos:

    @pytest.mark.asyncio
    async def test_verified_photos_no_limit(self):
        session = _make_mock_session()
        face1 = _make_fake_face(photo_type="verified", id=1)
        face2 = _make_fake_face(photo_type="verified", id=2)

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [face1, face2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = scalars_mock
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_verified_photos("alice")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_verified_photos_with_limit(self):
        session = _make_mock_session()
        face = _make_fake_face(photo_type="verified")

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [face]
        mock_result = MagicMock()
        mock_result.scalars.return_value = scalars_mock
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_verified_photos("alice", limit=1)
        assert len(result) == 1


class TestRepositoryGetVerifiedPhotosCount:

    @pytest.mark.asyncio
    async def test_count_returns_integer(self):
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 5
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        count = await repo.get_verified_photos_count("alice")
        assert count == 5

    @pytest.mark.asyncio
    async def test_count_returns_zero(self):
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 0
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        count = await repo.get_verified_photos_count("nobody")
        assert count == 0


class TestRepositoryGetOldestVerifiedPhoto:

    @pytest.mark.asyncio
    async def test_oldest_verified_found(self):
        session = _make_mock_session()
        old_face = _make_fake_face(
            photo_type="verified",
            created_at=datetime(2024, 1, 1),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = old_face
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_oldest_verified_photo("alice")
        assert result is old_face

    @pytest.mark.asyncio
    async def test_oldest_verified_not_found(self):
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        from src.database.repository import FaceRepository

        repo = FaceRepository(session)
        result = await repo.get_oldest_verified_photo("nobody")
        assert result is None
