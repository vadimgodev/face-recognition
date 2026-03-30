"""Tests for face recognition providers and factory."""

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.exceptions import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    ProviderError,
)
from src.providers.base import FaceMetadata


# ============================================================================
# InsightFace provider tests
# ============================================================================
class TestInsightFaceGetApp:
    """_get_app uses double-checked locking with threading.Lock."""

    @patch("src.providers.insightface_provider.FaceAnalysis")
    def test_get_app_uses_threading_lock(self, mock_fa_cls):
        """The provider must use a threading.Lock for lazy init."""
        from src.providers.insightface_provider import InsightFaceProvider

        provider = InsightFaceProvider(model_name="buffalo_l")
        assert isinstance(provider._lock, type(threading.Lock()))

    @patch("src.providers.insightface_provider.FaceAnalysis")
    def test_get_app_double_check_pattern(self, mock_fa_cls):
        """After acquiring the lock, _app is checked again (double-check)."""
        from src.providers.insightface_provider import InsightFaceProvider

        mock_app = MagicMock()
        mock_fa_cls.return_value = mock_app

        provider = InsightFaceProvider(model_name="buffalo_l")
        assert provider._app is None

        result = provider._get_app()
        assert result is mock_app

        # Calling again returns the cached instance, NOT re-instantiating
        mock_fa_cls.reset_mock()
        result2 = provider._get_app()
        assert result2 is mock_app
        mock_fa_cls.assert_not_called()

    @patch("src.providers.insightface_provider.FaceAnalysis")
    def test_get_app_fallback_on_assertion_error(self, mock_fa_cls):
        """When selective module loading fails, _get_app falls back to loading all modules."""
        from src.providers.insightface_provider import InsightFaceProvider

        # First call raises AssertionError (selective modules not supported),
        # second call succeeds (all modules)
        mock_app = MagicMock()
        mock_fa_cls.side_effect = [AssertionError("nope"), mock_app]

        provider = InsightFaceProvider(model_name="antelopev2")
        result = provider._get_app()

        assert result is mock_app
        assert mock_fa_cls.call_count == 2


class TestInsightFaceErrors:
    """InsightFace raises domain-specific errors, not ValueError."""

    @patch("src.providers.insightface_provider.get_redis_client")
    @patch("src.providers.insightface_provider.FaceAnalysis")
    @pytest.mark.asyncio
    async def test_no_face_raises_NoFaceDetectedError(self, mock_fa_cls, mock_redis_fn):
        from src.providers.insightface_provider import InsightFaceProvider

        # Setup mock redis to return no cached embedding
        mock_cache = AsyncMock()
        mock_cache.get_json = AsyncMock(return_value=None)
        mock_cache.set_json = AsyncMock()
        mock_redis_fn.return_value = mock_cache

        # Setup mock app that returns no faces
        mock_app = MagicMock()
        mock_app.get.return_value = []  # No faces detected
        mock_fa_cls.return_value = mock_app

        provider = InsightFaceProvider()
        provider._app = mock_app

        # Create a minimal valid image (1x1 white pixel PNG-ish data)
        # We need PIL to be able to open it, so mock at a deeper level
        fake_image = MagicMock()
        fake_image.convert.return_value = fake_image

        with patch("src.providers.insightface_provider.Image") as mock_pil:
            mock_pil.open.return_value = fake_image
            with patch("src.providers.insightface_provider.np") as mock_np:
                mock_np.array.return_value = MagicMock()

                with pytest.raises(NoFaceDetectedError):
                    await provider.extract_embedding(b"fake-image-bytes")

    @patch("src.providers.insightface_provider.get_redis_client")
    @patch("src.providers.insightface_provider.FaceAnalysis")
    @pytest.mark.asyncio
    async def test_multiple_faces_raises_MultipleFacesDetectedError(
        self, mock_fa_cls, mock_redis_fn
    ):
        from src.providers.insightface_provider import InsightFaceProvider

        mock_cache = AsyncMock()
        mock_cache.get_json = AsyncMock(return_value=None)
        mock_cache.set_json = AsyncMock()
        mock_redis_fn.return_value = mock_cache

        # Two faces detected
        face1 = MagicMock()
        face2 = MagicMock()
        mock_app = MagicMock()
        mock_app.get.return_value = [face1, face2]
        mock_fa_cls.return_value = mock_app

        provider = InsightFaceProvider()
        provider._app = mock_app

        fake_image = MagicMock()
        fake_image.convert.return_value = fake_image

        with patch("src.providers.insightface_provider.Image") as mock_pil:
            mock_pil.open.return_value = fake_image
            with patch("src.providers.insightface_provider.np") as mock_np:
                mock_np.array.return_value = MagicMock()

                with pytest.raises(MultipleFacesDetectedError):
                    await provider.extract_embedding(b"fake-image-bytes")


class TestInsightFaceProperties:

    @patch("src.providers.insightface_provider.FaceAnalysis")
    def test_provider_name(self, mock_fa_cls):
        from src.providers.insightface_provider import InsightFaceProvider

        provider = InsightFaceProvider()
        assert provider.provider_name == "insightface"

    @patch("src.providers.insightface_provider.FaceAnalysis")
    def test_supports_embeddings(self, mock_fa_cls):
        from src.providers.insightface_provider import InsightFaceProvider

        provider = InsightFaceProvider()
        assert provider.supports_embeddings is True


# ============================================================================
# AWS Rekognition provider tests
# ============================================================================
class TestAWSProviderErrors:

    @patch("src.providers.aws_rekognition.get_collection_manager")
    @patch("src.providers.aws_rekognition.settings")
    @patch("src.providers.aws_rekognition.boto3")
    def _make_provider(self, mock_boto3, mock_settings, mock_cm):
        """Helper to create an AWSRekognitionProvider with mocked deps."""
        mock_settings.aws_access_key_id = "fake"
        mock_settings.aws_secret_access_key = "fake"
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_rekognition_collection_id = "test-collection"

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        from src.providers.aws_rekognition import AWSRekognitionProvider

        provider = AWSRekognitionProvider(use_sharding=False)
        return provider, mock_client

    @pytest.mark.asyncio
    async def test_enroll_no_face_raises_NoFaceDetectedError(self):
        provider, mock_client = self._make_provider()

        # Simulate: collection exists, but no face records in response
        mock_client.describe_collection.return_value = {}
        mock_client.index_faces.return_value = {"FaceRecords": []}

        metadata = FaceMetadata(user_id="u1", user_name="Alice")

        with pytest.raises(NoFaceDetectedError):
            await provider.enroll_face(b"image-bytes", metadata)

    @pytest.mark.asyncio
    async def test_enroll_invalid_param_raises_InvalidImageError(self):
        provider, mock_client = self._make_provider()

        mock_client.describe_collection.return_value = {}
        mock_client.index_faces.side_effect = ClientError(
            {"Error": {"Code": "InvalidParameterException", "Message": "bad"}},
            "IndexFaces",
        )

        metadata = FaceMetadata(user_id="u1", user_name="Alice")

        with pytest.raises(InvalidImageError):
            await provider.enroll_face(b"bad-bytes", metadata)

    @pytest.mark.asyncio
    async def test_enroll_invalid_format_raises_InvalidImageError(self):
        provider, mock_client = self._make_provider()

        mock_client.describe_collection.return_value = {}
        mock_client.index_faces.side_effect = ClientError(
            {"Error": {"Code": "InvalidImageFormatException", "Message": "bad format"}},
            "IndexFaces",
        )

        metadata = FaceMetadata(user_id="u1", user_name="Alice")

        with pytest.raises(InvalidImageError):
            await provider.enroll_face(b"bad-bytes", metadata)

    @pytest.mark.asyncio
    async def test_enroll_other_client_error_raises_ProviderError(self):
        provider, mock_client = self._make_provider()

        mock_client.describe_collection.return_value = {}
        mock_client.index_faces.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
            "IndexFaces",
        )

        metadata = FaceMetadata(user_id="u1", user_name="Alice")

        with pytest.raises(ProviderError):
            await provider.enroll_face(b"image-bytes", metadata)


class TestAWSCompareFaces:

    @patch("src.providers.aws_rekognition.get_collection_manager")
    @patch("src.providers.aws_rekognition.settings")
    @patch("src.providers.aws_rekognition.boto3")
    def _make_provider(self, mock_boto3, mock_settings, mock_cm):
        mock_settings.aws_access_key_id = "fake"
        mock_settings.aws_secret_access_key = "fake"
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_rekognition_collection_id = "test-collection"

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        from src.providers.aws_rekognition import AWSRekognitionProvider

        provider = AWSRekognitionProvider(use_sharding=False)
        return provider, mock_client

    @pytest.mark.asyncio
    async def test_compare_faces_returns_score(self):
        provider, mock_client = self._make_provider()

        mock_client.compare_faces.return_value = {"FaceMatches": [{"Similarity": 95.5, "Face": {}}]}

        score = await provider.compare_faces(b"source", b"target", 0.0)
        assert score == pytest.approx(0.955)

    @pytest.mark.asyncio
    async def test_compare_faces_returns_none_when_no_match(self):
        provider, mock_client = self._make_provider()

        mock_client.compare_faces.return_value = {"FaceMatches": []}

        score = await provider.compare_faces(b"source", b"target", 0.0)
        assert score is None

    @pytest.mark.asyncio
    async def test_compare_faces_returns_none_on_exception(self):
        provider, mock_client = self._make_provider()

        mock_client.compare_faces.side_effect = Exception("Network error")

        score = await provider.compare_faces(b"source", b"target", 0.0)
        assert score is None


class TestAWSProperties:

    @patch("src.providers.aws_rekognition.get_collection_manager")
    @patch("src.providers.aws_rekognition.settings")
    @patch("src.providers.aws_rekognition.boto3")
    def test_provider_name(self, mock_boto3, mock_settings, mock_cm):
        mock_settings.aws_access_key_id = "fake"
        mock_settings.aws_secret_access_key = "fake"
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_rekognition_collection_id = "test-collection"
        mock_boto3.client.return_value = MagicMock()

        from src.providers.aws_rekognition import AWSRekognitionProvider

        provider = AWSRekognitionProvider(use_sharding=False)
        assert provider.provider_name == "aws_rekognition"

    @patch("src.providers.aws_rekognition.get_collection_manager")
    @patch("src.providers.aws_rekognition.settings")
    @patch("src.providers.aws_rekognition.boto3")
    def test_supports_embeddings_false(self, mock_boto3, mock_settings, mock_cm):
        mock_settings.aws_access_key_id = "fake"
        mock_settings.aws_secret_access_key = "fake"
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_rekognition_collection_id = "test-collection"
        mock_boto3.client.return_value = MagicMock()

        from src.providers.aws_rekognition import AWSRekognitionProvider

        provider = AWSRekognitionProvider(use_sharding=False)
        assert provider.supports_embeddings is False


# ============================================================================
# Provider Factory tests
# ============================================================================
class TestProviderFactory:

    @patch("src.providers.factory._insightface_cache", None)
    @patch("src.providers.factory._aws_cache", None)
    @patch("src.providers.factory.settings")
    def test_create_unknown_provider_raises_ValueError(self, mock_settings):
        from src.providers.factory import ProviderFactory

        mock_settings.face_provider = "nonexistent_provider"

        with pytest.raises(ValueError, match="Unsupported provider"):
            ProviderFactory.create_provider("nonexistent_provider")

    @patch("src.providers.factory._insightface_cache", None)
    @patch("src.providers.factory.settings")
    @patch("src.providers.factory.InsightFaceProvider")
    def test_create_insightface_returns_cached_singleton(self, mock_provider_cls, mock_settings):
        """Calling create_provider('insightface') twice returns the same instance."""
        import src.providers.factory as factory_module

        mock_settings.insightface_model = "buffalo_l"
        mock_settings.insightface_det_size = 640
        mock_settings.insightface_ctx_id = -1

        mock_instance = MagicMock()
        mock_provider_cls.return_value = mock_instance

        # Reset cache
        factory_module._insightface_cache = None

        first = factory_module.get_insightface_provider()
        second = factory_module.get_insightface_provider()

        assert first is second
        # InsightFaceProvider constructor called only once
        mock_provider_cls.assert_called_once()

        # Cleanup
        factory_module._insightface_cache = None

    @patch("src.providers.factory._aws_cache", None)
    @patch("src.providers.factory.settings")
    @patch("src.providers.factory.AWSRekognitionProvider")
    def test_create_aws_returns_cached_singleton(self, mock_provider_cls, mock_settings):
        """Calling get_aws_provider twice returns the same instance."""
        import src.providers.factory as factory_module

        mock_instance = MagicMock()
        mock_provider_cls.return_value = mock_instance

        factory_module._aws_cache = None

        first = factory_module.get_aws_provider()
        second = factory_module.get_aws_provider()

        assert first is second
        mock_provider_cls.assert_called_once()

        # Cleanup
        factory_module._aws_cache = None

    @patch("src.providers.factory._insightface_cache", None)
    @patch("src.providers.factory._aws_cache", None)
    @patch("src.providers.factory.settings")
    @patch("src.providers.factory.InsightFaceProvider")
    def test_create_provider_defaults_to_settings(self, mock_provider_cls, mock_settings):
        """When no provider_name given, uses settings.face_provider."""
        import src.providers.factory as factory_module

        mock_settings.face_provider = "insightface"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.insightface_det_size = 640
        mock_settings.insightface_ctx_id = -1

        mock_instance = MagicMock()
        mock_provider_cls.return_value = mock_instance
        factory_module._insightface_cache = None

        result = factory_module.ProviderFactory.create_provider()
        assert result is mock_instance

        # Cleanup
        factory_module._insightface_cache = None

    def test_get_available_providers(self):
        from src.providers.factory import ProviderFactory

        providers = ProviderFactory.get_available_providers()
        assert "aws_rekognition" in providers
        assert "insightface" in providers
