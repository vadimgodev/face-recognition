"""Provider contract tests. A new provider must subclass the matching
Contract class, provide the `provider` fixture, and pass unchanged.

`sample_image_bytes` is a shared fixture from tests/contracts/conftest.py —
see docs/extending.md for how a third-party provider reuses these contracts."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.providers.interfaces import CloudMatchProvider, EmbeddingProvider


class EmbeddingProviderContract:
    @pytest.fixture
    def provider(self):
        raise NotImplementedError("subclass provides the provider instance")

    def test_satisfies_protocol(self, provider):
        assert isinstance(provider, EmbeddingProvider)

    def test_declares_512_dim(self, provider):
        assert provider.embedding_dim == 512

    async def test_extract_embedding_returns_normalized_512(self, provider, sample_image_bytes):
        embedding = await provider.extract_embedding(sample_image_bytes)
        vector = np.asarray(embedding)
        assert vector.shape == (512,)
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)


class CloudMatchProviderContract:
    @pytest.fixture
    def provider(self):
        raise NotImplementedError("subclass provides the provider instance")

    def test_satisfies_protocol(self, provider):
        assert isinstance(provider, CloudMatchProvider)

    async def test_compare_faces_returns_unit_interval(self, provider, sample_image_bytes):
        score = await provider.compare_faces(sample_image_bytes, sample_image_bytes)
        assert score is not None
        assert 0.0 <= score <= 1.0


class TestInsightFaceContract(EmbeddingProviderContract):
    @pytest.fixture
    def provider(self):
        from src.providers.insightface_provider import InsightFaceProvider

        face = MagicMock()
        face.normed_embedding = (np.ones(512) / np.sqrt(512)).astype(np.float32)
        app = MagicMock()
        app.get.return_value = [face]

        mock_cache = AsyncMock()
        mock_cache.get_json = AsyncMock(return_value=None)
        mock_cache.set_json = AsyncMock()

        with patch("src.providers.insightface_provider.get_redis_client", return_value=mock_cache):
            p = InsightFaceProvider()
            p._app = app
            yield p


class TestAWSContract(CloudMatchProviderContract):
    @pytest.fixture
    def provider(self):
        with (
            patch("src.providers.aws_rekognition.get_collection_manager"),
            patch("src.providers.aws_rekognition.settings") as mock_settings,
            patch("src.providers.aws_rekognition.boto3") as mock_boto3,
        ):
            mock_settings.aws_access_key_id = "fake"
            mock_settings.aws_secret_access_key = "fake"
            mock_settings.aws_region = "us-east-1"
            mock_settings.aws_rekognition_collection_id = "test-collection"

            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            from src.providers.aws_rekognition import AWSRekognitionProvider

            p = AWSRekognitionProvider(use_sharding=False)

        p.client.compare_faces.return_value = {"FaceMatches": [{"Similarity": 93.0}]}
        return p
