"""Unit tests for recognition strategies and the create_strategy factory."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.recognition_strategies import (
    create_strategy,
    InsightFaceOnlyStrategy,
    SmartHybridStrategy,
    InsightFaceAWSStrategy,
    AWSOnlyStrategy,
    RecognitionResult,
    RecognitionStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized_vector(dim: int = 512, seed: int = 0):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float64)
    return (v / np.linalg.norm(v)).tolist()


def _make_face(
    user_name: str = "alice",
    embedding: list = None,
    photo_type: str = "enrolled",
    face_id: int = 1,
):
    face = MagicMock()
    face.id = face_id
    face.user_name = user_name
    face.embedding_insightface = embedding
    face.photo_type = photo_type
    face.image_path = f"faces/{user_name}/{face_id}.jpg"
    face.provider_face_id = f"prov_{face_id}"
    face.provider_collection_id = "default"
    return face


# ---------------------------------------------------------------------------
# create_strategy factory
# ---------------------------------------------------------------------------

class TestCreateStrategy:
    """Tests for the create_strategy factory function."""

    def test_insightface_only(self):
        strategy = create_strategy(
            mode="insightface_only",
            insightface_provider=MagicMock(),
            repository=MagicMock(),
            template_service=MagicMock(),
        )
        assert isinstance(strategy, InsightFaceOnlyStrategy)

    def test_smart_hybrid(self):
        strategy = create_strategy(
            mode="smart_hybrid",
            insightface_provider=MagicMock(),
            aws_provider=MagicMock(),
            repository=MagicMock(),
            template_service=MagicMock(),
            storage=MagicMock(),
        )
        assert isinstance(strategy, SmartHybridStrategy)

    def test_insightface_aws(self):
        strategy = create_strategy(
            mode="insightface_aws",
            insightface_provider=MagicMock(),
            aws_provider=MagicMock(),
            repository=MagicMock(),
            template_service=MagicMock(),
        )
        assert isinstance(strategy, InsightFaceAWSStrategy)

    def test_aws_only(self):
        strategy = create_strategy(
            mode="aws_only",
            aws_provider=MagicMock(),
            repository=MagicMock(),
        )
        assert isinstance(strategy, AWSOnlyStrategy)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown recognition mode"):
            create_strategy(mode="does_not_exist")

    def test_all_strategies_are_recognition_strategy(self):
        for cls in [
            InsightFaceOnlyStrategy,
            SmartHybridStrategy,
            InsightFaceAWSStrategy,
            AWSOnlyStrategy,
        ]:
            assert issubclass(cls, RecognitionStrategy)


# ---------------------------------------------------------------------------
# RecognitionResult
# ---------------------------------------------------------------------------

class TestRecognitionResult:
    """Tests for the RecognitionResult data class."""

    def test_creation(self):
        face = _make_face()
        rr = RecognitionResult(face=face, similarity=0.95)
        assert rr.face is face
        assert rr.similarity == 0.95
        assert rr.aws_verified is False

    def test_aws_verified_flag(self):
        face = _make_face()
        rr = RecognitionResult(face=face, similarity=0.8, aws_verified=True)
        assert rr.aws_verified is True


# ---------------------------------------------------------------------------
# InsightFaceOnlyStrategy
# ---------------------------------------------------------------------------

class TestInsightFaceOnlyStrategy:
    """Tests for InsightFaceOnlyStrategy."""

    @pytest.fixture
    def mock_provider(self):
        provider = AsyncMock()
        provider.extract_embedding.return_value = _normalized_vector(512, seed=1)
        return provider

    @pytest.fixture
    def mock_repo(self):
        return AsyncMock()

    @pytest.fixture
    def mock_template(self):
        return AsyncMock()

    @pytest.fixture
    def strategy(self, mock_provider, mock_repo, mock_template):
        return InsightFaceOnlyStrategy(mock_provider, mock_repo, mock_template)

    @pytest.mark.asyncio
    async def test_recognize_returns_tuples_with_aws_false(
        self, strategy, mock_provider, mock_repo, mock_template,
    ):
        emb = _normalized_vector(512, seed=1)
        face = _make_face("alice", embedding=emb)

        mock_repo.search_by_embedding.return_value = [(face, 0.9)]
        mock_template.compute_template_results.return_value = [(face, 0.92)]

        results = await strategy.recognize(
            image_data=b"fake_image",
            max_results=5,
            confidence_threshold=0.5,
        )

        assert len(results) == 1
        result_face, score, aws_used = results[0]
        assert result_face is face
        assert score == 0.92
        assert aws_used is False

    @pytest.mark.asyncio
    async def test_recognize_from_embedding(
        self, strategy, mock_repo, mock_template,
    ):
        emb = _normalized_vector(512, seed=2)
        face = _make_face("bob", embedding=emb)

        mock_repo.search_by_embedding.return_value = [(face, 0.85)]
        mock_template.compute_template_results.return_value = [(face, 0.88)]

        results = await strategy.recognize_from_embedding(
            embedding=emb,
            max_results=5,
            confidence_threshold=0.5,
        )

        assert len(results) == 1
        assert results[0] == (face, 0.88)

    @pytest.mark.asyncio
    async def test_recognize_empty_candidates(
        self, strategy, mock_repo, mock_template,
    ):
        mock_repo.search_by_embedding.return_value = []
        mock_template.compute_template_results.return_value = []

        results = await strategy.recognize(
            image_data=b"fake",
            max_results=5,
            confidence_threshold=0.5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_limit_is_5x_max_results(
        self, strategy, mock_repo, mock_template,
    ):
        mock_repo.search_by_embedding.return_value = []
        mock_template.compute_template_results.return_value = []

        await strategy.recognize(
            image_data=b"fake",
            max_results=3,
            confidence_threshold=0.5,
        )

        call_kwargs = mock_repo.search_by_embedding.call_args[1]
        assert call_kwargs["limit"] == 15  # 3 * 5


# ---------------------------------------------------------------------------
# SmartHybridStrategy - confidence tiers
# ---------------------------------------------------------------------------

class TestSmartHybridStrategy:
    """Tests for SmartHybridStrategy confidence tiers."""

    @pytest.fixture
    def mock_insightface(self):
        provider = AsyncMock()
        provider.extract_embedding.return_value = _normalized_vector(512, seed=1)
        return provider

    @pytest.fixture
    def mock_aws(self):
        return AsyncMock()

    @pytest.fixture
    def mock_repo(self):
        repo = AsyncMock()
        repo.get_photos_by_user_name.return_value = []
        return repo

    @pytest.fixture
    def mock_template(self):
        return AsyncMock()

    @pytest.fixture
    def mock_storage(self):
        return AsyncMock()

    @pytest.fixture
    def strategy(
        self, mock_insightface, mock_aws, mock_repo, mock_template, mock_storage,
    ):
        return SmartHybridStrategy(
            mock_insightface, mock_aws, mock_repo, mock_template, mock_storage,
        )

    @pytest.mark.asyncio
    @patch("src.services.recognition_strategies.settings")
    async def test_high_confidence_accepted_immediately(
        self, mock_settings, strategy, mock_repo, mock_template,
    ):
        mock_settings.insightface_high_confidence = 0.8
        mock_settings.insightface_medium_confidence = 0.6

        emb = _normalized_vector(512, seed=1)
        face = _make_face("alice", embedding=emb)
        face_enrolled = _make_face("alice", embedding=emb, photo_type="enrolled")

        mock_repo.search_by_embedding.return_value = [(face, 0.9)]
        mock_repo.get_photos_by_user_name.return_value = [face_enrolled]

        mock_template.compute_template_results_single_user.return_value = (
            face_enrolled, 0.91
        )

        results = await strategy.recognize(
            image_data=b"fake",
            max_results=5,
            confidence_threshold=0.5,
        )

        # High confidence -> template path, no AWS
        assert len(results) == 1
        _face, score, aws_used = results[0]
        assert aws_used is False

    @pytest.mark.asyncio
    @patch("src.services.recognition_strategies.settings")
    async def test_no_candidates_returns_empty(
        self, mock_settings, strategy, mock_repo,
    ):
        mock_settings.insightface_medium_confidence = 0.6

        mock_repo.search_by_embedding.return_value = []

        results = await strategy.recognize(
            image_data=b"fake",
            max_results=5,
            confidence_threshold=0.5,
        )

        assert results == []

    @pytest.mark.asyncio
    @patch("src.services.recognition_strategies.settings")
    async def test_low_confidence_rejected(
        self, mock_settings, strategy, mock_repo,
    ):
        """Candidates below medium threshold are filtered by the search threshold."""
        mock_settings.insightface_high_confidence = 0.8
        mock_settings.insightface_medium_confidence = 0.6

        # No candidates pass the medium threshold search
        mock_repo.search_by_embedding.return_value = []

        results = await strategy.recognize(
            image_data=b"fake",
            max_results=5,
            confidence_threshold=0.5,
        )

        assert results == []

    @pytest.mark.asyncio
    @patch("src.services.recognition_strategies.settings")
    async def test_recognize_from_embedding_only_high_confidence(
        self, mock_settings, strategy, mock_repo, mock_template,
    ):
        """recognize_from_embedding only keeps high-confidence results."""
        mock_settings.insightface_high_confidence = 0.8
        mock_settings.insightface_medium_confidence = 0.6

        emb = _normalized_vector(512, seed=1)
        face_high = _make_face("alice", embedding=emb, face_id=1)
        face_mid = _make_face("bob", embedding=emb, face_id=2)

        mock_repo.search_by_embedding.return_value = [
            (face_high, 0.85),
            (face_mid, 0.65),  # Below high threshold
        ]

        mock_template.compute_template_results_single_user.return_value = (
            face_high, 0.87
        )

        results = await strategy.recognize_from_embedding(
            embedding=emb,
            max_results=5,
            confidence_threshold=0.5,
        )

        # Only alice (0.85 >= 0.8) passes the high threshold filter
        assert len(results) == 1
        assert results[0][0].user_name == "alice"


# ---------------------------------------------------------------------------
# AWSOnlyStrategy
# ---------------------------------------------------------------------------

class TestAWSOnlyStrategy:
    """Tests for AWSOnlyStrategy."""

    def test_recognize_from_embedding_raises(self):
        strategy = AWSOnlyStrategy(
            aws_provider=MagicMock(),
            repository=MagicMock(),
        )
        with pytest.raises(ValueError, match="aws_only mode not supported"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                strategy.recognize_from_embedding(
                    embedding=[0.1] * 512,
                    max_results=5,
                    confidence_threshold=0.5,
                )
            )
