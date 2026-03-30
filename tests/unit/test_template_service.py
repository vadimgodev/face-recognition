"""Unit tests for TemplateService."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.services.template_service import TemplateService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_face(
    user_name: str = "alice",
    embedding: list = None,
    photo_type: str = "enrolled",
    face_id: int = 1,
):
    """Create a lightweight mock Face."""
    face = MagicMock()
    face.id = face_id
    face.user_name = user_name
    face.embedding_insightface = embedding
    face.photo_type = photo_type
    face.image_path = f"faces/{user_name}/{face_id}.jpg"
    return face


def _normalized_vector(dim: int = 512, seed: int = 0):
    """Return a normalized random vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float64)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# compute_cosine_similarity
# ---------------------------------------------------------------------------


class TestComputeCosineSimilarity:
    """Tests for the static cosine similarity method."""

    def test_identical_vectors(self):
        vec = _normalized_vector(512, seed=42)
        sim = TemplateService.compute_cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should give similarity ~0.5 on the 0-1 scale."""
        # Create two orthogonal unit vectors in 2D for clarity
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = TemplateService.compute_cosine_similarity(v1, v2)
        # cosine_sim=0, cosine_distance=1, similarity = 1 - 0.5 = 0.5
        assert sim == pytest.approx(0.5, abs=1e-6)

    def test_opposite_vectors(self):
        """Opposite vectors: cosine_sim=-1, distance=2, similarity=0."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        sim = TemplateService.compute_cosine_similarity(v1, v2)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_similar_vectors_high_score(self):
        """Slightly different vectors should score close to 1.0."""
        v1 = _normalized_vector(512, seed=10)
        v2 = list(v1)  # copy
        # Perturb slightly
        v2[0] += 0.001
        sim = TemplateService.compute_cosine_similarity(v1, v2)
        assert sim > 0.99

    def test_returns_float(self):
        v = _normalized_vector(4, seed=0)
        result = TemplateService.compute_cosine_similarity(v, v)
        assert isinstance(result, float)

    def test_different_random_vectors(self):
        """Two random high-dimensional vectors should have similarity near 0.5."""
        v1 = _normalized_vector(512, seed=1)
        v2 = _normalized_vector(512, seed=2)
        sim = TemplateService.compute_cosine_similarity(v1, v2)
        # Random unit vectors are nearly orthogonal in high dimensions
        assert 0.4 < sim < 0.6


# ---------------------------------------------------------------------------
# compute_template_results
# ---------------------------------------------------------------------------


class TestComputeTemplateResults:
    """Tests for compute_template_results with mocked repository."""

    @pytest.fixture
    def mock_repo(self):
        repo = AsyncMock()
        return repo

    @pytest.fixture
    def service(self, mock_repo):
        return TemplateService(repository=mock_repo)

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self, service):
        result = await service.compute_template_results(
            query_embedding=_normalized_vector(512),
            candidates=[],
            confidence_threshold=0.5,
            max_results=5,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_groups_by_user_name(self, service, mock_repo):
        """Two candidates for the same user should result in one output entry."""
        emb = _normalized_vector(512, seed=7)
        face1 = _make_face("bob", embedding=emb, face_id=1)
        face2 = _make_face("bob", embedding=emb, face_id=2)

        # Repository returns all faces for the batch query
        mock_repo.get_photos_by_user_names_batch.return_value = [face1, face2]

        result = await service.compute_template_results(
            query_embedding=emb,
            candidates=[(face1, 0.9), (face2, 0.85)],
            confidence_threshold=0.0,
            max_results=10,
        )

        assert len(result) == 1
        assert result[0][0].user_name == "bob"

    @pytest.mark.asyncio
    async def test_returns_template_similarity(self, service, mock_repo):
        """Template similarity should be computed from averaged embeddings."""
        emb = _normalized_vector(512, seed=3)
        face = _make_face("carol", embedding=emb, face_id=1)

        mock_repo.get_photos_by_user_names_batch.return_value = [face]

        result = await service.compute_template_results(
            query_embedding=emb,
            candidates=[(face, 0.9)],
            confidence_threshold=0.0,
            max_results=10,
        )

        assert len(result) == 1
        # Same embedding queried against itself -> similarity ~1.0
        assert result[0][1] == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.asyncio
    async def test_filters_by_confidence_threshold(self, service, mock_repo):
        """Users below the threshold should be excluded."""
        emb_query = _normalized_vector(512, seed=1)
        emb_distant = _normalized_vector(512, seed=2)
        face = _make_face("dave", embedding=emb_distant, face_id=1)

        mock_repo.get_photos_by_user_names_batch.return_value = [face]

        result = await service.compute_template_results(
            query_embedding=emb_query,
            candidates=[(face, 0.5)],
            confidence_threshold=0.99,
            max_results=10,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_max_results(self, service, mock_repo):
        """Should return at most max_results entries."""
        emb = _normalized_vector(512, seed=5)
        faces = []
        for i in range(5):
            name = f"user_{i}"
            f = _make_face(name, embedding=emb, face_id=i)
            faces.append(f)

        mock_repo.get_photos_by_user_names_batch.return_value = faces

        candidates = [(f, 0.9) for f in faces]
        result = await service.compute_template_results(
            query_embedding=emb,
            candidates=candidates,
            confidence_threshold=0.0,
            max_results=2,
        )

        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_skips_users_without_embeddings(self, service, mock_repo):
        """Users with no valid embeddings should be skipped."""
        face = _make_face("eve", embedding=None, face_id=1)

        mock_repo.get_photos_by_user_names_batch.return_value = [face]

        result = await service.compute_template_results(
            query_embedding=_normalized_vector(512),
            candidates=[(face, 0.9)],
            confidence_threshold=0.0,
            max_results=10,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_results_sorted_descending(self, service, mock_repo):
        """Results should be sorted by template similarity, highest first."""
        emb_query = _normalized_vector(512, seed=10)

        # Create two users with slightly different embeddings
        emb_close = list(emb_query)  # very similar
        rng = np.random.RandomState(99)
        emb_far = (rng.randn(512) / np.linalg.norm(rng.randn(512))).tolist()

        face_close = _make_face("close_user", embedding=emb_close, face_id=1)
        face_far = _make_face("far_user", embedding=emb_far, face_id=2)

        mock_repo.get_photos_by_user_names_batch.return_value = [face_close, face_far]

        result = await service.compute_template_results(
            query_embedding=emb_query,
            candidates=[(face_far, 0.5), (face_close, 0.9)],
            confidence_threshold=0.0,
            max_results=10,
        )

        if len(result) >= 2:
            assert result[0][1] >= result[1][1]

    @pytest.mark.asyncio
    async def test_batch_fetch_called_with_user_names(self, service, mock_repo):
        """Verify repository is called with the correct user names."""
        emb = _normalized_vector(512, seed=4)
        face_a = _make_face("alice", embedding=emb, face_id=1)
        face_b = _make_face("bob", embedding=emb, face_id=2)

        mock_repo.get_photos_by_user_names_batch.return_value = [face_a, face_b]

        await service.compute_template_results(
            query_embedding=emb,
            candidates=[(face_a, 0.9), (face_b, 0.8)],
            confidence_threshold=0.0,
            max_results=10,
        )

        call_args = mock_repo.get_photos_by_user_names_batch.call_args[0][0]
        assert set(call_args) == {"alice", "bob"}


# ---------------------------------------------------------------------------
# get_representative_face
# ---------------------------------------------------------------------------


class TestGetRepresentativeFace:
    """Tests for the static get_representative_face method."""

    def test_returns_enrolled_photo_if_available(self):
        enrolled = _make_face("alice", photo_type="enrolled", face_id=1)
        verified = _make_face("alice", photo_type="verified", face_id=2)
        result = TemplateService.get_representative_face([verified, enrolled])
        assert result.photo_type == "enrolled"

    def test_returns_first_if_no_enrolled(self):
        v1 = _make_face("alice", photo_type="verified", face_id=1)
        v2 = _make_face("alice", photo_type="verified", face_id=2)
        result = TemplateService.get_representative_face([v1, v2])
        assert result is v1

    def test_returns_fallback_if_empty(self):
        fallback = _make_face("alice", photo_type="enrolled", face_id=99)
        result = TemplateService.get_representative_face([], fallback=fallback)
        assert result is fallback

    def test_raises_if_empty_and_no_fallback(self):
        with pytest.raises(ValueError, match="No faces and no fallback"):
            TemplateService.get_representative_face([])

    def test_custom_prefer_type(self):
        enrolled = _make_face("alice", photo_type="enrolled", face_id=1)
        verified = _make_face("alice", photo_type="verified", face_id=2)
        result = TemplateService.get_representative_face(
            [enrolled, verified], prefer_type="verified"
        )
        assert result.photo_type == "verified"

    def test_single_face_returned(self):
        face = _make_face("solo", photo_type="enrolled", face_id=1)
        result = TemplateService.get_representative_face([face])
        assert result is face
