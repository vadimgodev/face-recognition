"""Tests for FaceRepository (src/database/repository.py)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime

from src.database.repository import FaceRepository


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_session():
    """Return a fully-mocked AsyncSession."""
    session = AsyncMock()
    return session


def _make_fake_face(**overrides):
    """Return a MagicMock that behaves like a Face instance."""
    defaults = dict(
        id=1,
        user_name="alice",
        user_email="alice@example.com",
        provider_name="insightface",
        provider_face_id="face_abc",
        image_path="/images/alice.jpg",
        image_storage="local",
        photo_type="enrolled",
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
    )
    defaults.update(overrides)
    face = MagicMock(**defaults)
    # Make sure .id attribute works correctly on the mock
    face.id = defaults["id"]
    return face


# ============================================================================
# Tests: create
# ============================================================================
class TestCreate:

    @pytest.mark.asyncio
    async def test_create_adds_and_commits(self):
        session = _make_mock_session()
        repo = FaceRepository(session)
        face = _make_fake_face()

        result = await repo.create(face)

        session.add.assert_called_once_with(face)
        session.commit.assert_awaited_once()
        session.refresh.assert_awaited_once_with(face)
        assert result is face


# ============================================================================
# Tests: get_by_id
# ============================================================================
class TestGetById:

    @pytest.mark.asyncio
    async def test_get_by_id_returns_face(self):
        session = _make_mock_session()
        face = _make_fake_face(id=42)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = face
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        result = await repo.get_by_id(42)

        assert result is face
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_missing(self):
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        result = await repo.get_by_id(999)

        assert result is None


# ============================================================================
# Tests: delete
# ============================================================================
class TestDelete:

    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self):
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        result = await repo.delete(42)

        assert result is True
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self):
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        result = await repo.delete(999)

        assert result is False


# ============================================================================
# Tests: list_all uses func.count (not len)
# ============================================================================
class TestListAll:
    """Verify list_all uses SQL COUNT, not Python len()."""

    @pytest.mark.asyncio
    async def test_list_all_issues_two_queries(self):
        """list_all must execute two queries: one COUNT, one SELECT."""
        session = _make_mock_session()

        face = _make_fake_face()

        # First execute call: count query
        count_result = MagicMock()
        count_result.scalar_one.return_value = 5

        # Second execute call: paginated results
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [face]
        paginated_result = MagicMock()
        paginated_result.scalars.return_value = scalars_mock

        session.execute.side_effect = [count_result, paginated_result]

        repo = FaceRepository(session)
        faces, total = await repo.list_all(limit=10, offset=0)

        assert total == 5
        assert faces == [face]
        assert session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_list_all_uses_func_count(self):
        """The count query must use func.count(Face.id), not len().

        We verify by inspecting the SQL statement passed to the first execute() call.
        The compiled SQL must contain a COUNT expression (server-side aggregate)
        rather than the code doing len(scalars().all()).
        """
        session = _make_mock_session()

        count_result = MagicMock()
        count_result.scalar_one.return_value = 3

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        paginated_result = MagicMock()
        paginated_result.scalars.return_value = scalars_mock

        session.execute.side_effect = [count_result, paginated_result]

        repo = FaceRepository(session)
        _, total = await repo.list_all()

        # Inspect the first query passed to execute (the count query)
        first_call_args = session.execute.call_args_list[0]
        stmt = first_call_args[0][0]  # The SQLAlchemy Select statement

        # Compile the statement to SQL text and verify it contains "count"
        from sqlalchemy.dialects import postgresql
        compiled = str(stmt.compile(dialect=postgresql.dialect()))
        assert "count" in compiled.lower(), (
            f"Expected SQL COUNT in query, got: {compiled}"
        )

    @pytest.mark.asyncio
    async def test_list_all_respects_pagination(self):
        session = _make_mock_session()

        count_result = MagicMock()
        count_result.scalar_one.return_value = 50

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        paginated_result = MagicMock()
        paginated_result.scalars.return_value = scalars_mock

        session.execute.side_effect = [count_result, paginated_result]

        repo = FaceRepository(session)
        faces, total = await repo.list_all(limit=10, offset=20)

        assert total == 50
        # Two queries were executed
        assert session.execute.await_count == 2


# ============================================================================
# Tests: search_by_embedding
# ============================================================================
class TestSearchByEmbedding:

    @pytest.mark.asyncio
    async def test_search_returns_face_similarity_tuples(self):
        session = _make_mock_session()
        face = _make_fake_face()

        mock_result = MagicMock()
        mock_result.all.return_value = [(face, 0.95)]
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        results = await repo.search_by_embedding(
            embedding=[0.1] * 512, threshold=0.7, limit=10
        )

        assert len(results) == 1
        assert results[0][0] is face
        assert results[0][1] == 0.95

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_matches(self):
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        results = await repo.search_by_embedding(
            embedding=[0.0] * 512, threshold=0.9
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_query_is_executed(self):
        """Verify the search query is actually sent to the session."""
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        await repo.search_by_embedding(embedding=[0.5] * 512, threshold=0.6, limit=5)

        session.execute.assert_awaited_once()


# ============================================================================
# Tests: get_by_provider_face_id
# ============================================================================
class TestGetByProviderFaceId:

    @pytest.mark.asyncio
    async def test_returns_face_for_matching_ids(self):
        session = _make_mock_session()
        face = _make_fake_face(provider_face_id="abc", provider_name="insightface")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = face
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        result = await repo.get_by_provider_face_id("abc", "insightface")

        assert result is face

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        repo = FaceRepository(session)
        result = await repo.get_by_provider_face_id("nope", "aws")

        assert result is None


# ============================================================================
# Tests: get_photos_by_user_names_batch
# ============================================================================
class TestBatchQueries:

    @pytest.mark.asyncio
    async def test_empty_user_names_returns_empty_list(self):
        session = _make_mock_session()
        repo = FaceRepository(session)
        result = await repo.get_photos_by_user_names_batch([])

        assert result == []
        # Should NOT execute any query
        session.execute.assert_not_awaited()
