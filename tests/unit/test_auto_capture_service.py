"""Unit tests for AutoCaptureService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestAutoCaptureService:
    """Tests for AutoCaptureService.capture_if_eligible."""

    @pytest.fixture
    def mock_repo(self):
        repo = AsyncMock()
        repo.get_verified_photos_count.return_value = 0
        repo.create.return_value = MagicMock()
        return repo

    @pytest.fixture
    def mock_storage(self):
        storage = AsyncMock()
        return storage

    @pytest.fixture
    def mock_insightface(self):
        provider = AsyncMock()
        provider.extract_embedding.return_value = [0.1] * 512
        return provider

    @pytest.fixture
    def matched_face(self):
        face = MagicMock()
        face.user_name = "alice"
        face.user_email = "alice@example.com"
        face.user_metadata = None
        face.provider_name = "insightface"
        face.provider_face_id = "face_1"
        face.provider_collection_id = "default"
        face.image_path = "faces/alice/enrolled.jpg"
        return face

    @pytest.fixture
    def image_data(self):
        return b"\xff\xd8\xff\xe0" + b"\x00" * 100

    def _make_service(self, repo, storage, insightface=None):
        """Import and instantiate AutoCaptureService inside the method
        so the settings patch is already active."""
        from src.services.auto_capture_service import AutoCaptureService
        return AutoCaptureService(
            repository=repo,
            storage=storage,
            insightface_provider=insightface,
        )

    @pytest.mark.asyncio
    @patch("src.services.auto_capture_service.settings")
    async def test_captures_when_under_limit(
        self, mock_settings, mock_repo, mock_storage, mock_insightface,
        matched_face, image_data,
    ):
        mock_settings.auto_capture_enabled = True
        mock_settings.auto_capture_confidence_threshold = 0.8
        mock_settings.auto_capture_max_verified_photos = 4
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        mock_repo.get_verified_photos_count.return_value = 1

        service = self._make_service(mock_repo, mock_storage, mock_insightface)
        result = await service.capture_if_eligible(
            image_data=image_data,
            matched_face=matched_face,
            confidence=0.95,
            processor="insightface",
        )

        assert result is True
        mock_storage.save.assert_awaited_once()
        mock_repo.create.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.auto_capture_service.settings")
    async def test_skips_when_disabled(
        self, mock_settings, mock_repo, mock_storage, matched_face, image_data,
    ):
        mock_settings.auto_capture_enabled = False

        service = self._make_service(mock_repo, mock_storage)
        result = await service.capture_if_eligible(
            image_data=image_data,
            matched_face=matched_face,
            confidence=0.95,
            processor="insightface",
        )

        assert result is False
        mock_repo.get_verified_photos_count.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.auto_capture_service.settings")
    async def test_skips_when_confidence_below_threshold(
        self, mock_settings, mock_repo, mock_storage, matched_face, image_data,
    ):
        mock_settings.auto_capture_enabled = True
        mock_settings.auto_capture_confidence_threshold = 0.9

        service = self._make_service(mock_repo, mock_storage)
        result = await service.capture_if_eligible(
            image_data=image_data,
            matched_face=matched_face,
            confidence=0.5,
            processor="insightface",
        )

        assert result is False
        mock_repo.get_verified_photos_count.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.auto_capture_service.settings")
    async def test_deletes_oldest_when_at_max(
        self, mock_settings, mock_repo, mock_storage, mock_insightface,
        matched_face, image_data,
    ):
        mock_settings.auto_capture_enabled = True
        mock_settings.auto_capture_confidence_threshold = 0.8
        mock_settings.auto_capture_max_verified_photos = 2
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        mock_repo.get_verified_photos_count.return_value = 2
        oldest = MagicMock()
        oldest.id = 999
        oldest.image_path = "faces/alice/old.jpg"
        mock_repo.get_oldest_verified_photo.return_value = oldest

        service = self._make_service(mock_repo, mock_storage, mock_insightface)
        result = await service.capture_if_eligible(
            image_data=image_data,
            matched_face=matched_face,
            confidence=0.95,
            processor="insightface",
        )

        assert result is True
        mock_repo.delete.assert_awaited_once_with(999)

    @pytest.mark.asyncio
    @patch("src.services.auto_capture_service.settings")
    async def test_returns_false_on_storage_error(
        self, mock_settings, mock_repo, mock_storage, matched_face, image_data,
    ):
        mock_settings.auto_capture_enabled = True
        mock_settings.auto_capture_confidence_threshold = 0.8
        mock_settings.auto_capture_max_verified_photos = 4
        mock_settings.storage_backend = "local"

        mock_repo.get_verified_photos_count.return_value = 0
        mock_storage.save.side_effect = IOError("disk full")

        service = self._make_service(mock_repo, mock_storage)
        result = await service.capture_if_eligible(
            image_data=image_data,
            matched_face=matched_face,
            confidence=0.95,
            processor="insightface",
        )

        assert result is False

    @pytest.mark.asyncio
    @patch("src.services.auto_capture_service.settings")
    async def test_no_embedding_when_no_provider(
        self, mock_settings, mock_repo, mock_storage, matched_face, image_data,
    ):
        mock_settings.auto_capture_enabled = True
        mock_settings.auto_capture_confidence_threshold = 0.8
        mock_settings.auto_capture_max_verified_photos = 4
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        mock_repo.get_verified_photos_count.return_value = 0

        service = self._make_service(mock_repo, mock_storage, insightface=None)
        result = await service.capture_if_eligible(
            image_data=image_data,
            matched_face=matched_face,
            confidence=0.95,
            processor="insightface",
        )

        assert result is True
        # Verify the Face created has no embedding
        created_face = mock_repo.create.call_args[0][0]
        assert created_face.embedding_insightface is None

    @pytest.mark.asyncio
    @patch("src.services.auto_capture_service.settings")
    async def test_storage_delete_failure_continues(
        self, mock_settings, mock_repo, mock_storage, mock_insightface,
        matched_face, image_data,
    ):
        """Even if deleting the old image from storage fails, capture proceeds."""
        mock_settings.auto_capture_enabled = True
        mock_settings.auto_capture_confidence_threshold = 0.8
        mock_settings.auto_capture_max_verified_photos = 1
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        mock_repo.get_verified_photos_count.return_value = 1
        oldest = MagicMock()
        oldest.id = 100
        oldest.image_path = "faces/alice/old.jpg"
        mock_repo.get_oldest_verified_photo.return_value = oldest

        # Storage delete fails but should not prevent capture
        mock_storage.delete.side_effect = Exception("storage unavailable")

        service = self._make_service(mock_repo, mock_storage, mock_insightface)
        result = await service.capture_if_eligible(
            image_data=image_data,
            matched_face=matched_face,
            confidence=0.95,
            processor="insightface",
        )

        assert result is True
        mock_repo.delete.assert_awaited_once_with(100)
        mock_repo.create.assert_awaited_once()
