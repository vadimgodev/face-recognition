"""Unit tests to increase coverage for door_service, hybrid_face_service,
multiface_service, and face_service."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_face(
    user_name: str = "alice",
    face_id: int = 1,
    provider_name: str = "insightface",
    provider_face_id: str = "prov_1",
    provider_collection_id: str = "default",
    image_path: str = "faces/alice/1.jpg",
    photo_type: str = "enrolled",
    embedding_insightface=None,
):
    face = MagicMock()
    face.id = face_id
    face.user_name = user_name
    face.user_email = f"{user_name}@example.com"
    face.user_metadata = None
    face.provider_name = provider_name
    face.provider_face_id = provider_face_id
    face.provider_collection_id = provider_collection_id
    face.image_path = image_path
    face.image_storage = "local"
    face.photo_type = photo_type
    face.embedding_insightface = embedding_insightface
    return face


def _jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    """Return minimal valid JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ===========================================================================
# DoorService tests
# ===========================================================================


class TestMockDoorProvider:
    """Tests for MockDoorProvider."""

    @pytest.mark.asyncio
    async def test_unlock_always_returns_true(self):
        from src.services.door_service import MockDoorProvider

        provider = MockDoorProvider()
        result = await provider.unlock("alice", 0.95)
        assert result is True

    @pytest.mark.asyncio
    async def test_unlock_with_different_inputs(self):
        from src.services.door_service import MockDoorProvider

        provider = MockDoorProvider()
        assert await provider.unlock("bob", 0.50) is True
        assert await provider.unlock("", 0.0) is True


class TestHttpDoorProvider:
    """Tests for HttpDoorProvider."""

    @pytest.mark.asyncio
    async def test_init_sets_url_and_timeout(self):
        from src.services.door_service import HttpDoorProvider

        provider = HttpDoorProvider(unlock_url="http://example.com/unlock", timeout=10)
        assert provider.unlock_url == "http://example.com/unlock"
        assert provider.timeout == 10

    @pytest.mark.asyncio
    async def test_unlock_success(self):
        from src.services.door_service import HttpDoorProvider

        provider = HttpDoorProvider(unlock_url="http://door/unlock")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.services.door_service.httpx.AsyncClient", return_value=mock_client):
            result = await provider.unlock("alice", 0.95)

        assert result is True
        mock_client.post.assert_awaited_once_with(
            "http://door/unlock",
            json={
                "user_name": "alice",
                "confidence": 0.95,
                "action": "unlock",
            },
            timeout=5,
        )

    @pytest.mark.asyncio
    async def test_unlock_http_error(self):
        import httpx

        from src.services.door_service import HttpDoorProvider

        provider = HttpDoorProvider(unlock_url="http://door/unlock")

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.services.door_service.httpx.AsyncClient", return_value=mock_client):
            result = await provider.unlock("alice", 0.95)

        assert result is False

    @pytest.mark.asyncio
    async def test_unlock_unexpected_error(self):
        from src.services.door_service import HttpDoorProvider

        provider = HttpDoorProvider(unlock_url="http://door/unlock")

        mock_client = AsyncMock()
        mock_client.post.side_effect = RuntimeError("connection lost")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.services.door_service.httpx.AsyncClient", return_value=mock_client):
            result = await provider.unlock("alice", 0.95)

        assert result is False


class TestGpioDoorProvider:
    """Tests for GpioDoorProvider."""

    def test_init_without_gpio_module(self):
        """When RPi.GPIO is not importable, _gpio_available should be False."""
        from src.services.door_service import GpioDoorProvider

        provider = GpioDoorProvider(pin=17, pulse_duration=0.5)
        assert provider._gpio_available is False
        assert provider.pin == 17
        assert provider.pulse_duration == 0.5

    @pytest.mark.asyncio
    async def test_unlock_returns_false_when_gpio_unavailable(self):
        from src.services.door_service import GpioDoorProvider

        provider = GpioDoorProvider(pin=17)
        result = await provider.unlock("alice", 0.95)
        assert result is False

    @pytest.mark.asyncio
    async def test_unlock_success_with_gpio(self):
        """When GPIO is available, unlock sets pin HIGH, sleeps, then LOW."""
        from src.services.door_service import GpioDoorProvider

        provider = GpioDoorProvider.__new__(GpioDoorProvider)
        provider.pin = 17
        provider.pulse_duration = 0.01
        provider._gpio_available = True

        mock_gpio = MagicMock()
        mock_gpio.HIGH = 1
        mock_gpio.LOW = 0
        provider.GPIO = mock_gpio

        result = await provider.unlock("alice", 0.95)
        assert result is True
        mock_gpio.output.assert_any_call(17, 1)
        mock_gpio.output.assert_any_call(17, 0)

    @pytest.mark.asyncio
    async def test_unlock_gpio_exception(self):
        """If GPIO.output raises, return False."""
        from src.services.door_service import GpioDoorProvider

        provider = GpioDoorProvider.__new__(GpioDoorProvider)
        provider.pin = 17
        provider.pulse_duration = 0.01
        provider._gpio_available = True

        mock_gpio = MagicMock()
        mock_gpio.HIGH = 1
        mock_gpio.output.side_effect = RuntimeError("GPIO fault")
        provider.GPIO = mock_gpio

        result = await provider.unlock("alice", 0.95)
        assert result is False

    def test_cleanup_when_gpio_available(self):
        from src.services.door_service import GpioDoorProvider

        provider = GpioDoorProvider.__new__(GpioDoorProvider)
        provider._gpio_available = True
        provider.GPIO = MagicMock()

        provider.cleanup()
        provider.GPIO.cleanup.assert_called_once()

    def test_cleanup_when_gpio_unavailable(self):
        from src.services.door_service import GpioDoorProvider

        provider = GpioDoorProvider.__new__(GpioDoorProvider)
        provider._gpio_available = False

        # Should not raise
        provider.cleanup()

    def test_init_with_mock_gpio_module(self):
        """Test GPIO initialization succeeds when RPi.GPIO is importable."""
        import sys

        from src.services.door_service import GpioDoorProvider

        mock_gpio = MagicMock()
        mock_gpio.BCM = 11
        mock_gpio.OUT = 0
        mock_gpio.LOW = 0

        fake_rpi = MagicMock()
        fake_rpi.GPIO = mock_gpio

        with patch.dict(sys.modules, {"RPi": fake_rpi, "RPi.GPIO": mock_gpio}):
            provider = GpioDoorProvider(pin=18, pulse_duration=2.0)

        assert provider._gpio_available is True
        assert provider.pin == 18

    def test_init_gpio_general_exception(self):
        """Test GPIO initialization handles general exceptions."""
        import sys

        from src.services.door_service import GpioDoorProvider

        mock_gpio = MagicMock()
        mock_gpio.BCM = 11
        mock_gpio.OUT = 0
        mock_gpio.LOW = 0
        # setup raises after setmode succeeds
        mock_gpio.setup.side_effect = RuntimeError("permission denied")

        fake_rpi = MagicMock()
        fake_rpi.GPIO = mock_gpio

        with patch.dict(sys.modules, {"RPi": fake_rpi, "RPi.GPIO": mock_gpio}):
            provider = GpioDoorProvider(pin=17)

        assert provider._gpio_available is False


class TestDoorService:
    """Tests for DoorService."""

    @patch("src.services.door_service.settings")
    def test_init_with_explicit_provider(self, mock_settings):
        from src.services.door_service import DoorService, MockDoorProvider

        provider = MockDoorProvider()
        service = DoorService(provider=provider)
        assert service.provider is provider

    @patch("src.services.door_service.settings")
    def test_create_provider_mock(self, mock_settings):
        from src.services.door_service import DoorService, MockDoorProvider

        mock_settings.door_unlock_provider = "mock"
        service = DoorService()
        assert isinstance(service.provider, MockDoorProvider)

    @patch("src.services.door_service.settings")
    def test_create_provider_http(self, mock_settings):
        from src.services.door_service import DoorService, HttpDoorProvider

        mock_settings.door_unlock_provider = "http"
        mock_settings.door_unlock_url = "http://door/api"
        service = DoorService()
        assert isinstance(service.provider, HttpDoorProvider)
        assert service.provider.unlock_url == "http://door/api"

    @patch("src.services.door_service.settings")
    def test_create_provider_gpio(self, mock_settings):
        from src.services.door_service import DoorService, GpioDoorProvider

        mock_settings.door_unlock_provider = "gpio"
        service = DoorService()
        assert isinstance(service.provider, GpioDoorProvider)

    @patch("src.services.door_service.settings")
    def test_create_provider_unknown_falls_back_to_mock(self, mock_settings):
        from src.services.door_service import DoorService, MockDoorProvider

        mock_settings.door_unlock_provider = "some_unknown_provider"
        service = DoorService()
        assert isinstance(service.provider, MockDoorProvider)

    @pytest.mark.asyncio
    @patch("src.services.door_service.settings")
    async def test_unlock_if_authorized_above_threshold(self, mock_settings):
        from src.services.door_service import DoorService

        mock_settings.door_unlock_confidence_threshold = 0.8

        mock_provider = AsyncMock()
        mock_provider.unlock.return_value = True

        service = DoorService(provider=mock_provider)
        success, action = await service.unlock_if_authorized("alice", 0.95)

        assert success is True
        assert action == "unlocked"
        mock_provider.unlock.assert_awaited_once_with("alice", 0.95)

    @pytest.mark.asyncio
    @patch("src.services.door_service.settings")
    async def test_unlock_if_authorized_below_threshold(self, mock_settings):
        from src.services.door_service import DoorService

        mock_settings.door_unlock_confidence_threshold = 0.9

        mock_provider = AsyncMock()
        service = DoorService(provider=mock_provider)

        success, action = await service.unlock_if_authorized("alice", 0.5)

        assert success is False
        assert action == "denied"
        mock_provider.unlock.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.door_service.settings")
    async def test_unlock_if_authorized_at_exact_threshold(self, mock_settings):
        from src.services.door_service import DoorService

        mock_settings.door_unlock_confidence_threshold = 0.85

        mock_provider = AsyncMock()
        mock_provider.unlock.return_value = True

        service = DoorService(provider=mock_provider)
        success, action = await service.unlock_if_authorized("alice", 0.85)

        assert success is True
        assert action == "unlocked"

    @pytest.mark.asyncio
    @patch("src.services.door_service.settings")
    async def test_unlock_if_authorized_provider_fails(self, mock_settings):
        from src.services.door_service import DoorService

        mock_settings.door_unlock_confidence_threshold = 0.8

        mock_provider = AsyncMock()
        mock_provider.unlock.return_value = False

        service = DoorService(provider=mock_provider)
        success, action = await service.unlock_if_authorized("alice", 0.95)

        assert success is False
        assert action == "denied"


# ===========================================================================
# HybridFaceService tests
# ===========================================================================


class TestHybridFaceServiceInit:
    """Tests for HybridFaceService initialization with various modes."""

    @patch("src.services.hybrid_face_service.create_face_detector")
    @patch("src.services.hybrid_face_service.AutoCaptureService")
    @patch("src.services.hybrid_face_service.create_strategy")
    @patch("src.services.hybrid_face_service.TemplateService")
    @patch("src.services.hybrid_face_service.get_redis_client")
    @patch("src.services.hybrid_face_service.get_storage")
    @patch("src.services.hybrid_face_service.get_insightface_provider")
    @patch("src.services.hybrid_face_service.get_aws_provider")
    @patch("src.services.hybrid_face_service.FaceRepository")
    @patch("src.services.hybrid_face_service.settings")
    def test_init_insightface_only_mode(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_get_redis,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.hybrid_face_service import HybridFaceService

        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.multiface_enabled = False

        mock_db = AsyncMock()
        service = HybridFaceService(mock_db)

        mock_insightface_factory.assert_called_once()
        mock_aws_factory.assert_not_called()
        assert service.face_detector is None

    @patch("src.services.hybrid_face_service.create_face_detector")
    @patch("src.services.hybrid_face_service.AutoCaptureService")
    @patch("src.services.hybrid_face_service.create_strategy")
    @patch("src.services.hybrid_face_service.TemplateService")
    @patch("src.services.hybrid_face_service.get_redis_client")
    @patch("src.services.hybrid_face_service.get_storage")
    @patch("src.services.hybrid_face_service.get_insightface_provider")
    @patch("src.services.hybrid_face_service.get_aws_provider")
    @patch("src.services.hybrid_face_service.FaceRepository")
    @patch("src.services.hybrid_face_service.settings")
    def test_init_insightface_aws_mode(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_get_redis,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.hybrid_face_service import HybridFaceService

        mock_settings.hybrid_mode = "insightface_aws"
        mock_settings.multiface_enabled = False

        mock_db = AsyncMock()
        HybridFaceService(mock_db)

        mock_insightface_factory.assert_called_once()
        mock_aws_factory.assert_called_once()

    @patch("src.services.hybrid_face_service.create_face_detector")
    @patch("src.services.hybrid_face_service.AutoCaptureService")
    @patch("src.services.hybrid_face_service.create_strategy")
    @patch("src.services.hybrid_face_service.TemplateService")
    @patch("src.services.hybrid_face_service.get_redis_client")
    @patch("src.services.hybrid_face_service.get_storage")
    @patch("src.services.hybrid_face_service.get_insightface_provider")
    @patch("src.services.hybrid_face_service.get_aws_provider")
    @patch("src.services.hybrid_face_service.FaceRepository")
    @patch("src.services.hybrid_face_service.settings")
    def test_init_aws_only_mode(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_get_redis,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.hybrid_face_service import HybridFaceService

        mock_settings.hybrid_mode = "aws_only"
        mock_settings.multiface_enabled = False

        mock_db = AsyncMock()
        HybridFaceService(mock_db)

        mock_insightface_factory.assert_not_called()
        mock_aws_factory.assert_called_once()

    @patch("src.services.hybrid_face_service.create_face_detector")
    @patch("src.services.hybrid_face_service.AutoCaptureService")
    @patch("src.services.hybrid_face_service.create_strategy")
    @patch("src.services.hybrid_face_service.TemplateService")
    @patch("src.services.hybrid_face_service.get_redis_client")
    @patch("src.services.hybrid_face_service.get_storage")
    @patch("src.services.hybrid_face_service.get_insightface_provider")
    @patch("src.services.hybrid_face_service.get_aws_provider")
    @patch("src.services.hybrid_face_service.FaceRepository")
    @patch("src.services.hybrid_face_service.settings")
    def test_init_smart_hybrid_aws_fails_gracefully(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_get_redis,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.hybrid_face_service import HybridFaceService

        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.multiface_enabled = False
        mock_aws_factory.side_effect = Exception("no AWS credentials")

        mock_db = AsyncMock()
        service = HybridFaceService(mock_db)

        # Should not raise, aws_provider becomes None
        assert service.aws_provider is None
        mock_insightface_factory.assert_called_once()

    @patch("src.services.hybrid_face_service.create_face_detector")
    @patch("src.services.hybrid_face_service.AutoCaptureService")
    @patch("src.services.hybrid_face_service.create_strategy")
    @patch("src.services.hybrid_face_service.TemplateService")
    @patch("src.services.hybrid_face_service.get_redis_client")
    @patch("src.services.hybrid_face_service.get_storage")
    @patch("src.services.hybrid_face_service.get_insightface_provider")
    @patch("src.services.hybrid_face_service.get_aws_provider")
    @patch("src.services.hybrid_face_service.FaceRepository")
    @patch("src.services.hybrid_face_service.settings")
    def test_init_with_multiface_enabled(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_get_redis,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.hybrid_face_service import HybridFaceService

        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.multiface_enabled = True
        mock_settings.face_detection_method = "dnn"
        mock_settings.min_face_size = 80
        mock_settings.detection_confidence_threshold = 0.5

        mock_db = AsyncMock()
        service = HybridFaceService(mock_db)

        mock_create_detector.assert_called_once_with(
            method="dnn",
            min_face_size=80,
            confidence_threshold=0.5,
        )
        assert service.face_detector is not None


class TestHybridFaceServiceRecognize:
    """Tests for HybridFaceService.recognize_face."""

    def _build_service(self):
        """Build a HybridFaceService with all dependencies mocked."""
        from src.services.hybrid_face_service import HybridFaceService

        service = HybridFaceService.__new__(HybridFaceService)
        service.repository = AsyncMock()
        service.storage = AsyncMock()
        service.cache = AsyncMock()
        service.insightface_provider = AsyncMock()
        service.aws_provider = None
        service.template_service = AsyncMock()
        service.strategy = AsyncMock()
        service.auto_capture = AsyncMock()
        service.face_detector = None
        service.multiface_service = AsyncMock()
        return service

    @pytest.mark.asyncio
    @patch("src.services.hybrid_face_service.settings")
    async def test_recognize_face_with_matches(self, mock_settings):
        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.insightface_model = "buffalo_l"

        service = self._build_service()
        face = _make_face()
        service.strategy.recognize.return_value = [(face, 0.95, False)]
        service.auto_capture.capture_if_eligible.return_value = True

        results, processor = await service.recognize_face(b"image_data")

        assert len(results) == 1
        result_face, score, photo_captured, proc = results[0]
        assert result_face is face
        assert score == 0.95
        assert photo_captured is True
        assert processor == "insightface_buffalo_l"

    @pytest.mark.asyncio
    @patch("src.services.hybrid_face_service.settings")
    async def test_recognize_face_no_matches(self, mock_settings):
        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.insightface_model = "buffalo_l"

        service = self._build_service()
        service.strategy.recognize.return_value = []

        results, processor = await service.recognize_face(b"image_data")

        assert results == []
        service.auto_capture.capture_if_eligible.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.hybrid_face_service.settings")
    async def test_recognize_face_multiple_matches_only_first_has_photo(self, mock_settings):
        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.insightface_model = "buffalo_l"

        service = self._build_service()
        face1 = _make_face("alice", face_id=1)
        face2 = _make_face("bob", face_id=2)
        service.strategy.recognize.return_value = [
            (face1, 0.95, True),
            (face2, 0.80, False),
        ]
        service.auto_capture.capture_if_eligible.return_value = True

        results, processor = await service.recognize_face(b"image_data")

        assert len(results) == 2
        # Only first match gets photo_captured=True
        assert results[0][2] is True
        assert results[1][2] is False


class TestHybridFaceServiceEnroll:
    """Tests for HybridFaceService.enroll_face."""

    def _build_service(self):
        from src.services.hybrid_face_service import HybridFaceService

        service = HybridFaceService.__new__(HybridFaceService)
        service.repository = AsyncMock()
        service.storage = AsyncMock()
        service.cache = AsyncMock()
        service.insightface_provider = AsyncMock()
        service.aws_provider = None
        service.template_service = AsyncMock()
        service.strategy = AsyncMock()
        service.auto_capture = AsyncMock()
        service.face_detector = None
        service.multiface_service = AsyncMock()
        return service

    @pytest.mark.asyncio
    @patch("src.services.hybrid_face_service.settings")
    async def test_enroll_insightface_only(self, mock_settings):
        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = self._build_service()
        service.insightface_provider.extract_embedding.return_value = [0.1] * 512

        created_face = _make_face()
        service.repository.create.return_value = created_face

        result = await service.enroll_face(
            image_data=_jpeg_bytes(),
            user_name="alice",
            user_email="alice@example.com",
        )

        assert result is created_face
        service.insightface_provider.extract_embedding.assert_awaited_once()
        service.storage.save.assert_awaited_once()
        service.repository.create.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.hybrid_face_service.get_collection_manager")
    @patch("src.services.hybrid_face_service.settings")
    async def test_enroll_insightface_aws(self, mock_settings, mock_get_cm):
        mock_settings.hybrid_mode = "insightface_aws"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        mock_cm = MagicMock()
        mock_cm.get_collection_for_user.return_value = "coll_1"
        mock_get_cm.return_value = mock_cm

        service = self._build_service()
        service.insightface_provider.extract_embedding.return_value = [0.1] * 512
        service.aws_provider = AsyncMock()

        enrollment_result = MagicMock()
        enrollment_result.face_id = "aws_face_123"
        service.aws_provider.enroll_face.return_value = enrollment_result

        created_face = _make_face()
        service.repository.create.return_value = created_face

        result = await service.enroll_face(
            image_data=_jpeg_bytes(),
            user_name="bob",
        )

        assert result is created_face
        service.aws_provider.enroll_face.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.hybrid_face_service.settings")
    async def test_enroll_no_insightface_provider(self, mock_settings):
        mock_settings.hybrid_mode = "aws_only"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = self._build_service()
        service.insightface_provider = None
        service.aws_provider = None

        created_face = _make_face()
        service.repository.create.return_value = created_face

        result = await service.enroll_face(
            image_data=_jpeg_bytes(),
            user_name="carol",
        )

        # No embedding should be extracted
        assert result is created_face


class TestHybridFaceServiceCRUD:
    """Tests for HybridFaceService CRUD helpers."""

    def _build_service(self):
        from src.services.hybrid_face_service import HybridFaceService

        service = HybridFaceService.__new__(HybridFaceService)
        service.repository = AsyncMock()
        service.storage = AsyncMock()
        service.aws_provider = None
        return service

    @pytest.mark.asyncio
    async def test_get_face_by_id(self):
        service = self._build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face

        result = await service.get_face_by_id(1)
        assert result is face

    @pytest.mark.asyncio
    async def test_list_faces(self):
        service = self._build_service()
        faces = [_make_face(face_id=i) for i in range(3)]
        service.repository.list_all.return_value = (faces, 3)

        result_faces, total = await service.list_faces(limit=10, offset=0)
        assert len(result_faces) == 3
        assert total == 3

    @pytest.mark.asyncio
    async def test_delete_face_success(self):
        service = self._build_service()
        face = _make_face()
        face.provider_face_id = None
        face.provider_collection_id = None
        service.repository.get_by_id.return_value = face
        service.repository.delete.return_value = True

        result = await service.delete_face(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_face_not_found(self):
        service = self._build_service()
        service.repository.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Face not found"):
            await service.delete_face(999)

    @pytest.mark.asyncio
    async def test_delete_face_with_aws_cleanup(self):
        service = self._build_service()
        service.aws_provider = AsyncMock()

        face = _make_face(provider_face_id="aws_123", provider_collection_id="coll_1")
        service.repository.get_by_id.return_value = face
        service.repository.delete.return_value = True

        result = await service.delete_face(1)
        assert result is True
        service.aws_provider.delete_face.assert_awaited_once_with("aws_123", "coll_1")

    @pytest.mark.asyncio
    async def test_delete_face_aws_cleanup_failure_continues(self):
        service = self._build_service()
        service.aws_provider = AsyncMock()
        service.aws_provider.delete_face.side_effect = Exception("AWS error")

        face = _make_face(provider_face_id="aws_123", provider_collection_id="coll_1")
        service.repository.get_by_id.return_value = face
        service.repository.delete.return_value = True

        # Should not raise
        result = await service.delete_face(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_face_storage_failure_continues(self):
        service = self._build_service()
        face = _make_face()
        face.provider_face_id = None
        face.provider_collection_id = None
        service.repository.get_by_id.return_value = face
        service.storage.delete.side_effect = Exception("storage failure")
        service.repository.delete.return_value = True

        result = await service.delete_face(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_face_image(self):
        service = self._build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face
        service.storage.read.return_value = b"image_bytes"

        result = await service.get_face_image(1)
        assert result == b"image_bytes"

    @pytest.mark.asyncio
    async def test_get_face_image_not_found(self):
        service = self._build_service()
        service.repository.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Face not found"):
            await service.get_face_image(999)

    @pytest.mark.asyncio
    async def test_get_user_photos(self):
        service = self._build_service()
        photos = [_make_face(face_id=i) for i in range(2)]
        service.repository.get_photos_by_user_name.return_value = photos

        result = await service.get_user_photos("alice")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_recognize_multiple_faces_delegates(self):
        from src.services.hybrid_face_service import HybridFaceService

        service = HybridFaceService.__new__(HybridFaceService)
        service.multiface_service = AsyncMock()
        service.multiface_service.recognize_multiple.return_value = ([], "processor")

        results, processor = await service.recognize_multiple_faces(b"image", 5, 0.8)
        assert results == []
        assert processor == "processor"


class TestHybridHelperFunctions:
    """Tests for module-level helper functions in hybrid_face_service."""

    @patch("src.services.hybrid_face_service.settings")
    def test_base_processor_name_insightface_only(self, mock_settings):
        from src.services.hybrid_face_service import _base_processor_name

        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.insightface_model = "buffalo_l"
        assert _base_processor_name() == "insightface_buffalo_l"

    @patch("src.services.hybrid_face_service.settings")
    def test_base_processor_name_smart_hybrid(self, mock_settings):
        from src.services.hybrid_face_service import _base_processor_name

        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _base_processor_name() == "smart_hybrid_buffalo_l"

    @patch("src.services.hybrid_face_service.settings")
    def test_base_processor_name_insightface_aws(self, mock_settings):
        from src.services.hybrid_face_service import _base_processor_name

        mock_settings.hybrid_mode = "insightface_aws"
        mock_settings.insightface_model = "buffalo_l"
        assert _base_processor_name() == "hybrid_buffalo_l+aws"

    @patch("src.services.hybrid_face_service.settings")
    def test_base_processor_name_aws_only(self, mock_settings):
        from src.services.hybrid_face_service import _base_processor_name

        mock_settings.hybrid_mode = "aws_only"
        assert _base_processor_name() == "aws_rekognition"

    @patch("src.services.hybrid_face_service.settings")
    def test_match_processor_name_smart_hybrid_aws(self, mock_settings):
        from src.services.hybrid_face_service import _match_processor_name

        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _match_processor_name(True) == "buffalo_l+aws"

    @patch("src.services.hybrid_face_service.settings")
    def test_match_processor_name_smart_hybrid_no_aws(self, mock_settings):
        from src.services.hybrid_face_service import _match_processor_name

        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _match_processor_name(False) == "buffalo_l"

    @patch("src.services.hybrid_face_service.settings")
    def test_match_processor_name_insightface_aws(self, mock_settings):
        from src.services.hybrid_face_service import _match_processor_name

        mock_settings.hybrid_mode = "insightface_aws"
        mock_settings.insightface_model = "buffalo_l"
        assert _match_processor_name(False) == "buffalo_l+aws"

    @patch("src.services.hybrid_face_service.settings")
    def test_match_processor_name_insightface_only(self, mock_settings):
        from src.services.hybrid_face_service import _match_processor_name

        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.insightface_model = "buffalo_l"
        assert _match_processor_name(False) == "buffalo_l"

    @patch("src.services.hybrid_face_service.settings")
    def test_match_processor_name_aws_only(self, mock_settings):
        from src.services.hybrid_face_service import _match_processor_name

        mock_settings.hybrid_mode = "aws_only"
        assert _match_processor_name(True) == "aws_rekognition"


# ===========================================================================
# MultiFaceService tests
# ===========================================================================


class TestMultiFaceServiceRecognizeMultiple:
    """Tests for MultiFaceService.recognize_multiple."""

    def _make_bbox(self, face_id="face_0", confidence=0.99, x1=10, y1=10, x2=90, y2=90):
        from src.utils.face_processing import BoundingBox

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, face_id=face_id)

    @pytest.mark.asyncio
    async def test_raises_without_insightface_provider(self):
        from src.services.multiface_service import MultiFaceService

        service = MultiFaceService(
            insightface_provider=None,
            face_detector=MagicMock(),
            strategy=AsyncMock(),
            auto_capture=AsyncMock(),
        )

        with pytest.raises(ValueError, match="Multi-face recognition requires InsightFace"):
            await service.recognize_multiple(b"image", 5, 0.8)

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_no_faces_detected(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.hybrid_mode = "insightface_only"

        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = []

        service = MultiFaceService(
            insightface_provider=AsyncMock(),
            face_detector=mock_detector,
            strategy=AsyncMock(),
            auto_capture=AsyncMock(),
        )

        results, processor = await service.recognize_multiple(_jpeg_bytes(), 5, 0.8)
        assert results == []
        assert "detection:dnn" in processor

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_fallback_to_insightface_detection(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.hybrid_mode = "insightface_only"

        mock_provider = AsyncMock()
        mock_provider.detect_multiple_faces.return_value = []

        service = MultiFaceService(
            insightface_provider=mock_provider,
            face_detector=None,  # No detector -> fallback
            strategy=AsyncMock(),
            auto_capture=AsyncMock(),
        )

        results, processor = await service.recognize_multiple(_jpeg_bytes(), 5, 0.8)
        assert results == []
        mock_provider.detect_multiple_faces.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_limits_faces_to_max(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.max_faces_per_frame = 2
        mock_settings.face_crop_padding = 0.2
        mock_settings.insightface_model = "buffalo_l"

        bboxes = [
            self._make_bbox("face_0", x1=0, y1=0, x2=50, y2=50),
            self._make_bbox("face_1", x1=10, y1=10, x2=90, y2=90),
            self._make_bbox("face_2", x1=5, y1=5, x2=60, y2=60),
        ]

        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = bboxes

        mock_provider = AsyncMock()
        mock_provider.extract_embedding.return_value = [0.1] * 512

        mock_strategy = AsyncMock()
        mock_strategy.recognize_from_embedding.return_value = []

        service = MultiFaceService(
            insightface_provider=mock_provider,
            face_detector=mock_detector,
            strategy=mock_strategy,
            auto_capture=AsyncMock(),
        )

        results, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        # Should be limited to max_faces_per_frame=2
        assert len(results) == 2

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_single_face_with_match(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.max_faces_per_frame = 10
        mock_settings.face_crop_padding = 0.2
        mock_settings.insightface_model = "buffalo_l"

        bbox = self._make_bbox("face_0", x1=10, y1=10, x2=80, y2=80)

        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = [bbox]

        face = _make_face()
        mock_provider = AsyncMock()
        mock_provider.extract_embedding.return_value = [0.1] * 512

        mock_strategy = AsyncMock()
        mock_strategy.recognize_from_embedding.return_value = [(face, 0.92)]

        mock_auto_capture = AsyncMock()
        mock_auto_capture.capture_if_eligible.return_value = False

        service = MultiFaceService(
            insightface_provider=mock_provider,
            face_detector=mock_detector,
            strategy=mock_strategy,
            auto_capture=mock_auto_capture,
        )

        results, processor = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        assert len(results) == 1
        assert results[0]["face_id"] == "face_0"
        assert len(results[0]["matches"]) == 1

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_embedding_extraction_failure(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.max_faces_per_frame = 10
        mock_settings.face_crop_padding = 0.2
        mock_settings.insightface_model = "buffalo_l"

        bbox = self._make_bbox("face_0", x1=10, y1=10, x2=80, y2=80)

        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = [bbox]

        mock_provider = AsyncMock()
        mock_provider.extract_embedding.side_effect = ValueError("No face found")

        service = MultiFaceService(
            insightface_provider=mock_provider,
            face_detector=mock_detector,
            strategy=AsyncMock(),
            auto_capture=AsyncMock(),
        )

        results, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        assert len(results) == 1
        assert results[0]["matches"] == []

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_auto_capture_triggered_on_best_match(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.max_faces_per_frame = 10
        mock_settings.face_crop_padding = 0.2
        mock_settings.insightface_model = "buffalo_l"

        bbox = self._make_bbox("face_0", x1=10, y1=10, x2=80, y2=80)
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = [bbox]

        face = _make_face()
        mock_provider = AsyncMock()
        mock_provider.extract_embedding.return_value = [0.1] * 512

        mock_strategy = AsyncMock()
        mock_strategy.recognize_from_embedding.return_value = [(face, 0.95)]

        mock_auto_capture = AsyncMock()
        mock_auto_capture.capture_if_eligible.return_value = True

        service = MultiFaceService(
            insightface_provider=mock_provider,
            face_detector=mock_detector,
            strategy=mock_strategy,
            auto_capture=mock_auto_capture,
        )

        results, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        assert len(results) == 1
        # The first match should have photo_captured=True
        assert results[0]["matches"][0][2] is True
        mock_auto_capture.capture_if_eligible.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_match_with_three_element_tuple(self, mock_settings):
        """Strategy returns 3-tuples: (face, similarity, aws_used)."""
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.max_faces_per_frame = 10
        mock_settings.face_crop_padding = 0.2
        mock_settings.insightface_model = "buffalo_l"

        bbox = self._make_bbox("face_0", x1=10, y1=10, x2=80, y2=80)
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = [bbox]

        face = _make_face()
        mock_provider = AsyncMock()
        mock_provider.extract_embedding.return_value = [0.1] * 512

        mock_strategy = AsyncMock()
        mock_strategy.recognize_from_embedding.return_value = [(face, 0.90, True)]

        mock_auto_capture = AsyncMock()
        mock_auto_capture.capture_if_eligible.return_value = False

        service = MultiFaceService(
            insightface_provider=mock_provider,
            face_detector=mock_detector,
            strategy=mock_strategy,
            auto_capture=mock_auto_capture,
        )

        results, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        assert len(results[0]["matches"]) == 1
        # Match processor should include +aws for smart_hybrid with aws_used=True
        assert "+aws" in results[0]["matches"][0][3]


class TestComputeMatchProcessor:
    """Tests for _compute_match_processor module-level function."""

    @patch("src.services.multiface_service.settings")
    def test_smart_hybrid_aws_used(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _compute_match_processor(True) == "buffalo_l+aws"

    @patch("src.services.multiface_service.settings")
    def test_smart_hybrid_no_aws(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.hybrid_mode = "smart_hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _compute_match_processor(False) == "buffalo_l"

    @patch("src.services.multiface_service.settings")
    def test_insightface_aws_mode(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.hybrid_mode = "insightface_aws"
        mock_settings.insightface_model = "buffalo_l"
        assert _compute_match_processor(False) == "buffalo_l+aws"

    @patch("src.services.multiface_service.settings")
    def test_insightface_only_mode(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.hybrid_mode = "insightface_only"
        mock_settings.insightface_model = "buffalo_l"
        assert _compute_match_processor(False) == "buffalo_l"

    @patch("src.services.multiface_service.settings")
    def test_aws_only_mode(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.hybrid_mode = "aws_only"
        assert _compute_match_processor(False) == "aws_rekognition"


# ===========================================================================
# FaceService tests
# ===========================================================================


class TestFaceServiceInit:
    """Tests for FaceService initialization."""

    @patch("src.services.face_service.get_storage")
    @patch("src.services.face_service.get_face_provider")
    @patch("src.services.face_service.FaceRepository")
    def test_init_defaults(self, mock_repo_cls, mock_get_provider, mock_get_storage):
        from src.services.face_service import FaceService

        db = AsyncMock()
        service = FaceService(db)

        mock_repo_cls.assert_called_once_with(db)
        mock_get_provider.assert_called_once()
        mock_get_storage.assert_called_once()
        assert service._hybrid_service is None
        assert service._liveness_provider is None


class TestFaceServiceHybridDelegation:
    """Tests for FaceService delegating to HybridFaceService."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @patch("src.services.face_service.settings")
    def test_get_hybrid_service_disabled(self, mock_settings):
        mock_settings.use_hybrid_recognition = False
        service = self._build_service()
        assert service._get_hybrid_service() is None

    @patch("src.services.face_service.settings")
    def test_get_hybrid_service_enabled_caches(self, mock_settings):
        mock_settings.use_hybrid_recognition = True

        service = self._build_service()
        mock_hybrid = MagicMock()

        with patch(
            "src.services.hybrid_face_service.HybridFaceService",
            mock_hybrid,
            create=True,
        ):
            # Patch the lazy import by directly injecting into the module
            import src.services.hybrid_face_service as hfs_mod

            original = getattr(hfs_mod, "HybridFaceService", None)
            hfs_mod.HybridFaceService = MagicMock(return_value=mock_hybrid)
            try:
                result1 = service._get_hybrid_service()
                result2 = service._get_hybrid_service()
            finally:
                if original is not None:
                    hfs_mod.HybridFaceService = original

        assert result1 is mock_hybrid
        # Second call should return cached instance, not re-create
        assert result2 is mock_hybrid

    @patch("src.services.face_service.settings")
    def test_get_liveness_provider_disabled(self, mock_settings):
        mock_settings.liveness_enabled = False
        service = self._build_service()
        assert service._get_liveness_provider() is None

    @patch("src.services.face_service.settings")
    def test_get_liveness_provider_enabled_caches(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_device_id = -1
        mock_settings.liveness_model_dir = "./models"
        mock_settings.liveness_detector_path = "./detectors"

        service = self._build_service()
        mock_liveness = MagicMock()

        # Patch the lazy-imported function at its source module
        import src.providers.silent_face_liveness as liveness_mod

        original = getattr(liveness_mod, "get_liveness_provider", None)
        liveness_mod.get_liveness_provider = MagicMock(return_value=mock_liveness)
        try:
            result1 = service._get_liveness_provider()
            result2 = service._get_liveness_provider()
        finally:
            if original is not None:
                liveness_mod.get_liveness_provider = original

        assert result1 is mock_liveness
        assert result2 is mock_liveness


class TestFaceServiceEnroll:
    """Tests for FaceService.enroll_face."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.provider.provider_name = "aws_rekognition"
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_collection_manager")
    @patch("src.services.face_service.settings")
    async def test_enroll_non_hybrid(self, mock_settings, mock_get_cm):
        mock_settings.use_hybrid_recognition = False
        mock_settings.liveness_enabled = False
        mock_settings.storage_backend = "local"

        mock_cm = MagicMock()
        mock_cm.get_collection_for_user.return_value = "coll_default"
        mock_get_cm.return_value = mock_cm

        service = self._build_service()

        enrollment_result = MagicMock()
        enrollment_result.face_id = "face_abc"
        enrollment_result.confidence = 0.98
        enrollment_result.quality_score = 0.95
        enrollment_result.embedding = [0.1] * 512
        service.provider.enroll_face.return_value = enrollment_result

        created_face = _make_face()
        service.repository.create.return_value = created_face

        result = await service.enroll_face(
            image_data=_jpeg_bytes(),
            user_name="alice",
            user_email="alice@example.com",
            additional_metadata={"group": "admin"},
        )

        assert result is created_face
        service.provider.enroll_face.assert_awaited_once()
        service.storage.save.assert_awaited_once()
        service.repository.create.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_enroll_hybrid_delegates(self, mock_settings):
        mock_settings.use_hybrid_recognition = True
        mock_settings.liveness_enabled = False

        service = self._build_service()
        mock_hybrid = AsyncMock()
        mock_hybrid.enroll_face.return_value = _make_face()

        with patch.object(service, "_get_hybrid_service", return_value=mock_hybrid):
            await service.enroll_face(
                image_data=b"image",
                user_name="alice",
            )

        mock_hybrid.enroll_face.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_enroll_with_liveness_check(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_on_enrollment = True
        mock_settings.use_hybrid_recognition = True

        service = self._build_service()
        mock_hybrid = AsyncMock()
        mock_hybrid.enroll_face.return_value = _make_face()

        with (
            patch.object(service, "_get_hybrid_service", return_value=mock_hybrid),
            patch.object(service, "_check_liveness", new_callable=AsyncMock) as mock_check,
        ):
            await service.enroll_face(image_data=b"image", user_name="alice")
            mock_check.assert_awaited_once_with(b"image")


class TestFaceServiceRecognize:
    """Tests for FaceService.recognize_face."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.provider.provider_name = "aws_rekognition"
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_non_hybrid(self, mock_settings):
        mock_settings.use_hybrid_recognition = False
        mock_settings.liveness_enabled = False

        service = self._build_service()

        match = MagicMock()
        match.face_id = "face_abc"
        match.similarity = 0.92
        service.provider.recognize_face.return_value = [match]

        face = _make_face()
        service.repository.get_by_provider_face_id.return_value = face

        results = await service.recognize_face(b"image", max_results=5, confidence_threshold=0.5)

        assert len(results) == 1
        assert results[0][0] is face
        assert results[0][1] == 0.92

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_non_hybrid_no_db_match(self, mock_settings):
        """Provider match but no corresponding DB record -> result filtered."""
        mock_settings.use_hybrid_recognition = False
        mock_settings.liveness_enabled = False

        service = self._build_service()

        match = MagicMock()
        match.face_id = "face_missing"
        match.similarity = 0.9
        service.provider.recognize_face.return_value = [match]
        service.repository.get_by_provider_face_id.return_value = None

        results = await service.recognize_face(b"image")
        assert results == []

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_hybrid_delegates(self, mock_settings):
        mock_settings.use_hybrid_recognition = True
        mock_settings.liveness_enabled = False

        service = self._build_service()
        mock_hybrid = AsyncMock()
        mock_hybrid.recognize_face.return_value = "hybrid_result"

        with patch.object(service, "_get_hybrid_service", return_value=mock_hybrid):
            result = await service.recognize_face(b"image")

        assert result == "hybrid_result"

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_with_liveness_check(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_on_recognition = True
        mock_settings.use_hybrid_recognition = True

        service = self._build_service()
        mock_hybrid = AsyncMock()
        mock_hybrid.recognize_face.return_value = []

        with (
            patch.object(service, "_get_hybrid_service", return_value=mock_hybrid),
            patch.object(service, "_check_liveness", new_callable=AsyncMock) as mock_check,
        ):
            await service.recognize_face(b"image")
            mock_check.assert_awaited_once_with(b"image")


class TestFaceServiceDelete:
    """Tests for FaceService.delete_face."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    async def test_delete_success(self):
        service = self._build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face
        service.repository.delete.return_value = True

        result = await service.delete_face(1)

        assert result is True
        service.provider.delete_face.assert_awaited_once_with(face.provider_face_id)
        service.storage.delete.assert_awaited_once_with(face.image_path)
        service.repository.delete.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        service = self._build_service()
        service.repository.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Face not found"):
            await service.delete_face(999)

    @pytest.mark.asyncio
    async def test_delete_storage_failure_continues(self):
        service = self._build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face
        service.storage.delete.side_effect = Exception("storage down")
        service.repository.delete.return_value = True

        result = await service.delete_face(1)
        assert result is True


class TestFaceServiceGetImage:
    """Tests for FaceService.get_face_image."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    async def test_get_face_image_success(self):
        service = self._build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face
        service.storage.read.return_value = b"jpeg_data"

        result = await service.get_face_image(1)
        assert result == b"jpeg_data"

    @pytest.mark.asyncio
    async def test_get_face_image_not_found(self):
        service = self._build_service()
        service.repository.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Face not found"):
            await service.get_face_image(999)


class TestFaceServiceListAndGet:
    """Tests for FaceService.list_faces and get_face_by_id."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    async def test_get_face_by_id(self):
        service = self._build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face

        result = await service.get_face_by_id(1)
        assert result is face

    @pytest.mark.asyncio
    async def test_list_faces(self):
        service = self._build_service()
        faces = [_make_face(face_id=i) for i in range(3)]
        service.repository.list_all.return_value = (faces, 3)

        result_faces, total = await service.list_faces(limit=50, offset=10)
        assert len(result_faces) == 3
        assert total == 3
        service.repository.list_all.assert_awaited_once_with(50, 10)


class TestFaceServiceGetUserPhotos:
    """Tests for FaceService.get_user_photos."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_get_user_photos_non_hybrid(self, mock_settings):
        mock_settings.use_hybrid_recognition = False

        service = self._build_service()
        photos = [_make_face(face_id=1), _make_face(face_id=2)]
        service.repository.get_photos_by_user_name.return_value = photos

        result = await service.get_user_photos("alice")
        assert len(result) == 2

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_get_user_photos_hybrid_delegates(self, mock_settings):
        mock_settings.use_hybrid_recognition = True

        service = self._build_service()
        mock_hybrid = AsyncMock()
        mock_hybrid.get_user_photos.return_value = [_make_face()]

        with patch.object(service, "_get_hybrid_service", return_value=mock_hybrid):
            await service.get_user_photos("alice")

        mock_hybrid.get_user_photos.assert_awaited_once_with("alice")


class TestFaceServiceRecognizeMultiple:
    """Tests for FaceService.recognize_multiple_faces."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_multiple_requires_hybrid(self, mock_settings):
        mock_settings.use_hybrid_recognition = False
        mock_settings.liveness_enabled = False

        service = self._build_service()

        with pytest.raises(ValueError, match="Multi-face recognition requires hybrid mode"):
            await service.recognize_multiple_faces(b"image")

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_multiple_delegates_to_hybrid(self, mock_settings):
        mock_settings.use_hybrid_recognition = True
        mock_settings.liveness_enabled = False

        service = self._build_service()
        mock_hybrid = AsyncMock()
        mock_hybrid.recognize_multiple_faces.return_value = ([], "processor")

        with patch.object(service, "_get_hybrid_service", return_value=mock_hybrid):
            results, processor = await service.recognize_multiple_faces(b"image", 5, 0.8)

        assert results == []
        assert processor == "processor"

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_multiple_with_liveness(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_on_recognition = True
        mock_settings.use_hybrid_recognition = True

        service = self._build_service()
        mock_hybrid = AsyncMock()
        mock_hybrid.recognize_multiple_faces.return_value = ([], "processor")

        with (
            patch.object(service, "_get_hybrid_service", return_value=mock_hybrid),
            patch.object(service, "_check_liveness", new_callable=AsyncMock) as mock_check,
        ):
            await service.recognize_multiple_faces(b"image")
            mock_check.assert_awaited_once_with(b"image")


class TestFaceServiceCheckLiveness:
    """Tests for FaceService._check_liveness."""

    def _build_service(self):
        from src.services.face_service import FaceService

        service = FaceService.__new__(FaceService)
        service.db_session = AsyncMock()
        service.repository = AsyncMock()
        service.provider = AsyncMock()
        service.storage = AsyncMock()
        service._hybrid_service = None
        service._liveness_provider = None
        return service

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_liveness_disabled_returns_none(self, mock_settings):
        mock_settings.liveness_enabled = False

        service = self._build_service()
        # _get_liveness_provider returns None
        with patch.object(service, "_get_liveness_provider", return_value=None):
            # Should not raise
            await service._check_liveness(b"image")

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_enabled_but_provider_none_raises(self, mock_settings, mock_redis):
        mock_settings.liveness_enabled = True

        service = self._build_service()
        with patch.object(service, "_get_liveness_provider", return_value=None):
            with pytest.raises(RuntimeError, match="SECURITY CRITICAL"):
                await service._check_liveness(b"image")

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_cache_hit_pass(self, mock_settings, mock_get_redis):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_threshold = 0.5

        mock_cache = AsyncMock()
        mock_cache.get_json.return_value = {"is_real": True, "error": None}
        mock_get_redis.return_value = mock_cache

        mock_liveness = AsyncMock()

        service = self._build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            # Should not raise
            await service._check_liveness(b"image")
            mock_liveness.check_liveness.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_cache_hit_fail(self, mock_settings, mock_get_redis):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_threshold = 0.5

        mock_cache = AsyncMock()
        mock_cache.get_json.return_value = {"is_real": False, "error": "spoofing detected"}
        mock_get_redis.return_value = mock_cache

        mock_liveness = AsyncMock()

        service = self._build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            with pytest.raises(ValueError, match="spoofing detected"):
                await service._check_liveness(b"image")

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_pass(self, mock_settings, mock_get_redis):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_threshold = 0.5

        mock_cache = AsyncMock()
        mock_cache.get_json.return_value = None  # Cache miss
        mock_get_redis.return_value = mock_cache

        mock_result = MagicMock()
        mock_result.is_real = True
        mock_result.confidence = 0.95

        mock_liveness = AsyncMock()
        mock_liveness.check_liveness.return_value = mock_result

        service = self._build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            await service._check_liveness(b"image")
            mock_cache.set_json.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_fail(self, mock_settings, mock_get_redis):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_threshold = 0.5

        mock_cache = AsyncMock()
        mock_cache.get_json.return_value = None  # Cache miss
        mock_get_redis.return_value = mock_cache

        mock_spoofing_type = MagicMock()
        mock_spoofing_type.value = "print_attack"

        mock_result = MagicMock()
        mock_result.is_real = False
        mock_result.confidence = 0.9
        mock_result.spoofing_type = mock_spoofing_type

        mock_liveness = AsyncMock()
        mock_liveness.check_liveness.return_value = mock_result

        service = self._build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            with pytest.raises(ValueError, match="Liveness check failed"):
                await service._check_liveness(b"image")

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_unexpected_error(self, mock_settings, mock_get_redis):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_threshold = 0.5

        mock_cache = AsyncMock()
        mock_cache.get_json.return_value = None
        mock_get_redis.return_value = mock_cache

        mock_liveness = AsyncMock()
        mock_liveness.check_liveness.side_effect = RuntimeError("model crashed")

        service = self._build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            with pytest.raises(ValueError, match="Liveness detection failed"):
                await service._check_liveness(b"image")

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_custom_threshold(self, mock_settings, mock_get_redis):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_threshold = 0.5

        mock_cache = AsyncMock()
        mock_cache.get_json.return_value = None
        mock_get_redis.return_value = mock_cache

        mock_result = MagicMock()
        mock_result.is_real = True
        mock_result.confidence = 0.99

        mock_liveness = AsyncMock()
        mock_liveness.check_liveness.return_value = mock_result

        service = self._build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            await service._check_liveness(b"image", threshold=0.9)
            mock_liveness.check_liveness.assert_awaited_once_with(b"image", 0.9)
