"""Unit tests to increase coverage for face_service and multiface_service."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.exceptions import ConfigurationError, LivenessCheckFailedError

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
    embedding_local=None,
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
    face.embedding_local = embedding_local
    return face


def _jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    """Return minimal valid JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _build_service():
    """Build a FaceService with all dependencies mocked."""
    from src.services.face_service import FaceService

    service = FaceService.__new__(FaceService)
    service.db_session = AsyncMock()
    service.repository = AsyncMock()
    service.storage = AsyncMock()
    service.provider = AsyncMock()
    service.insightface_provider = AsyncMock()
    service.aws_provider = None
    service.template_service = AsyncMock()
    service.strategy = AsyncMock()
    service.auto_capture = AsyncMock()
    service.face_detector = None
    service.multiface_service = AsyncMock()
    service._liveness_provider = None
    return service


# ===========================================================================
# FaceService tests
# ===========================================================================


class TestFaceServiceInit:
    """Tests for FaceService initialization with various modes."""

    @patch("src.services.face_service.create_face_detector")
    @patch("src.services.face_service.AutoCaptureService")
    @patch("src.services.face_service.create_strategy")
    @patch("src.services.face_service.TemplateService")
    @patch("src.services.face_service.get_storage")
    @patch("src.services.face_service.get_insightface_provider")
    @patch("src.services.face_service.get_aws_provider")
    @patch("src.services.face_service.FaceRepository")
    @patch("src.services.face_service.settings")
    def test_init_local_mode(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.face_service import FaceService

        mock_settings.recognition_mode = "local"
        mock_settings.multiface_enabled = False

        mock_db = AsyncMock()
        service = FaceService(mock_db)

        mock_insightface_factory.assert_called_once()
        mock_aws_factory.assert_not_called()
        assert service.face_detector is None

    @patch("src.services.face_service.create_face_detector")
    @patch("src.services.face_service.AutoCaptureService")
    @patch("src.services.face_service.create_strategy")
    @patch("src.services.face_service.TemplateService")
    @patch("src.services.face_service.get_storage")
    @patch("src.services.face_service.get_insightface_provider")
    @patch("src.services.face_service.get_aws_provider")
    @patch("src.services.face_service.FaceRepository")
    @patch("src.services.face_service.settings")
    def test_init_hybrid_mode(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.face_service import FaceService

        mock_settings.recognition_mode = "hybrid"
        mock_settings.multiface_enabled = False

        mock_db = AsyncMock()
        FaceService(mock_db)

        mock_insightface_factory.assert_called_once()
        mock_aws_factory.assert_called_once()

    @patch("src.services.face_service.create_face_detector")
    @patch("src.services.face_service.AutoCaptureService")
    @patch("src.services.face_service.create_strategy")
    @patch("src.services.face_service.TemplateService")
    @patch("src.services.face_service.get_storage")
    @patch("src.services.face_service.get_insightface_provider")
    @patch("src.services.face_service.get_aws_provider")
    @patch("src.services.face_service.FaceRepository")
    @patch("src.services.face_service.settings")
    def test_init_cloud_mode(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.face_service import FaceService

        mock_settings.recognition_mode = "cloud"
        mock_settings.multiface_enabled = False

        mock_db = AsyncMock()
        FaceService(mock_db)

        mock_insightface_factory.assert_not_called()
        mock_aws_factory.assert_called_once()

    @patch("src.services.face_service.create_face_detector")
    @patch("src.services.face_service.AutoCaptureService")
    @patch("src.services.face_service.create_strategy")
    @patch("src.services.face_service.TemplateService")
    @patch("src.services.face_service.get_storage")
    @patch("src.services.face_service.get_insightface_provider")
    @patch("src.services.face_service.get_aws_provider")
    @patch("src.services.face_service.FaceRepository")
    @patch("src.services.face_service.settings")
    def test_init_hybrid_aws_fails_gracefully(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.face_service import FaceService

        mock_settings.recognition_mode = "hybrid"
        mock_settings.multiface_enabled = False
        mock_aws_factory.side_effect = Exception("no AWS credentials")

        mock_db = AsyncMock()
        service = FaceService(mock_db)

        # Should not raise, aws_provider becomes None
        assert service.aws_provider is None
        mock_insightface_factory.assert_called_once()

    @patch("src.services.face_service.create_face_detector")
    @patch("src.services.face_service.AutoCaptureService")
    @patch("src.services.face_service.create_strategy")
    @patch("src.services.face_service.TemplateService")
    @patch("src.services.face_service.get_storage")
    @patch("src.services.face_service.get_insightface_provider")
    @patch("src.services.face_service.get_aws_provider")
    @patch("src.services.face_service.FaceRepository")
    @patch("src.services.face_service.settings")
    def test_init_with_multiface_enabled(
        self,
        mock_settings,
        mock_repo_cls,
        mock_aws_factory,
        mock_insightface_factory,
        mock_get_storage,
        mock_template_cls,
        mock_create_strategy,
        mock_auto_capture_cls,
        mock_create_detector,
    ):
        from src.services.face_service import FaceService

        mock_settings.recognition_mode = "local"
        mock_settings.multiface_enabled = True
        mock_settings.face_detection_method = "dnn"
        mock_settings.min_face_size = 80
        mock_settings.detection_confidence_threshold = 0.5

        mock_db = AsyncMock()
        service = FaceService(mock_db)

        mock_create_detector.assert_called_once_with(
            method="dnn",
            min_face_size=80,
            confidence_threshold=0.5,
        )
        assert service.face_detector is not None


class TestFaceServiceRecognize:
    """Tests for FaceService.recognize_face."""

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_face_with_matches(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"

        service = _build_service()
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
    @patch("src.services.face_service.settings")
    async def test_recognize_face_no_matches(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"

        service = _build_service()
        service.strategy.recognize.return_value = []

        results, processor = await service.recognize_face(b"image_data")

        assert results == []
        service.auto_capture.capture_if_eligible.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_face_multiple_matches_only_first_has_photo(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "hybrid"
        mock_settings.insightface_model = "buffalo_l"

        service = _build_service()
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

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_with_liveness_check(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_on_recognition = True
        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"

        service = _build_service()
        service.strategy.recognize.return_value = []

        with patch.object(service, "_check_liveness", new_callable=AsyncMock) as mock_check:
            await service.recognize_face(b"image")
            mock_check.assert_awaited_once_with(b"image")


class TestFaceServiceEnroll:
    """Tests for FaceService.enroll_face."""

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_enroll_insightface_only(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = _build_service()
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
    @patch("src.services.face_service.settings")
    async def test_enroll_hybrid(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "hybrid"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = _build_service()
        service.insightface_provider.extract_embedding.return_value = [0.1] * 512
        service.aws_provider = AsyncMock()

        created_face = _make_face()
        service.repository.create.return_value = created_face

        result = await service.enroll_face(
            image_data=_jpeg_bytes(),
            user_name="bob",
        )

        assert result is created_face
        service.insightface_provider.extract_embedding.assert_awaited_once()
        # hybrid is collection-free: no upfront AWS indexing
        service.aws_provider.enroll_face.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_collection_manager")
    @patch("src.services.face_service.settings")
    async def test_enroll_cloud_indexes_aws(self, mock_settings, mock_get_cm):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "cloud"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        mock_cm = MagicMock()
        mock_cm.get_collection_for_user.return_value = "coll_1"
        mock_get_cm.return_value = mock_cm

        service = _build_service()
        service.insightface_provider = None
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
    @patch("src.services.face_service.settings")
    async def test_enroll_no_insightface_provider(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "cloud"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = _build_service()
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

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_enroll_with_liveness_check(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_on_enrollment = True
        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = _build_service()
        service.repository.create.return_value = _make_face()

        with patch.object(service, "_check_liveness", new_callable=AsyncMock) as mock_check:
            await service.enroll_face(image_data=b"image", user_name="alice")
            mock_check.assert_awaited_once_with(b"image")


class TestFaceServiceUniqueProviderFaceId:
    """Tests for the single Face-record creation path."""

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_enroll_generates_unique_provider_face_ids(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = _build_service()
        service.insightface_provider.extract_embedding.return_value = [0.1] * 512
        service.repository.create.side_effect = lambda face: face

        image = _jpeg_bytes()
        face1 = await service.enroll_face(image_data=image, user_name="alice")
        face2 = await service.enroll_face(image_data=image, user_name="alice")

        assert face1.provider_face_id.startswith("insightface_")
        assert face2.provider_face_id.startswith("insightface_")
        assert face1.provider_face_id != face2.provider_face_id

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_create_record_failure_rolls_back_and_cleans_up_image(self, mock_settings):
        mock_settings.liveness_enabled = False
        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"
        mock_settings.storage_backend = "local"

        service = _build_service()
        service.insightface_provider.extract_embedding.return_value = [0.1] * 512
        service.repository.create.side_effect = RuntimeError("insert failed")

        with pytest.raises(RuntimeError, match="insert failed"):
            await service.enroll_face(image_data=_jpeg_bytes(), user_name="alice")

        service.repository.session.rollback.assert_awaited_once()
        service.storage.delete.assert_awaited_once()


class TestFaceServiceDelete:
    """Tests for FaceService.delete_face."""

    @pytest.mark.asyncio
    async def test_delete_success(self):
        service = _build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face
        service.repository.delete.return_value = True

        result = await service.delete_face(1)

        assert result is True
        service.provider.delete_face.assert_awaited_once_with(
            face.provider_face_id, collection_id=face.provider_collection_id
        )
        service.storage.delete.assert_awaited_once_with(face.image_path)
        service.repository.delete.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    async def test_delete_cloud_mode_passes_sharded_collection_id(self):
        service = _build_service()
        face = _make_face(
            provider_name="aws_rekognition",
            provider_face_id="aws-face-uuid",
            provider_collection_id="faces-collection-shard-03",
        )
        service.repository.get_by_id.return_value = face
        service.repository.delete.return_value = True

        result = await service.delete_face(1)

        assert result is True
        service.provider.delete_face.assert_awaited_once_with(
            "aws-face-uuid", collection_id="faces-collection-shard-03"
        )

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        service = _build_service()
        service.repository.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Face not found"):
            await service.delete_face(999)

    @pytest.mark.asyncio
    async def test_delete_storage_failure_continues(self):
        service = _build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face
        service.storage.delete.side_effect = Exception("storage down")
        service.repository.delete.return_value = True

        result = await service.delete_face(1)
        assert result is True


class TestFaceServiceGetImage:
    """Tests for FaceService.get_face_image."""

    @pytest.mark.asyncio
    async def test_get_face_image_success(self):
        service = _build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face
        service.storage.read.return_value = b"jpeg_data"

        result = await service.get_face_image(1)
        assert result == b"jpeg_data"

    @pytest.mark.asyncio
    async def test_get_face_image_not_found(self):
        service = _build_service()
        service.repository.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Face not found"):
            await service.get_face_image(999)


class TestFaceServiceListAndGet:
    """Tests for FaceService.list_faces and get_face_by_id."""

    @pytest.mark.asyncio
    async def test_get_face_by_id(self):
        service = _build_service()
        face = _make_face()
        service.repository.get_by_id.return_value = face

        result = await service.get_face_by_id(1)
        assert result is face

    @pytest.mark.asyncio
    async def test_list_faces(self):
        service = _build_service()
        faces = [_make_face(face_id=i) for i in range(3)]
        service.repository.list_all.return_value = (faces, 3)

        result_faces, total = await service.list_faces(limit=50, offset=10)
        assert len(result_faces) == 3
        assert total == 3
        service.repository.list_all.assert_awaited_once_with(50, 10)


class TestFaceServiceGetUserPhotos:
    """Tests for FaceService.get_user_photos."""

    @pytest.mark.asyncio
    async def test_get_user_photos(self):
        service = _build_service()
        photos = [_make_face(face_id=1), _make_face(face_id=2)]
        service.repository.get_photos_by_user_name.return_value = photos

        result = await service.get_user_photos("alice")
        assert len(result) == 2


class TestFaceServiceRecognizeMultiple:
    """Tests for FaceService.recognize_multiple_faces."""

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_multiple_delegates(self, mock_settings):
        mock_settings.liveness_enabled = False

        service = _build_service()
        service.multiface_service.recognize_multiple.return_value = ([], "processor", 0.01, 0.02)

        results, processor, detection_time, recognition_time = (
            await service.recognize_multiple_faces(b"image", 5, 0.8)
        )

        assert results == []
        assert processor == "processor"
        assert detection_time == 0.01
        assert recognition_time == 0.02

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_recognize_multiple_with_liveness(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_on_recognition = True

        service = _build_service()
        service.multiface_service.recognize_multiple.return_value = ([], "processor", 0.01, 0.02)

        with patch.object(service, "_check_liveness", new_callable=AsyncMock) as mock_check:
            await service.recognize_multiple_faces(b"image")
            mock_check.assert_awaited_once_with(b"image")


class TestFaceServiceLivenessProvider:
    """Tests for FaceService._get_liveness_provider."""

    @patch("src.services.face_service.settings")
    def test_get_liveness_provider_disabled(self, mock_settings):
        mock_settings.liveness_enabled = False
        service = _build_service()
        assert service._get_liveness_provider() is None

    @patch("src.services.face_service.settings")
    def test_get_liveness_provider_enabled_caches(self, mock_settings):
        mock_settings.liveness_enabled = True
        mock_settings.liveness_device_id = -1
        mock_settings.liveness_model_dir = "./models"
        mock_settings.liveness_detector_path = "./detectors"

        service = _build_service()
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


class TestFaceServiceCheckLiveness:
    """Tests for FaceService._check_liveness."""

    @pytest.mark.asyncio
    @patch("src.services.face_service.settings")
    async def test_liveness_disabled_returns_none(self, mock_settings):
        mock_settings.liveness_enabled = False

        service = _build_service()
        # _get_liveness_provider returns None
        with patch.object(service, "_get_liveness_provider", return_value=None):
            # Should not raise
            await service._check_liveness(b"image")

    @pytest.mark.asyncio
    @patch("src.services.face_service.get_redis_client")
    @patch("src.services.face_service.settings")
    async def test_liveness_enabled_but_provider_none_raises(self, mock_settings, mock_redis):
        mock_settings.liveness_enabled = True

        service = _build_service()
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

        service = _build_service()
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

        service = _build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            with pytest.raises(LivenessCheckFailedError, match="Liveness check failed"):
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

        service = _build_service()
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

        service = _build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            with pytest.raises(LivenessCheckFailedError, match="Liveness check failed"):
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

        service = _build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            with pytest.raises(LivenessCheckFailedError, match="Liveness check failed"):
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

        service = _build_service()
        with patch.object(service, "_get_liveness_provider", return_value=mock_liveness):
            await service._check_liveness(b"image", threshold=0.9)
            mock_liveness.check_liveness.assert_awaited_once_with(b"image", 0.9)


class TestProcessorHelperFunctions:
    """Tests for module-level processor-name helpers in face_service."""

    @patch("src.services.face_service.settings")
    def test_base_processor_name_local(self, mock_settings):
        from src.services.face_service import _base_processor_name

        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"
        assert _base_processor_name() == "insightface_buffalo_l"

    @patch("src.services.face_service.settings")
    def test_base_processor_name_hybrid(self, mock_settings):
        from src.services.face_service import _base_processor_name

        mock_settings.recognition_mode = "hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _base_processor_name() == "hybrid_buffalo_l"

    @patch("src.services.face_service.settings")
    def test_base_processor_name_cloud(self, mock_settings):
        from src.services.face_service import _base_processor_name

        mock_settings.recognition_mode = "cloud"
        assert _base_processor_name() == "aws_rekognition"

    @patch("src.services.face_service.settings")
    def test_match_processor_name_hybrid_aws(self, mock_settings):
        from src.services.face_service import _match_processor_name

        mock_settings.recognition_mode = "hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _match_processor_name(True) == "buffalo_l+aws"

    @patch("src.services.face_service.settings")
    def test_match_processor_name_hybrid_no_aws(self, mock_settings):
        from src.services.face_service import _match_processor_name

        mock_settings.recognition_mode = "hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _match_processor_name(False) == "buffalo_l"

    @patch("src.services.face_service.settings")
    def test_match_processor_name_local(self, mock_settings):
        from src.services.face_service import _match_processor_name

        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"
        assert _match_processor_name(False) == "buffalo_l"

    @patch("src.services.face_service.settings")
    def test_match_processor_name_cloud(self, mock_settings):
        from src.services.face_service import _match_processor_name

        mock_settings.recognition_mode = "cloud"
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

        with pytest.raises(
            ConfigurationError, match="Multi-face recognition requires a local embedding"
        ):
            await service.recognize_multiple(b"image", 5, 0.8)

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_no_faces_detected(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.recognition_mode = "local"

        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = []

        service = MultiFaceService(
            insightface_provider=AsyncMock(),
            face_detector=mock_detector,
            strategy=AsyncMock(),
            auto_capture=AsyncMock(),
        )

        results, processor, detection_time, recognition_time = await service.recognize_multiple(
            _jpeg_bytes(), 5, 0.8
        )
        assert results == []
        assert "detection:dnn" in processor
        assert isinstance(detection_time, float)
        assert recognition_time == 0.0

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_fallback_to_insightface_detection(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.recognition_mode = "local"

        mock_provider = AsyncMock()
        mock_provider.detect_multiple_faces.return_value = []

        service = MultiFaceService(
            insightface_provider=mock_provider,
            face_detector=None,  # No detector -> fallback
            strategy=AsyncMock(),
            auto_capture=AsyncMock(),
        )

        results, processor, _, _ = await service.recognize_multiple(_jpeg_bytes(), 5, 0.8)
        assert results == []
        mock_provider.detect_multiple_faces.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_limits_faces_to_max(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.recognition_mode = "local"
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

        results, _, _, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        # Should be limited to max_faces_per_frame=2
        assert len(results) == 2

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_single_face_with_match(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.recognition_mode = "local"
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

        results, processor, detection_time, recognition_time = await service.recognize_multiple(
            _jpeg_bytes(200, 200), 5, 0.8
        )
        assert len(results) == 1
        assert results[0]["face_id"] == "face_0"
        assert len(results[0]["matches"]) == 1
        assert isinstance(detection_time, float)
        assert isinstance(recognition_time, float)

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_embedding_extraction_failure(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.recognition_mode = "local"
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

        results, _, _, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        assert len(results) == 1
        assert results[0]["matches"] == []

    @pytest.mark.asyncio
    @patch("src.services.multiface_service.settings")
    async def test_auto_capture_triggered_on_best_match(self, mock_settings):
        from src.services.multiface_service import MultiFaceService

        mock_settings.face_detection_method = "dnn"
        mock_settings.detection_confidence_threshold = 0.5
        mock_settings.recognition_mode = "local"
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

        results, _, _, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
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
        mock_settings.recognition_mode = "hybrid"
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

        results, _, _, _ = await service.recognize_multiple(_jpeg_bytes(200, 200), 5, 0.8)
        assert len(results[0]["matches"]) == 1
        # Match processor should include +aws for hybrid with aws_used=True
        assert "+aws" in results[0]["matches"][0][3]


class TestComputeMatchProcessor:
    """Tests for _compute_match_processor module-level function."""

    @patch("src.services.multiface_service.settings")
    def test_hybrid_aws_used(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.recognition_mode = "hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _compute_match_processor(True) == "buffalo_l+aws"

    @patch("src.services.multiface_service.settings")
    def test_hybrid_no_aws(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.recognition_mode = "hybrid"
        mock_settings.insightface_model = "buffalo_l"
        assert _compute_match_processor(False) == "buffalo_l"

    @patch("src.services.multiface_service.settings")
    def test_local_mode(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.recognition_mode = "local"
        mock_settings.insightface_model = "buffalo_l"
        assert _compute_match_processor(False) == "buffalo_l"

    @patch("src.services.multiface_service.settings")
    def test_cloud_mode(self, mock_settings):
        from src.services.multiface_service import _compute_match_processor

        mock_settings.recognition_mode = "cloud"
        assert _compute_match_processor(False) == "aws_rekognition"
