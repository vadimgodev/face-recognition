"""Shared pytest configuration and fixtures."""

import os
import sys
import tempfile
from importlib import reload
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with proper paths."""
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    test_storage_path = Path(temp_dir) / "test_images"
    test_storage_path.mkdir(parents=True, exist_ok=True)

    # Override environment variables for testing BEFORE importing settings
    os.environ["STORAGE_LOCAL_PATH"] = str(test_storage_path)
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["USE_HYBRID_RECOGNITION"] = "false"
    os.environ["FACE_PROVIDER"] = "insightface"

    # Reload settings to pick up new environment variables
    import src.config.settings as settings_module

    reload(settings_module)

    # Update the global settings object
    reload(sys.modules["src.config.settings"])

    yield

    # Cleanup is optional - temp dirs are cleaned by OS


@pytest.fixture(scope="session")
def test_api_token():
    """Return test API token."""
    return "test-api-token-for-testing"


@pytest.fixture
def sample_image_path():
    """Return path to sample image for testing."""
    return Path(__file__).parent.parent / "sample_data" / "multi.jpg"


@pytest.fixture
def sample_embedding():
    """Return a 512-dimensional normalized numpy array."""
    rng = np.random.RandomState(42)
    vec = rng.randn(512).astype(np.float64)
    vec = vec / np.linalg.norm(vec)
    return vec


@pytest.fixture
def mock_face():
    """Return a mock Face object with common attributes populated."""
    face = MagicMock()
    face.id = 1
    face.user_name = "test_user"
    face.user_email = "test@example.com"
    face.user_metadata = None
    face.provider_name = "insightface"
    face.provider_face_id = "face_001"
    face.provider_collection_id = "default"
    rng = np.random.RandomState(42)
    emb = rng.randn(512).astype(np.float64)
    face.embedding_insightface = (emb / np.linalg.norm(emb)).tolist()
    face.photo_type = "enrolled"
    face.image_path = "faces/test_user/enrolled.jpg"
    face.image_storage = "local"
    face.quality_score = 0.95
    face.confidence_score = 0.9
    return face


@pytest.fixture
def mock_repository():
    """Return an AsyncMock of FaceRepository with default return values."""
    from unittest.mock import AsyncMock

    repo = AsyncMock()
    repo.create.return_value = MagicMock()
    repo.get_by_id.return_value = None
    repo.get_by_provider_face_id.return_value = None
    repo.list_all.return_value = ([], 0)
    repo.search_by_embedding.return_value = []
    repo.get_photos_by_user_name.return_value = []
    repo.get_photos_by_user_names_batch.return_value = []
    repo.get_verified_photos_count.return_value = 0
    repo.get_oldest_verified_photo.return_value = None
    repo.delete.return_value = None
    return repo
