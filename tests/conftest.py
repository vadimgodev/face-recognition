"""Shared pytest configuration and fixtures."""
import pytest
import sys
import os
import tempfile
from pathlib import Path
from importlib import reload

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
    from src.config import settings
    reload(sys.modules['src.config.settings'])

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
