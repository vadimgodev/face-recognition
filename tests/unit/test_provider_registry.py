"""Registry resolution tests."""

import pytest

from src.exceptions import ConfigurationError
from src.providers import registry


class TestRegistry:
    def test_register_and_resolve_local(self):
        @registry.register_local("dummy-local")
        class Dummy:
            embedding_dim = 512

        try:
            assert registry.resolve_local("dummy-local") is Dummy
            assert Dummy.name == "dummy-local"
        finally:
            registry._LOCAL.pop("dummy-local")

    def test_unknown_local_lists_available(self):
        with pytest.raises(ConfigurationError, match="insightface"):
            registry.resolve_local("nope")

    def test_unknown_cloud(self):
        with pytest.raises(ConfigurationError, match="aws_rekognition"):
            registry.resolve_cloud("nope")

    def test_builtin_names_registered(self):
        assert "insightface" in registry.available_local()
        assert "aws_rekognition" in registry.available_cloud()
