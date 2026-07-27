"""Shared fixtures for provider contract tests.

`sample_image_bytes` lives here (not in test_provider_contracts.py) so that a
third-party provider's own test file under this directory picks it up
automatically via pytest's conftest discovery — it only needs to subclass the
contract classes and override the `provider` fixture. See docs/extending.md.
"""

import io

import pytest
from PIL import Image


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (640, 480), color=(73, 109, 137))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()
