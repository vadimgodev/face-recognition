"""Tests for storage backends (src/storage/local.py, src/storage/s3.py)."""

import os
from unittest.mock import patch

import pytest

from src.storage.local import LocalStorageBackend


# ============================================================================
# Path-traversal validation (CRITICAL security tests)
# ============================================================================
class TestValidatePath:
    """_validate_path must reject any path-traversal attempt."""

    def _make_storage(self, tmp_path):
        """Create a LocalStorageBackend rooted in tmp_path."""
        return LocalStorageBackend(str(tmp_path))

    def test_reject_dot_dot_prefix(self, tmp_path):
        storage = self._make_storage(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("../../etc/passwd")

    def test_reject_absolute_path(self, tmp_path):
        storage = self._make_storage(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("/etc/passwd")

    def test_reject_embedded_dot_dot(self, tmp_path):
        storage = self._make_storage(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("foo/../../bar")

    def test_accept_normal_path(self, tmp_path):
        storage = self._make_storage(tmp_path)
        result = storage._validate_path("normal/path.jpg")
        assert result == os.path.normpath("normal/path.jpg")

    def test_accept_nested_path(self, tmp_path):
        storage = self._make_storage(tmp_path)
        result = storage._validate_path("faces/user1/photo.jpg")
        assert result == os.path.normpath("faces/user1/photo.jpg")

    def test_accept_simple_filename(self, tmp_path):
        storage = self._make_storage(tmp_path)
        result = storage._validate_path("photo.jpg")
        assert result == "photo.jpg"

    def test_reject_backslash_traversal(self, tmp_path):
        """On systems where os.sep is /, double-dot with backslash in split still caught."""
        storage = self._make_storage(tmp_path)
        # The path "foo/../bar" normalises to "bar" but the raw split catches ".."
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("foo/../bar")


# ============================================================================
# S3StorageBackend path validation (same logic, separate class)
# ============================================================================
class TestS3ValidatePath:
    """S3StorageBackend._validate_path mirrors LocalStorageBackend logic."""

    def _make_storage(self):
        """Create an S3StorageBackend with mocked S3FS."""
        with patch("src.storage.s3.S3FS"):
            from src.storage.s3 import S3StorageBackend

            return S3StorageBackend(
                bucket_name="test-bucket",
                region="us-east-1",
                aws_access_key_id="fake",
                aws_secret_access_key="fake",
            )

    def test_reject_dot_dot(self):
        storage = self._make_storage()
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("../../etc/passwd")

    def test_reject_absolute(self):
        storage = self._make_storage()
        with pytest.raises(ValueError, match="path traversal"):
            storage._validate_path("/etc/passwd")

    def test_accept_normal(self):
        storage = self._make_storage()
        result = storage._validate_path("faces/user1/photo.jpg")
        assert result == os.path.normpath("faces/user1/photo.jpg")


# ============================================================================
# LocalStorage save / read / delete / exists cycle
# ============================================================================
class TestLocalStorageCrud:
    """End-to-end CRUD using a real temp directory."""

    def _make_storage(self, tmp_path):
        return LocalStorageBackend(str(tmp_path))

    @pytest.mark.asyncio
    async def test_save_and_read(self, tmp_path):
        storage = self._make_storage(tmp_path)
        content = b"hello world"

        full_path = await storage.save("test/file.txt", content)
        assert str(tmp_path) in full_path

        data = await storage.read("test/file.txt")
        assert data == content

    @pytest.mark.asyncio
    async def test_exists_true_after_save(self, tmp_path):
        storage = self._make_storage(tmp_path)
        await storage.save("a.txt", b"data")
        assert await storage.exists("a.txt") is True

    @pytest.mark.asyncio
    async def test_exists_false_before_save(self, tmp_path):
        storage = self._make_storage(tmp_path)
        assert await storage.exists("nonexistent.txt") is False

    @pytest.mark.asyncio
    async def test_delete_existing_file(self, tmp_path):
        storage = self._make_storage(tmp_path)
        await storage.save("to_delete.txt", b"bye")
        result = await storage.delete("to_delete.txt")
        assert result is True
        assert await storage.exists("to_delete.txt") is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, tmp_path):
        storage = self._make_storage(tmp_path)
        result = await storage.delete("no_such_file.txt")
        assert result is False

    @pytest.mark.asyncio
    async def test_read_nonexistent_raises(self, tmp_path):
        storage = self._make_storage(tmp_path)
        with pytest.raises(FileNotFoundError):
            await storage.read("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_get_url(self, tmp_path):
        storage = self._make_storage(tmp_path)
        url = storage.get_url("faces/photo.jpg")
        assert url == str(tmp_path / "faces" / "photo.jpg")

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path):
        """save -> exists -> read -> delete -> exists cycle."""
        storage = self._make_storage(tmp_path)
        content = b"\x89PNG fake image data"

        await storage.save("lifecycle.png", content)
        assert await storage.exists("lifecycle.png") is True
        assert await storage.read("lifecycle.png") == content

        assert await storage.delete("lifecycle.png") is True
        assert await storage.exists("lifecycle.png") is False


# ============================================================================
# Validate path is called before operations
# ============================================================================
class TestValidatePathCalledBeforeOps:
    """Each storage operation must call _validate_path first."""

    @pytest.mark.asyncio
    async def test_save_calls_validate(self, tmp_path):
        storage = LocalStorageBackend(str(tmp_path))
        with patch.object(storage, "_validate_path", wraps=storage._validate_path) as spy:
            await storage.save("ok.txt", b"data")
            spy.assert_called_once_with("ok.txt")

    @pytest.mark.asyncio
    async def test_read_calls_validate(self, tmp_path):
        storage = LocalStorageBackend(str(tmp_path))
        await storage.save("ok.txt", b"data")
        with patch.object(storage, "_validate_path", wraps=storage._validate_path) as spy:
            await storage.read("ok.txt")
            spy.assert_called_once_with("ok.txt")

    @pytest.mark.asyncio
    async def test_delete_calls_validate(self, tmp_path):
        storage = LocalStorageBackend(str(tmp_path))
        with patch.object(storage, "_validate_path", wraps=storage._validate_path) as spy:
            await storage.delete("ok.txt")
            spy.assert_called_once_with("ok.txt")

    @pytest.mark.asyncio
    async def test_exists_calls_validate(self, tmp_path):
        storage = LocalStorageBackend(str(tmp_path))
        with patch.object(storage, "_validate_path", wraps=storage._validate_path) as spy:
            await storage.exists("ok.txt")
            spy.assert_called_once_with("ok.txt")

    def test_get_url_calls_validate(self, tmp_path):
        storage = LocalStorageBackend(str(tmp_path))
        with patch.object(storage, "_validate_path", wraps=storage._validate_path) as spy:
            storage.get_url("ok.txt")
            spy.assert_called_once_with("ok.txt")
