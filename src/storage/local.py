import os
from pathlib import Path

from fs import open_fs
from fs.base import FS

from src.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage using PyFilesystem2."""

    def __init__(self, base_path: str):
        """
        Initialize local storage.

        Args:
            base_path: Base directory path for storing files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create PyFilesystem2 OSFS instance
        self.fs: FS = open_fs(str(self.base_path))

    def _validate_path(self, file_path: str) -> str:
        """Validate file path to prevent path traversal attacks.

        Args:
            file_path: The file path to validate

        Returns:
            The validated and normalized path

        Raises:
            ValueError: If path traversal is detected
        """
        if ".." in file_path.split("/") or ".." in file_path.split(os.sep):
            raise ValueError("Invalid storage path: path traversal detected")
        if file_path.startswith("/"):
            raise ValueError("Invalid storage path: path traversal detected")
        normalized = os.path.normpath(file_path)
        if ".." in normalized.split(os.sep):
            raise ValueError("Invalid storage path: path traversal detected")
        return normalized

    async def save(self, file_path: str, file_data: bytes) -> str:
        """Save file to local filesystem."""
        file_path = self._validate_path(file_path)
        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            self.fs.makedirs(parent_dir, recreate=True)

        # Write file
        self.fs.writebytes(file_path, file_data)

        # Return full path
        return str(self.base_path / file_path)

    async def read(self, file_path: str) -> bytes:
        """Read file from local filesystem."""
        file_path = self._validate_path(file_path)
        if not self.fs.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return self.fs.readbytes(file_path)

    async def delete(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        file_path = self._validate_path(file_path)
        try:
            if self.fs.exists(file_path):
                self.fs.remove(file_path)
                return True
            return False
        except Exception:
            return False

    async def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        file_path = self._validate_path(file_path)
        return self.fs.exists(file_path)

    def get_url(self, file_path: str) -> str:
        """Get local file path."""
        file_path = self._validate_path(file_path)
        return str(self.base_path / file_path)
