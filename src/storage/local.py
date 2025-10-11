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

    async def save(self, file_path: str, file_data: bytes) -> str:
        """Save file to local filesystem."""
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
        if not self.fs.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return self.fs.readbytes(file_path)

    async def delete(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            if self.fs.exists(file_path):
                self.fs.remove(file_path)
                return True
            return False
        except Exception:
            return False

    async def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        return self.fs.exists(file_path)

    def get_url(self, file_path: str) -> str:
        """Get local file path."""
        return str(self.base_path / file_path)
