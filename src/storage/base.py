from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save(self, file_path: str, file_data: bytes) -> str:
        """
        Save file to storage.

        Args:
            file_path: Relative path where file should be saved
            file_data: File content as bytes

        Returns:
            Full path/URL to saved file
        """
        pass

    @abstractmethod
    async def read(self, file_path: str) -> bytes:
        """
        Read file from storage.

        Args:
            file_path: Path to file

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        pass

    @abstractmethod
    async def delete(self, file_path: str) -> bool:
        """
        Delete file from storage.

        Args:
            file_path: Path to file

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def exists(self, file_path: str) -> bool:
        """
        Check if file exists.

        Args:
            file_path: Path to file

        Returns:
            True if file exists
        """
        pass

    @abstractmethod
    def get_url(self, file_path: str) -> str:
        """
        Get URL/path for accessing the file.

        Args:
            file_path: Path to file

        Returns:
            URL or path to file
        """
        pass
