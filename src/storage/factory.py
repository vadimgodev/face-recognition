from src.storage.base import StorageBackend
from src.storage.local import LocalStorageBackend
from src.storage.s3 import S3StorageBackend
from src.config.settings import settings


class StorageFactory:
    """Factory for creating storage backend instances."""

    @classmethod
    def create_storage(cls, backend_type: str = None) -> StorageBackend:
        """
        Create a storage backend instance.

        Args:
            backend_type: Type of storage ('local' or 's3')
                         Defaults to settings.storage_backend

        Returns:
            StorageBackend instance

        Raises:
            ValueError: If storage backend is not supported
        """
        if backend_type is None:
            backend_type = settings.storage_backend

        if backend_type == "local":
            return LocalStorageBackend(base_path=settings.storage_local_path)
        elif backend_type == "s3":
            if not settings.storage_s3_bucket:
                raise ValueError("S3 bucket name not configured")
            return S3StorageBackend(
                bucket_name=settings.storage_s3_bucket,
                region=settings.storage_s3_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
        else:
            raise ValueError(
                f"Unsupported storage backend: {backend_type}. "
                f"Supported: 'local', 's3'"
            )


# Convenience function for getting the default storage backend
def get_storage() -> StorageBackend:
    """Get the configured storage backend instance."""
    return StorageFactory.create_storage()
