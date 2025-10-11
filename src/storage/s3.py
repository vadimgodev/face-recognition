from fs_s3fs import S3FS

from src.storage.base import StorageBackend


class S3StorageBackend(StorageBackend):
    """S3 storage using PyFilesystem2 S3FS."""

    def __init__(self, bucket_name: str, region: str, aws_access_key_id: str = None, aws_secret_access_key: str = None):
        """
        Initialize S3 storage.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            aws_access_key_id: AWS access key (optional, uses boto3 defaults)
            aws_secret_access_key: AWS secret key (optional, uses boto3 defaults)
        """
        self.bucket_name = bucket_name
        self.region = region

        # Create S3FS instance
        # If credentials are not provided, boto3 will use default credential chain
        if aws_access_key_id and aws_secret_access_key:
            self.fs = S3FS(
                bucket_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region=region,
            )
        else:
            self.fs = S3FS(bucket_name, region=region)

    async def save(self, file_path: str, file_data: bytes) -> str:
        """Save file to S3."""
        # PyFilesystem2 handles directory creation automatically
        self.fs.writebytes(file_path, file_data)

        # Return S3 URL
        return f"s3://{self.bucket_name}/{file_path}"

    async def read(self, file_path: str) -> bytes:
        """Read file from S3."""
        if not self.fs.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return self.fs.readbytes(file_path)

    async def delete(self, file_path: str) -> bool:
        """Delete file from S3."""
        try:
            if self.fs.exists(file_path):
                self.fs.remove(file_path)
                return True
            return False
        except Exception:
            return False

    async def exists(self, file_path: str) -> bool:
        """Check if file exists in S3."""
        return self.fs.exists(file_path)

    def get_url(self, file_path: str) -> str:
        """Get S3 URL."""
        return f"s3://{self.bucket_name}/{file_path}"

    def get_https_url(self, file_path: str) -> str:
        """Get HTTPS URL for S3 object."""
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_path}"
