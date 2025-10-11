import hashlib
from typing import List, Optional
from dataclasses import dataclass

from src.config.settings import settings


@dataclass
class CollectionInfo:
    """Information about a face collection."""

    collection_id: str
    shard_index: int
    estimated_faces: int = 0
    is_active: bool = True


class CollectionManager:
    """
    Manages multiple AWS Rekognition collections for sharding.

    Uses consistent hashing to distribute users across collections.
    This ensures:
    - Each user's faces are always in the same collection
    - Fast search (only search one collection per user)
    - Even distribution across collections
    """

    def __init__(self, num_collections: int = 10, base_collection_id: str = None):
        """
        Initialize collection manager.

        Args:
            num_collections: Number of collections to shard across (default: 10)
            base_collection_id: Base name for collections (default: from settings)
        """
        self.num_collections = num_collections
        self.base_collection_id = (
            base_collection_id or settings.aws_rekognition_collection_id
        )
        self.collections = self._generate_collections()

    def _generate_collections(self) -> List[CollectionInfo]:
        """Generate collection info for all shards."""
        collections = []
        for i in range(self.num_collections):
            collection_id = f"{self.base_collection_id}-shard-{i:02d}"
            collections.append(
                CollectionInfo(
                    collection_id=collection_id,
                    shard_index=i,
                    is_active=True,
                )
            )
        return collections

    def get_collection_for_user(self, user_id: str) -> str:
        """
        Get the collection ID for a specific user using consistent hashing.

        Args:
            user_id: User identifier

        Returns:
            Collection ID where this user's faces should be stored
        """
        # Use SHA256 hash for consistent distribution
        hash_value = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        shard_index = hash_value % self.num_collections

        return self.collections[shard_index].collection_id

    def get_all_collection_ids(self) -> List[str]:
        """Get all collection IDs."""
        return [coll.collection_id for coll in self.collections if coll.is_active]

    def get_collection_by_index(self, index: int) -> Optional[str]:
        """Get collection ID by shard index."""
        if 0 <= index < len(self.collections):
            return self.collections[index].collection_id
        return None

    def get_shard_index_for_user(self, user_id: str) -> int:
        """Get the shard index for a user."""
        hash_value = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        return hash_value % self.num_collections

    def get_collection_stats(self) -> dict:
        """
        Get statistics about collections.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "total_collections": self.num_collections,
            "active_collections": sum(1 for c in self.collections if c.is_active),
            "base_collection_id": self.base_collection_id,
            "collections": [
                {
                    "collection_id": coll.collection_id,
                    "shard_index": coll.shard_index,
                    "is_active": coll.is_active,
                }
                for coll in self.collections
            ],
        }


# Global collection manager instance
def get_collection_manager() -> CollectionManager:
    """Get the global collection manager instance."""
    # Can be configured via environment variable
    num_collections = getattr(settings, "num_rekognition_collections", 10)
    return CollectionManager(num_collections=num_collections)
