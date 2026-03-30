from typing import Optional, List

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Face


class FaceRepository:
    """Repository for Face model database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, face: Face) -> Face:
        """Create a new face record."""
        self.session.add(face)
        await self.session.commit()
        await self.session.refresh(face)
        return face

    async def get_by_id(self, face_id: int) -> Optional[Face]:
        """Get face by ID."""
        result = await self.session.execute(select(Face).where(Face.id == face_id))
        return result.scalar_one_or_none()

    async def get_by_provider_face_id(
        self, provider_face_id: str, provider_name: str
    ) -> Optional[Face]:
        """Get face by provider face ID and provider name."""
        result = await self.session.execute(
            select(Face).where(
                Face.provider_face_id == provider_face_id,
                Face.provider_name == provider_name,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[List[Face], int]:
        """
        List all faces with pagination.

        Returns:
            Tuple of (faces, total_count)
        """
        # Get total count
        count_result = await self.session.execute(select(func.count(Face.id)))
        total = count_result.scalar_one()

        # Get paginated results
        result = await self.session.execute(
            select(Face)
            .order_by(Face.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        faces = list(result.scalars().all())

        return faces, total

    async def delete(self, face_id: int) -> bool:
        """
        Delete a face record.

        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            delete(Face).where(Face.id == face_id)
        )
        await self.session.commit()
        return result.rowcount > 0

    async def search_by_embedding(
        self, embedding: List[float], threshold: float = 0.7, limit: int = 10
    ) -> List[tuple[Face, float]]:
        """
        Search for similar faces by InsightFace embedding vector using pgvector.

        Uses cosine distance for similarity:
        - Distance 0 = identical vectors
        - Distance 2 = completely opposite
        - Similarity = 1 - (distance / 2) to convert to 0-1 scale

        Args:
            embedding: InsightFace embedding vector (512 dimensions)
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results

        Returns:
            List of tuples (Face, similarity_score) sorted by similarity descending
        """
        # pgvector cosine_distance returns 0-2 scale
        # We convert to similarity score: similarity = 1 - (distance / 2)
        # This gives us 0-1 where 1 is identical

        # Build query using pgvector's cosine_distance operator
        similarity_expr = (1 - Face.embedding_insightface.cosine_distance(embedding) / 2)

        query = (
            select(
                Face,
                similarity_expr.label("similarity"),
            )
            .where(Face.embedding_insightface.isnot(None))
            .where(similarity_expr >= threshold)
            .order_by(Face.embedding_insightface.cosine_distance(embedding))
            .limit(limit)
        )

        result = await self.session.execute(query)
        return [(row[0], float(row[1])) for row in result.all()]

    async def get_photos_by_user_name(
        self, user_name: str, photo_type: Optional[str] = None
    ) -> List[Face]:
        """
        Get all photos for a user, optionally filtered by photo_type.

        Args:
            user_name: User's name
            photo_type: Optional filter ('enrolled' or 'verified')

        Returns:
            List of Face records ordered by photo_type (enrolled first) then created_at
        """
        query = select(Face).where(Face.user_name == user_name)

        if photo_type:
            query = query.where(Face.photo_type == photo_type)

        # Order: enrolled first, then verified by creation date descending
        query = query.order_by(Face.photo_type.desc(), Face.created_at.desc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_photos_by_user_names_batch(
        self, user_names: List[str], photo_type: Optional[str] = None
    ) -> List[Face]:
        """
        Get all photos for multiple users in a single query.

        This method prevents N+1 query problems when fetching photos for multiple users.

        Args:
            user_names: List of user names
            photo_type: Optional filter ('enrolled' or 'verified')

        Returns:
            List of Face records for all users, ordered by user_name, photo_type, and created_at
        """
        if not user_names:
            return []

        query = select(Face).where(Face.user_name.in_(user_names))

        if photo_type:
            query = query.where(Face.photo_type == photo_type)

        # Order: user_name, enrolled first, then verified by creation date descending
        query = query.order_by(Face.user_name, Face.photo_type.desc(), Face.created_at.desc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_enrollment_photo(self, user_name: str) -> Optional[Face]:
        """
        Get the enrollment photo for a user.

        Args:
            user_name: User's name

        Returns:
            Face record with photo_type='enrolled' or None
        """
        result = await self.session.execute(
            select(Face).where(
                Face.user_name == user_name, Face.photo_type == "enrolled"
            )
        )
        return result.scalar_one_or_none()

    async def get_verified_photos(
        self, user_name: str, limit: Optional[int] = None
    ) -> List[Face]:
        """
        Get verified photos for a user, ordered by creation date (newest first).

        Args:
            user_name: User's name
            limit: Optional maximum number of photos to return

        Returns:
            List of Face records with photo_type='verified'
        """
        query = (
            select(Face)
            .where(Face.user_name == user_name, Face.photo_type == "verified")
            .order_by(Face.created_at.desc())
        )

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_verified_photos_count(self, user_name: str) -> int:
        """
        Count verified photos for a user.

        Args:
            user_name: User's name

        Returns:
            Number of verified photos
        """
        result = await self.session.execute(
            select(func.count(Face.id)).where(
                Face.user_name == user_name, Face.photo_type == "verified"
            )
        )
        return result.scalar_one()

    async def get_oldest_verified_photo(self, user_name: str) -> Optional[Face]:
        """
        Get the oldest verified photo for a user (for FIFO deletion).

        Args:
            user_name: User's name

        Returns:
            Oldest Face record with photo_type='verified' or None
        """
        result = await self.session.execute(
            select(Face)
            .where(Face.user_name == user_name, Face.photo_type == "verified")
            .order_by(Face.created_at.asc())
            .limit(1)
        )
        return result.scalar_one_or_none()
