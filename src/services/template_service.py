"""
Template averaging service for face recognition.

Computes average template embeddings across a user's enrolled + verified photos
and calculates similarity scores against query embeddings.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from src.database.models import Face
from src.database.repository import FaceRepository

logger = logging.getLogger(__name__)


class TemplateService:
    """Computes average template embeddings and similarity scores."""

    def __init__(self, repository: FaceRepository):
        self.repository = repository

    @staticmethod
    def compute_cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Formula: similarity = 1 - (cosine_distance / 2)
        This matches pgvector's similarity calculation.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        cosine_sim = dot_product / (norm1 * norm2)
        cosine_distance = 1 - cosine_sim

        # Convert to 0-1 similarity scale (matching pgvector)
        similarity = 1 - (cosine_distance / 2)

        return float(similarity)

    async def compute_template_results(
        self,
        query_embedding: list[float],
        candidates: list[tuple[Face, float]],
        confidence_threshold: float,
        max_results: int,
    ) -> list[tuple[Face, float]]:
        """
        For each unique user in candidates:
        1. Fetch all their enrolled faces from the DB
        2. Compute average template embedding
        3. Calculate template similarity vs query
        4. Return (representative_face, template_similarity) for users above threshold

        This replaces the duplicated template logic from the recognition methods.

        Args:
            query_embedding: The query face embedding
            candidates: List of (Face, score) tuples from vector search
            confidence_threshold: Minimum similarity to include
            max_results: Maximum number of results to return

        Returns:
            List of (representative_face, template_similarity) sorted descending
        """
        # Group candidates by user_name
        user_groups: dict[str, list[Face]] = {}
        for face, _score in candidates:
            if face.user_name not in user_groups:
                user_groups[face.user_name] = []
            user_groups[face.user_name].append(face)

        # Batch fetch all user photos in a single query (prevents N+1 problem)
        user_names = list(user_groups.keys())
        all_faces_batch = await self.repository.get_photos_by_user_names_batch(user_names)

        # Group fetched faces by user in memory
        user_faces_map: dict[str, list[Face]] = defaultdict(list)
        for face in all_faces_batch:
            user_faces_map[face.user_name].append(face)

        # Compute template similarity for each user
        template_results: list[tuple[Face, float]] = []

        for user_name in user_groups:
            all_user_faces = user_faces_map.get(user_name, [])
            embeddings = [
                f.embedding_insightface
                for f in all_user_faces
                if f.embedding_insightface is not None
            ]

            if not embeddings:
                continue

            # Compute average template embedding
            template_embedding = np.mean(embeddings, axis=0).tolist()

            # Compute similarity with template
            template_similarity = self.compute_cosine_similarity(
                query_embedding, template_embedding
            )

            # Only include if meets threshold
            if template_similarity >= confidence_threshold:
                representative_face = self.get_representative_face(
                    all_user_faces,
                    fallback=user_groups[user_name][0],
                )
                template_results.append((representative_face, template_similarity))

            if len(template_results) >= max_results:
                break

        # Sort by template similarity descending
        template_results.sort(key=lambda x: x[1], reverse=True)
        return template_results[:max_results]

    async def compute_template_results_single_user(
        self,
        query_embedding: list[float],
        user_name: str,
        fallback_face: Face,
    ) -> tuple[Face, float] | None:
        """
        Compute template similarity for a single user.

        Used by smart_hybrid when processing users one at a time.

        Args:
            query_embedding: The query face embedding
            user_name: User to compute template for
            fallback_face: Face to use if no user faces found

        Returns:
            (representative_face, template_similarity) or None
        """
        all_user_faces = await self.repository.get_photos_by_user_name(user_name)
        embeddings = [
            f.embedding_insightface for f in all_user_faces if f.embedding_insightface is not None
        ]

        if not embeddings:
            return None

        template_embedding = np.mean(embeddings, axis=0).tolist()
        template_similarity = self.compute_cosine_similarity(query_embedding, template_embedding)

        representative_face = self.get_representative_face(all_user_faces, fallback=fallback_face)
        return representative_face, template_similarity

    @staticmethod
    def get_representative_face(
        faces: list[Face],
        prefer_type: str = "enrolled",
        fallback: Face | None = None,
    ) -> Face:
        """
        Return enrolled photo if available, else first face, else fallback.

        Args:
            faces: List of Face records for a user
            prefer_type: Photo type to prefer (default "enrolled")
            fallback: Face to return if faces list is empty

        Returns:
            The most representative Face record
        """
        if not faces:
            if fallback is not None:
                return fallback
            raise ValueError("No faces and no fallback provided")

        preferred = next((f for f in faces if f.photo_type == prefer_type), None)
        return preferred if preferred is not None else faces[0]
