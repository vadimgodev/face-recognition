"""Provider role contracts. See docs/extending.md for how to add one."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.providers.base import EnrollmentResult, FaceMatch, FaceMetadata


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    embedding_dim: int

    async def extract_embedding(self, image_bytes: bytes) -> np.ndarray: ...

    async def detect_multiple_faces(self, image_bytes: bytes) -> list: ...


@runtime_checkable
class CloudMatchProvider(Protocol):
    name: str

    async def enroll_face(self, image_bytes: bytes, metadata: FaceMetadata) -> EnrollmentResult: ...

    async def recognize_face(
        self, image_bytes: bytes, max_results: int, confidence_threshold: float
    ) -> list[FaceMatch]: ...

    async def compare_faces(self, source_bytes: bytes, target_bytes: bytes) -> float | None: ...

    async def delete_face(self, face_id: str, collection_id: str | None = None) -> bool: ...
