"""
InsightFace Provider for local face recognition with embedding extraction.

This provider uses InsightFace to:
1. Detect faces in images
2. Extract 512-dimensional embeddings
3. Perform local similarity matching

Much faster than AWS Rekognition for bulk operations and doesn't require API calls.
"""

import asyncio
import logging
import hashlib
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis

from src.providers.base import FaceProvider, FaceMatch, EnrollmentResult, FaceMetadata
from src.utils.face_processing import BoundingBox, convert_insightface_bbox
from src.cache.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class InsightFaceProvider(FaceProvider):
    """
    InsightFace provider for local face recognition.

    Uses pre-trained models to extract face embeddings locally.
    No API calls, no costs, runs entirely on your hardware.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        ctx_id: int = -1,  # -1 for CPU, 0+ for GPU
    ):
        """
        Initialize InsightFace provider.

        Args:
            model_name: Model to use ('buffalo_l', 'buffalo_s', 'antelopev2')
            det_size: Detection size (larger = more accurate but slower)
            ctx_id: Context ID (-1 for CPU, 0+ for GPU device ID)
        """
        self.model_name = model_name
        self.det_size = det_size
        self.ctx_id = ctx_id
        self._app: Optional[FaceAnalysis] = None

    def _get_app(self) -> FaceAnalysis:
        """Lazy load the FaceAnalysis app.

        Note: Models are baked into Docker image during build with correct structure,
        so no directory fixing or retry logic is needed.
        """
        if self._app is None:
            logger.info(f"Loading InsightFace model: {self.model_name}")

            # Load only detection and recognition modules (if model supports it)
            # Some models like antelopev2 may not support module filtering
            try:
                self._app = FaceAnalysis(
                    name=self.model_name,
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"]
                    if self.ctx_id < 0
                    else ["CUDAExecutionProvider"],
                )
            except (AssertionError, KeyError):
                # Fallback: load all modules if selective loading fails
                logger.warning(f"Could not load {self.model_name} with selective modules, loading all modules")
                self._app = FaceAnalysis(
                    name=self.model_name,
                    providers=["CPUExecutionProvider"]
                    if self.ctx_id < 0
                    else ["CUDAExecutionProvider"],
                )
            self._app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

            logger.info(f"✅ Successfully loaded InsightFace model: {self.model_name}")

        return self._app

    async def initialize_collection(self, collection_id: str) -> bool:
        """
        Initialize collection (no-op for InsightFace).

        InsightFace doesn't need collections - all embeddings stored in database.

        Args:
            collection_id: Collection identifier (ignored)

        Returns:
            True (always succeeds)
        """
        # No initialization needed - embeddings stored in database
        return True

    async def initialize_all_collections(self) -> dict:
        """
        Initialize all collections (no-op for InsightFace).

        Returns:
            Dict with initialization results (always successful)
        """
        # Ensure model is loaded
        await asyncio.get_event_loop().run_in_executor(None, self._get_app)
        return {"initialized": ["insightface-local"], "failed": []}

    async def extract_embedding(
        self, image_bytes: bytes, allow_multiple: bool = False
    ) -> List[float]:
        """
        Extract 512-dimensional embedding from face image.

        Args:
            image_bytes: Face image as bytes
            allow_multiple: If True, extract first face even if multiple detected

        Returns:
            List of 512 float values (embedding vector)

        Raises:
            ValueError: If no face detected, or multiple faces (when allow_multiple=False)
        """
        # Check cache first (embeddings are deterministic)
        cache = get_redis_client()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        cache_key = f"embedding:{image_hash}"

        cached_embedding = await cache.get_json(cache_key)
        if cached_embedding is not None:
            logger.debug(f"Embedding cache HIT for image hash {image_hash[:16]}")
            return cached_embedding

        logger.debug(f"Embedding cache MISS for image hash {image_hash[:16]}")

        def _extract():
            # Load image
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image.convert("RGB"))

            # Detect and extract faces
            app = self._get_app()
            faces = app.get(image_np)

            if len(faces) == 0:
                raise ValueError("No face detected in image")
            if len(faces) > 1 and not allow_multiple:
                raise ValueError(
                    f"Multiple faces detected ({len(faces)}). Please provide image with single face."
                )

            # Get embedding from first face
            face = faces[0]
            embedding = face.normed_embedding  # Already L2 normalized

            return embedding.tolist()

        # Run in thread pool since InsightFace is CPU/GPU intensive
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, _extract)

        # Cache the embedding (TTL: 1 hour - deterministic operation)
        await cache.set_json(cache_key, embedding, ex=3600)

        return embedding

    async def detect_multiple_faces(
        self, image_bytes: bytes
    ) -> List[Dict[str, Any]]:
        """
        Detect all faces in an image and return metadata.

        Args:
            image_bytes: Image data as bytes

        Returns:
            List of face dictionaries with bbox, embedding, confidence, etc.
        """

        def _detect():
            # Load image
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image.convert("RGB"))

            # Detect all faces
            app = self._get_app()
            faces = app.get(image_np)

            results = []
            for idx, face in enumerate(faces):
                # Convert bbox to BoundingBox object
                bbox = convert_insightface_bbox(face.bbox)

                face_data = {
                    "face_id": f"face_{idx}",
                    "bbox": bbox,
                    "embedding": face.normed_embedding.tolist(),
                    "confidence": float(face.det_score),
                    "det_score": float(face.det_score),
                }

                # Add optional attributes if available
                if hasattr(face, "age"):
                    face_data["age"] = int(face.age)
                if hasattr(face, "gender"):
                    face_data["gender"] = int(face.gender)

                results.append(face_data)

            logger.info(f"Detected {len(results)} faces in image")
            return results

        # Run in thread pool
        loop = asyncio.get_event_loop()
        faces = await loop.run_in_executor(None, _detect)

        return faces

    async def extract_multiple_embeddings(
        self, image_bytes: bytes
    ) -> List[List[float]]:
        """
        Extract embeddings from all faces in an image.

        Args:
            image_bytes: Image data as bytes

        Returns:
            List of embeddings (each is a list of 512 floats)

        Raises:
            ValueError: If no faces detected
        """
        faces = await self.detect_multiple_faces(image_bytes)

        if len(faces) == 0:
            raise ValueError("No faces detected in image")

        embeddings = [face["embedding"] for face in faces]

        logger.info(f"Extracted {len(embeddings)} embeddings from image")
        return embeddings

    async def delete_face(
        self, face_id: str, collection_id: Optional[str] = None
    ) -> bool:
        """
        Delete face (no-op for InsightFace).

        InsightFace stores embeddings in database - deletion handled by repository.

        Args:
            face_id: Face ID to delete
            collection_id: Optional collection ID (ignored)

        Returns:
            True (always succeeds)
        """
        # Deletion handled by database - no action needed here
        return True

    async def enroll_face(
        self, image_bytes: bytes, metadata: FaceMetadata
    ) -> EnrollmentResult:
        """
        Enroll a face by extracting embedding.

        Args:
            image_bytes: Image data as bytes
            metadata: User metadata

        Returns:
            EnrollmentResult with embedding

        Raises:
            ValueError: If no face or multiple faces detected
        """
        # Extract embedding
        embedding = await self.extract_embedding(image_bytes)

        # Generate a unique face ID
        import hashlib
        from datetime import datetime
        face_id = f"insightface_{metadata.user_id}_{datetime.utcnow().timestamp()}"

        return EnrollmentResult(
            face_id=face_id,
            confidence=1.0,  # InsightFace doesn't provide confidence
            quality_score=None,
            bounding_box=None,
            embedding=embedding,
        )

    async def recognize_face(
        self, image_bytes: bytes, max_results: int = 10, confidence_threshold: float = 0.8
    ) -> List[FaceMatch]:
        """
        Recognize faces in an image by extracting embedding.

        Note: This method only extracts the embedding. The actual similarity search
        is performed by the service layer using the database's pgvector extension.

        Args:
            image_bytes: Image data as bytes
            max_results: Maximum number of matches to return (not used here)
            confidence_threshold: Minimum confidence threshold (not used here)

        Returns:
            Empty list (matching is done in service layer with pgvector)

        Raises:
            ValueError: If no face detected in image
        """
        # Just verify a face exists and can be extracted
        # Actual recognition is done via vector similarity search in service layer
        await self.extract_embedding(image_bytes)

        # Return empty list - service layer handles the actual matching
        return []

    async def get_face_details(self, face_id: str) -> Optional[dict]:
        """
        Get face details (not applicable for InsightFace).

        InsightFace doesn't store face data - it's stored in database.

        Args:
            face_id: Face ID

        Returns:
            None (details stored in database)
        """
        return None

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "insightface"

    @property
    def supports_embeddings(self) -> bool:
        """Return whether provider supports embeddings."""
        return True
