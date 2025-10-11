from typing import List, Optional
import logging

import boto3
from botocore.exceptions import ClientError

from src.providers.base import (
    FaceProvider,
    FaceMatch,
    EnrollmentResult,
    FaceMetadata,
)
from src.providers.collection_manager import get_collection_manager
from src.config.settings import settings

logger = logging.getLogger(__name__)


class AWSRekognitionProvider(FaceProvider):
    """
    AWS Rekognition implementation of FaceProvider with multi-collection sharding.

    Uses CollectionManager to shard faces across multiple collections based on user_id.
    This enables:
    - Scaling beyond 20M faces (AWS limit per collection)
    - Faster searches (only search user's specific collection)
    - Better load distribution
    """

    def __init__(self, use_sharding: bool = True):
        """
        Initialize AWS Rekognition client.

        Args:
            use_sharding: If True, use multi-collection sharding (default: True)
        """
        self.client = boto3.client(
            "rekognition",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        self.use_sharding = use_sharding
        self.collection_manager = get_collection_manager() if use_sharding else None
        self.collection_id = settings.aws_rekognition_collection_id  # Legacy single collection

    def _get_collection_for_user(self, user_id: str) -> str:
        """
        Get the appropriate collection ID for a user.

        Args:
            user_id: User identifier

        Returns:
            Collection ID (sharded if enabled, otherwise default)
        """
        if self.use_sharding and self.collection_manager:
            return self.collection_manager.get_collection_for_user(user_id)
        return self.collection_id

    async def initialize_collection(self, collection_id: str) -> bool:
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            self.client.describe_collection(CollectionId=collection_id)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Collection doesn't exist, create it
                try:
                    self.client.create_collection(CollectionId=collection_id)
                    return True
                except ClientError as create_error:
                    raise Exception(
                        f"Failed to create collection: {create_error}"
                    ) from create_error
            else:
                raise Exception(f"Error checking collection: {e}") from e

    async def initialize_all_collections(self) -> dict:
        """
        Initialize all sharded collections.

        Returns:
            Dictionary with initialization results
        """
        if not self.use_sharding or not self.collection_manager:
            # Single collection mode
            await self.initialize_collection(self.collection_id)
            return {"initialized": [self.collection_id]}

        # Initialize all sharded collections
        results = {"initialized": [], "failed": []}
        for collection_id in self.collection_manager.get_all_collection_ids():
            try:
                await self.initialize_collection(collection_id)
                results["initialized"].append(collection_id)
            except Exception as e:
                results["failed"].append({"collection_id": collection_id, "error": str(e)})

        return results

    async def enroll_face(
        self, image_bytes: bytes, metadata: FaceMetadata
    ) -> EnrollmentResult:
        """
        Enroll a face using AWS Rekognition IndexFaces API.

        AWS Rekognition stores faces in collections and returns a FaceId.
        Uses sharding to distribute faces across multiple collections based on user_id.
        """
        # Get the appropriate collection for this user
        collection_id = self._get_collection_for_user(metadata.user_id)

        # Ensure collection exists
        await self.initialize_collection(collection_id)

        # Prepare external image ID and metadata
        external_image_id = metadata.user_id

        # Store metadata as JSON string (AWS Rekognition doesn't support complex metadata)
        face_metadata = {
            "user_id": metadata.user_id,
            "user_name": metadata.user_name,
        }
        if metadata.user_email:
            face_metadata["user_email"] = metadata.user_email
        if metadata.additional_data:
            face_metadata.update(metadata.additional_data)

        try:
            response = self.client.index_faces(
                CollectionId=collection_id,  # Use sharded collection
                Image={"Bytes": image_bytes},
                ExternalImageId=external_image_id,
                MaxFaces=1,  # Only index one face
                QualityFilter="AUTO",  # Filter low quality faces
                DetectionAttributes=["ALL"],
            )

            # Check if face was detected
            if not response["FaceRecords"]:
                raise ValueError("No face detected in image")

            face_record = response["FaceRecords"][0]
            face_detail = face_record["Face"]
            face_id = face_detail["FaceId"]
            confidence = face_detail["Confidence"]

            # Extract bounding box
            bbox = face_detail.get("BoundingBox", {})
            bounding_box = {
                "left": bbox.get("Left", 0),
                "top": bbox.get("Top", 0),
                "width": bbox.get("Width", 0),
                "height": bbox.get("Height", 0),
            }

            # Calculate quality score from face detail
            quality = face_record.get("FaceDetail", {}).get("Quality", {})
            quality_score = (
                quality.get("Brightness", 50) + quality.get("Sharpness", 50)
            ) / 2

            return EnrollmentResult(
                face_id=face_id,
                confidence=confidence / 100.0,  # Convert to 0-1 range
                bounding_box=bounding_box,
                quality_score=quality_score / 100.0,  # Convert to 0-1 range
                embedding=None,  # AWS Rekognition doesn't expose embeddings
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "InvalidParameterException":
                raise ValueError(f"Invalid image: {e}") from e
            elif error_code == "InvalidImageFormatException":
                raise ValueError("Invalid image format") from e
            else:
                raise Exception(f"AWS Rekognition error: {e}") from e

    async def recognize_face(
        self, image_bytes: bytes, max_results: int = 10, confidence_threshold: float = 0.8, user_id: Optional[str] = None
    ) -> List[FaceMatch]:
        """
        Recognize faces using AWS Rekognition SearchFacesByImage API.

        Args:
            image_bytes: Image data
            max_results: Maximum number of matches
            confidence_threshold: Minimum confidence (0-1)
            user_id: If provided, search only in this user's collection (fast).
                    If None, search all collections (slower but finds anyone).

        Returns:
            List of FaceMatch objects
        """
        # Determine which collection(s) to search
        if user_id:
            # Fast path: search only user's collection
            collections_to_search = [self._get_collection_for_user(user_id)]
        elif self.use_sharding and self.collection_manager:
            # Slow path: search all collections
            collections_to_search = self.collection_manager.get_all_collection_ids()
        else:
            # Single collection mode
            collections_to_search = [self.collection_id]

        all_matches = []

        # Search each collection
        for collection_id in collections_to_search:
            try:
                response = self.client.search_faces_by_image(
                    CollectionId=collection_id,
                    Image={"Bytes": image_bytes},
                    MaxFaces=max_results,
                    FaceMatchThreshold=confidence_threshold * 100,  # Convert to 0-100 range
                )

                # Check if face was detected
                if "SearchedFaceBoundingBox" not in response:
                    continue  # No face in this image, skip to next collection

                # Parse matches from this collection
                for match in response.get("FaceMatches", []):
                    face = match["Face"]
                    similarity = match["Similarity"] / 100.0  # Convert to 0-1 range
                    confidence = face["Confidence"] / 100.0

                    # Extract user_id from ExternalImageId
                    match_user_id = face.get("ExternalImageId")

                    # Extract bounding box
                    bbox = face.get("BoundingBox", {})
                    bounding_box = {
                        "left": bbox.get("Left", 0),
                        "top": bbox.get("Top", 0),
                        "width": bbox.get("Width", 0),
                        "height": bbox.get("Height", 0),
                    }

                    all_matches.append(
                        FaceMatch(
                            face_id=face["FaceId"],
                            confidence=confidence,
                            similarity=similarity,
                            user_id=match_user_id,
                            bounding_box=bounding_box,
                        )
                    )

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ResourceNotFoundException":
                    # Collection doesn't exist yet, skip it
                    continue
                elif error_code == "InvalidParameterException":
                    raise ValueError(f"Invalid image: {e}") from e
                elif error_code == "InvalidImageFormatException":
                    raise ValueError("Invalid image format") from e
                else:
                    # Log error but continue with other collections
                    logger.error(f"Error searching collection {collection_id}: {e}")
                    continue

        # Sort all matches by similarity (descending)
        all_matches.sort(key=lambda x: x.similarity, reverse=True)

        # Return top max_results matches
        return all_matches[:max_results]

    async def delete_face(self, face_id: str) -> bool:
        """Delete a face from the collection."""
        try:
            response = self.client.delete_faces(
                CollectionId=self.collection_id, FaceIds=[face_id]
            )
            deleted_faces = response.get("DeletedFaces", [])
            return face_id in deleted_faces
        except ClientError as e:
            raise Exception(f"Failed to delete face: {e}") from e

    async def get_face_details(self, face_id: str) -> Optional[dict]:
        """
        Get face details. AWS Rekognition doesn't have a direct API for this,
        so we return basic information.
        """
        try:
            # AWS Rekognition doesn't have a direct "get face" API
            # We can use list_faces with a token, but it's not efficient
            # This is a limitation of AWS Rekognition
            response = self.client.list_faces(
                CollectionId=self.collection_id, MaxResults=1000
            )

            for face in response.get("Faces", []):
                if face["FaceId"] == face_id:
                    return {
                        "face_id": face["FaceId"],
                        "external_image_id": face.get("ExternalImageId"),
                        "confidence": face.get("Confidence", 0) / 100.0,
                        "image_id": face.get("ImageId"),
                    }

            return None

        except ClientError as e:
            raise Exception(f"Failed to get face details: {e}") from e

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "aws_rekognition"

    @property
    def supports_embeddings(self) -> bool:
        """AWS Rekognition doesn't expose raw embeddings."""
        return False
