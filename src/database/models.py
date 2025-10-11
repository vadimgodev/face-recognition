from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Float, Text, Index, BigInteger
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from src.database.base import Base


class Face(Base):
    """Face model for storing face enrollment data."""

    __tablename__ = "faces"

    # Primary key - auto-increment BIGINT
    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )

    # User information
    user_name: Mapped[str] = mapped_column(String(255), nullable=False)
    user_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    user_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Face provider data
    provider_name: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # e.g., 'aws_rekognition'
    provider_face_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )  # Provider's face ID
    provider_collection_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # Collection ID (for AWS Rekognition)

    # Face embeddings (optional, for providers that expose it)
    # AWS Rekognition doesn't expose embeddings, but InsightFace does
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(512), nullable=True, comment="Generic embedding vector"
    )  # 512-dimensional vector
    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="Model used for generic embedding"
    )  # Model used for embedding

    # InsightFace specific embedding (for hybrid approach)
    embedding_insightface: Mapped[Optional[list[float]]] = mapped_column(
        Vector(512), nullable=True, comment="InsightFace embedding for fast search"
    )  # 512-dimensional InsightFace embedding

    # Image information
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    image_storage: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'local' or 's3'

    # Quality metrics
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Auto-capture fields
    photo_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="enrolled", index=True
    )  # 'enrolled' or 'verified'
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="When photo was verified during recognition"
    )
    verified_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Recognition confidence score"
    )
    verified_by_processor: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="Recognition processor used (e.g., 'antelopev2', 'aws_rekognition')"
    )  # Tracks which model/service was used for recognition

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Indexes
    __table_args__ = (
        Index(
            "ix_faces_provider_face_id_provider",
            "provider_face_id",
            "provider_name",
            unique=True,
        ),
        Index("ix_faces_user_name_photo_type", "user_name", "photo_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<Face(id={self.id}, user_name={self.user_name}, "
            f"photo_type={self.photo_type}, provider={self.provider_name})>"
        )
