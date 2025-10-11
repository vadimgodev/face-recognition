"""add_insightface_embedding_column

Adds InsightFace embedding column and creates vector index for fast similarity search.

Revision ID: f8495daf9f9f
Revises: 001
Create Date: 2025-10-12 21:16:04.447701

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'f8495daf9f9f'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add InsightFace embedding column and vector index."""
    # Add embedding_insightface column
    op.add_column(
        'faces',
        sa.Column(
            'embedding_insightface',
            Vector(512),
            nullable=True,
            comment='InsightFace embedding for fast vector search'
        )
    )

    # Create HNSW index for fast cosine similarity search
    # HNSW (Hierarchical Navigable Small World) is optimal for large-scale vector search
    # Parameters:
    # - m=16: number of connections per layer (higher = more accurate but slower build)
    # - ef_construction=64: size of dynamic candidate list (higher = better index quality)
    op.execute("""
        CREATE INDEX idx_faces_embedding_insightface_hnsw
        ON faces
        USING hnsw (embedding_insightface vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)


def downgrade() -> None:
    """Remove InsightFace embedding column and index."""
    # Drop index first
    op.execute('DROP INDEX IF EXISTS idx_faces_embedding_insightface_hnsw;')

    # Drop column
    op.drop_column('faces', 'embedding_insightface')
