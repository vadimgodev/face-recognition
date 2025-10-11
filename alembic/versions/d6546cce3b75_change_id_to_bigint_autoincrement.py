"""change_id_to_bigint_autoincrement

Revision ID: d6546cce3b75
Revises: f8495daf9f9f
Create Date: 2025-10-20 22:23:53.892488

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'd6546cce3b75'
down_revision: Union[str, None] = 'f8495daf9f9f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Change ID from UUID to BIGINT with auto-increment.

    Strategy: Create new table with BIGINT ID, migrate data, drop old table, rename new table.
    """
    # Create new table with BIGINT ID
    op.create_table(
        'faces_new',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('user_name', sa.String(length=255), nullable=False),
        sa.Column('user_email', sa.String(length=255), nullable=True),
        sa.Column('user_metadata', sa.Text(), nullable=True),
        sa.Column('provider_name', sa.String(length=50), nullable=False),
        sa.Column('provider_face_id', sa.String(length=255), nullable=False),
        sa.Column('provider_collection_id', sa.String(length=255), nullable=True),
        sa.Column('embedding', Vector(dim=512), nullable=True, comment='Generic embedding vector'),
        sa.Column('embedding_model', sa.String(length=100), nullable=True, comment='Model used for generic embedding'),
        sa.Column('embedding_insightface', Vector(dim=512), nullable=True, comment='InsightFace embedding for fast search'),
        sa.Column('image_path', sa.String(length=500), nullable=False),
        sa.Column('image_storage', sa.String(length=50), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes on new table
    op.create_index('ix_faces_new_user_id', 'faces_new', ['user_id'])
    op.create_index('ix_faces_new_provider_name', 'faces_new', ['provider_name'])
    op.create_index('ix_faces_new_provider_face_id', 'faces_new', ['provider_face_id'])
    op.create_index('ix_faces_new_user_id_provider', 'faces_new', ['user_id', 'provider_name'])
    op.create_index('ix_faces_new_provider_face_id_provider', 'faces_new', ['provider_face_id', 'provider_name'], unique=True)

    # Migrate data from old table to new table (excluding id column)
    op.execute("""
        INSERT INTO faces_new (
            user_id, user_name, user_email, user_metadata,
            provider_name, provider_face_id, provider_collection_id,
            embedding, embedding_model, embedding_insightface,
            image_path, image_storage,
            quality_score, confidence_score,
            created_at, updated_at
        )
        SELECT
            user_id, user_name, user_email, user_metadata,
            provider_name, provider_face_id, provider_collection_id,
            embedding, embedding_model, embedding_insightface,
            image_path, image_storage,
            quality_score, confidence_score,
            created_at, updated_at
        FROM faces
        ORDER BY created_at
    """)

    # Drop old table
    op.drop_table('faces')

    # Rename new table to faces
    op.rename_table('faces_new', 'faces')


def downgrade() -> None:
    """
    Revert back to UUID primary key.

    Warning: This will lose auto-increment IDs and regenerate UUIDs.
    """
    # Create table with UUID ID
    op.create_table(
        'faces_new',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('user_name', sa.String(length=255), nullable=False),
        sa.Column('user_email', sa.String(length=255), nullable=True),
        sa.Column('user_metadata', sa.Text(), nullable=True),
        sa.Column('provider_name', sa.String(length=50), nullable=False),
        sa.Column('provider_face_id', sa.String(length=255), nullable=False),
        sa.Column('provider_collection_id', sa.String(length=255), nullable=True),
        sa.Column('embedding', Vector(dim=512), nullable=True, comment='Generic embedding vector'),
        sa.Column('embedding_model', sa.String(length=100), nullable=True, comment='Model used for generic embedding'),
        sa.Column('embedding_insightface', Vector(dim=512), nullable=True, comment='InsightFace embedding for fast search'),
        sa.Column('image_path', sa.String(length=500), nullable=False),
        sa.Column('image_storage', sa.String(length=50), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_faces_new_user_id', 'faces_new', ['user_id'])
    op.create_index('ix_faces_new_provider_name', 'faces_new', ['provider_name'])
    op.create_index('ix_faces_new_provider_face_id', 'faces_new', ['provider_face_id'])
    op.create_index('ix_faces_new_user_id_provider', 'faces_new', ['user_id', 'provider_name'])
    op.create_index('ix_faces_new_provider_face_id_provider', 'faces_new', ['provider_face_id', 'provider_name'], unique=True)

    # Migrate data (generate new UUIDs)
    op.execute("""
        INSERT INTO faces_new (
            id, user_id, user_name, user_email, user_metadata,
            provider_name, provider_face_id, provider_collection_id,
            embedding, embedding_model, embedding_insightface,
            image_path, image_storage,
            quality_score, confidence_score,
            created_at, updated_at
        )
        SELECT
            gen_random_uuid(), user_id, user_name, user_email, user_metadata,
            provider_name, provider_face_id, provider_collection_id,
            embedding, embedding_model, embedding_insightface,
            image_path, image_storage,
            quality_score, confidence_score,
            created_at, updated_at
        FROM faces
    """)

    # Drop old table
    op.drop_table('faces')

    # Rename new table
    op.rename_table('faces_new', 'faces')
