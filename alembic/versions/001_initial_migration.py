"""Initial migration with pgvector extension

Revision ID: 001
Revises:
Create Date: 2025-01-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create faces table
    op.create_table(
        'faces',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('user_name', sa.String(length=255), nullable=False),
        sa.Column('user_email', sa.String(length=255), nullable=True),
        sa.Column('user_metadata', sa.Text(), nullable=True),
        sa.Column('provider_name', sa.String(length=50), nullable=False),
        sa.Column('provider_face_id', sa.String(length=255), nullable=False),
        sa.Column('provider_collection_id', sa.String(length=255), nullable=True),
        sa.Column('embedding', Vector(512), nullable=True),
        sa.Column('embedding_model', sa.String(length=100), nullable=True),
        sa.Column('image_path', sa.String(length=500), nullable=False),
        sa.Column('image_storage', sa.String(length=50), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_faces_user_id', 'faces', ['user_id'])
    op.create_index('ix_faces_provider_name', 'faces', ['provider_name'])
    op.create_index('ix_faces_provider_face_id', 'faces', ['provider_face_id'])
    op.create_index(
        'ix_faces_user_id_provider',
        'faces',
        ['user_id', 'provider_name']
    )
    op.create_index(
        'ix_faces_provider_face_id_provider',
        'faces',
        ['provider_face_id', 'provider_name'],
        unique=True
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_faces_provider_face_id_provider', table_name='faces')
    op.drop_index('ix_faces_user_id_provider', table_name='faces')
    op.drop_index('ix_faces_provider_face_id', table_name='faces')
    op.drop_index('ix_faces_provider_name', table_name='faces')
    op.drop_index('ix_faces_user_id', table_name='faces')

    # Drop table
    op.drop_table('faces')

    # Drop pgvector extension (optional, might want to keep it)
    # op.execute('DROP EXTENSION IF EXISTS vector')
