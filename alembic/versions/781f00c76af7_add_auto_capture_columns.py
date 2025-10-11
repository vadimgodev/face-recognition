"""add_auto_capture_columns

Revision ID: 781f00c76af7
Revises: 31714ff8d01c
Create Date: 2025-10-21 09:54:59.068832

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '781f00c76af7'
down_revision: Union[str, None] = '31714ff8d01c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add photo_type column (enrolled or verified)
    op.add_column('faces', sa.Column('photo_type', sa.String(20), nullable=True))

    # Add verified_at timestamp (when photo was verified during recognition)
    op.add_column('faces', sa.Column('verified_at', sa.DateTime(), nullable=True))

    # Add verified_confidence (recognition confidence score)
    op.add_column('faces', sa.Column('verified_confidence', sa.Float(), nullable=True))

    # Set existing records as 'enrolled' type
    op.execute("UPDATE faces SET photo_type = 'enrolled' WHERE photo_type IS NULL")

    # Make photo_type non-nullable after setting defaults
    op.alter_column('faces', 'photo_type', nullable=False)

    # Create index on photo_type for faster queries
    op.create_index('ix_faces_photo_type', 'faces', ['photo_type'])

    # Create composite index for user_name + photo_type queries
    op.create_index('ix_faces_user_name_photo_type', 'faces', ['user_name', 'photo_type'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_faces_user_name_photo_type', table_name='faces')
    op.drop_index('ix_faces_photo_type', table_name='faces')

    # Drop columns
    op.drop_column('faces', 'verified_confidence')
    op.drop_column('faces', 'verified_at')
    op.drop_column('faces', 'photo_type')
