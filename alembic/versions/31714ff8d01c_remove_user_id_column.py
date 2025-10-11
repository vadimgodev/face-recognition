"""remove_user_id_column

Revision ID: 31714ff8d01c
Revises: d6546cce3b75
Create Date: 2025-10-20 22:36:07.377093

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '31714ff8d01c'
down_revision: Union[str, None] = 'd6546cce3b75'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop indexes that reference user_id
    op.drop_index('ix_faces_new_user_id', table_name='faces')
    op.drop_index('ix_faces_new_user_id_provider', table_name='faces')

    # Drop user_id column
    op.drop_column('faces', 'user_id')


def downgrade() -> None:
    # Add user_id column back
    op.add_column('faces', sa.Column('user_id', sa.String(255), nullable=True))

    # Populate with default values (id as string)
    op.execute("UPDATE faces SET user_id = 'user_' || id::text WHERE user_id IS NULL")

    # Make it non-nullable
    op.alter_column('faces', 'user_id', nullable=False)

    # Recreate indexes
    op.create_index('ix_faces_user_id', 'faces', ['user_id'])
    op.create_index('ix_faces_user_id_provider', 'faces', ['user_id', 'provider_name'])
