"""rename embedding_insightface to embedding_local

Revision ID: 3d2f4f07018a
Revises: f1f65fa53cc6
Create Date: 2026-07-27 17:21:23.213179

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '3d2f4f07018a'
down_revision: str | None = 'f1f65fa53cc6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("faces", "embedding_insightface", new_column_name="embedding_local")


def downgrade() -> None:
    op.alter_column("faces", "embedding_local", new_column_name="embedding_insightface")
