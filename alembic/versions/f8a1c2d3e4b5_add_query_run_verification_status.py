"""add query run verification status

Revision ID: f8a1c2d3e4b5
Revises: b1c9d8e7f234
Create Date: 2026-06-04 14:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "f8a1c2d3e4b5"
down_revision: str | Sequence[str] | None = "b1c9d8e7f234"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "query_runs",
        sa.Column(
            "verification_status",
            sa.String(length=32),
            server_default="not_required",
            nullable=False,
        ),
    )
    op.add_column(
        "query_runs",
        sa.Column("verification_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("query_runs", "verification_reason")
    op.drop_column("query_runs", "verification_status")
