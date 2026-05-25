"""add request id to query runs

Revision ID: c4d7a2f9b8e1
Revises: 9f2a6c1e8b43
Create Date: 2026-05-25 12:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c4d7a2f9b8e1"
down_revision: str | Sequence[str] | None = "9f2a6c1e8b43"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "query_runs",
        sa.Column("request_id", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_query_runs_request_id", "query_runs", ["request_id"])
    op.create_index(
        "ix_query_runs_tenant_request_id",
        "query_runs",
        ["tenant_id", "request_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_query_runs_tenant_request_id", table_name="query_runs")
    op.drop_index("ix_query_runs_request_id", table_name="query_runs")
    op.drop_column("query_runs", "request_id")
