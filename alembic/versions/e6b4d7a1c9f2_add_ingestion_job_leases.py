"""add ingestion job leases

Revision ID: e6b4d7a1c9f2
Revises: c4d7a2f9b8e1
Create Date: 2026-05-27 18:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e6b4d7a1c9f2"
down_revision: str | Sequence[str] | None = "c4d7a2f9b8e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("locked_by", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("ix_ingestion_jobs_locked_by", "ingestion_jobs", ["locked_by"])
    op.create_index(
        "ix_ingestion_jobs_lease_expires_at",
        "ingestion_jobs",
        ["lease_expires_at"],
    )
    op.create_index(
        "ix_ingestion_jobs_next_retry_at",
        "ingestion_jobs",
        ["next_retry_at"],
    )
    op.create_index(
        "ix_ingestion_jobs_tenant_status_lease",
        "ingestion_jobs",
        ["tenant_id", "status", "lease_expires_at"],
    )
    op.create_index(
        "ix_ingestion_jobs_tenant_status_retry",
        "ingestion_jobs",
        ["tenant_id", "status", "next_retry_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_jobs_tenant_status_retry", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_tenant_status_lease", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_next_retry_at", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_lease_expires_at", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_locked_by", table_name="ingestion_jobs")

    op.drop_column("ingestion_jobs", "next_retry_at")
    op.drop_column("ingestion_jobs", "lease_expires_at")
    op.drop_column("ingestion_jobs", "locked_at")
    op.drop_column("ingestion_jobs", "locked_by")
