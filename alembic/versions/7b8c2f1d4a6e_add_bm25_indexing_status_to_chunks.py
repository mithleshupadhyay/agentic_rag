"""add bm25 indexing status to chunks

Revision ID: 7b8c2f1d4a6e
Revises: 057a3e979a93
Create Date: 2026-05-24 10:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "7b8c2f1d4a6e"
down_revision: str | Sequence[str] | None = "057a3e979a93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column(
            "bm25_index_status",
            sa.String(length=32),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "document_chunks",
        sa.Column("bm25_index_name", sa.String(length=256), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("bm25_indexed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("bm25_index_content_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("bm25_index_error", sa.Text(), nullable=True),
    )

    op.create_index(
        "ix_document_chunks_bm25_index_status",
        "document_chunks",
        ["bm25_index_status"],
    )
    op.create_index(
        "ix_document_chunks_bm25_index_name",
        "document_chunks",
        ["bm25_index_name"],
    )
    op.create_index(
        "ix_document_chunks_bm25_index_content_hash",
        "document_chunks",
        ["bm25_index_content_hash"],
    )
    op.create_index(
        "ix_document_chunks_tenant_bm25_status",
        "document_chunks",
        ["tenant_id", "bm25_index_status"],
    )
    op.create_index(
        "ix_document_chunks_tenant_bm25_index",
        "document_chunks",
        ["tenant_id", "bm25_index_name"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_tenant_bm25_index", table_name="document_chunks")
    op.drop_index("ix_document_chunks_tenant_bm25_status", table_name="document_chunks")
    op.drop_index("ix_document_chunks_bm25_index_content_hash", table_name="document_chunks")
    op.drop_index("ix_document_chunks_bm25_index_name", table_name="document_chunks")
    op.drop_index("ix_document_chunks_bm25_index_status", table_name="document_chunks")

    op.drop_column("document_chunks", "bm25_index_error")
    op.drop_column("document_chunks", "bm25_index_content_hash")
    op.drop_column("document_chunks", "bm25_indexed_at")
    op.drop_column("document_chunks", "bm25_index_name")
    op.drop_column("document_chunks", "bm25_index_status")
