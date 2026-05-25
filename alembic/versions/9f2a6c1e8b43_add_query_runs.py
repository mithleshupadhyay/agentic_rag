"""add query runs

Revision ID: 9f2a6c1e8b43
Revises: 7b8c2f1d4a6e
Create Date: 2026-05-24 18:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "9f2a6c1e8b43"
down_revision: str | Sequence[str] | None = "7b8c2f1d4a6e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> sa.types.TypeEngine[object]:
    return postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def _timestamps() -> list[sa.Column[object]]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "query_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=True),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("conversation_id", sa.String(length=256), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("filters", _jsonb(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="running", nullable=False),
        sa.Column("retrieval_strategy", sa.String(length=32), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("citations", _jsonb(), nullable=False),
        sa.Column("candidates", _jsonb(), nullable=False),
        sa.Column("context", _jsonb(), nullable=False),
        sa.Column("response_payload", _jsonb(), nullable=False),
        sa.Column("retrieval_limit", sa.Integer(), server_default="20", nullable=False),
        sa.Column("max_context_chunks", sa.Integer(), server_default="12", nullable=False),
        sa.Column("max_context_tokens", sa.Integer(), server_default="6000", nullable=False),
        sa.Column("context_token_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("synthesis_enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("llm_provider", sa.String(length=128), nullable=True),
        sa.Column("llm_model", sa.String(length=256), nullable=True),
        sa.Column("llm_input_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("llm_output_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("llm_cost_estimate", sa.Float(), server_default="0", nullable=False),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_runs_tenant_id", "query_runs", ["tenant_id"])
    op.create_index("ix_query_runs_workspace_id", "query_runs", ["workspace_id"])
    op.create_index("ix_query_runs_user_id", "query_runs", ["user_id"])
    op.create_index("ix_query_runs_conversation_id", "query_runs", ["conversation_id"])
    op.create_index("ix_query_runs_status", "query_runs", ["status"])
    op.create_index("ix_query_runs_retrieval_strategy", "query_runs", ["retrieval_strategy"])
    op.create_index("ix_query_runs_tenant_status", "query_runs", ["tenant_id", "status"])
    op.create_index(
        "ix_query_runs_tenant_user_created",
        "query_runs",
        ["tenant_id", "user_id", "created_at"],
    )
    op.create_index(
        "ix_query_runs_tenant_workspace_created",
        "query_runs",
        ["tenant_id", "workspace_id", "created_at"],
    )
    op.create_index(
        "ix_query_runs_tenant_conversation",
        "query_runs",
        ["tenant_id", "conversation_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_query_runs_tenant_conversation", table_name="query_runs")
    op.drop_index("ix_query_runs_tenant_workspace_created", table_name="query_runs")
    op.drop_index("ix_query_runs_tenant_user_created", table_name="query_runs")
    op.drop_index("ix_query_runs_tenant_status", table_name="query_runs")
    op.drop_index("ix_query_runs_retrieval_strategy", table_name="query_runs")
    op.drop_index("ix_query_runs_status", table_name="query_runs")
    op.drop_index("ix_query_runs_conversation_id", table_name="query_runs")
    op.drop_index("ix_query_runs_user_id", table_name="query_runs")
    op.drop_index("ix_query_runs_workspace_id", table_name="query_runs")
    op.drop_index("ix_query_runs_tenant_id", table_name="query_runs")
    op.drop_table("query_runs")
