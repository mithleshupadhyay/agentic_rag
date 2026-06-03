"""add agent run persistence

Revision ID: b1c9d8e7f234
Revises: e6b4d7a1c9f2
Create Date: 2026-06-03 17:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "b1c9d8e7f234"
down_revision: str | Sequence[str] | None = "e6b4d7a1c9f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=True),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="running", nullable=False),
        sa.Column("retrieval_strategy", sa.String(length=32), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("total_steps", sa.Integer(), server_default="0", nullable=False),
        sa.Column("total_tool_calls", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "limits",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("timeout_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_runs_tenant_id", "agent_runs", ["tenant_id"])
    op.create_index("ix_agent_runs_workspace_id", "agent_runs", ["workspace_id"])
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])
    op.create_index(
        "ix_agent_runs_retrieval_strategy",
        "agent_runs",
        ["retrieval_strategy"],
    )
    op.create_index("ix_agent_runs_tenant_status", "agent_runs", ["tenant_id", "status"])
    op.create_index(
        "ix_agent_runs_tenant_user_created",
        "agent_runs",
        ["tenant_id", "user_id", "created_at"],
    )
    op.create_index(
        "ix_agent_runs_tenant_workspace_created",
        "agent_runs",
        ["tenant_id", "workspace_id", "created_at"],
    )
    op.create_index(
        "ix_agent_runs_tenant_timeout",
        "agent_runs",
        ["tenant_id", "timeout_at"],
    )

    op.create_table(
        "agent_steps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("node_name", sa.String(length=64), nullable=False),
        sa.Column("step_number", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=True),
        sa.Column(
            "tool_input",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("tool_output_summary", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="completed", nullable=False),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_run_id",
            "step_number",
            name="uq_agent_steps_run_step_number",
        ),
    )
    op.create_index("ix_agent_steps_tenant_id", "agent_steps", ["tenant_id"])
    op.create_index("ix_agent_steps_agent_run_id", "agent_steps", ["agent_run_id"])
    op.create_index("ix_agent_steps_node_name", "agent_steps", ["node_name"])
    op.create_index("ix_agent_steps_tool_name", "agent_steps", ["tool_name"])
    op.create_index("ix_agent_steps_status", "agent_steps", ["status"])
    op.create_index(
        "ix_agent_steps_tenant_run",
        "agent_steps",
        ["tenant_id", "agent_run_id"],
    )
    op.create_index(
        "ix_agent_steps_tenant_node",
        "agent_steps",
        ["tenant_id", "node_name"],
    )
    op.create_index(
        "ix_agent_steps_tenant_status",
        "agent_steps",
        ["tenant_id", "status"],
    )

    op.create_table(
        "agent_checkpoints",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("checkpoint_key", sa.String(length=128), nullable=False),
        sa.Column(
            "state",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
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
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_run_id",
            "checkpoint_key",
            name="uq_agent_checkpoints_run_key",
        ),
    )
    op.create_index(
        "ix_agent_checkpoints_tenant_id",
        "agent_checkpoints",
        ["tenant_id"],
    )
    op.create_index(
        "ix_agent_checkpoints_agent_run_id",
        "agent_checkpoints",
        ["agent_run_id"],
    )
    op.create_index(
        "ix_agent_checkpoints_checkpoint_key",
        "agent_checkpoints",
        ["checkpoint_key"],
    )
    op.create_index(
        "ix_agent_checkpoints_tenant_run",
        "agent_checkpoints",
        ["tenant_id", "agent_run_id"],
    )
    op.create_index(
        "ix_agent_checkpoints_tenant_key",
        "agent_checkpoints",
        ["tenant_id", "checkpoint_key"],
    )
    op.create_index(
        "ix_agent_checkpoints_tenant_created",
        "agent_checkpoints",
        ["tenant_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_checkpoints_tenant_created", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_tenant_key", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_tenant_run", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_checkpoint_key", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_agent_run_id", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_tenant_id", table_name="agent_checkpoints")
    op.drop_table("agent_checkpoints")

    op.drop_index("ix_agent_steps_tenant_status", table_name="agent_steps")
    op.drop_index("ix_agent_steps_tenant_node", table_name="agent_steps")
    op.drop_index("ix_agent_steps_tenant_run", table_name="agent_steps")
    op.drop_index("ix_agent_steps_status", table_name="agent_steps")
    op.drop_index("ix_agent_steps_tool_name", table_name="agent_steps")
    op.drop_index("ix_agent_steps_node_name", table_name="agent_steps")
    op.drop_index("ix_agent_steps_agent_run_id", table_name="agent_steps")
    op.drop_index("ix_agent_steps_tenant_id", table_name="agent_steps")
    op.drop_table("agent_steps")

    op.drop_index("ix_agent_runs_tenant_timeout", table_name="agent_runs")
    op.drop_index("ix_agent_runs_tenant_workspace_created", table_name="agent_runs")
    op.drop_index("ix_agent_runs_tenant_user_created", table_name="agent_runs")
    op.drop_index("ix_agent_runs_tenant_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_retrieval_strategy", table_name="agent_runs")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_user_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_workspace_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_tenant_id", table_name="agent_runs")
    op.drop_table("agent_runs")
