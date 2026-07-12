from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.shared.db.base import (
    Base,
    JsonDict,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    jsonb_type,
)


class AgentRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "agent_runs"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        index=True,
    )
    workspace_id: Mapped[str | None] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="running",
        server_default="running",
        index=True,
    )
    retrieval_strategy: Mapped[str | None] = mapped_column(String(32), index=True)
    confidence_score: Mapped[float | None] = mapped_column()
    total_steps: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    total_tool_calls: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    limits: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    timeout_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)

    tenant = relationship("Tenant", back_populates="agent_runs", lazy="select")
    steps = relationship(
        "AgentStep",
        back_populates="agent_run",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="AgentStep.step_number",
    )
    checkpoints = relationship(
        "AgentCheckpoint",
        back_populates="agent_run",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="AgentCheckpoint.created_at",
    )

    __table_args__ = (
        Index("ix_agent_runs_tenant_status", "tenant_id", "status"),
        Index("ix_agent_runs_tenant_user_created", "tenant_id", "user_id", "created_at"),
        Index(
            "ix_agent_runs_tenant_workspace_created",
            "tenant_id",
            "workspace_id",
            "created_at",
        ),
        Index("ix_agent_runs_tenant_timeout", "tenant_id", "timeout_at"),
        Index(
            "ix_agent_runs_tenant_department_created",
            "tenant_id",
            "department_id",
            "created_at",
        ),
    )


class AgentStep(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "agent_steps"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    node_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    step_number: Mapped[int] = mapped_column(nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(128), index=True)
    tool_input: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    tool_output_summary: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column()
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="completed",
        server_default="completed",
        index=True,
    )
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)

    agent_run = relationship("AgentRun", back_populates="steps", lazy="select")

    __table_args__ = (
        UniqueConstraint(
            "agent_run_id",
            "step_number",
            name="uq_agent_steps_run_step_number",
        ),
        Index("ix_agent_steps_tenant_run", "tenant_id", "agent_run_id"),
        Index("ix_agent_steps_tenant_node", "tenant_id", "node_name"),
        Index("ix_agent_steps_tenant_status", "tenant_id", "status"),
    )


class AgentCheckpoint(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "agent_checkpoints"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    checkpoint_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    state: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )

    agent_run = relationship("AgentRun", back_populates="checkpoints", lazy="select")

    __table_args__ = (
        UniqueConstraint(
            "agent_run_id",
            "checkpoint_key",
            name="uq_agent_checkpoints_run_key",
        ),
        Index("ix_agent_checkpoints_tenant_run", "tenant_id", "agent_run_id"),
        Index("ix_agent_checkpoints_tenant_key", "tenant_id", "checkpoint_key"),
        Index("ix_agent_checkpoints_tenant_created", "tenant_id", "created_at"),
    )
