from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, false
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.shared.db.base import (
    Base,
    JsonDict,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    jsonb_type,
)


class QueryRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "query_runs"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str | None] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(256), index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    filters: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="running",
        server_default="running",
        index=True,
    )
    retrieval_strategy: Mapped[str | None] = mapped_column(String(32), index=True)
    answer: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    candidates: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    context: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    response_payload: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    retrieval_limit: Mapped[int] = mapped_column(
        nullable=False,
        default=20,
        server_default="20",
    )
    max_context_chunks: Mapped[int] = mapped_column(
        nullable=False,
        default=12,
        server_default="12",
    )
    max_context_tokens: Mapped[int] = mapped_column(
        nullable=False,
        default=6000,
        server_default="6000",
    )
    context_token_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    confidence_score: Mapped[float | None] = mapped_column()
    latency_ms: Mapped[int | None] = mapped_column()
    synthesis_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=false(),
    )
    llm_provider: Mapped[str | None] = mapped_column(String(128))
    llm_model: Mapped[str | None] = mapped_column(String(256))
    llm_input_tokens: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    llm_output_tokens: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    llm_cost_estimate: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
        server_default="0",
    )
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tenant = relationship("Tenant", back_populates="query_runs", lazy="select")

    __table_args__ = (
        Index("ix_query_runs_tenant_status", "tenant_id", "status"),
        Index("ix_query_runs_tenant_user_created", "tenant_id", "user_id", "created_at"),
        Index("ix_query_runs_tenant_workspace_created", "tenant_id", "workspace_id", "created_at"),
        Index("ix_query_runs_tenant_conversation", "tenant_id", "conversation_id"),
    )
