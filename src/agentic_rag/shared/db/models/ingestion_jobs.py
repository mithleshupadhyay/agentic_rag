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


class IngestionJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str | None] = mapped_column(String(128), index=True)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    object_key: Mapped[str | None] = mapped_column(String(1024), index=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    current_stage: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="created",
        server_default="created",
    )
    retry_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    max_retries: Mapped[int] = mapped_column(
        nullable=False,
        default=3,
        server_default="3",
    )
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    locked_by: Mapped[str | None] = mapped_column(String(128), index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    metadata_: Mapped[JsonDict] = mapped_column(
        "metadata",
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    created_by: Mapped[str | None] = mapped_column(String(256), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tenant = relationship("Tenant", back_populates="ingestion_jobs", lazy="select")
    document = relationship("Document", back_populates="ingestion_jobs", lazy="select")

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_ingestion_jobs_tenant_idempotency",
        ),
        Index("ix_ingestion_jobs_tenant_status", "tenant_id", "status"),
        Index("ix_ingestion_jobs_tenant_stage", "tenant_id", "current_stage"),
        Index("ix_ingestion_jobs_tenant_created", "tenant_id", "created_at"),
        Index(
            "ix_ingestion_jobs_tenant_status_lease",
            "tenant_id",
            "status",
            "lease_expires_at",
        ),
        Index(
            "ix_ingestion_jobs_tenant_status_retry",
            "tenant_id",
            "status",
            "next_retry_at",
        ),
    )
