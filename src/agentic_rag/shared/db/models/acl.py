from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.shared.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    jsonb_type,
)


class DocumentAcl(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document_acl"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    visibility: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="private",
        server_default="private",
    )
    allowed_user_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    allowed_group_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    allowed_roles: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    denied_user_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    denied_group_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    acl_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )

    document = relationship("Document", back_populates="acl", lazy="select")

    __table_args__ = (
        UniqueConstraint("document_id", name="uq_document_acl_document_id"),
        Index("ix_document_acl_tenant_version", "tenant_id", "acl_version"),
        Index("ix_document_acl_tenant_visibility", "tenant_id", "visibility"),
    )


class ChunkAcl(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "chunk_acl"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    visibility: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="private",
        server_default="private",
    )
    allowed_user_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    allowed_group_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    allowed_roles: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    denied_user_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    denied_group_ids: Mapped[list[str]] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=list,
    )
    acl_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )

    chunk = relationship("DocumentChunk", back_populates="acl", lazy="select")

    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunk_acl_chunk_id"),
        Index("ix_chunk_acl_tenant_version", "tenant_id", "acl_version"),
        Index("ix_chunk_acl_tenant_visibility", "tenant_id", "visibility"),
    )
