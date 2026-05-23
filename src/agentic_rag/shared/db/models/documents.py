from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.shared.db.base import (
    Base,
    JsonDict,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    jsonb_type,
)


class Document(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "documents"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str | None] = mapped_column(String(128), index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_uri: Mapped[str | None] = mapped_column(Text)
    object_key: Mapped[str | None] = mapped_column(String(1024), index=True)
    title: Mapped[str | None] = mapped_column(String(512), index=True)
    file_name: Mapped[str | None] = mapped_column(String(512))
    mime_type: Mapped[str | None] = mapped_column(String(128))
    byte_size: Mapped[int | None] = mapped_column(BigInteger)
    content_hash: Mapped[str | None] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    owner_user_id: Mapped[str | None] = mapped_column(String(256), index=True)
    acl_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    classification_level: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="internal",
        server_default="internal",
    )
    metadata_: Mapped[JsonDict] = mapped_column(
        "metadata",
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    created_by: Mapped[str | None] = mapped_column(String(256), index=True)

    tenant = relationship("Tenant", back_populates="documents", lazy="select")
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    acl = relationship(
        "DocumentAcl",
        back_populates="document",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
    ingestion_jobs = relationship(
        "IngestionJob",
        back_populates="document",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_documents_tenant_status_deleted", "tenant_id", "status", "is_deleted"),
        Index("ix_documents_tenant_created_at", "tenant_id", "created_at"),
        Index("ix_documents_tenant_source_status", "tenant_id", "source_type", "status"),
        Index("ix_documents_tenant_content_hash", "tenant_id", "content_hash"),
        Index("ix_documents_tenant_acl_version", "tenant_id", "acl_version"),
    )


class DocumentChunk(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "document_chunks"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str | None] = mapped_column(String(128), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    token_count: Mapped[int] = mapped_column(nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(1024))
    page_number: Mapped[int | None] = mapped_column()
    start_offset: Mapped[int | None] = mapped_column()
    end_offset: Mapped[int | None] = mapped_column()
    metadata_: Mapped[JsonDict] = mapped_column(
        "metadata",
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    acl_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    classification_level: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="internal",
        server_default="internal",
    )
    bm25_index_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
    )
    bm25_index_name: Mapped[str | None] = mapped_column(String(256), index=True)
    bm25_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    bm25_index_content_hash: Mapped[str | None] = mapped_column(String(128), index=True)
    bm25_index_error: Mapped[str | None] = mapped_column(Text)

    document = relationship("Document", back_populates="chunks", lazy="select")
    acl = relationship(
        "ChunkAcl",
        back_populates="chunk",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
    embeddings = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
        Index("ix_document_chunks_tenant_document", "tenant_id", "document_id"),
        Index("ix_document_chunks_tenant_hash", "tenant_id", "content_hash"),
        Index("ix_document_chunks_tenant_acl_version", "tenant_id", "acl_version"),
        Index("ix_document_chunks_tenant_deleted", "tenant_id", "is_deleted"),
        Index("ix_document_chunks_tenant_bm25_status", "tenant_id", "bm25_index_status"),
        Index("ix_document_chunks_tenant_bm25_index", "tenant_id", "bm25_index_name"),
    )


class ChunkEmbedding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "chunk_embeddings"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str | None] = mapped_column(String(128), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(
        nullable=False,
        default=768,
        server_default="768",
    )
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    vector_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    metadata_: Mapped[JsonDict] = mapped_column(
        "metadata",
        jsonb_type(),
        nullable=False,
        default=dict,
    )

    chunk = relationship("DocumentChunk", back_populates="embeddings", lazy="select")

    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "embedding_model",
            "vector_version",
            name="uq_chunk_embeddings_chunk_model_version",
        ),
        Index("ix_chunk_embeddings_tenant_model", "tenant_id", "embedding_model"),
        Index("ix_chunk_embeddings_tenant_document", "tenant_id", "document_id"),
    )
