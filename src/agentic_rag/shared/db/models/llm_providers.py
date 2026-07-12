from __future__ import annotations

from sqlalchemy import ForeignKey, Index, String, Text, false, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.shared.db.base import (
    Base,
    JsonDict,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    jsonb_type,
)


class LLMProvider(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    SoftDeleteMixin,
    Base,
):
    __tablename__ = "llm_providers"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_type: Mapped[str] = mapped_column(String(64), nullable=False)
    chat_model: Mapped[str | None] = mapped_column(String(256))
    embedding_model: Mapped[str | None] = mapped_column(String(256))
    embedding_dimension: Mapped[int | None] = mapped_column()
    base_url: Mapped[str | None] = mapped_column(String(2048))
    encrypted_api_key: Mapped[str | None] = mapped_column(Text)
    config: Mapped[JsonDict] = mapped_column(
        jsonb_type(),
        nullable=False,
        default=dict,
    )
    is_active: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default=true(),
    )
    is_default_chat: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=false(),
    )
    is_default_embedding: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=false(),
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)

    tenant = relationship("Tenant", back_populates="llm_providers", lazy="select")

    __table_args__ = (
        Index(
            "uq_llm_providers_tenant_active_name",
            "tenant_id",
            "name",
            unique=True,
            postgresql_where=text("is_deleted = false"),
            sqlite_where=text("is_deleted = 0"),
        ),
        Index(
            "uq_llm_providers_tenant_default_chat",
            "tenant_id",
            unique=True,
            postgresql_where=text(
                "is_default_chat = true AND is_active = true AND is_deleted = false"
            ),
            sqlite_where=text(
                "is_default_chat = 1 AND is_active = 1 AND is_deleted = 0"
            ),
        ),
        Index(
            "uq_llm_providers_tenant_default_embedding",
            "tenant_id",
            unique=True,
            postgresql_where=text(
                "is_default_embedding = true "
                "AND is_active = true AND is_deleted = false"
            ),
            sqlite_where=text(
                "is_default_embedding = 1 AND is_active = 1 AND is_deleted = 0"
            ),
        ),
        Index(
            "ix_llm_providers_tenant_active",
            "tenant_id",
            "is_active",
            "is_deleted",
        ),
        Index(
            "ix_llm_providers_tenant_type",
            "tenant_id",
            "provider_type",
        ),
    )
