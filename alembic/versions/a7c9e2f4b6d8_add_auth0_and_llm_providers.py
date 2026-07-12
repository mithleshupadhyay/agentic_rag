"""add Auth0 tenant mapping and LLM providers

Revision ID: a7c9e2f4b6d8
Revises: f8a1c2d3e4b5
Create Date: 2026-07-10 10:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "a7c9e2f4b6d8"
down_revision: str | Sequence[str] | None = "f8a1c2d3e4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> sa.types.TypeEngine[object]:
    return postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("identity_provider", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "external_organization_id",
            sa.String(length=128),
            nullable=True,
        ),
    )
    op.create_unique_constraint(
        "uq_tenants_identity_organization",
        "tenants",
        ["identity_provider", "external_organization_id"],
    )
    op.create_index(
        "ix_tenants_identity_organization",
        "tenants",
        ["identity_provider", "external_organization_id"],
    )

    op.create_table(
        "llm_providers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("provider_type", sa.String(length=64), nullable=False),
        sa.Column("chat_model", sa.String(length=256), nullable=True),
        sa.Column("embedding_model", sa.String(length=256), nullable=True),
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
        sa.Column("base_url", sa.String(length=2048), nullable=True),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("config", _jsonb(), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column(
            "is_default_chat",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "is_default_embedding",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=256), nullable=False),
        sa.Column("updated_by", sa.String(length=256), nullable=False),
        sa.Column(
            "is_deleted",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
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
    op.create_index("ix_llm_providers_tenant_id", "llm_providers", ["tenant_id"])
    op.create_index("ix_llm_providers_is_deleted", "llm_providers", ["is_deleted"])
    op.create_index(
        "ix_llm_providers_tenant_active",
        "llm_providers",
        ["tenant_id", "is_active", "is_deleted"],
    )
    op.create_index(
        "ix_llm_providers_tenant_type",
        "llm_providers",
        ["tenant_id", "provider_type"],
    )
    op.create_index(
        "uq_llm_providers_tenant_active_name",
        "llm_providers",
        ["tenant_id", "name"],
        unique=True,
        postgresql_where=sa.text("is_deleted = false"),
        sqlite_where=sa.text("is_deleted = 0"),
    )
    op.create_index(
        "uq_llm_providers_tenant_default_chat",
        "llm_providers",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text(
            "is_default_chat = true AND is_active = true AND is_deleted = false"
        ),
        sqlite_where=sa.text(
            "is_default_chat = 1 AND is_active = 1 AND is_deleted = 0"
        ),
    )
    op.create_index(
        "uq_llm_providers_tenant_default_embedding",
        "llm_providers",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text(
            "is_default_embedding = true "
            "AND is_active = true AND is_deleted = false"
        ),
        sqlite_where=sa.text(
            "is_default_embedding = 1 AND is_active = 1 AND is_deleted = 0"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_llm_providers_tenant_default_embedding",
        table_name="llm_providers",
    )
    op.drop_index(
        "uq_llm_providers_tenant_default_chat",
        table_name="llm_providers",
    )
    op.drop_index(
        "uq_llm_providers_tenant_active_name",
        table_name="llm_providers",
    )
    op.drop_index("ix_llm_providers_tenant_type", table_name="llm_providers")
    op.drop_index("ix_llm_providers_tenant_active", table_name="llm_providers")
    op.drop_index("ix_llm_providers_is_deleted", table_name="llm_providers")
    op.drop_index("ix_llm_providers_tenant_id", table_name="llm_providers")
    op.drop_table("llm_providers")

    op.drop_index("ix_tenants_identity_organization", table_name="tenants")
    op.drop_constraint(
        "uq_tenants_identity_organization",
        "tenants",
        type_="unique",
    )
    op.drop_column("tenants", "external_organization_id")
    op.drop_column("tenants", "identity_provider")
