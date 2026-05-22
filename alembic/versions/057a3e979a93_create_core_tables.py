"""create core tables

Revision ID: 057a3e979a93
Revises:
Create Date: 2026-05-22 10:45:20.594996

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = "057a3e979a93"
down_revision: str | Sequence[str] | None = None
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
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("slug", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="active",
            nullable=False,
        ),
        sa.Column("data_region", sa.String(length=32), nullable=True),
        sa.Column("metadata", _jsonb(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
        sa.UniqueConstraint("tenant_id", name="uq_tenants_tenant_id"),
    )
    op.create_index("ix_tenants_status_region", "tenants", ["status", "data_region"])

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("external_subject", sa.String(length=256), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("display_name", sa.String(length=256), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="active",
            nullable=False,
        ),
        sa.Column("acl_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("metadata", _jsonb(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "email",
            name="uq_users_tenant_email",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "external_subject",
            name="uq_users_tenant_external_subject",
        ),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_index("ix_users_tenant_status", "users", ["tenant_id", "status"])
    op.create_index(
        "ix_users_tenant_acl_version",
        "users",
        ["tenant_id", "acl_version"],
    )

    op.create_table(
        "roles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=True),
        sa.Column("is_system", sa.Boolean(), server_default=sa.false(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_roles_tenant_name"),
    )
    op.create_index("ix_roles_tenant_id", "roles", ["tenant_id"])

    op.create_table(
        "groups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_groups_tenant_name"),
    )
    op.create_index("ix_groups_tenant_id", "groups", ["tenant_id"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=True),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("object_key", sa.String(length=1024), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("file_name", sa.String(length=512), nullable=True),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("byte_size", sa.BigInteger(), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="queued",
            nullable=False,
        ),
        sa.Column("owner_user_id", sa.String(length=256), nullable=True),
        sa.Column("acl_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "classification_level",
            sa.String(length=32),
            server_default="internal",
            nullable=False,
        ),
        sa.Column("metadata", _jsonb(), nullable=False),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        *_timestamps(),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index("ix_documents_workspace_id", "documents", ["workspace_id"])
    op.create_index("ix_documents_source_type", "documents", ["source_type"])
    op.create_index("ix_documents_object_key", "documents", ["object_key"])
    op.create_index("ix_documents_title", "documents", ["title"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_owner_user_id", "documents", ["owner_user_id"])
    op.create_index("ix_documents_created_by", "documents", ["created_by"])
    op.create_index("ix_documents_is_deleted", "documents", ["is_deleted"])
    op.create_index(
        "ix_documents_tenant_status_deleted",
        "documents",
        ["tenant_id", "status", "is_deleted"],
    )
    op.create_index(
        "ix_documents_tenant_created_at",
        "documents",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_documents_tenant_source_status",
        "documents",
        ["tenant_id", "source_type", "status"],
    )
    op.create_index(
        "ix_documents_tenant_content_hash",
        "documents",
        ["tenant_id", "content_hash"],
    )
    op.create_index(
        "ix_documents_tenant_acl_version",
        "documents",
        ["tenant_id", "acl_version"],
    )

    op.create_table(
        "user_roles",
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role_id", sa.Uuid(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id", "user_id", "role_id"),
    )
    op.create_index(
        "ix_user_roles_tenant_user",
        "user_roles",
        ["tenant_id", "user_id"],
    )
    op.create_index(
        "ix_user_roles_tenant_role",
        "user_roles",
        ["tenant_id", "role_id"],
    )

    op.create_table(
        "user_groups",
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["group_id"], ["groups.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id", "user_id", "group_id"),
    )
    op.create_index(
        "ix_user_groups_tenant_user",
        "user_groups",
        ["tenant_id", "user_id"],
    )
    op.create_index(
        "ix_user_groups_tenant_group",
        "user_groups",
        ["tenant_id", "group_id"],
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=True),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("section_path", sa.String(length=1024), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("metadata", _jsonb(), nullable=False),
        sa.Column("acl_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "classification_level",
            sa.String(length=32),
            server_default="internal",
            nullable=False,
        ),
        *_timestamps(),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
    )
    op.create_index("ix_document_chunks_tenant_id", "document_chunks", ["tenant_id"])
    op.create_index(
        "ix_document_chunks_workspace_id",
        "document_chunks",
        ["workspace_id"],
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )
    op.create_index(
        "ix_document_chunks_content_hash",
        "document_chunks",
        ["content_hash"],
    )
    op.create_index(
        "ix_document_chunks_is_deleted",
        "document_chunks",
        ["is_deleted"],
    )
    op.create_index(
        "ix_document_chunks_tenant_document",
        "document_chunks",
        ["tenant_id", "document_id"],
    )
    op.create_index(
        "ix_document_chunks_tenant_hash",
        "document_chunks",
        ["tenant_id", "content_hash"],
    )
    op.create_index(
        "ix_document_chunks_tenant_acl_version",
        "document_chunks",
        ["tenant_id", "acl_version"],
    )
    op.create_index(
        "ix_document_chunks_tenant_deleted",
        "document_chunks",
        ["tenant_id", "is_deleted"],
    )

    op.create_table(
        "document_acl",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column(
            "visibility",
            sa.String(length=32),
            server_default="private",
            nullable=False,
        ),
        sa.Column("allowed_user_ids", _jsonb(), nullable=False),
        sa.Column("allowed_group_ids", _jsonb(), nullable=False),
        sa.Column("allowed_roles", _jsonb(), nullable=False),
        sa.Column("denied_user_ids", _jsonb(), nullable=False),
        sa.Column("denied_group_ids", _jsonb(), nullable=False),
        sa.Column("acl_version", sa.Integer(), server_default="1", nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", name="uq_document_acl_document_id"),
    )
    op.create_index("ix_document_acl_tenant_id", "document_acl", ["tenant_id"])
    op.create_index(
        "ix_document_acl_document_id",
        "document_acl",
        ["document_id"],
    )
    op.create_index(
        "ix_document_acl_tenant_version",
        "document_acl",
        ["tenant_id", "acl_version"],
    )
    op.create_index(
        "ix_document_acl_tenant_visibility",
        "document_acl",
        ["tenant_id", "visibility"],
    )

    op.create_table(
        "chunk_acl",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column(
            "visibility",
            sa.String(length=32),
            server_default="private",
            nullable=False,
        ),
        sa.Column("allowed_user_ids", _jsonb(), nullable=False),
        sa.Column("allowed_group_ids", _jsonb(), nullable=False),
        sa.Column("allowed_roles", _jsonb(), nullable=False),
        sa.Column("denied_user_ids", _jsonb(), nullable=False),
        sa.Column("denied_group_ids", _jsonb(), nullable=False),
        sa.Column("acl_version", sa.Integer(), server_default="1", nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", name="uq_chunk_acl_chunk_id"),
    )
    op.create_index("ix_chunk_acl_tenant_id", "chunk_acl", ["tenant_id"])
    op.create_index("ix_chunk_acl_chunk_id", "chunk_acl", ["chunk_id"])
    op.create_index(
        "ix_chunk_acl_tenant_version",
        "chunk_acl",
        ["tenant_id", "acl_version"],
    )
    op.create_index(
        "ix_chunk_acl_tenant_visibility",
        "chunk_acl",
        ["tenant_id", "visibility"],
    )

    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=True),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column(
            "embedding_dimension",
            sa.Integer(),
            server_default="768",
            nullable=False,
        ),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("vector_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("metadata", _jsonb(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chunk_id",
            "embedding_model",
            "vector_version",
            name="uq_chunk_embeddings_chunk_model_version",
        ),
    )
    op.create_index("ix_chunk_embeddings_tenant_id", "chunk_embeddings", ["tenant_id"])
    op.create_index(
        "ix_chunk_embeddings_workspace_id",
        "chunk_embeddings",
        ["workspace_id"],
    )
    op.create_index(
        "ix_chunk_embeddings_document_id",
        "chunk_embeddings",
        ["document_id"],
    )
    op.create_index("ix_chunk_embeddings_chunk_id", "chunk_embeddings", ["chunk_id"])
    op.create_index(
        "ix_chunk_embeddings_tenant_model",
        "chunk_embeddings",
        ["tenant_id", "embedding_model"],
    )
    op.create_index(
        "ix_chunk_embeddings_tenant_document",
        "chunk_embeddings",
        ["tenant_id", "document_id"],
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=True),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("object_key", sa.String(length=1024), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="queued",
            nullable=False,
        ),
        sa.Column(
            "current_stage",
            sa.String(length=32),
            server_default="created",
            nullable=False,
        ),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_retries", sa.Integer(), server_default="3", nullable=False),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("metadata", _jsonb(), nullable=False),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_ingestion_jobs_tenant_idempotency",
        ),
    )
    op.create_index("ix_ingestion_jobs_tenant_id", "ingestion_jobs", ["tenant_id"])
    op.create_index("ix_ingestion_jobs_workspace_id", "ingestion_jobs", ["workspace_id"])
    op.create_index("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])
    op.create_index("ix_ingestion_jobs_object_key", "ingestion_jobs", ["object_key"])
    op.create_index("ix_ingestion_jobs_created_by", "ingestion_jobs", ["created_by"])
    op.create_index(
        "ix_ingestion_jobs_tenant_status",
        "ingestion_jobs",
        ["tenant_id", "status"],
    )
    op.create_index(
        "ix_ingestion_jobs_tenant_stage",
        "ingestion_jobs",
        ["tenant_id", "current_stage"],
    )
    op.create_index(
        "ix_ingestion_jobs_tenant_created",
        "ingestion_jobs",
        ["tenant_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
    op.drop_table("chunk_embeddings")
    op.drop_table("chunk_acl")
    op.drop_table("document_acl")
    op.drop_table("document_chunks")
    op.drop_table("user_groups")
    op.drop_table("user_roles")
    op.drop_table("documents")
    op.drop_table("groups")
    op.drop_table("roles")
    op.drop_table("users")
    op.drop_table("tenants")
