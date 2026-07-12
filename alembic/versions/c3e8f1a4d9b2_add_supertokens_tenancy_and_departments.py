"""add SuperTokens identity, memberships, departments, and authorization

Revision ID: c3e8f1a4d9b2
Revises: a7c9e2f4b6d8
Create Date: 2026-07-11 12:00:00.000000

"""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID, uuid4

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "c3e8f1a4d9b2"
down_revision: str | Sequence[str] | None = "a7c9e2f4b6d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TENANT_PERMISSIONS = {
    "tenant.view": "View company settings.",
    "tenant.update": "Update company settings.",
    "tenant.archive": "Archive the company.",
    "tenant.delete": "Permanently delete the company.",
    "tenant.members.view": "View company members.",
    "tenant.members.invite": "Invite company members.",
    "tenant.members.update": "Update company memberships.",
    "tenant.members.remove": "Remove company members.",
    "tenant.departments.view": "View company departments.",
    "tenant.departments.create": "Create company departments.",
    "tenant.departments.update": "Update company departments.",
    "tenant.departments.archive": "Archive company departments.",
    "tenant.roles.view": "View company roles.",
    "tenant.roles.manage": "Manage company roles.",
    "tenant.data.view_all": "View data in every company department.",
    "tenant.data.manage_all": "Manage data in every company department.",
    "tenant.audit.view": "View company audit events.",
    "tenant.billing.manage": "Manage company billing.",
}

DEPARTMENT_PERMISSIONS = {
    "department.view": "View a department.",
    "department.update": "Update a department.",
    "department.archive": "Archive a department.",
    "department.members.view": "View department members.",
    "department.members.invite": "Invite department members.",
    "department.members.update": "Update department memberships.",
    "department.members.remove": "Remove department members.",
    "workspaces.view": "View department workspaces.",
    "workspaces.create": "Create department workspaces.",
    "workspaces.update": "Update department workspaces.",
    "workspaces.archive": "Archive department workspaces.",
    "documents.view": "View department documents.",
    "documents.upload": "Upload department documents.",
    "documents.update": "Update department documents.",
    "documents.delete": "Delete department documents.",
    "collections.view": "View department collections.",
    "collections.manage": "Manage department collections.",
    "rag.query": "Run RAG queries against department data.",
    "conversations.view": "View department conversations.",
    "conversations.create": "Create department conversations.",
    "conversations.delete": "Delete department conversations.",
}

DEFAULT_ROLE_PERMISSIONS = {
    "tenant-owner": tuple(TENANT_PERMISSIONS),
    "tenant-admin": (
        "tenant.view",
        "tenant.update",
        "tenant.members.view",
        "tenant.members.invite",
        "tenant.members.update",
        "tenant.members.remove",
        "tenant.departments.view",
        "tenant.departments.create",
        "tenant.departments.update",
        "tenant.departments.archive",
        "tenant.roles.view",
        "tenant.roles.manage",
        "tenant.data.view_all",
        "tenant.data.manage_all",
        "tenant.audit.view",
    ),
    "tenant-member": ("tenant.view", "tenant.departments.view"),
    "tenant-auditor": (
        "tenant.view",
        "tenant.members.view",
        "tenant.departments.view",
        "tenant.roles.view",
        "tenant.audit.view",
    ),
    "department-admin": tuple(DEPARTMENT_PERMISSIONS),
    "editor": (
        "department.view",
        "workspaces.view",
        "documents.view",
        "documents.upload",
        "documents.update",
        "documents.delete",
        "collections.view",
        "collections.manage",
        "rag.query",
        "conversations.view",
        "conversations.create",
        "conversations.delete",
    ),
    "contributor": (
        "department.view",
        "workspaces.view",
        "documents.view",
        "documents.upload",
        "collections.view",
        "rag.query",
        "conversations.view",
        "conversations.create",
    ),
    "viewer": (
        "department.view",
        "workspaces.view",
        "documents.view",
        "collections.view",
        "rag.query",
        "conversations.view",
    ),
    "chat-only": ("department.view", "rag.query", "conversations.create"),
}


def _jsonb() -> sa.types.TypeEngine[object]:
    return postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def _timestamps() -> tuple[sa.Column[datetime], sa.Column[datetime]]:
    return (
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
    )


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("plan", sa.String(length=32), server_default="free", nullable=False),
    )
    op.add_column("tenants", sa.Column("created_by", sa.Uuid(), nullable=True))

    op.alter_column("users", "tenant_id", existing_type=sa.String(64), nullable=True)
    op.alter_column(
        "users",
        "external_subject",
        existing_type=sa.String(256),
        nullable=True,
    )
    op.add_column("users", sa.Column("primary_email", sa.String(320), nullable=True))
    op.add_column("users", sa.Column("normalized_email", sa.String(320), nullable=True))
    op.add_column("users", sa.Column("username", sa.String(128), nullable=True))
    op.add_column("users", sa.Column("normalized_username", sa.String(128), nullable=True))
    op.add_column(
        "users",
        sa.Column(
            "must_change_password",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "email_verified",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        "users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "uq_users_normalized_email",
        "users",
        ["normalized_email"],
        unique=True,
    )
    op.create_index(
        "uq_users_normalized_username",
        "users",
        ["normalized_username"],
        unique=True,
    )

    op.alter_column("roles", "tenant_id", existing_type=sa.String(64), nullable=True)
    op.add_column("roles", sa.Column("tenant_uuid", sa.Uuid(), nullable=True))
    op.add_column("roles", sa.Column("slug", sa.String(128), nullable=True))
    op.add_column(
        "roles",
        sa.Column("scope", sa.String(32), server_default="tenant", nullable=False),
    )
    op.add_column(
        "roles",
        sa.Column(
            "is_mutable",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
    )
    op.add_column("roles", sa.Column("created_by", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_roles_tenant_uuid_tenants",
        "roles",
        "tenants",
        ["tenant_uuid"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_roles_tenant_scope_slug",
        "roles",
        ["tenant_uuid", "scope", "slug"],
    )
    op.create_index("ix_roles_tenant_scope", "roles", ["tenant_uuid", "scope"])

    op.create_table(
        "auth_identities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("provider_user_id", sa.String(256), nullable=False),
        sa.Column("provider_email", sa.String(320), nullable=True),
        sa.Column(
            "provider_email_verified",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "provider_user_id",
            name="uq_auth_identities_provider_user",
        ),
    )
    op.create_index("ix_auth_identities_user_id", "auth_identities", ["user_id"])
    op.create_index("ix_auth_identities_provider", "auth_identities", ["provider"])
    op.create_index(
        "ix_auth_identities_provider_email",
        "auth_identities",
        ["provider", "provider_email"],
    )

    op.create_table(
        "permissions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("code", sa.String(128), nullable=False),
        sa.Column("scope", sa.String(32), nullable=False),
        sa.Column("description", sa.String(1024), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code", name="uq_permissions_code"),
    )
    op.create_index("ix_permissions_scope", "permissions", ["scope"])

    op.create_table(
        "role_permissions",
        sa.Column("role_id", sa.Uuid(), nullable=False),
        sa.Column("permission_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["permission_id"],
            ["permissions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("role_id", "permission_id"),
    )

    op.create_table(
        "tenant_memberships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_key", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(32), server_default="active", nullable=False),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "user_id",
            name="uq_tenant_memberships_tenant_user",
        ),
    )
    op.create_index("ix_tenant_memberships_tenant_id", "tenant_memberships", ["tenant_id"])
    op.create_index("ix_tenant_memberships_tenant_key", "tenant_memberships", ["tenant_key"])
    op.create_index("ix_tenant_memberships_user_id", "tenant_memberships", ["user_id"])
    op.create_index("ix_tenant_memberships_role_id", "tenant_memberships", ["role_id"])
    op.create_index("ix_tenant_memberships_status", "tenant_memberships", ["status"])
    op.create_index(
        "ix_tenant_memberships_tenant_status",
        "tenant_memberships",
        ["tenant_id", "status"],
    )

    op.create_table(
        "departments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_key", sa.String(64), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("slug", sa.String(128), nullable=False),
        sa.Column("description", sa.String(1024), nullable=True),
        sa.Column("status", sa.String(32), server_default="active", nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "slug", name="uq_departments_tenant_slug"),
    )
    op.create_index("ix_departments_tenant_id", "departments", ["tenant_id"])
    op.create_index("ix_departments_tenant_key", "departments", ["tenant_key"])
    op.create_index("ix_departments_status", "departments", ["status"])
    op.create_index(
        "ix_departments_tenant_status",
        "departments",
        ["tenant_id", "status"],
    )

    op.create_table(
        "department_memberships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("department_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(32), server_default="active", nullable=False),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["department_id"],
            ["departments.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "department_id",
            "user_id",
            name="uq_department_memberships_department_user",
        ),
    )
    for column_name in ("tenant_id", "department_id", "user_id", "role_id", "status"):
        op.create_index(
            f"ix_department_memberships_{column_name}",
            "department_memberships",
            [column_name],
        )
    op.create_index(
        "ix_department_memberships_tenant_user_status",
        "department_memberships",
        ["tenant_id", "user_id", "status"],
    )

    op.create_table(
        "workspaces",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_key", sa.String(64), nullable=False),
        sa.Column("department_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_key", sa.String(128), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("slug", sa.String(128), nullable=False),
        sa.Column("description", sa.String(1024), nullable=True),
        sa.Column("status", sa.String(32), server_default="active", nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["department_id"],
            ["departments.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "workspace_key", name="uq_workspaces_tenant_key"),
        sa.UniqueConstraint(
            "department_id",
            "slug",
            name="uq_workspaces_department_slug",
        ),
    )
    for column_name in ("tenant_id", "tenant_key", "department_id", "status"):
        op.create_index(
            f"ix_workspaces_{column_name}",
            "workspaces",
            [column_name],
        )
    op.create_index(
        "ix_workspaces_tenant_department_status",
        "workspaces",
        ["tenant_id", "department_id", "status"],
    )

    op.create_table(
        "invitations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_key", sa.String(64), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("normalized_email", sa.String(320), nullable=False),
        sa.Column("username", sa.String(128), nullable=True),
        sa.Column("display_name", sa.String(256), nullable=True),
        sa.Column("tenant_role_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column(
            "provisioning_mode",
            sa.String(32),
            server_default="invitation_link",
            nullable=False,
        ),
        sa.Column("invited_by", sa.Uuid(), nullable=False),
        sa.Column("personal_message", sa.String(1000), nullable=True),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tenant_role_id"], ["roles.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["invited_by"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_invitations_token_hash"),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_invitations_tenant_idempotency",
        ),
    )
    for column_name in ("tenant_id", "tenant_key", "email", "status"):
        op.create_index(f"ix_invitations_{column_name}", "invitations", [column_name])
    op.create_index(
        "ix_invitations_tenant_status_created",
        "invitations",
        ["tenant_id", "status", "created_at"],
    )
    op.create_index(
        "ix_invitations_tenant_email_status",
        "invitations",
        ["tenant_id", "normalized_email", "status"],
    )

    op.create_table(
        "invitation_department_assignments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("invitation_id", sa.Uuid(), nullable=False),
        sa.Column("department_id", sa.Uuid(), nullable=False),
        sa.Column("role_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["invitation_id"],
            ["invitations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["department_id"],
            ["departments.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "invitation_id",
            "department_id",
            name="uq_invitation_assignments_department",
        ),
    )
    op.create_index(
        "ix_invitation_department_assignments_invitation_id",
        "invitation_department_assignments",
        ["invitation_id"],
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_key", sa.String(64), nullable=False),
        sa.Column("department_id", sa.Uuid(), nullable=True),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("target_user_id", sa.Uuid(), nullable=True),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("resource_id", sa.Uuid(), nullable=True),
        sa.Column("metadata", _jsonb(), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=True),
        sa.Column("ip_address", sa.String(64), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["department_id"],
            ["departments.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["target_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column_name in (
        "tenant_id",
        "tenant_key",
        "department_id",
        "actor_user_id",
        "action",
        "request_id",
    ):
        op.create_index(f"ix_audit_events_{column_name}", "audit_events", [column_name])
    op.create_index(
        "ix_audit_events_tenant_created",
        "audit_events",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_audit_events_tenant_department_created",
        "audit_events",
        ["tenant_id", "department_id", "created_at"],
    )

    op.create_table(
        "email_outbox",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=True),
        sa.Column("email_type", sa.String(64), nullable=False),
        sa.Column("recipient", sa.String(320), nullable=False),
        sa.Column("subject", sa.String(512), nullable=False),
        sa.Column("template_data", _jsonb(), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_email_outbox_idempotency_key"),
    )
    for column_name in ("tenant_id", "email_type", "status", "available_at"):
        op.create_index(f"ix_email_outbox_{column_name}", "email_outbox", [column_name])
    op.create_index(
        "ix_email_outbox_status_available",
        "email_outbox",
        ["status", "available_at"],
    )

    department_resources = (
        "documents",
        "document_chunks",
        "chunk_embeddings",
        "ingestion_jobs",
        "query_runs",
        "agent_runs",
    )
    for table_name in department_resources:
        op.add_column(table_name, sa.Column("department_id", sa.Uuid(), nullable=True))
        op.create_foreign_key(
            f"fk_{table_name}_department_id_departments",
            table_name,
            "departments",
            ["department_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        op.create_index(
            f"ix_{table_name}_department_id",
            table_name,
            ["department_id"],
        )

    op.create_index(
        "ix_documents_tenant_department_status",
        "documents",
        ["tenant_id", "department_id", "status", "is_deleted"],
    )
    op.create_index(
        "ix_documents_tenant_department_workspace",
        "documents",
        ["tenant_id", "department_id", "workspace_id"],
    )
    op.create_index(
        "ix_document_chunks_tenant_department_document",
        "document_chunks",
        ["tenant_id", "department_id", "document_id"],
    )
    op.create_index(
        "ix_chunk_embeddings_tenant_department_model",
        "chunk_embeddings",
        ["tenant_id", "department_id", "embedding_model"],
    )
    op.create_index(
        "ix_ingestion_jobs_tenant_department_status",
        "ingestion_jobs",
        ["tenant_id", "department_id", "status"],
    )
    op.create_index(
        "ix_query_runs_tenant_department_created",
        "query_runs",
        ["tenant_id", "department_id", "created_at"],
    )
    op.create_index(
        "ix_agent_runs_tenant_department_created",
        "agent_runs",
        ["tenant_id", "department_id", "created_at"],
    )

    connection = op.get_bind()
    now = datetime.now(timezone.utc)

    permission_ids: dict[str, UUID] = {}
    for scope, catalog in (
        ("tenant", TENANT_PERMISSIONS),
        ("department", DEPARTMENT_PERMISSIONS),
    ):
        for code, description in catalog.items():
            permission_id = uuid4()
            permission_ids[code] = permission_id
            connection.execute(
                sa.text(
                    "INSERT INTO permissions "
                    "(id, code, scope, description, created_at, updated_at) "
                    "VALUES (:id, :code, :scope, :description, :now, :now)"
                ),
                {
                    "id": permission_id,
                    "code": code,
                    "scope": scope,
                    "description": description,
                    "now": now,
                },
            )

    tenant_rows = connection.execute(
        sa.text("SELECT id, tenant_id, identity_provider FROM tenants")
    ).mappings()
    for tenant in tenant_rows:
        tenant_uuid = tenant["id"]
        tenant_key = tenant["tenant_id"]
        connection.execute(
            sa.text(
                "UPDATE roles SET tenant_uuid = :tenant_uuid, "
                "slug = lower(replace(name, ' ', '-')), scope = 'tenant' "
                "WHERE tenant_id = :tenant_key"
            ),
            {"tenant_uuid": tenant_uuid, "tenant_key": tenant_key},
        )

        role_ids: dict[str, UUID] = {}
        for role_slug, permission_codes in DEFAULT_ROLE_PERMISSIONS.items():
            role_id = uuid4()
            role_ids[role_slug] = role_id
            role_scope = (
                "tenant" if role_slug.startswith("tenant-") else "department"
            )
            role_name = role_slug.replace("-", " ").title()
            connection.execute(
                sa.text(
                    "INSERT INTO roles "
                    "(id, tenant_id, tenant_uuid, name, slug, scope, description, "
                    "is_system, is_mutable, created_at, updated_at) "
                    "VALUES (:id, :tenant_key, :tenant_uuid, :name, :slug, :scope, "
                    ":description, true, false, :now, :now)"
                ),
                {
                    "id": role_id,
                    "tenant_key": tenant_key,
                    "tenant_uuid": tenant_uuid,
                    "name": role_name,
                    "slug": role_slug,
                    "scope": role_scope,
                    "description": f"Built-in {role_name} role.",
                    "now": now,
                },
            )
            for permission_code in permission_codes:
                connection.execute(
                    sa.text(
                        "INSERT INTO role_permissions "
                        "(role_id, permission_id, created_at) "
                        "VALUES (:role_id, :permission_id, :now)"
                    ),
                    {
                        "role_id": role_id,
                        "permission_id": permission_ids[permission_code],
                        "now": now,
                    },
                )

        general_department_id = uuid4()
        connection.execute(
            sa.text(
                "INSERT INTO departments "
                "(id, tenant_id, tenant_key, name, slug, description, status, "
                "created_at, updated_at) "
                "VALUES (:id, :tenant_id, :tenant_key, 'General', 'general', "
                "'Default department for existing and unassigned data.', 'active', "
                ":now, :now)"
            ),
            {
                "id": general_department_id,
                "tenant_id": tenant_uuid,
                "tenant_key": tenant_key,
                "now": now,
            },
        )

        workspace_rows = connection.execute(
            sa.text(
                "SELECT DISTINCT workspace_id FROM ("
                "SELECT workspace_id FROM documents WHERE tenant_id = :tenant_key "
                "UNION SELECT workspace_id FROM ingestion_jobs WHERE tenant_id = :tenant_key "
                "UNION SELECT workspace_id FROM query_runs WHERE tenant_id = :tenant_key "
                "UNION SELECT workspace_id FROM agent_runs WHERE tenant_id = :tenant_key"
                ") existing_workspaces WHERE workspace_id IS NOT NULL"
            ),
            {"tenant_key": tenant_key},
        ).scalars()
        workspace_keys = {str(value) for value in workspace_rows if value}
        if not workspace_keys:
            workspace_keys.add("default")
        for workspace_key in workspace_keys:
            normalized_slug = workspace_key.strip().lower().replace("_", "-").replace(" ", "-")
            connection.execute(
                sa.text(
                    "INSERT INTO workspaces "
                    "(id, tenant_id, tenant_key, department_id, workspace_key, name, "
                    "slug, status, created_at, updated_at) "
                    "VALUES (:id, :tenant_id, :tenant_key, :department_id, "
                    ":workspace_key, :name, :slug, 'active', :now, :now)"
                ),
                {
                    "id": uuid4(),
                    "tenant_id": tenant_uuid,
                    "tenant_key": tenant_key,
                    "department_id": general_department_id,
                    "workspace_key": workspace_key,
                    "name": workspace_key,
                    "slug": normalized_slug[:128] or "default",
                    "now": now,
                },
            )

        for table_name in department_resources:
            connection.execute(
                sa.text(
                    f"UPDATE {table_name} SET department_id = :department_id "
                    "WHERE tenant_id = :tenant_key AND department_id IS NULL"
                ),
                {
                    "department_id": general_department_id,
                    "tenant_key": tenant_key,
                },
            )

        user_rows = connection.execute(
            sa.text(
                "SELECT id, external_subject, email, status FROM users "
                "WHERE tenant_id = :tenant_key"
            ),
            {"tenant_key": tenant_key},
        ).mappings()
        for user in user_rows:
            normalized_email = (
                str(user["email"]).strip().lower() if user["email"] else None
            )
            connection.execute(
                sa.text(
                    "UPDATE users SET primary_email = COALESCE(primary_email, :email), "
                    "normalized_email = COALESCE(normalized_email, :email), "
                    "email_verified = CASE WHEN :email IS NULL THEN false ELSE true END "
                    "WHERE id = :user_id"
                ),
                {"email": normalized_email, "user_id": user["id"]},
            )

            is_admin = connection.execute(
                sa.text(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM user_roles ur JOIN roles r ON r.id = ur.role_id "
                    "WHERE ur.user_id = :user_id AND ur.tenant_id = :tenant_key "
                    "AND lower(r.name) = 'admin')"
                ),
                {"user_id": user["id"], "tenant_key": tenant_key},
            ).scalar_one()
            membership_role_id = (
                role_ids["tenant-owner"] if is_admin else role_ids["tenant-member"]
            )
            membership_status = (
                "active" if user["status"] == "active" else user["status"]
            )
            if membership_status not in {"active", "invited", "suspended", "removed"}:
                membership_status = "suspended"
            connection.execute(
                sa.text(
                    "INSERT INTO tenant_memberships "
                    "(id, tenant_id, tenant_key, user_id, role_id, status, joined_at, "
                    "created_at, updated_at) "
                    "VALUES (:id, :tenant_id, :tenant_key, :user_id, :role_id, "
                    ":status, :now, :now, :now)"
                ),
                {
                    "id": uuid4(),
                    "tenant_id": tenant_uuid,
                    "tenant_key": tenant_key,
                    "user_id": user["id"],
                    "role_id": membership_role_id,
                    "status": membership_status,
                    "now": now,
                },
            )

            department_role_id = (
                role_ids["department-admin"] if is_admin else role_ids["viewer"]
            )
            if membership_status == "active":
                connection.execute(
                    sa.text(
                        "INSERT INTO department_memberships "
                        "(id, tenant_id, department_id, user_id, role_id, status, "
                        "joined_at, created_at, updated_at) "
                        "VALUES (:id, :tenant_id, :department_id, :user_id, :role_id, "
                        "'active', :now, :now, :now)"
                    ),
                    {
                        "id": uuid4(),
                        "tenant_id": tenant_uuid,
                        "department_id": general_department_id,
                        "user_id": user["id"],
                        "role_id": department_role_id,
                        "now": now,
                    },
                )

            if user["external_subject"]:
                identity_provider = str(tenant["identity_provider"] or "local")
                provider = {
                    "auth0": "legacy_auth0",
                    "oidc": "legacy_oidc",
                    "keycloak": "legacy_keycloak",
                    "local": "local",
                }.get(identity_provider, f"legacy_{identity_provider}")
                existing_identity = connection.execute(
                    sa.text(
                        "SELECT id FROM auth_identities "
                        "WHERE provider = :provider AND provider_user_id = :subject"
                    ),
                    {"provider": provider, "subject": user["external_subject"]},
                ).first()
                if existing_identity is None:
                    connection.execute(
                        sa.text(
                            "INSERT INTO auth_identities "
                            "(id, user_id, provider, provider_user_id, provider_email, "
                            "provider_email_verified, created_at, updated_at) "
                            "VALUES (:id, :user_id, :provider, :subject, :email, "
                            ":verified, :now, :now)"
                        ),
                        {
                            "id": uuid4(),
                            "user_id": user["id"],
                            "provider": provider,
                            "subject": user["external_subject"],
                            "email": normalized_email,
                            "verified": normalized_email is not None,
                            "now": now,
                        },
                    )

    if connection.dialect.name == "postgresql":
        rls_tables_with_string_tenant = (
            "documents",
            "document_chunks",
            "chunk_embeddings",
            "ingestion_jobs",
            "query_runs",
            "agent_runs",
            "departments",
            "workspaces",
            "tenant_memberships",
            "invitations",
            "audit_events",
        )
        for table_name in rls_tables_with_string_tenant:
            tenant_column = "tenant_id"
            if table_name in {"departments", "workspaces", "tenant_memberships", "invitations", "audit_events"}:
                tenant_column = "tenant_key"
            op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
                f"USING (current_user = 'agentic_rag_worker' OR {tenant_column} = "
                "current_setting('app.tenant_id', true)) "
                f"WITH CHECK (current_user = 'agentic_rag_worker' OR {tenant_column} = "
                "current_setting('app.tenant_id', true))"
            )

        for table_name in ("department_memberships",):
            op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY {table_name}_tenant_isolation ON {table_name} "
                "USING (current_user = 'agentic_rag_worker' OR tenant_id::text = "
                "current_setting('app.tenant_uuid', true)) "
                "WITH CHECK (current_user = 'agentic_rag_worker' OR tenant_id::text = "
                "current_setting('app.tenant_uuid', true))"
            )


def downgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        for table_name in (
            "department_memberships",
            "audit_events",
            "invitations",
            "tenant_memberships",
            "workspaces",
            "departments",
            "agent_runs",
            "query_runs",
            "ingestion_jobs",
            "chunk_embeddings",
            "document_chunks",
            "documents",
        ):
            op.execute(
                f"DROP POLICY IF EXISTS {table_name}_tenant_isolation ON {table_name}"
            )
            op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY")

    op.drop_index("ix_agent_runs_tenant_department_created", table_name="agent_runs")
    op.drop_index("ix_query_runs_tenant_department_created", table_name="query_runs")
    op.drop_index("ix_ingestion_jobs_tenant_department_status", table_name="ingestion_jobs")
    op.drop_index("ix_chunk_embeddings_tenant_department_model", table_name="chunk_embeddings")
    op.drop_index("ix_document_chunks_tenant_department_document", table_name="document_chunks")
    op.drop_index("ix_documents_tenant_department_workspace", table_name="documents")
    op.drop_index("ix_documents_tenant_department_status", table_name="documents")

    for table_name in (
        "agent_runs",
        "query_runs",
        "ingestion_jobs",
        "chunk_embeddings",
        "document_chunks",
        "documents",
    ):
        op.drop_index(f"ix_{table_name}_department_id", table_name=table_name)
        op.drop_constraint(
            f"fk_{table_name}_department_id_departments",
            table_name,
            type_="foreignkey",
        )
        op.drop_column(table_name, "department_id")

    op.drop_table("email_outbox")
    op.drop_table("audit_events")
    op.drop_table("invitation_department_assignments")
    op.drop_table("invitations")
    op.drop_table("workspaces")
    op.drop_table("department_memberships")
    op.drop_table("departments")
    op.drop_table("tenant_memberships")
    op.drop_table("role_permissions")
    op.drop_table("permissions")
    op.drop_table("auth_identities")

    op.drop_index("ix_roles_tenant_scope", table_name="roles")
    op.drop_constraint("uq_roles_tenant_scope_slug", "roles", type_="unique")
    op.drop_constraint("fk_roles_tenant_uuid_tenants", "roles", type_="foreignkey")
    op.drop_column("roles", "created_by")
    op.drop_column("roles", "is_mutable")
    op.drop_column("roles", "scope")
    op.drop_column("roles", "slug")
    op.drop_column("roles", "tenant_uuid")
    op.alter_column("roles", "tenant_id", existing_type=sa.String(64), nullable=False)

    op.drop_index("uq_users_normalized_username", table_name="users")
    op.drop_index("uq_users_normalized_email", table_name="users")
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "email_verified")
    op.drop_column("users", "must_change_password")
    op.drop_column("users", "normalized_username")
    op.drop_column("users", "username")
    op.drop_column("users", "normalized_email")
    op.drop_column("users", "primary_email")
    op.alter_column(
        "users",
        "external_subject",
        existing_type=sa.String(256),
        nullable=False,
    )
    op.alter_column("users", "tenant_id", existing_type=sa.String(64), nullable=False)

    op.drop_column("tenants", "created_by")
    op.drop_column("tenants", "plan")

