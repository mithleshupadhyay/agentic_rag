from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    Uuid,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.shared.db.base import (
    Base,
    JsonDict,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    jsonb_type,
)


class Tenant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "tenants"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        server_default="active",
    )
    data_region: Mapped[str | None] = mapped_column(String(32))
    plan: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="free",
        server_default="free",
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True))
    identity_provider: Mapped[str | None] = mapped_column(String(32))
    external_organization_id: Mapped[str | None] = mapped_column(String(128))
    metadata_: Mapped[JsonDict] = mapped_column(
        "metadata",
        jsonb_type(),
        nullable=False,
        default=dict,
    )

    users = relationship("User", back_populates="tenant", lazy="selectin")
    roles = relationship(
        "Role",
        back_populates="tenant",
        foreign_keys="Role.tenant_id",
        lazy="selectin",
    )
    groups = relationship("Group", back_populates="tenant", lazy="selectin")
    documents = relationship("Document", back_populates="tenant", lazy="selectin")
    ingestion_jobs = relationship(
        "IngestionJob",
        back_populates="tenant",
        lazy="selectin",
    )
    query_runs = relationship(
        "QueryRun",
        back_populates="tenant",
        lazy="selectin",
    )
    agent_runs = relationship(
        "AgentRun",
        back_populates="tenant",
        lazy="selectin",
    )
    llm_providers = relationship(
        "LLMProvider",
        back_populates="tenant",
        lazy="selectin",
    )
    memberships = relationship(
        "TenantMembership",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    departments = relationship(
        "Department",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    workspaces = relationship(
        "Workspace",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "identity_provider",
            "external_organization_id",
            name="uq_tenants_identity_organization",
        ),
        Index("ix_tenants_status_region", "status", "data_region"),
        Index(
            "ix_tenants_identity_organization",
            "identity_provider",
            "external_organization_id",
        ),
    )


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    tenant_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    external_subject: Mapped[str | None] = mapped_column(String(256))
    email: Mapped[str | None] = mapped_column(String(320))
    primary_email: Mapped[str | None] = mapped_column(String(320))
    normalized_email: Mapped[str | None] = mapped_column(
        String(320),
        unique=True,
        index=True,
    )
    username: Mapped[str | None] = mapped_column(String(128))
    normalized_username: Mapped[str | None] = mapped_column(
        String(128),
        unique=True,
        index=True,
    )
    display_name: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        server_default="active",
    )
    acl_version: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
        server_default="1",
    )
    must_change_password: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[JsonDict] = mapped_column(
        "metadata",
        jsonb_type(),
        nullable=False,
        default=dict,
    )

    tenant = relationship("Tenant", back_populates="users", lazy="select")
    role_links = relationship(
        "UserRole",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    group_links = relationship(
        "UserGroup",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    identities = relationship(
        "AuthIdentity",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    tenant_memberships = relationship(
        "TenantMembership",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    department_memberships = relationship(
        "DepartmentMembership",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "external_subject",
            name="uq_users_tenant_external_subject",
        ),
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
        Index("ix_users_tenant_status", "tenant_id", "status"),
        Index("ix_users_tenant_acl_version", "tenant_id", "acl_version"),
    )


class Role(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "roles"

    tenant_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    tenant_uuid: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    slug: Mapped[str | None] = mapped_column(String(128))
    scope: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="tenant",
        server_default="tenant",
    )
    description: Mapped[str | None] = mapped_column(String(1024))
    is_system: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
    is_mutable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True))

    tenant = relationship(
        "Tenant",
        back_populates="roles",
        foreign_keys=[tenant_id],
        lazy="select",
    )
    user_links = relationship(
        "UserRole",
        back_populates="role",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    permission_links = relationship(
        "RolePermission",
        back_populates="role",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_roles_tenant_name"),
        UniqueConstraint(
            "tenant_uuid",
            "scope",
            "slug",
            name="uq_roles_tenant_scope_slug",
        ),
        Index("ix_roles_tenant_scope", "tenant_uuid", "scope"),
    )


class Group(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "groups"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1024))

    tenant = relationship("Tenant", back_populates="groups", lazy="select")
    user_links = relationship(
        "UserGroup",
        back_populates="group",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_groups_tenant_name"),
    )


class UserRole(TimestampMixin, Base):
    __tablename__ = "user_roles"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    )

    user = relationship("User", back_populates="role_links", lazy="select")
    role = relationship("Role", back_populates="user_links", lazy="select")

    __table_args__ = (
        Index("ix_user_roles_tenant_user", "tenant_id", "user_id"),
        Index("ix_user_roles_tenant_role", "tenant_id", "role_id"),
    )


class UserGroup(TimestampMixin, Base):
    __tablename__ = "user_groups"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("groups.id", ondelete="CASCADE"),
        primary_key=True,
    )

    user = relationship("User", back_populates="group_links", lazy="select")
    group = relationship("Group", back_populates="user_links", lazy="select")

    __table_args__ = (
        Index("ix_user_groups_tenant_user", "tenant_id", "user_id"),
        Index("ix_user_groups_tenant_group", "tenant_id", "group_id"),
    )
