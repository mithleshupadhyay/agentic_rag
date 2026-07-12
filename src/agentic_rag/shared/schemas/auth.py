from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, ORMModel, PageResponse


class TokenType(StrEnum):
    USER = "user"
    SERVICE = "service"


class Visibility(StrEnum):
    PRIVATE = "private"
    GROUP = "group"
    TENANT = "tenant"
    PUBLIC = "public"


class PermissionAction(StrEnum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class TenantUserRole(StrEnum):
    VIEWER = "viewer"
    USER = "user"
    ADMIN = "admin"


TENANT_ROLE_SCOPES: dict[TenantUserRole, tuple[str, ...]] = {
    TenantUserRole.VIEWER: (
        "documents:read",
        "query:run",
    ),
    TenantUserRole.USER: (
        "documents:read",
        "documents:write",
        "ingestion:write",
        "query:run",
    ),
    TenantUserRole.ADMIN: (
        "documents:delete",
        "documents:read",
        "documents:write",
        "ingestion:write",
        "query:run",
    ),
}


class AuthContext(APIModel):
    user_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    department_id: UUID | None = None
    workspace_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    acl_version: int = Field(default=1, ge=1)
    data_region: str | None = None
    request_id: str | None = None
    token_type: TokenType = TokenType.USER

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


class AuthConfigurationResponse(APIModel):
    mode: Literal["local", "supertokens", "auth0"]
    provider: str = Field(..., min_length=1)
    issuer_url: str | None = None
    audience: str | None = None
    client_id: str | None = None
    scope: str = Field(default="openid profile email", min_length=1)
    identity_connections: list[str] = Field(default_factory=list)
    api_base_path: str = "/auth"
    website_base_path: str = "/auth"
    public_tenant_signup_enabled: bool = False
    social_providers: list[str] = Field(default_factory=list)


class AuthSessionResponse(APIModel):
    user_id: str = Field(..., min_length=1)
    tenant_id: str = ""
    tenant_uuid: UUID | None = None
    department_id: UUID | None = None
    workspace_id: str | None = None
    email: str | None = None
    roles: list[str] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    tenant_permissions: list[str] = Field(default_factory=list)
    department_permissions: dict[UUID, list[str]] = Field(default_factory=dict)
    must_change_password: bool = False
    acl_version: int = Field(default=1, ge=1)
    auth_provider: str = Field(..., min_length=1)


class UserInvitationRequest(APIModel):
    email: str = Field(
        ...,
        min_length=3,
        max_length=320,
        pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$",
    )
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    role: TenantUserRole = TenantUserRole.USER
    workspace_id: str | None = Field(default=None, min_length=1, max_length=64)


class TenantUserRead(APIModel):
    id: UUID
    tenant_id: str = Field(..., min_length=1)
    external_subject: str = Field(..., min_length=1)
    email: str | None = None
    display_name: str | None = None
    status: str = Field(..., min_length=1)
    roles: list[str] = Field(default_factory=list)
    workspace_id: str | None = None
    acl_version: int = Field(default=1, ge=1)
    created_at: datetime
    updated_at: datetime


class UserInvitationResponse(APIModel):
    user: TenantUserRead
    invitation_email_sent: bool
    identity_invitation_id: str = Field(..., min_length=1)


class TenantUserListResponse(APIModel):
    items: list[TenantUserRead] = Field(default_factory=list)
    page: PageResponse


class TokenClaims(APIModel):
    subject: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    acl_version: int = Field(default=1, ge=1)
    token_type: TokenType = TokenType.USER
    expires_at: datetime | None = None


class AclPolicy(APIModel):
    visibility: Visibility = Visibility.PRIVATE
    allowed_user_ids: list[str] = Field(default_factory=list)
    allowed_group_ids: list[str] = Field(default_factory=list)
    allowed_roles: list[str] = Field(default_factory=list)
    denied_user_ids: list[str] = Field(default_factory=list)
    denied_group_ids: list[str] = Field(default_factory=list)
    acl_version: int = Field(default=1, ge=1)


class AclDecision(APIModel):
    allowed: bool
    reason: str
    acl_version: int = Field(..., ge=1)
    denied_by: str | None = None


class AclFilterRequest(APIModel):
    auth: AuthContext
    resource_ids: list[UUID] = Field(default_factory=list)
    action: PermissionAction = PermissionAction.READ


class AclFilterResponse(APIModel):
    allowed_ids: list[UUID] = Field(default_factory=list)
    denied: dict[UUID, str] = Field(default_factory=dict)


class TenantRead(ORMModel):
    id: UUID
    tenant_id: str
    name: str
    status: str
    data_region: str | None = None
    identity_provider: str | None = None
    external_organization_id: str | None = None
    created_at: datetime
    updated_at: datetime


class UserRead(ORMModel):
    id: UUID
    tenant_id: str
    external_subject: str
    email: str | None = None
    display_name: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
