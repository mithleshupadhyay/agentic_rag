from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, ORMModel


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


class AuthContext(APIModel):
    user_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
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
    name: str
    status: str
    data_region: str | None = None
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

