from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, JsonObject, PageResponse


class TenantStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class MembershipStatus(StrEnum):
    INVITED = "invited"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REMOVED = "removed"


class DepartmentStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class RoleScope(StrEnum):
    TENANT = "tenant"
    DEPARTMENT = "department"


class InvitationStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ProvisioningMode(StrEnum):
    INVITATION_LINK = "invitation_link"
    TEMPORARY_PASSWORD = "temporary_password"


class PermissionRead(APIModel):
    id: UUID
    code: str
    scope: RoleScope
    description: str


class RoleRead(APIModel):
    id: UUID
    tenant_id: UUID | None = None
    name: str
    slug: str
    scope: RoleScope
    description: str | None = None
    is_system: bool
    is_mutable: bool
    permissions: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class RoleCreate(APIModel):
    name: str = Field(..., min_length=2, max_length=128)
    slug: str = Field(..., min_length=2, max_length=128, pattern=r"^[a-z0-9-]+$")
    scope: RoleScope
    description: str | None = Field(default=None, max_length=1024)
    permission_codes: list[str] = Field(..., min_length=1, max_length=100)


class RoleUpdate(APIModel):
    name: str | None = Field(default=None, min_length=2, max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    permission_codes: list[str] | None = Field(default=None, min_length=1, max_length=100)


class TenantCreate(APIModel):
    name: str = Field(..., min_length=2, max_length=256)
    slug: str = Field(..., min_length=2, max_length=128, pattern=r"^[a-z0-9-]+$")


class TenantUpdate(APIModel):
    name: str | None = Field(default=None, min_length=2, max_length=256)
    plan: str | None = Field(default=None, min_length=2, max_length=32)


class TenantRead(APIModel):
    id: UUID
    tenant_key: str
    name: str
    slug: str
    status: TenantStatus
    plan: str
    role: RoleRead | None = None
    permissions: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class TenantListResponse(APIModel):
    items: list[TenantRead] = Field(default_factory=list)


class DepartmentCreate(APIModel):
    name: str = Field(..., min_length=2, max_length=256)
    slug: str = Field(..., min_length=2, max_length=128, pattern=r"^[a-z0-9-]+$")
    description: str | None = Field(default=None, max_length=1024)
    workspace_name: str | None = Field(default=None, min_length=2, max_length=256)


class DepartmentUpdate(APIModel):
    name: str | None = Field(default=None, min_length=2, max_length=256)
    description: str | None = Field(default=None, max_length=1024)


class DepartmentRead(APIModel):
    id: UUID
    tenant_id: UUID
    name: str
    slug: str
    description: str | None = None
    status: DepartmentStatus
    role: RoleRead | None = None
    permissions: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class DepartmentListResponse(APIModel):
    items: list[DepartmentRead] = Field(default_factory=list)
    page: PageResponse


class WorkspaceRead(APIModel):
    id: UUID
    tenant_id: UUID
    department_id: UUID
    workspace_key: str
    name: str
    slug: str
    status: str


class AuthIdentityRead(APIModel):
    provider: str
    provider_email: str | None = None
    provider_email_verified: bool
    last_login_at: datetime | None = None


class DepartmentAccessRead(APIModel):
    department: DepartmentRead
    role: RoleRead
    permissions: list[str] = Field(default_factory=list)


class CurrentUserRead(APIModel):
    id: UUID
    primary_email: str
    username: str | None = None
    display_name: str | None = None
    status: str
    email_verified: bool
    must_change_password: bool
    identities: list[AuthIdentityRead] = Field(default_factory=list)
    active_tenant_id: UUID | None = None
    active_department_id: UUID | None = None


class EffectiveAccessResponse(APIModel):
    user: CurrentUserRead
    tenant: TenantRead | None = None
    tenant_permissions: list[str] = Field(default_factory=list)
    departments: list[DepartmentAccessRead] = Field(default_factory=list)


class DepartmentRoleAssignment(APIModel):
    department_id: UUID
    role_id: UUID


class InvitationCreate(APIModel):
    email: str = Field(..., min_length=3, max_length=320)
    username: str | None = Field(default=None, min_length=2, max_length=128)
    display_name: str | None = Field(default=None, min_length=2, max_length=256)
    tenant_role_id: UUID
    department_assignments: list[DepartmentRoleAssignment] = Field(
        default_factory=list,
        max_length=100,
    )
    personal_message: str | None = Field(default=None, max_length=1000)


class InvitationDepartmentRead(APIModel):
    department_id: UUID
    department_name: str
    role_id: UUID
    role_name: str


class InvitationRead(APIModel):
    id: UUID
    tenant_id: UUID
    email: str
    username: str | None = None
    display_name: str | None = None
    tenant_role_id: UUID
    tenant_role_name: str
    status: InvitationStatus
    provisioning_mode: ProvisioningMode
    assignments: list[InvitationDepartmentRead] = Field(default_factory=list)
    expires_at: datetime
    accepted_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime


class InvitationCreateResponse(APIModel):
    invitation: InvitationRead
    acceptance_url: str


class InvitationListResponse(APIModel):
    items: list[InvitationRead] = Field(default_factory=list)
    page: PageResponse


class InvitationResolveResponse(APIModel):
    tenant_name: str
    invited_email: str
    display_name: str | None = None
    status: InvitationStatus
    expires_at: datetime
    assignments: list[InvitationDepartmentRead] = Field(default_factory=list)
    available_identity_providers: list[str] = Field(default_factory=list)


class InvitationAcceptRequest(APIModel):
    tenant_id: UUID | None = None


class InvitationAcceptResponse(APIModel):
    tenant_id: UUID
    tenant_name: str
    department_ids: list[UUID] = Field(default_factory=list)
    status: str = "accepted"


class MemberRead(APIModel):
    id: UUID
    primary_email: str
    username: str | None = None
    display_name: str | None = None
    user_status: str
    membership_status: MembershipStatus
    tenant_role: RoleRead
    departments: list[DepartmentAccessRead] = Field(default_factory=list)
    email_verified: bool
    must_change_password: bool
    joined_at: datetime | None = None
    last_login_at: datetime | None = None


class MemberListResponse(APIModel):
    items: list[MemberRead] = Field(default_factory=list)
    page: PageResponse


class MemberUpdate(APIModel):
    tenant_role_id: UUID | None = None
    status: MembershipStatus | None = None
    display_name: str | None = Field(default=None, min_length=2, max_length=256)
    department_assignments: list[DepartmentRoleAssignment] | None = Field(
        default=None,
        max_length=100,
    )


class MemberProvisionRequest(APIModel):
    email: str = Field(..., min_length=3, max_length=320)
    username: str | None = Field(default=None, min_length=2, max_length=128)
    display_name: str | None = Field(default=None, min_length=2, max_length=256)
    tenant_role_id: UUID
    department_assignments: list[DepartmentRoleAssignment] = Field(
        default_factory=list,
        max_length=100,
    )


class TemporaryPasswordResponse(APIModel):
    member: MemberRead
    temporary_password: str
    retrievable_again: bool = False


class DepartmentMemberCreate(APIModel):
    user_id: UUID
    role_id: UUID


class DepartmentMemberUpdate(APIModel):
    role_id: UUID | None = None
    status: MembershipStatus | None = None


class PasswordChangeRequest(APIModel):
    current_password: str = Field(..., min_length=1, max_length=1024)
    new_password: str = Field(..., min_length=12, max_length=1024)


class AuditEventRead(APIModel):
    id: UUID
    tenant_id: UUID
    department_id: UUID | None = None
    actor_user_id: UUID | None = None
    target_user_id: UUID | None = None
    action: str
    resource_type: str
    resource_id: UUID | None = None
    metadata: JsonObject = Field(default_factory=dict)
    request_id: str | None = None
    created_at: datetime


class AuditEventListResponse(APIModel):
    items: list[AuditEventRead] = Field(default_factory=list)
    page: PageResponse

