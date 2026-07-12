import logging
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from supertokens_python.asyncio import delete_user as delete_supertokens_user
from supertokens_python.asyncio import get_user as get_supertokens_user
from supertokens_python.recipe.emailpassword.asyncio import (
    sign_up as supertokens_sign_up,
    update_email_or_password,
    verify_credentials,
)
from supertokens_python.recipe.emailpassword.interfaces import (
    EmailAlreadyExistsError,
    PasswordPolicyViolationError,
    SignInOkResult,
    SignUpOkResult,
    UpdateEmailOrPasswordOkResult,
)
from supertokens_python.recipe.emailverification.asyncio import (
    send_email_verification_email,
)
from supertokens_python.recipe.session.asyncio import (
    get_all_session_handles_for_user,
    revoke_all_sessions_for_user,
    revoke_multiple_sessions,
)

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.authorization import (
    get_accessible_department_ids,
    require_department_permission,
    require_tenant_permission,
)
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.identity import (
    accept_invitation,
    add_or_update_tenant_member,
    build_user_context,
    create_custom_role,
    create_department,
    create_invitation,
    create_tenant_for_user,
    decrypt_outbox_value,
    encrypt_outbox_value,
    generate_temporary_password,
    get_active_tenant,
    get_invitation_for_tenant,
    get_tenant_member,
    list_audit_events,
    list_departments,
    list_invitations,
    list_permissions,
    list_tenant_members,
    list_tenant_roles,
    list_user_tenants,
    reconcile_authenticated_identity,
    remove_tenant_member,
    resend_invitation,
    resolve_invitation,
    revoke_invitation,
    update_custom_role,
)
from agentic_rag.shared.db.models import (
    AuditEvent,
    AuthIdentity,
    Department,
    DepartmentMembership,
    Invitation,
    Role,
    RolePermission,
    Tenant,
    TenantMembership,
    User,
)
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.common import PageResponse
from agentic_rag.shared.schemas.tenancy import (
    AuditEventListResponse,
    AuditEventRead,
    AuthIdentityRead,
    CurrentUserRead,
    DepartmentAccessRead,
    DepartmentCreate,
    DepartmentListResponse,
    DepartmentMemberCreate,
    DepartmentMemberUpdate,
    DepartmentRead,
    DepartmentRoleAssignment,
    DepartmentUpdate,
    EffectiveAccessResponse,
    InvitationAcceptResponse,
    InvitationCreate,
    InvitationCreateResponse,
    InvitationDepartmentRead,
    InvitationListResponse,
    InvitationRead,
    InvitationResolveResponse,
    MemberListResponse,
    MemberProvisionRequest,
    MemberRead,
    MemberUpdate,
    PasswordChangeRequest,
    PermissionRead,
    RoleCreate,
    RoleRead,
    RoleUpdate,
    TemporaryPasswordResponse,
    TenantCreate,
    TenantListResponse,
    TenantRead,
    TenantUpdate,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tenancy"])

INVITATION_COOKIE_NAME = "agentic_rag_invitation"


def _role_response(role: Role) -> RoleRead:
    permissions = sorted(
        {
            link.permission.code
            for link in role.permission_links
            if link.permission is not None
        }
    )
    return RoleRead(
        id=role.id,
        tenant_id=role.tenant_uuid,
        name=role.name,
        slug=role.slug or role.name.lower().replace(" ", "-"),
        scope=role.scope,
        description=role.description,
        is_system=role.is_system,
        is_mutable=role.is_mutable,
        permissions=permissions,
        created_at=role.created_at,
        updated_at=role.updated_at,
    )


def _department_response(
    department: Department,
    user_context: UserContext | None = None,
    role: Role | None = None,
) -> DepartmentRead:
    permissions: list[str] = []
    if user_context is not None:
        permissions = user_context.department_permissions.get(department.id, [])
        if (
            not permissions
            and "tenant.data.view_all" in user_context.tenant_permissions
        ):
            permissions = ["department.view", "documents.view", "rag.query"]
        if "tenant.data.manage_all" in user_context.tenant_permissions:
            permissions = [
                "department.view",
                "department.update",
                "department.archive",
                "department.members.view",
                "department.members.invite",
                "department.members.update",
                "department.members.remove",
                "workspaces.view",
                "workspaces.create",
                "workspaces.update",
                "workspaces.archive",
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
            ]
    return DepartmentRead(
        id=department.id,
        tenant_id=department.tenant_id,
        name=department.name,
        slug=department.slug,
        description=department.description,
        status=department.status,
        role=_role_response(role) if role is not None else None,
        permissions=sorted(set(permissions)),
        created_at=department.created_at,
        updated_at=department.updated_at,
    )


def _tenant_response(
    tenant: Tenant,
    membership: TenantMembership | None = None,
) -> TenantRead:
    role = membership.role if membership is not None else None
    permissions = []
    if role is not None:
        permissions = sorted(
            {
                link.permission.code
                for link in role.permission_links
                if link.permission is not None
            }
        )
    return TenantRead(
        id=tenant.id,
        tenant_key=tenant.tenant_id,
        name=tenant.name,
        slug=tenant.slug,
        status=tenant.status,
        plan=tenant.plan,
        role=_role_response(role) if role is not None else None,
        permissions=permissions,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


def _invitation_response(invitation: Invitation) -> InvitationRead:
    return InvitationRead(
        id=invitation.id,
        tenant_id=invitation.tenant_id,
        email=invitation.email,
        username=invitation.username,
        display_name=invitation.display_name,
        tenant_role_id=invitation.tenant_role_id,
        tenant_role_name=invitation.tenant_role.name,
        status=invitation.status,
        provisioning_mode=invitation.provisioning_mode,
        assignments=[
            InvitationDepartmentRead(
                department_id=assignment.department_id,
                department_name=assignment.department.name,
                role_id=assignment.role_id,
                role_name=assignment.role.name,
            )
            for assignment in invitation.assignments
        ],
        expires_at=invitation.expires_at,
        accepted_at=invitation.accepted_at,
        revoked_at=invitation.revoked_at,
        created_at=invitation.created_at,
    )


def _member_response(membership: TenantMembership) -> MemberRead:
    user = membership.user
    department_access = []
    for department_membership in user.department_memberships:
        if (
            department_membership.tenant_id != membership.tenant_id
            or department_membership.status == "removed"
        ):
            continue
        role = department_membership.role
        permissions = sorted(
            {
                link.permission.code
                for link in role.permission_links
                if link.permission is not None
            }
        )
        department_access.append(
            DepartmentAccessRead(
                department=_department_response(
                    department_membership.department,
                    role=role,
                ),
                role=_role_response(role),
                permissions=permissions,
            )
        )
    return MemberRead(
        id=user.id,
        primary_email=user.primary_email or user.email or "",
        username=user.username,
        display_name=user.display_name,
        user_status=user.status,
        membership_status=membership.status,
        tenant_role=_role_response(membership.role),
        departments=department_access,
        email_verified=user.email_verified,
        must_change_password=user.must_change_password,
        joined_at=membership.joined_at,
        last_login_at=user.last_login_at,
    )


def _get_application_user(db: Session, user_context: UserContext) -> User:
    if user_context.app_user_id is None:
        raise HTTPException(
            status_code=409,
            detail="This legacy session has not been migrated to an application user.",
        )
    user = (
        db.query(User)
        .options(selectinload(User.identities))
        .filter(User.id == user_context.app_user_id, User.status != "deleted")
        .first()
    )
    if user is None:
        raise HTTPException(status_code=401, detail="Application user not found.")
    return user


@router.get("/me", response_model=CurrentUserRead)
def get_current_application_user_endpoint(
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> CurrentUserRead:
    user = _get_application_user(db, user_context)
    return CurrentUserRead(
        id=user.id,
        primary_email=user.primary_email or user.email or "",
        username=user.username,
        display_name=user.display_name,
        status=user.status,
        email_verified=user.email_verified,
        must_change_password=user.must_change_password,
        identities=[
            AuthIdentityRead(
                provider=identity.provider,
                provider_email=identity.provider_email,
                provider_email_verified=identity.provider_email_verified,
                last_login_at=identity.last_login_at,
            )
            for identity in user.identities
        ],
        active_tenant_id=user_context.tenant_uuid,
        active_department_id=user_context.department_id,
    )


@router.get("/me/tenants", response_model=TenantListResponse)
def list_current_user_tenants_endpoint(
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> TenantListResponse:
    user = _get_application_user(db, user_context)
    memberships = list_user_tenants(db, user.id)
    return TenantListResponse(
        items=[
            _tenant_response(membership.tenant, membership)
            for membership in memberships
        ]
    )


@router.get("/me/context", response_model=EffectiveAccessResponse)
def get_effective_access_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID | None = None,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> EffectiveAccessResponse:
    user = _get_application_user(db, user_context)
    effective_context = build_user_context(
        db,
        user=user,
        supertokens_user_id=user_context.supertokens_user_id or "",
        requested_tenant_id=tenant_id,
        requested_department_id=department_id,
    )
    membership = (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.tenant),
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == user.id,
            TenantMembership.status == "active",
        )
        .one()
    )
    department_memberships = (
        db.query(DepartmentMembership)
        .options(
            selectinload(DepartmentMembership.department),
            selectinload(DepartmentMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .filter(
            DepartmentMembership.tenant_id == tenant_id,
            DepartmentMembership.user_id == user.id,
            DepartmentMembership.status == "active",
        )
        .all()
    )
    department_access = [
        DepartmentAccessRead(
            department=_department_response(
                department_membership.department,
                effective_context,
                department_membership.role,
            ),
            role=_role_response(department_membership.role),
            permissions=effective_context.department_permissions.get(
                department_membership.department_id,
                [],
            ),
        )
        for department_membership in department_memberships
    ]
    current_user = get_current_application_user_endpoint(db, effective_context)
    return EffectiveAccessResponse(
        user=current_user,
        tenant=_tenant_response(membership.tenant, membership),
        tenant_permissions=effective_context.tenant_permissions,
        departments=department_access,
    )


@router.get("/me/departments", response_model=DepartmentListResponse)
def list_current_user_departments_endpoint(
    tenant_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> DepartmentListResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.departments.view")
    accessible = get_accessible_department_ids(
        user_context,
        tenant_id,
        "department.view",
    )
    departments, total = list_departments(db, tenant_id, set(accessible), page, size)
    return DepartmentListResponse(
        items=[_department_response(item, user_context) for item in departments],
        page=PageResponse(page=page, size=size, total=total),
    )


@router.post("/me/change-initial-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_initial_password_endpoint(
    data: PasswordChangeRequest,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> None:
    user = _get_application_user(db, user_context)
    if not user.must_change_password:
        raise HTTPException(
            status_code=409, detail="No initial password change is required."
        )
    if not user_context.supertokens_user_id:
        raise HTTPException(
            status_code=409, detail="Password changes require SuperTokens."
        )

    identity = next(
        (
            item
            for item in user.identities
            if item.provider == "supertokens_email_password"
        ),
        None,
    )
    if identity is None or not user.primary_email:
        raise HTTPException(
            status_code=409,
            detail="This account does not use email and password authentication.",
        )
    verified = await verify_credentials(
        tenant_id="public",
        email=user.primary_email,
        password=data.current_password,
    )
    if not isinstance(verified, SignInOkResult):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    updated = await update_email_or_password(
        recipe_user_id=verified.recipe_user_id,
        password=data.new_password,
        apply_password_policy=True,
        tenant_id_for_password_policy="public",
    )
    if isinstance(updated, PasswordPolicyViolationError):
        raise HTTPException(status_code=422, detail=updated.failure_reason)
    if not isinstance(updated, UpdateEmailOrPasswordOkResult):
        raise HTTPException(status_code=409, detail="Password could not be updated.")

    await revoke_all_sessions_for_user(user_context.supertokens_user_id)
    user.must_change_password = False
    user.status = "active"
    (
        db.query(TenantMembership)
        .filter(
            TenantMembership.user_id == user.id,
            TenantMembership.status == "invited",
        )
        .update({TenantMembership.status: "active"}, synchronize_session=False)
    )
    for membership in list_user_tenants(db, user.id):
        db.add(
            AuditEvent(
                tenant_id=membership.tenant_id,
                tenant_key=membership.tenant_key,
                actor_user_id=user.id,
                target_user_id=user.id,
                action="member.initial_password_changed",
                resource_type="user",
                resource_id=user.id,
                metadata_={},
            )
        )
    db.commit()


@router.post("/me/revoke-other-sessions", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_other_sessions_endpoint(
    request: Request,
    user_context: UserContext = Depends(get_current_user),
) -> None:
    if not user_context.supertokens_user_id:
        raise HTTPException(
            status_code=409, detail="Session revocation requires SuperTokens."
        )
    current_session_handle = getattr(
        request.state,
        "supertokens_session_handle",
        None,
    )
    session_handles = await get_all_session_handles_for_user(
        user_context.supertokens_user_id
    )
    other_session_handles = [
        session_handle
        for session_handle in session_handles
        if session_handle != current_session_handle
    ]
    if other_session_handles:
        await revoke_multiple_sessions(other_session_handles)


@router.post("/tenants", response_model=TenantRead, status_code=status.HTTP_201_CREATED)
def create_tenant_endpoint(
    data: TenantCreate,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> TenantRead:
    user = _get_application_user(db, user_context)
    tenant, _, _ = create_tenant_for_user(
        db,
        user=user,
        name=data.name,
        slug=data.slug,
        request_id=getattr(request.state, "request_id", None),
    )
    membership = list_user_tenants(db, user.id)
    owner_membership = next(item for item in membership if item.tenant_id == tenant.id)
    return _tenant_response(tenant, owner_membership)


@router.get("/tenants/{tenant_id}", response_model=TenantRead)
def get_tenant_endpoint(
    tenant_id: uuid.UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> TenantRead:
    require_tenant_permission(user_context, tenant_id, "tenant.view")
    tenant = get_active_tenant(db, tenant_id)
    membership = (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission)
        )
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == user_context.app_user_id,
        )
        .one()
    )
    return _tenant_response(tenant, membership)


@router.patch("/tenants/{tenant_id}", response_model=TenantRead)
def update_tenant_endpoint(
    tenant_id: uuid.UUID,
    data: TenantUpdate,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> TenantRead:
    require_tenant_permission(user_context, tenant_id, "tenant.update")
    tenant = get_active_tenant(db, tenant_id)
    if data.name is not None:
        tenant.name = data.name.strip()
    if data.plan is not None:
        require_tenant_permission(user_context, tenant_id, "tenant.billing.manage")
        tenant.plan = data.plan.strip().lower()
    db.add(
        AuditEvent(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            actor_user_id=user_context.app_user_id,
            action="tenant.updated",
            resource_type="tenant",
            resource_id=tenant.id,
            metadata_={},
            request_id=getattr(request.state, "request_id", None),
        )
    )
    db.commit()
    return get_tenant_endpoint(tenant_id, db, user_context)


@router.delete("/tenants/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
def archive_tenant_endpoint(
    tenant_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> None:
    require_tenant_permission(user_context, tenant_id, "tenant.archive")
    tenant = get_active_tenant(db, tenant_id)
    tenant.status = "archived"
    db.add(
        AuditEvent(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            actor_user_id=user_context.app_user_id,
            action="tenant.archived",
            resource_type="tenant",
            resource_id=tenant.id,
            metadata_={},
            request_id=getattr(request.state, "request_id", None),
        )
    )
    db.commit()


@router.post(
    "/tenants/{tenant_id}/departments",
    response_model=DepartmentRead,
    status_code=status.HTTP_201_CREATED,
)
def create_department_endpoint(
    tenant_id: uuid.UUID,
    data: DepartmentCreate,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> DepartmentRead:
    require_tenant_permission(user_context, tenant_id, "tenant.departments.create")
    tenant = get_active_tenant(db, tenant_id)
    department = create_department(
        db,
        tenant=tenant,
        actor_user_id=user_context.app_user_id,
        name=data.name,
        slug=data.slug,
        description=data.description,
        workspace_name=data.workspace_name,
        request_id=getattr(request.state, "request_id", None),
    )
    return _department_response(department, user_context)


@router.get(
    "/tenants/{tenant_id}/departments",
    response_model=DepartmentListResponse,
)
def list_tenant_departments_endpoint(
    tenant_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> DepartmentListResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.departments.view")
    can_view_all = "tenant.data.view_all" in user_context.tenant_permissions
    can_manage_all = "tenant.data.manage_all" in user_context.tenant_permissions
    accessible = None
    if not can_view_all and not can_manage_all:
        accessible = set(
            get_accessible_department_ids(user_context, tenant_id, "department.view")
        )
    departments, total = list_departments(db, tenant_id, accessible, page, size)
    return DepartmentListResponse(
        items=[_department_response(item, user_context) for item in departments],
        page=PageResponse(page=page, size=size, total=total),
    )


@router.get(
    "/tenants/{tenant_id}/departments/{department_id}",
    response_model=DepartmentRead,
)
def get_department_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> DepartmentRead:
    require_department_permission(
        user_context,
        tenant_id,
        department_id,
        "department.view",
    )
    department = (
        db.query(Department)
        .filter(
            Department.id == department_id,
            Department.tenant_id == tenant_id,
            Department.status == "active",
        )
        .first()
    )
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found.")
    return _department_response(department, user_context)


@router.patch(
    "/tenants/{tenant_id}/departments/{department_id}",
    response_model=DepartmentRead,
)
def update_department_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID,
    data: DepartmentUpdate,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> DepartmentRead:
    require_department_permission(
        user_context,
        tenant_id,
        department_id,
        "department.update",
    )
    department = (
        db.query(Department)
        .filter(Department.id == department_id, Department.tenant_id == tenant_id)
        .first()
    )
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found.")
    if data.name is not None:
        department.name = data.name.strip()
    if data.description is not None:
        department.description = data.description.strip()
    db.add(
        AuditEvent(
            tenant_id=tenant_id,
            tenant_key=department.tenant_key,
            department_id=department.id,
            actor_user_id=user_context.app_user_id,
            action="department.updated",
            resource_type="department",
            resource_id=department.id,
            metadata_={},
            request_id=getattr(request.state, "request_id", None),
        )
    )
    db.commit()
    db.refresh(department)
    return _department_response(department, user_context)


@router.delete(
    "/tenants/{tenant_id}/departments/{department_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def archive_department_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> None:
    require_department_permission(
        user_context,
        tenant_id,
        department_id,
        "department.archive",
    )
    department = (
        db.query(Department)
        .filter(Department.id == department_id, Department.tenant_id == tenant_id)
        .first()
    )
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found.")
    if department.slug == "general":
        raise HTTPException(
            status_code=409, detail="The General department cannot be archived."
        )
    department.status = "archived"
    db.add(
        AuditEvent(
            tenant_id=tenant_id,
            tenant_key=department.tenant_key,
            department_id=department.id,
            actor_user_id=user_context.app_user_id,
            action="department.archived",
            resource_type="department",
            resource_id=department.id,
            metadata_={},
            request_id=getattr(request.state, "request_id", None),
        )
    )
    db.commit()


@router.get("/permissions", response_model=list[PermissionRead])
def list_permissions_endpoint(
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> list[PermissionRead]:
    if user_context.tenant_uuid is None:
        raise HTTPException(status_code=403, detail="Missing company context.")
    require_tenant_permission(
        user_context,
        user_context.tenant_uuid,
        "tenant.roles.view",
    )
    return [
        PermissionRead(
            id=permission.id,
            code=permission.code,
            scope=permission.scope,
            description=permission.description,
        )
        for permission in list_permissions(db)
    ]


@router.get("/tenants/{tenant_id}/roles", response_model=list[RoleRead])
def list_roles_endpoint(
    tenant_id: uuid.UUID,
    scope: str | None = Query(default=None, pattern="^(tenant|department)$"),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> list[RoleRead]:
    require_tenant_permission(user_context, tenant_id, "tenant.roles.view")
    return [_role_response(role) for role in list_tenant_roles(db, tenant_id, scope)]


@router.post(
    "/tenants/{tenant_id}/roles",
    response_model=RoleRead,
    status_code=status.HTTP_201_CREATED,
)
def create_role_endpoint(
    tenant_id: uuid.UUID,
    data: RoleCreate,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> RoleRead:
    require_tenant_permission(user_context, tenant_id, "tenant.roles.manage")
    tenant = get_active_tenant(db, tenant_id)
    actor_permissions = set(user_context.tenant_permissions)
    if data.scope == "department":
        for permissions in user_context.department_permissions.values():
            actor_permissions.update(permissions)
        if "tenant.data.manage_all" in user_context.tenant_permissions:
            actor_permissions.update(
                permission.code
                for permission in list_permissions(db)
                if permission.scope == "department"
            )
    role = create_custom_role(
        db,
        tenant=tenant,
        actor_user_id=user_context.app_user_id,
        actor_permissions=actor_permissions,
        name=data.name,
        slug=data.slug,
        role_scope=data.scope,
        description=data.description,
        permission_codes=data.permission_codes,
        request_id=getattr(request.state, "request_id", None),
    )
    return _role_response(role)


@router.patch("/tenants/{tenant_id}/roles/{role_id}", response_model=RoleRead)
def update_role_endpoint(
    tenant_id: uuid.UUID,
    role_id: uuid.UUID,
    data: RoleUpdate,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> RoleRead:
    require_tenant_permission(user_context, tenant_id, "tenant.roles.manage")
    role = (
        db.query(Role)
        .options(
            selectinload(Role.permission_links).selectinload(RolePermission.permission)
        )
        .filter(Role.id == role_id, Role.tenant_uuid == tenant_id)
        .first()
    )
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found.")
    actor_permissions = set(user_context.tenant_permissions)
    for permissions in user_context.department_permissions.values():
        actor_permissions.update(permissions)
    updated = update_custom_role(
        db,
        role=role,
        actor_user_id=user_context.app_user_id,
        actor_permissions=actor_permissions,
        name=data.name,
        description=data.description,
        permission_codes=data.permission_codes,
        request_id=getattr(request.state, "request_id", None),
    )
    return _role_response(updated)


@router.delete(
    "/tenants/{tenant_id}/roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_role_endpoint(
    tenant_id: uuid.UUID,
    role_id: uuid.UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> None:
    require_tenant_permission(user_context, tenant_id, "tenant.roles.manage")
    role = (
        db.query(Role).filter(Role.id == role_id, Role.tenant_uuid == tenant_id).first()
    )
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found.")
    if not role.is_mutable:
        raise HTTPException(status_code=409, detail="Built-in roles cannot be deleted.")
    try:
        db.delete(role)
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="This role is assigned to one or more members.",
        ) from error


@router.post(
    "/tenants/{tenant_id}/invitations",
    response_model=InvitationCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_invitation_endpoint(
    tenant_id: uuid.UUID,
    data: InvitationCreate,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> InvitationCreateResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.members.invite")
    tenant = get_active_tenant(db, tenant_id)
    inviter = _get_application_user(db, user_context)
    invitation, acceptance_url = create_invitation(
        db,
        tenant=tenant,
        invited_by=inviter,
        email=data.email,
        username=data.username,
        display_name=data.display_name,
        tenant_role_id=data.tenant_role_id,
        assignments=data.department_assignments,
        personal_message=data.personal_message,
        idempotency_key=idempotency_key,
        request_id=getattr(request.state, "request_id", None),
    )
    return InvitationCreateResponse(
        invitation=_invitation_response(invitation),
        acceptance_url=acceptance_url,
    )


@router.get(
    "/tenants/{tenant_id}/invitations",
    response_model=InvitationListResponse,
)
def list_invitations_endpoint(
    tenant_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> InvitationListResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.members.view")
    invitations, total = list_invitations(db, tenant_id, page, size)
    return InvitationListResponse(
        items=[_invitation_response(item) for item in invitations],
        page=PageResponse(page=page, size=size, total=total),
    )


@router.post(
    "/tenants/{tenant_id}/invitations/{invitation_id}/resend",
    response_model=InvitationCreateResponse,
)
def resend_invitation_endpoint(
    tenant_id: uuid.UUID,
    invitation_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> InvitationCreateResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.members.invite")
    invitation = get_invitation_for_tenant(db, tenant_id, invitation_id)
    invitation, acceptance_url = resend_invitation(
        db,
        invitation,
        actor_user_id=user_context.app_user_id,
        request_id=getattr(request.state, "request_id", None),
    )
    return InvitationCreateResponse(
        invitation=_invitation_response(invitation),
        acceptance_url=acceptance_url,
    )


@router.post(
    "/tenants/{tenant_id}/invitations/{invitation_id}/revoke",
    response_model=InvitationRead,
)
def revoke_invitation_endpoint(
    tenant_id: uuid.UUID,
    invitation_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> InvitationRead:
    require_tenant_permission(user_context, tenant_id, "tenant.members.invite")
    invitation = get_invitation_for_tenant(db, tenant_id, invitation_id)
    invitation = revoke_invitation(
        db,
        invitation,
        actor_user_id=user_context.app_user_id,
        request_id=getattr(request.state, "request_id", None),
    )
    return _invitation_response(invitation)


@router.get("/invitations/resolve", response_model=InvitationResolveResponse)
def resolve_invitation_endpoint(
    response: Response,
    token: str = Query(..., min_length=32, max_length=512),
    db: Session = Depends(get_session),
) -> InvitationResolveResponse:
    invitation = resolve_invitation(db, token)
    response.set_cookie(
        key=INVITATION_COOKIE_NAME,
        value=encrypt_outbox_value(token),
        max_age=settings.invitation_context_ttl_seconds,
        httponly=True,
        secure=settings.supertokens_cookie_secure,
        samesite=settings.supertokens_cookie_same_site,
        path="/api/v1/invitations",
    )
    social_providers = []
    if settings.google_client_id and settings.google_client_secret:
        social_providers.append("google")
    if settings.github_client_id and settings.github_client_secret:
        social_providers.append("github")
    social_providers.append("emailpassword")
    return InvitationResolveResponse(
        tenant_name=invitation.tenant.name,
        invited_email=invitation.email,
        display_name=invitation.display_name,
        status=invitation.status,
        expires_at=invitation.expires_at,
        assignments=[
            InvitationDepartmentRead(
                department_id=assignment.department_id,
                department_name=assignment.department.name,
                role_id=assignment.role_id,
                role_name=assignment.role.name,
            )
            for assignment in invitation.assignments
        ],
        available_identity_providers=social_providers,
    )


@router.post("/invitations/accept", response_model=InvitationAcceptResponse)
def accept_invitation_endpoint(
    request: Request,
    response: Response,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> InvitationAcceptResponse:
    encrypted_token = request.cookies.get(INVITATION_COOKIE_NAME)
    if not encrypted_token:
        raise HTTPException(status_code=400, detail="Invitation context is missing.")
    try:
        raw_token = decrypt_outbox_value(encrypted_token)
    except Exception as error:
        raise HTTPException(
            status_code=400, detail="Invitation context is invalid."
        ) from error
    user = _get_application_user(db, user_context)
    invitation = accept_invitation(
        db,
        raw_token=raw_token,
        user=user,
        request_id=getattr(request.state, "request_id", None),
    )
    response.delete_cookie(INVITATION_COOKIE_NAME, path="/api/v1/invitations")
    return InvitationAcceptResponse(
        tenant_id=invitation.tenant_id,
        tenant_name=invitation.tenant.name,
        department_ids=[item.department_id for item in invitation.assignments],
    )


@router.get("/tenants/{tenant_id}/members", response_model=MemberListResponse)
def list_members_endpoint(
    tenant_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> MemberListResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.members.view")
    memberships, total = list_tenant_members(db, tenant_id, page, size)
    return MemberListResponse(
        items=[_member_response(item) for item in memberships],
        page=PageResponse(page=page, size=size, total=total),
    )


@router.get("/tenants/{tenant_id}/members/{user_id}", response_model=MemberRead)
def get_member_endpoint(
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> MemberRead:
    require_tenant_permission(user_context, tenant_id, "tenant.members.view")
    membership = get_tenant_member(db, tenant_id, user_id)
    return _member_response(membership)


@router.post(
    "/tenants/{tenant_id}/members/provision",
    response_model=TemporaryPasswordResponse,
    status_code=status.HTTP_201_CREATED,
)
async def provision_member_endpoint(
    tenant_id: uuid.UUID,
    data: MemberProvisionRequest,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> TemporaryPasswordResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.members.invite")
    if not settings.temporary_password_provisioning_enabled:
        raise HTTPException(
            status_code=403, detail="Temporary provisioning is disabled."
        )
    tenant = get_active_tenant(db, tenant_id)
    temporary_password = generate_temporary_password()
    sign_up_result = await supertokens_sign_up(
        tenant_id="public",
        email=data.email.strip().lower(),
        password=temporary_password,
    )
    if isinstance(sign_up_result, EmailAlreadyExistsError):
        raise HTTPException(
            status_code=409,
            detail="An authentication account already exists for this email.",
        )
    if not isinstance(sign_up_result, SignUpOkResult):
        raise HTTPException(
            status_code=502, detail="Authentication account creation failed."
        )

    supertokens_user_id = sign_up_result.user.id
    try:
        user = reconcile_authenticated_identity(
            db,
            supertokens_user_id=supertokens_user_id,
            provider="supertokens_email_password",
            email=data.email,
            email_verified=False,
            display_name=data.display_name,
            allow_existing_email_link=True,
        )
        user.username = data.username.strip() if data.username else None
        user.normalized_username = (
            data.username.strip().lower() if data.username else None
        )
        await send_email_verification_email(
            tenant_id="public",
            user_id=supertokens_user_id,
            recipe_user_id=sign_up_result.recipe_user_id,
            email=data.email.strip().lower(),
        )
        membership = add_or_update_tenant_member(
            db,
            tenant=tenant,
            user=user,
            tenant_role_id=data.tenant_role_id,
            assignments=data.department_assignments,
            actor_user_id=user_context.app_user_id,
            must_change_password=True,
            request_id=getattr(request.state, "request_id", None),
        )
        return TemporaryPasswordResponse(
            member=_member_response(membership),
            temporary_password=temporary_password,
        )
    except Exception:
        db.rollback()
        await delete_supertokens_user(supertokens_user_id)
        raise


@router.patch(
    "/tenants/{tenant_id}/members/{user_id}",
    response_model=MemberRead,
)
def update_member_endpoint(
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    data: MemberUpdate,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> MemberRead:
    require_tenant_permission(user_context, tenant_id, "tenant.members.update")
    tenant = get_active_tenant(db, tenant_id)
    membership = (
        db.query(TenantMembership)
        .options(selectinload(TenantMembership.user))
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == user_id,
            TenantMembership.status != "removed",
        )
        .first()
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Member not found.")
    target_user = membership.user
    if data.display_name is not None:
        target_user.display_name = data.display_name.strip()
    if data.status is not None:
        if data.status == "removed":
            remove_tenant_member(
                db,
                tenant,
                membership,
                actor_user_id=user_context.app_user_id,
                request_id=getattr(request.state, "request_id", None),
            )
            raise HTTPException(status_code=410, detail="Member was removed.")
        membership.status = data.status

    assignments = data.department_assignments
    if assignments is None:
        assignments = [
            DepartmentRoleAssignment(
                department_id=item.department_id,
                role_id=item.role_id,
            )
            for item in (
                db.query(DepartmentMembership)
                .filter(
                    DepartmentMembership.tenant_id == tenant_id,
                    DepartmentMembership.user_id == user_id,
                    DepartmentMembership.status == "active",
                )
                .all()
            )
        ]
    updated = add_or_update_tenant_member(
        db,
        tenant=tenant,
        user=target_user,
        tenant_role_id=data.tenant_role_id or membership.role_id,
        assignments=assignments,
        actor_user_id=user_context.app_user_id,
        must_change_password=target_user.must_change_password,
        request_id=getattr(request.state, "request_id", None),
    )
    return _member_response(updated)


@router.delete(
    "/tenants/{tenant_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_member_endpoint(
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> None:
    require_tenant_permission(user_context, tenant_id, "tenant.members.remove")
    tenant = get_active_tenant(db, tenant_id)
    membership = (
        db.query(TenantMembership)
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == user_id,
            TenantMembership.status != "removed",
        )
        .first()
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Member not found.")
    remove_tenant_member(
        db,
        tenant,
        membership,
        actor_user_id=user_context.app_user_id,
        request_id=getattr(request.state, "request_id", None),
    )


@router.post(
    "/tenants/{tenant_id}/members/{user_id}/revoke-sessions",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_member_sessions_endpoint(
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> None:
    require_tenant_permission(user_context, tenant_id, "tenant.members.update")
    identity = (
        db.query(AuthIdentity)
        .join(TenantMembership, TenantMembership.user_id == AuthIdentity.user_id)
        .filter(
            TenantMembership.tenant_id == tenant_id,
            AuthIdentity.user_id == user_id,
            AuthIdentity.provider_user_id.is_not(None),
        )
        .first()
    )
    if identity is None:
        raise HTTPException(status_code=404, detail="Member identity not found.")
    await revoke_all_sessions_for_user(identity.provider_user_id)


@router.post(
    "/tenants/{tenant_id}/members/{user_id}/reset-temporary-password",
    response_model=TemporaryPasswordResponse,
)
async def reset_member_temporary_password_endpoint(
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> TemporaryPasswordResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.members.update")
    if not settings.temporary_password_provisioning_enabled:
        raise HTTPException(
            status_code=403, detail="Temporary provisioning is disabled."
        )

    membership = get_tenant_member(db, tenant_id, user_id)
    identity = next(
        (
            item
            for item in membership.user.identities
            if item.provider == "supertokens_email_password"
        ),
        None,
    )
    if identity is None:
        raise HTTPException(
            status_code=409,
            detail="This member does not use email and password authentication.",
        )

    supertokens_user = await get_supertokens_user(identity.provider_user_id)
    if supertokens_user is None:
        raise HTTPException(status_code=404, detail="Authentication account not found.")
    login_method = next(
        (
            method
            for method in supertokens_user.login_methods
            if method.recipe_id == "emailpassword"
        ),
        None,
    )
    if login_method is None:
        raise HTTPException(
            status_code=409,
            detail="This member does not use email and password authentication.",
        )

    temporary_password = generate_temporary_password()
    updated = await update_email_or_password(
        recipe_user_id=login_method.recipe_user_id,
        password=temporary_password,
        apply_password_policy=True,
        tenant_id_for_password_policy="public",
    )
    if isinstance(updated, PasswordPolicyViolationError):
        raise HTTPException(status_code=422, detail=updated.failure_reason)
    if not isinstance(updated, UpdateEmailOrPasswordOkResult):
        raise HTTPException(
            status_code=502, detail="Temporary password could not be set."
        )

    membership.user.must_change_password = True
    membership.user.status = "pending"
    db.add(
        AuditEvent(
            tenant_id=tenant_id,
            tenant_key=membership.tenant_key,
            actor_user_id=user_context.app_user_id,
            target_user_id=membership.user_id,
            action="member.temporary_password_reset",
            resource_type="user",
            resource_id=membership.user_id,
            metadata_={},
            request_id=getattr(request.state, "request_id", None),
        )
    )
    try:
        db.commit()
    except Exception:
        db.rollback()
        await revoke_all_sessions_for_user(identity.provider_user_id)
        raise

    await revoke_all_sessions_for_user(identity.provider_user_id)
    membership = get_tenant_member(db, tenant_id, user_id)
    return TemporaryPasswordResponse(
        member=_member_response(membership),
        temporary_password=temporary_password,
    )


@router.post(
    "/tenants/{tenant_id}/departments/{department_id}/members",
    response_model=MemberRead,
)
def add_department_member_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID,
    data: DepartmentMemberCreate,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> MemberRead:
    require_department_permission(
        user_context,
        tenant_id,
        department_id,
        "department.members.update",
    )
    membership = (
        db.query(TenantMembership)
        .options(selectinload(TenantMembership.user))
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == data.user_id,
            TenantMembership.status == "active",
        )
        .first()
    )
    role = (
        db.query(Role)
        .filter(
            Role.id == data.role_id,
            Role.tenant_uuid == tenant_id,
            Role.scope == "department",
        )
        .first()
    )
    department = (
        db.query(Department)
        .filter(Department.id == department_id, Department.tenant_id == tenant_id)
        .first()
    )
    if membership is None or role is None or department is None:
        raise HTTPException(
            status_code=404, detail="Member, role, or department not found."
        )
    existing = (
        db.query(DepartmentMembership)
        .filter(
            DepartmentMembership.department_id == department_id,
            DepartmentMembership.user_id == data.user_id,
        )
        .first()
    )
    if existing is None:
        db.add(
            DepartmentMembership(
                tenant_id=tenant_id,
                department_id=department_id,
                user_id=data.user_id,
                role_id=role.id,
                status="active",
                joined_at=datetime.now(timezone.utc),
            )
        )
    else:
        existing.role_id = role.id
        existing.status = "active"
    db.commit()
    return get_member_endpoint(tenant_id, data.user_id, db, user_context)


@router.get(
    "/tenants/{tenant_id}/departments/{department_id}/members",
    response_model=MemberListResponse,
)
def list_department_members_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> MemberListResponse:
    require_department_permission(
        user_context,
        tenant_id,
        department_id,
        "department.members.view",
    )
    memberships, _ = list_tenant_members(db, tenant_id, 1, 500)
    filtered = [
        membership
        for membership in memberships
        if any(
            item.department_id == department_id and item.status == "active"
            for item in membership.user.department_memberships
        )
    ]
    start = (page - 1) * size
    return MemberListResponse(
        items=[_member_response(item) for item in filtered[start : start + size]],
        page=PageResponse(page=page, size=size, total=len(filtered)),
    )


@router.patch(
    "/tenants/{tenant_id}/departments/{department_id}/members/{user_id}",
    response_model=MemberRead,
)
def update_department_member_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID,
    user_id: uuid.UUID,
    data: DepartmentMemberUpdate,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> MemberRead:
    require_department_permission(
        user_context,
        tenant_id,
        department_id,
        "department.members.update",
    )
    membership = (
        db.query(DepartmentMembership)
        .filter(
            DepartmentMembership.tenant_id == tenant_id,
            DepartmentMembership.department_id == department_id,
            DepartmentMembership.user_id == user_id,
        )
        .first()
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Department member not found.")
    if data.role_id is not None:
        role = (
            db.query(Role)
            .filter(
                Role.id == data.role_id,
                Role.tenant_uuid == tenant_id,
                Role.scope == "department",
            )
            .first()
        )
        if role is None:
            raise HTTPException(status_code=404, detail="Department role not found.")
        membership.role_id = role.id
    if data.status is not None:
        membership.status = data.status
    db.commit()
    return get_member_endpoint(tenant_id, user_id, db, user_context)


@router.delete(
    "/tenants/{tenant_id}/departments/{department_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_department_member_endpoint(
    tenant_id: uuid.UUID,
    department_id: uuid.UUID,
    user_id: uuid.UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> None:
    require_department_permission(
        user_context,
        tenant_id,
        department_id,
        "department.members.remove",
    )
    membership = (
        db.query(DepartmentMembership)
        .filter(
            DepartmentMembership.tenant_id == tenant_id,
            DepartmentMembership.department_id == department_id,
            DepartmentMembership.user_id == user_id,
        )
        .first()
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Department member not found.")
    membership.status = "removed"
    db.commit()


@router.get(
    "/tenants/{tenant_id}/audit-events",
    response_model=AuditEventListResponse,
)
def list_audit_events_endpoint(
    tenant_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(get_current_user),
) -> AuditEventListResponse:
    require_tenant_permission(user_context, tenant_id, "tenant.audit.view")
    events, total = list_audit_events(db, tenant_id, page, size)
    return AuditEventListResponse(
        items=[
            AuditEventRead(
                id=event.id,
                tenant_id=event.tenant_id,
                department_id=event.department_id,
                actor_user_id=event.actor_user_id,
                target_user_id=event.target_user_id,
                action=event.action,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                metadata=event.metadata_,
                request_id=event.request_id,
                created_at=event.created_at,
            )
            for event in events
        ],
        page=PageResponse(page=page, size=size, total=total),
    )
