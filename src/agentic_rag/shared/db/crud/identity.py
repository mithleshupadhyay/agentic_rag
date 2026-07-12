import base64
import hashlib
import logging
import secrets
import string
import uuid
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet
from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from agentic_rag.core.authorization import (
    DEFAULT_ROLE_PERMISSIONS,
    DEPARTMENT_PERMISSION_CODES,
    TENANT_PERMISSION_CODES,
)
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.models import (
    AuditEvent,
    AuthIdentity,
    Department,
    DepartmentMembership,
    EmailOutbox,
    Invitation,
    InvitationDepartmentAssignment,
    Permission,
    Role,
    RolePermission,
    Tenant,
    TenantMembership,
    User,
    Workspace,
)
from agentic_rag.shared.schemas.tenancy import DepartmentRoleAssignment


logger = logging.getLogger(__name__)


def normalize_email(email: str) -> str:
    normalized_email = email.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A valid recovery email address is required.",
        )
    return normalized_email


def normalize_username(username: str | None) -> str | None:
    if username is None:
        return None
    normalized_username = username.strip().lower()
    if len(normalized_username) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Username must contain at least two characters.",
        )
    return normalized_username


def hash_invitation_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def encrypt_outbox_value(value: str) -> str:
    encryption_secret = (
        settings.invitation_context_secret.strip()
        or settings.supertokens_api_key.strip()
    )
    if not encryption_secret:
        raise RuntimeError(
            "INVITATION_CONTEXT_SECRET is required for invitation delivery."
        )
    key_material = hashlib.sha256(encryption_secret.encode("utf-8")).digest()
    fernet_key = base64.urlsafe_b64encode(key_material)
    return Fernet(fernet_key).encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_outbox_value(value: str) -> str:
    encryption_secret = (
        settings.invitation_context_secret.strip()
        or settings.supertokens_api_key.strip()
    )
    if not encryption_secret:
        raise RuntimeError(
            "INVITATION_CONTEXT_SECRET is required for invitation delivery."
        )
    key_material = hashlib.sha256(encryption_secret.encode("utf-8")).digest()
    fernet_key = base64.urlsafe_b64encode(key_material)
    return Fernet(fernet_key).decrypt(value.encode("ascii")).decode("utf-8")


def generate_temporary_password(length: int = 24) -> str:
    if length < 16:
        raise ValueError("Temporary passwords must contain at least 16 characters.")

    alphabet = string.ascii_letters + string.digits + "!@#$%^&*_-+="
    password_characters = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*_-+="),
    ]
    password_characters.extend(
        secrets.choice(alphabet) for _ in range(length - len(password_characters))
    )
    secrets.SystemRandom().shuffle(password_characters)
    return "".join(password_characters)


def get_user_by_supertokens_id(
    db: Session,
    supertokens_user_id: str,
) -> User | None:
    identity = (
        db.query(AuthIdentity)
        .options(
            selectinload(AuthIdentity.user).selectinload(User.identities),
            selectinload(AuthIdentity.user).selectinload(User.tenant_memberships),
        )
        .filter(AuthIdentity.provider_user_id == supertokens_user_id)
        .first()
    )
    return identity.user if identity is not None else None


def reconcile_authenticated_identity(
    db: Session,
    supertokens_user_id: str,
    provider: str,
    email: str,
    email_verified: bool,
    display_name: str | None = None,
    allow_existing_email_link: bool = False,
) -> User:
    normalized_email = normalize_email(email)
    existing_identity = (
        db.query(AuthIdentity)
        .options(selectinload(AuthIdentity.user).selectinload(User.identities))
        .filter(AuthIdentity.provider_user_id == supertokens_user_id)
        .first()
    )
    if existing_identity is not None:
        user = existing_identity.user
        existing_identity.last_login_at = datetime.now(timezone.utc)
        existing_identity.provider_email = normalized_email
        existing_identity.provider_email_verified = email_verified
        user.last_login_at = datetime.now(timezone.utc)
        user.email_verified = user.email_verified or email_verified
        if display_name and not user.display_name:
            user.display_name = display_name.strip()
        db.flush()
        return user

    user = (
        db.query(User)
        .options(selectinload(User.identities))
        .filter(User.normalized_email == normalized_email)
        .first()
    )
    if user is not None and user.identities and not allow_existing_email_link:
        providers = sorted({identity.provider for identity in user.identities})
        logger.warning(
            f"[Auth] Explicit account linking required email={normalized_email} "
            f"existing_providers={providers} requested_provider={provider}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An account already exists for this email using another sign-in "
                "method. Sign in with the original method or contact an administrator."
            ),
        )

    try:
        now = datetime.now(timezone.utc)
        if user is None:
            user = User(
                tenant_id=None,
                external_subject=None,
                email=normalized_email,
                primary_email=normalized_email,
                normalized_email=normalized_email,
                display_name=display_name.strip() if display_name else None,
                status="active",
                email_verified=email_verified,
                must_change_password=False,
                last_login_at=now,
                acl_version=1,
                metadata_={"identity_source": "supertokens"},
            )
            db.add(user)
            db.flush()
        else:
            user.primary_email = normalized_email
            user.email = normalized_email
            user.email_verified = user.email_verified or email_verified
            user.last_login_at = now

        db.add(
            AuthIdentity(
                user_id=user.id,
                provider=provider,
                provider_user_id=supertokens_user_id,
                provider_email=normalized_email,
                provider_email_verified=email_verified,
                last_login_at=now,
            )
        )
        db.flush()
        logger.info(
            f"[DB] Reconciled authenticated identity user={user.id} provider={provider}"
        )
        return user

    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Identity reconciliation conflict provider={provider}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The authenticated identity is already assigned to another user.",
        ) from error


def build_user_context(
    db: Session,
    user: User,
    supertokens_user_id: str,
    requested_tenant_id: uuid.UUID | None = None,
    requested_department_id: uuid.UUID | None = None,
    requested_workspace_id: str | None = None,
) -> UserContext:
    membership_query = (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.tenant),
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .join(Tenant, Tenant.id == TenantMembership.tenant_id)
        .filter(
            TenantMembership.user_id == user.id,
            TenantMembership.status == "active",
            Tenant.status == "active",
        )
    )
    if requested_tenant_id is not None:
        membership_query = membership_query.filter(
            TenantMembership.tenant_id == requested_tenant_id
        )
    tenant_membership = membership_query.order_by(TenantMembership.created_at).first()

    if tenant_membership is None:
        if requested_tenant_id is not None:
            logger.warning(
                f"[AuthZ] User has no active company membership user={user.id} "
                f"tenant={requested_tenant_id}"
            )
            raise HTTPException(status_code=404, detail="Company not found.")
        return UserContext(
            id=str(user.id),
            app_user_id=user.id,
            supertokens_user_id=supertokens_user_id,
            customer_id="",
            tenant_id="",
            email=user.primary_email or user.email,
            email_verified=user.email_verified,
            must_change_password=user.must_change_password,
            acl_version=user.acl_version,
        )

    tenant_permissions = sorted(
        {
            link.permission.code
            for link in tenant_membership.role.permission_links
            if link.permission is not None
        }
    )
    department_memberships = (
        db.query(DepartmentMembership)
        .options(
            selectinload(DepartmentMembership.department),
            selectinload(DepartmentMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .join(Department, Department.id == DepartmentMembership.department_id)
        .filter(
            DepartmentMembership.tenant_id == tenant_membership.tenant_id,
            DepartmentMembership.user_id == user.id,
            DepartmentMembership.status == "active",
            Department.status == "active",
        )
        .all()
    )
    department_permissions = {
        membership.department_id: sorted(
            {
                link.permission.code
                for link in membership.role.permission_links
                if link.permission is not None
            }
        )
        for membership in department_memberships
    }

    if (
        "tenant.data.view_all" in tenant_permissions
        or "tenant.data.manage_all" in tenant_permissions
    ):
        all_departments = (
            db.query(Department.id)
            .filter(
                Department.tenant_id == tenant_membership.tenant_id,
                Department.status == "active",
            )
            .all()
        )
        accessible_department_ids = [row[0] for row in all_departments]
    else:
        accessible_department_ids = list(department_permissions)

    active_department_id = requested_department_id
    if (
        active_department_id is not None
        and active_department_id not in accessible_department_ids
    ):
        logger.warning(
            f"[AuthZ] Requested department is not accessible user={user.id} "
            f"tenant={tenant_membership.tenant_id} department={active_department_id}"
        )
        raise HTTPException(status_code=404, detail="Department not found.")
    if active_department_id is None and accessible_department_ids:
        active_department_id = accessible_department_ids[0]

    active_workspace_id = requested_workspace_id
    if active_workspace_id:
        workspace_query = db.query(Workspace).filter(
            Workspace.tenant_id == tenant_membership.tenant_id,
            Workspace.workspace_key == active_workspace_id,
            Workspace.status == "active",
        )
        if active_department_id is not None:
            workspace_query = workspace_query.filter(
                Workspace.department_id == active_department_id
            )
        if workspace_query.first() is None:
            logger.warning(
                f"[AuthZ] Requested workspace is not accessible user={user.id} "
                f"tenant={tenant_membership.tenant_id} workspace={active_workspace_id}"
            )
            raise HTTPException(status_code=404, detail="Workspace not found.")
    elif active_department_id is not None:
        active_workspace = (
            db.query(Workspace)
            .filter(
                Workspace.tenant_id == tenant_membership.tenant_id,
                Workspace.department_id == active_department_id,
                Workspace.status == "active",
            )
            .order_by(Workspace.created_at)
            .first()
        )
        active_workspace_id = (
            active_workspace.workspace_key if active_workspace is not None else None
        )

    legacy_scopes: set[str] = set()
    effective_permissions = set(tenant_permissions)
    if active_department_id is not None:
        effective_permissions.update(
            department_permissions.get(active_department_id, [])
        )
    if (
        "documents.view" in effective_permissions
        or "tenant.data.view_all" in effective_permissions
    ):
        legacy_scopes.add("documents:read")
    if (
        "documents.upload" in effective_permissions
        or "tenant.data.manage_all" in effective_permissions
    ):
        legacy_scopes.update({"documents:write", "ingestion:write"})
    if (
        "documents.delete" in effective_permissions
        or "tenant.data.manage_all" in effective_permissions
    ):
        legacy_scopes.add("documents:delete")
    if (
        "rag.query" in effective_permissions
        or "tenant.data.view_all" in effective_permissions
    ):
        legacy_scopes.add("query:run")

    return UserContext(
        id=str(user.id),
        app_user_id=user.id,
        supertokens_user_id=supertokens_user_id,
        customer_id=tenant_membership.tenant_key,
        tenant_id=tenant_membership.tenant_key,
        tenant_uuid=tenant_membership.tenant_id,
        department_id=active_department_id,
        workspace_id=active_workspace_id,
        email=user.primary_email or user.email,
        email_verified=user.email_verified,
        roles=[tenant_membership.role.slug or tenant_membership.role.name],
        scopes=sorted(legacy_scopes),
        tenant_permissions=tenant_permissions,
        department_permissions=department_permissions,
        accessible_department_ids=accessible_department_ids,
        must_change_password=user.must_change_password,
        acl_version=user.acl_version,
    )


def list_user_tenants(db: Session, user_id: uuid.UUID) -> list[TenantMembership]:
    return (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.tenant),
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .join(Tenant, Tenant.id == TenantMembership.tenant_id)
        .filter(
            TenantMembership.user_id == user_id,
            TenantMembership.status == "active",
            Tenant.status == "active",
        )
        .order_by(Tenant.name)
        .all()
    )


def seed_default_roles(db: Session, tenant: Tenant) -> dict[str, Role]:
    permission_catalog = {
        **{
            code: "Tenant authorization permission." for code in TENANT_PERMISSION_CODES
        },
        **{
            code: "Department authorization permission."
            for code in DEPARTMENT_PERMISSION_CODES
        },
    }
    permissions_by_code = {
        permission.code: permission for permission in db.query(Permission).all()
    }
    for permission_code, description in permission_catalog.items():
        if permission_code not in permissions_by_code:
            permission = Permission(
                code=permission_code,
                scope=(
                    "tenant"
                    if permission_code in TENANT_PERMISSION_CODES
                    else "department"
                ),
                description=description,
            )
            db.add(permission)
            db.flush()
            permissions_by_code[permission_code] = permission

    roles_by_slug: dict[str, Role] = {}
    for role_slug, permission_codes in DEFAULT_ROLE_PERMISSIONS.items():
        role = (
            db.query(Role)
            .filter(
                Role.tenant_uuid == tenant.id,
                Role.scope
                == ("tenant" if role_slug.startswith("tenant-") else "department"),
                Role.slug == role_slug,
            )
            .first()
        )
        if role is None:
            role_name = role_slug.replace("-", " ").title()
            role = Role(
                tenant_id=tenant.tenant_id,
                tenant_uuid=tenant.id,
                name=role_name,
                slug=role_slug,
                scope=("tenant" if role_slug.startswith("tenant-") else "department"),
                description=f"Built-in {role_name} role.",
                is_system=True,
                is_mutable=False,
            )
            db.add(role)
            db.flush()

        existing_permission_ids = {link.permission_id for link in role.permission_links}
        for permission_code in permission_codes:
            permission = permissions_by_code[permission_code]
            if permission.id not in existing_permission_ids:
                db.add(
                    RolePermission(
                        role_id=role.id,
                        permission_id=permission.id,
                    )
                )
        roles_by_slug[role_slug] = role
    db.flush()
    return roles_by_slug


def create_tenant_for_user(
    db: Session,
    user: User,
    name: str,
    slug: str,
    request_id: str | None = None,
) -> tuple[Tenant, Department, Workspace]:
    if not settings.public_tenant_signup_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public company signup is disabled.",
        )
    if user.must_change_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Change the temporary password before creating a company.",
        )

    normalized_slug = slug.strip().lower()
    try:
        tenant = Tenant(
            tenant_id=f"tenant-{uuid.uuid4().hex[:24]}",
            name=name.strip(),
            slug=normalized_slug,
            status="active",
            plan="free",
            created_by=user.id,
            identity_provider="supertokens",
            metadata_={"onboarding": "self_service"},
        )
        db.add(tenant)
        db.flush()

        roles = seed_default_roles(db, tenant)
        now = datetime.now(timezone.utc)
        db.add(
            TenantMembership(
                tenant_id=tenant.id,
                tenant_key=tenant.tenant_id,
                user_id=user.id,
                role_id=roles["tenant-owner"].id,
                status="active",
                joined_at=now,
            )
        )
        department = Department(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            name="General",
            slug="general",
            description="Default department for company knowledge.",
            status="active",
            created_by=user.id,
        )
        db.add(department)
        db.flush()
        db.add(
            DepartmentMembership(
                tenant_id=tenant.id,
                department_id=department.id,
                user_id=user.id,
                role_id=roles["department-admin"].id,
                status="active",
                joined_at=now,
            )
        )
        workspace = Workspace(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            department_id=department.id,
            workspace_key="default",
            name="General Knowledge",
            slug="general-knowledge",
            status="active",
            created_by=user.id,
        )
        db.add(workspace)
        db.flush()
        db.add(
            AuditEvent(
                tenant_id=tenant.id,
                tenant_key=tenant.tenant_id,
                actor_user_id=user.id,
                action="tenant.created",
                resource_type="tenant",
                resource_id=tenant.id,
                metadata_={"slug": tenant.slug, "plan": tenant.plan},
                request_id=request_id,
            )
        )
        db.commit()
        db.refresh(tenant)
        db.refresh(department)
        db.refresh(workspace)
        logger.info(
            f"[DB] Company onboarding completed tenant={tenant.id} owner={user.id}"
        )
        return tenant, department, workspace

    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Company onboarding conflict slug={normalized_slug}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A company with this slug already exists.",
        ) from error


def create_department(
    db: Session,
    tenant: Tenant,
    actor_user_id: uuid.UUID,
    name: str,
    slug: str,
    description: str | None,
    workspace_name: str | None,
    request_id: str | None = None,
) -> Department:
    normalized_slug = slug.strip().lower()
    try:
        department = Department(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            name=name.strip(),
            slug=normalized_slug,
            description=description.strip() if description else None,
            status="active",
            created_by=actor_user_id,
        )
        db.add(department)
        db.flush()
        if workspace_name:
            workspace_key = f"{normalized_slug}-{uuid.uuid4().hex[:8]}"
            db.add(
                Workspace(
                    tenant_id=tenant.id,
                    tenant_key=tenant.tenant_id,
                    department_id=department.id,
                    workspace_key=workspace_key,
                    name=workspace_name.strip(),
                    slug=normalized_slug,
                    status="active",
                    created_by=actor_user_id,
                )
            )
        db.add(
            AuditEvent(
                tenant_id=tenant.id,
                tenant_key=tenant.tenant_id,
                department_id=department.id,
                actor_user_id=actor_user_id,
                action="department.created",
                resource_type="department",
                resource_id=department.id,
                metadata_={"slug": department.slug},
                request_id=request_id,
            )
        )
        db.commit()
        db.refresh(department)
        return department
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A department with this slug already exists in the company.",
        ) from error


def list_departments(
    db: Session,
    tenant_id: uuid.UUID,
    accessible_department_ids: set[uuid.UUID] | None,
    page: int,
    size: int,
) -> tuple[list[Department], int]:
    query = db.query(Department).filter(
        Department.tenant_id == tenant_id,
        Department.status != "archived",
    )
    if accessible_department_ids is not None:
        if not accessible_department_ids:
            return [], 0
        query = query.filter(Department.id.in_(accessible_department_ids))
    total = query.count()
    items = query.order_by(Department.name).offset((page - 1) * size).limit(size).all()
    return items, total


def validate_role_assignment_ceiling(
    db: Session,
    tenant_id: uuid.UUID,
    actor_user_id: uuid.UUID,
    tenant_role: Role,
    department_assignments: list[tuple[Department, Role]],
) -> None:
    actor_membership = (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission)
        )
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == actor_user_id,
            TenantMembership.status == "active",
        )
        .first()
    )
    if actor_membership is None:
        raise HTTPException(
            status_code=403, detail="Active company membership required."
        )

    actor_tenant_permissions = {
        link.permission.code
        for link in actor_membership.role.permission_links
        if link.permission is not None
    }
    target_tenant_permissions = {
        link.permission.code
        for link in tenant_role.permission_links
        if link.permission is not None
    }
    if not target_tenant_permissions.issubset(actor_tenant_permissions):
        raise HTTPException(
            status_code=403,
            detail="You cannot assign a company role with permissions you do not have.",
        )
    if (
        tenant_role.slug == "tenant-owner"
        and actor_membership.role.slug != "tenant-owner"
    ):
        raise HTTPException(
            status_code=403,
            detail="Only a company owner can assign the company owner role.",
        )

    if "tenant.data.manage_all" in actor_tenant_permissions:
        return

    actor_department_memberships = (
        db.query(DepartmentMembership)
        .options(
            selectinload(DepartmentMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission)
        )
        .filter(
            DepartmentMembership.tenant_id == tenant_id,
            DepartmentMembership.user_id == actor_user_id,
            DepartmentMembership.status == "active",
        )
        .all()
    )
    actor_permissions_by_department = {
        membership.department_id: {
            link.permission.code
            for link in membership.role.permission_links
            if link.permission is not None
        }
        for membership in actor_department_memberships
    }
    for department, role in department_assignments:
        target_permissions = {
            link.permission.code
            for link in role.permission_links
            if link.permission is not None
        }
        actor_permissions = actor_permissions_by_department.get(department.id, set())
        if not target_permissions.issubset(actor_permissions):
            raise HTTPException(
                status_code=403,
                detail=(
                    "You cannot assign a department role with permissions you do "
                    "not have in that department."
                ),
            )


def create_invitation(
    db: Session,
    tenant: Tenant,
    invited_by: User,
    email: str,
    username: str | None,
    display_name: str | None,
    tenant_role_id: uuid.UUID,
    assignments: list[DepartmentRoleAssignment],
    personal_message: str | None,
    idempotency_key: str | None,
    request_id: str | None,
) -> tuple[Invitation, str]:
    normalized_email = normalize_email(email)
    normalized_username = normalize_username(username)
    tenant_role = (
        db.query(Role)
        .filter(
            Role.id == tenant_role_id,
            Role.tenant_uuid == tenant.id,
            Role.scope == "tenant",
        )
        .first()
    )
    if tenant_role is None:
        raise HTTPException(status_code=422, detail="Invalid company role.")

    validated_assignments: list[tuple[Department, Role]] = []
    for assignment in assignments:
        department = (
            db.query(Department)
            .filter(
                Department.id == assignment.department_id,
                Department.tenant_id == tenant.id,
                Department.status == "active",
            )
            .first()
        )
        role = (
            db.query(Role)
            .filter(
                Role.id == assignment.role_id,
                Role.tenant_uuid == tenant.id,
                Role.scope == "department",
            )
            .first()
        )
        if department is None or role is None:
            raise HTTPException(
                status_code=422,
                detail="Every department assignment must belong to this company.",
            )
        validated_assignments.append((department, role))

    validate_role_assignment_ceiling(
        db,
        tenant_id=tenant.id,
        actor_user_id=invited_by.id,
        tenant_role=tenant_role,
        department_assignments=validated_assignments,
    )

    existing_membership = (
        db.query(TenantMembership)
        .join(User, User.id == TenantMembership.user_id)
        .filter(
            TenantMembership.tenant_id == tenant.id,
            User.normalized_email == normalized_email,
            TenantMembership.status.in_(["active", "invited"]),
        )
        .first()
    )
    if existing_membership is not None:
        raise HTTPException(
            status_code=409,
            detail="This user is already a member or has an active membership.",
        )

    existing_invitation = (
        db.query(Invitation)
        .filter(
            Invitation.tenant_id == tenant.id,
            Invitation.normalized_email == normalized_email,
            Invitation.status == "pending",
            Invitation.expires_at > datetime.now(timezone.utc),
        )
        .first()
    )
    if existing_invitation is not None:
        raise HTTPException(
            status_code=409,
            detail="A pending invitation already exists. Resend or revoke it.",
        )

    raw_token = secrets.token_urlsafe(48)
    acceptance_url = (
        f"{settings.frontend_public_url.rstrip('/')}/invite?token={raw_token}"
    )
    now = datetime.now(timezone.utc)
    try:
        invitation = Invitation(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            email=normalized_email,
            normalized_email=normalized_email,
            username=normalized_username,
            display_name=display_name.strip() if display_name else None,
            tenant_role_id=tenant_role.id,
            token_hash=hash_invitation_token(raw_token),
            status="pending",
            provisioning_mode="invitation_link",
            invited_by=invited_by.id,
            personal_message=personal_message.strip() if personal_message else None,
            idempotency_key=idempotency_key,
            expires_at=now + timedelta(hours=settings.invite_expiry_hours),
        )
        db.add(invitation)
        db.flush()
        for department, role in validated_assignments:
            db.add(
                InvitationDepartmentAssignment(
                    invitation_id=invitation.id,
                    department_id=department.id,
                    role_id=role.id,
                )
            )

        db.add(
            EmailOutbox(
                tenant_id=tenant.id,
                email_type="invitation",
                recipient=normalized_email,
                subject=f"You are invited to {tenant.name}",
                template_data={
                    "tenant_name": tenant.name,
                    "inviter_name": invited_by.display_name
                    or invited_by.primary_email
                    or "Company administrator",
                    "assignments": [
                        {"department": department.name, "role": role.name}
                        for department, role in validated_assignments
                    ],
                    "expires_at": invitation.expires_at.isoformat(),
                    "personal_message": invitation.personal_message,
                    "encrypted_acceptance_url": encrypt_outbox_value(acceptance_url),
                    "support_email": settings.support_email,
                },
                idempotency_key=f"invitation:{invitation.id}:created",
            )
        )
        db.add(
            AuditEvent(
                tenant_id=tenant.id,
                tenant_key=tenant.tenant_id,
                actor_user_id=invited_by.id,
                action="invitation.created",
                resource_type="invitation",
                resource_id=invitation.id,
                metadata_={
                    "email_hash": hashlib.sha256(
                        normalized_email.encode("utf-8")
                    ).hexdigest(),
                    "department_count": len(validated_assignments),
                },
                request_id=request_id,
            )
        )
        db.commit()
        invitation = (
            db.query(Invitation)
            .options(
                selectinload(Invitation.tenant_role),
                selectinload(Invitation.assignments).selectinload(
                    InvitationDepartmentAssignment.department
                ),
                selectinload(Invitation.assignments).selectinload(
                    InvitationDepartmentAssignment.role
                ),
            )
            .filter(Invitation.id == invitation.id)
            .one()
        )
        logger.info(
            f"[DB] Invitation created invitation={invitation.id} tenant={tenant.id}"
        )
        return invitation, acceptance_url
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="The invitation could not be created because it already exists.",
        ) from error


def resolve_invitation(db: Session, raw_token: str) -> Invitation:
    invitation = (
        db.query(Invitation)
        .options(
            selectinload(Invitation.tenant),
            selectinload(Invitation.tenant_role),
            selectinload(Invitation.assignments).selectinload(
                InvitationDepartmentAssignment.department
            ),
            selectinload(Invitation.assignments).selectinload(
                InvitationDepartmentAssignment.role
            ),
        )
        .filter(Invitation.token_hash == hash_invitation_token(raw_token))
        .first()
    )
    if invitation is None:
        raise HTTPException(status_code=404, detail="Invitation is invalid or expired.")
    if invitation.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Invitation is {invitation.status}.",
        )
    if invitation.expires_at <= datetime.now(timezone.utc):
        invitation.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Invitation has expired.")
    return invitation


def accept_invitation(
    db: Session,
    raw_token: str,
    user: User,
    request_id: str | None,
) -> Invitation:
    normalized_email = normalize_email(user.primary_email or user.email or "")
    invitation = (
        db.query(Invitation)
        .options(
            selectinload(Invitation.tenant),
            selectinload(Invitation.assignments),
        )
        .filter(Invitation.token_hash == hash_invitation_token(raw_token))
        .with_for_update()
        .first()
    )
    if invitation is None:
        raise HTTPException(status_code=404, detail="Invitation is invalid or expired.")
    if invitation.status == "accepted":
        existing_membership = (
            db.query(TenantMembership)
            .filter(
                TenantMembership.tenant_id == invitation.tenant_id,
                TenantMembership.user_id == user.id,
                TenantMembership.status == "active",
            )
            .first()
        )
        if existing_membership is not None:
            return invitation
        raise HTTPException(status_code=409, detail="Invitation has already been used.")
    if invitation.status != "pending":
        raise HTTPException(
            status_code=409, detail=f"Invitation is {invitation.status}."
        )
    if invitation.expires_at <= datetime.now(timezone.utc):
        invitation.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Invitation has expired.")
    if invitation.normalized_email != normalized_email:
        logger.warning(
            f"[AuthZ] Invitation email mismatch invitation={invitation.id} user={user.id}"
        )
        raise HTTPException(
            status_code=403,
            detail="Sign in with the email address that received this invitation.",
        )
    if not user.email_verified:
        raise HTTPException(
            status_code=403,
            detail="Verify the invited email address before accepting the invitation.",
        )

    try:
        now = datetime.now(timezone.utc)
        membership = (
            db.query(TenantMembership)
            .filter(
                TenantMembership.tenant_id == invitation.tenant_id,
                TenantMembership.user_id == user.id,
            )
            .first()
        )
        if membership is None:
            membership = TenantMembership(
                tenant_id=invitation.tenant_id,
                tenant_key=invitation.tenant_key,
                user_id=user.id,
                role_id=invitation.tenant_role_id,
                status="active",
                joined_at=now,
            )
            db.add(membership)
        else:
            membership.role_id = invitation.tenant_role_id
            membership.status = "active"
            membership.joined_at = membership.joined_at or now

        for assignment in invitation.assignments:
            department_membership = (
                db.query(DepartmentMembership)
                .filter(
                    DepartmentMembership.department_id == assignment.department_id,
                    DepartmentMembership.user_id == user.id,
                )
                .first()
            )
            if department_membership is None:
                db.add(
                    DepartmentMembership(
                        tenant_id=invitation.tenant_id,
                        department_id=assignment.department_id,
                        user_id=user.id,
                        role_id=assignment.role_id,
                        status="active",
                        joined_at=now,
                    )
                )
            else:
                department_membership.role_id = assignment.role_id
                department_membership.status = "active"

        invitation.status = "accepted"
        invitation.accepted_at = now
        user.status = "active"
        db.add(
            AuditEvent(
                tenant_id=invitation.tenant_id,
                tenant_key=invitation.tenant_key,
                actor_user_id=user.id,
                target_user_id=user.id,
                action="invitation.accepted",
                resource_type="invitation",
                resource_id=invitation.id,
                metadata_={"department_count": len(invitation.assignments)},
                request_id=request_id,
            )
        )
        db.commit()
        logger.info(
            f"[DB] Invitation accepted invitation={invitation.id} user={user.id}"
        )
        return invitation
    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Invitation finalization conflict invitation={invitation.id}: {error}"
        )
        raise HTTPException(
            status_code=409,
            detail="Invitation acceptance conflicted with another membership update.",
        ) from error


def get_active_tenant(db: Session, tenant_id: uuid.UUID) -> Tenant:
    tenant = (
        db.query(Tenant)
        .filter(Tenant.id == tenant_id, Tenant.status == "active")
        .first()
    )
    if tenant is None:
        raise HTTPException(status_code=404, detail="Company not found.")
    return tenant


def list_tenant_roles(
    db: Session,
    tenant_id: uuid.UUID,
    scope: str | None = None,
) -> list[Role]:
    query = (
        db.query(Role)
        .options(
            selectinload(Role.permission_links).selectinload(RolePermission.permission)
        )
        .filter(Role.tenant_uuid == tenant_id)
    )
    if scope:
        query = query.filter(Role.scope == scope)
    return query.order_by(Role.scope, Role.name).all()


def list_permissions(db: Session) -> list[Permission]:
    return db.query(Permission).order_by(Permission.scope, Permission.code).all()


def create_custom_role(
    db: Session,
    tenant: Tenant,
    actor_user_id: uuid.UUID,
    actor_permissions: set[str],
    name: str,
    slug: str,
    role_scope: str,
    description: str | None,
    permission_codes: list[str],
    request_id: str | None,
) -> Role:
    requested_permissions = set(permission_codes)
    if not requested_permissions.issubset(actor_permissions):
        raise HTTPException(
            status_code=403,
            detail="A role cannot grant permissions the current user does not possess.",
        )
    permissions = (
        db.query(Permission)
        .filter(
            Permission.code.in_(requested_permissions),
            Permission.scope == role_scope,
        )
        .all()
    )
    if len(permissions) != len(requested_permissions):
        raise HTTPException(
            status_code=422,
            detail="One or more permissions are invalid for the selected role scope.",
        )

    try:
        role = Role(
            tenant_id=tenant.tenant_id,
            tenant_uuid=tenant.id,
            name=name.strip(),
            slug=slug.strip().lower(),
            scope=role_scope,
            description=description.strip() if description else None,
            is_system=False,
            is_mutable=True,
            created_by=actor_user_id,
        )
        db.add(role)
        db.flush()
        for permission in permissions:
            db.add(
                RolePermission(
                    role_id=role.id,
                    permission_id=permission.id,
                )
            )
        db.add(
            AuditEvent(
                tenant_id=tenant.id,
                tenant_key=tenant.tenant_id,
                actor_user_id=actor_user_id,
                action="role.created",
                resource_type="role",
                resource_id=role.id,
                metadata_={"scope": role.scope, "permission_count": len(permissions)},
                request_id=request_id,
            )
        )
        db.commit()
        return (
            db.query(Role)
            .options(
                selectinload(Role.permission_links).selectinload(
                    RolePermission.permission
                )
            )
            .filter(Role.id == role.id)
            .one()
        )
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A role with this slug already exists in this scope.",
        ) from error


def update_custom_role(
    db: Session,
    role: Role,
    actor_user_id: uuid.UUID,
    actor_permissions: set[str],
    name: str | None,
    description: str | None,
    permission_codes: list[str] | None,
    request_id: str | None,
) -> Role:
    if not role.is_mutable:
        raise HTTPException(
            status_code=409, detail="Built-in roles cannot be modified."
        )

    if name is not None:
        role.name = name.strip()
    if description is not None:
        role.description = description.strip()
    if permission_codes is not None:
        requested_permissions = set(permission_codes)
        if not requested_permissions.issubset(actor_permissions):
            raise HTTPException(
                status_code=403,
                detail="A role cannot grant permissions the current user does not possess.",
            )
        permissions = (
            db.query(Permission)
            .filter(
                Permission.code.in_(requested_permissions),
                Permission.scope == role.scope,
            )
            .all()
        )
        if len(permissions) != len(requested_permissions):
            raise HTTPException(
                status_code=422,
                detail="One or more permissions are invalid for this role scope.",
            )
        role.permission_links.clear()
        db.flush()
        for permission in permissions:
            role.permission_links.append(
                RolePermission(
                    role_id=role.id,
                    permission_id=permission.id,
                )
            )

    db.add(
        AuditEvent(
            tenant_id=role.tenant_uuid,
            tenant_key=role.tenant_id or "",
            actor_user_id=actor_user_id,
            action="role.updated",
            resource_type="role",
            resource_id=role.id,
            metadata_={"scope": role.scope},
            request_id=request_id,
        )
    )
    db.commit()
    return (
        db.query(Role)
        .options(
            selectinload(Role.permission_links).selectinload(RolePermission.permission)
        )
        .filter(Role.id == role.id)
        .one()
    )


def list_invitations(
    db: Session,
    tenant_id: uuid.UUID,
    page: int,
    size: int,
) -> tuple[list[Invitation], int]:
    query = (
        db.query(Invitation)
        .options(
            selectinload(Invitation.tenant_role),
            selectinload(Invitation.assignments).selectinload(
                InvitationDepartmentAssignment.department
            ),
            selectinload(Invitation.assignments).selectinload(
                InvitationDepartmentAssignment.role
            ),
        )
        .filter(Invitation.tenant_id == tenant_id)
    )
    total = query.count()
    invitations = (
        query.order_by(Invitation.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return invitations, total


def resend_invitation(
    db: Session,
    invitation: Invitation,
    actor_user_id: uuid.UUID,
    request_id: str | None,
) -> tuple[Invitation, str]:
    if invitation.status != "pending":
        raise HTTPException(
            status_code=409,
            detail="Only a pending invitation can be resent.",
        )

    raw_token = secrets.token_urlsafe(48)
    acceptance_url = (
        f"{settings.frontend_public_url.rstrip('/')}/invite?token={raw_token}"
    )
    invitation.token_hash = hash_invitation_token(raw_token)
    invitation.expires_at = datetime.now(timezone.utc) + timedelta(
        hours=settings.invite_expiry_hours
    )
    try:
        db.add(
            EmailOutbox(
                tenant_id=invitation.tenant_id,
                email_type="invitation_resend",
                recipient=invitation.email,
                subject=f"Your invitation to {invitation.tenant.name}",
                template_data={
                    "tenant_name": invitation.tenant.name,
                    "expires_at": invitation.expires_at.isoformat(),
                    "encrypted_acceptance_url": encrypt_outbox_value(acceptance_url),
                    "support_email": settings.support_email,
                },
                idempotency_key=(
                    f"invitation:{invitation.id}:resend:{uuid.uuid4().hex}"
                ),
            )
        )
        db.add(
            AuditEvent(
                tenant_id=invitation.tenant_id,
                tenant_key=invitation.tenant_key,
                actor_user_id=actor_user_id,
                action="invitation.resent",
                resource_type="invitation",
                resource_id=invitation.id,
                metadata_={"expires_at": invitation.expires_at.isoformat()},
                request_id=request_id,
            )
        )
        db.commit()
        db.refresh(invitation)
        return invitation, acceptance_url
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="The invitation could not be resent.",
        ) from error


def revoke_invitation(
    db: Session,
    invitation: Invitation,
    actor_user_id: uuid.UUID,
    request_id: str | None,
) -> Invitation:
    if invitation.status == "accepted":
        raise HTTPException(
            status_code=409, detail="Accepted invitations cannot be revoked."
        )
    if invitation.status == "revoked":
        return invitation

    invitation.status = "revoked"
    invitation.revoked_at = datetime.now(timezone.utc)
    db.add(
        AuditEvent(
            tenant_id=invitation.tenant_id,
            tenant_key=invitation.tenant_key,
            actor_user_id=actor_user_id,
            action="invitation.revoked",
            resource_type="invitation",
            resource_id=invitation.id,
            metadata_={},
            request_id=request_id,
        )
    )
    db.commit()
    db.refresh(invitation)
    return invitation


def get_invitation_for_tenant(
    db: Session,
    tenant_id: uuid.UUID,
    invitation_id: uuid.UUID,
) -> Invitation:
    invitation = (
        db.query(Invitation)
        .options(
            selectinload(Invitation.tenant),
            selectinload(Invitation.tenant_role),
            selectinload(Invitation.assignments).selectinload(
                InvitationDepartmentAssignment.department
            ),
            selectinload(Invitation.assignments).selectinload(
                InvitationDepartmentAssignment.role
            ),
        )
        .filter(
            Invitation.id == invitation_id,
            Invitation.tenant_id == tenant_id,
        )
        .first()
    )
    if invitation is None:
        raise HTTPException(status_code=404, detail="Invitation not found.")
    return invitation


def add_or_update_tenant_member(
    db: Session,
    tenant: Tenant,
    user: User,
    tenant_role_id: uuid.UUID,
    assignments: list[DepartmentRoleAssignment],
    actor_user_id: uuid.UUID,
    must_change_password: bool,
    request_id: str | None,
) -> TenantMembership:
    tenant_role = (
        db.query(Role)
        .filter(
            Role.id == tenant_role_id,
            Role.tenant_uuid == tenant.id,
            Role.scope == "tenant",
        )
        .first()
    )
    if tenant_role is None:
        raise HTTPException(status_code=422, detail="Invalid company role.")

    validated_assignments: list[tuple[Department, Role]] = []
    for assignment in assignments:
        department = (
            db.query(Department)
            .filter(
                Department.id == assignment.department_id,
                Department.tenant_id == tenant.id,
                Department.status == "active",
            )
            .first()
        )
        department_role = (
            db.query(Role)
            .filter(
                Role.id == assignment.role_id,
                Role.tenant_uuid == tenant.id,
                Role.scope == "department",
            )
            .first()
        )
        if department is None or department_role is None:
            raise HTTPException(
                status_code=422,
                detail="Every department assignment must belong to this company.",
            )
        validated_assignments.append((department, department_role))

    validate_role_assignment_ceiling(
        db,
        tenant_id=tenant.id,
        actor_user_id=actor_user_id,
        tenant_role=tenant_role,
        department_assignments=validated_assignments,
    )

    now = datetime.now(timezone.utc)
    membership = (
        db.query(TenantMembership)
        .filter(
            TenantMembership.tenant_id == tenant.id,
            TenantMembership.user_id == user.id,
        )
        .first()
    )
    if membership is None:
        membership = TenantMembership(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            user_id=user.id,
            role_id=tenant_role.id,
            status="active" if not must_change_password else "invited",
            joined_at=now,
        )
        db.add(membership)
    else:
        current_role = db.query(Role).filter(Role.id == membership.role_id).first()
        if (
            current_role is not None
            and current_role.slug == "tenant-owner"
            and tenant_role.slug != "tenant-owner"
        ):
            active_owner_count = (
                db.query(TenantMembership)
                .join(Role, Role.id == TenantMembership.role_id)
                .filter(
                    TenantMembership.tenant_id == tenant.id,
                    TenantMembership.status == "active",
                    Role.slug == "tenant-owner",
                )
                .count()
            )
            if active_owner_count <= 1:
                raise HTTPException(
                    status_code=409,
                    detail="The last active company owner cannot be demoted.",
                )
        membership.role_id = tenant_role.id
        membership.status = "active" if not must_change_password else "invited"

    user.must_change_password = must_change_password
    user.status = "pending" if must_change_password else "active"
    assigned_department_ids = {department.id for department, _ in validated_assignments}
    existing_department_memberships = (
        db.query(DepartmentMembership)
        .filter(
            DepartmentMembership.tenant_id == tenant.id,
            DepartmentMembership.user_id == user.id,
        )
        .all()
    )
    for existing_membership in existing_department_memberships:
        if existing_membership.department_id not in assigned_department_ids:
            existing_membership.status = "removed"

    for department, department_role in validated_assignments:
        department_membership = (
            db.query(DepartmentMembership)
            .filter(
                DepartmentMembership.department_id == department.id,
                DepartmentMembership.user_id == user.id,
            )
            .first()
        )
        if department_membership is None:
            db.add(
                DepartmentMembership(
                    tenant_id=tenant.id,
                    department_id=department.id,
                    user_id=user.id,
                    role_id=department_role.id,
                    status="active",
                    joined_at=now,
                )
            )
        else:
            department_membership.role_id = department_role.id
            department_membership.status = "active"

    db.add(
        AuditEvent(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            actor_user_id=actor_user_id,
            target_user_id=user.id,
            action=(
                "member.temporary_password_provisioned"
                if must_change_password
                else "member.updated"
            ),
            resource_type="user",
            resource_id=user.id,
            metadata_={"department_count": len(validated_assignments)},
            request_id=request_id,
        )
    )
    db.commit()
    return (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.user)
            .selectinload(User.department_memberships)
            .selectinload(DepartmentMembership.department),
            selectinload(TenantMembership.user)
            .selectinload(User.department_memberships)
            .selectinload(DepartmentMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .filter(TenantMembership.id == membership.id)
        .one()
    )


def list_tenant_members(
    db: Session,
    tenant_id: uuid.UUID,
    page: int,
    size: int,
) -> tuple[list[TenantMembership], int]:
    query = (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.user).selectinload(User.identities),
            selectinload(TenantMembership.user)
            .selectinload(User.department_memberships)
            .selectinload(DepartmentMembership.department),
            selectinload(TenantMembership.user)
            .selectinload(User.department_memberships)
            .selectinload(DepartmentMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.status != "removed",
        )
    )
    total = query.count()
    memberships = (
        query.join(User, User.id == TenantMembership.user_id)
        .order_by(User.display_name, User.primary_email)
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return memberships, total


def get_tenant_member(
    db: Session,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
) -> TenantMembership:
    membership = (
        db.query(TenantMembership)
        .options(
            selectinload(TenantMembership.user).selectinload(User.identities),
            selectinload(TenantMembership.user)
            .selectinload(User.department_memberships)
            .selectinload(DepartmentMembership.department),
            selectinload(TenantMembership.user)
            .selectinload(User.department_memberships)
            .selectinload(DepartmentMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
            selectinload(TenantMembership.role)
            .selectinload(Role.permission_links)
            .selectinload(RolePermission.permission),
        )
        .filter(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == user_id,
            TenantMembership.status != "removed",
        )
        .first()
    )
    if membership is None:
        raise HTTPException(status_code=404, detail="Member not found.")
    return membership


def remove_tenant_member(
    db: Session,
    tenant: Tenant,
    member: TenantMembership,
    actor_user_id: uuid.UUID,
    request_id: str | None,
) -> None:
    role = db.query(Role).filter(Role.id == member.role_id).first()
    if role is not None and role.slug == "tenant-owner":
        active_owner_count = (
            db.query(TenantMembership)
            .join(Role, Role.id == TenantMembership.role_id)
            .filter(
                TenantMembership.tenant_id == tenant.id,
                TenantMembership.status == "active",
                Role.slug == "tenant-owner",
            )
            .count()
        )
        if active_owner_count <= 1:
            raise HTTPException(
                status_code=409,
                detail="The last active company owner cannot be removed.",
            )

    member.status = "removed"
    (
        db.query(DepartmentMembership)
        .filter(
            DepartmentMembership.tenant_id == tenant.id,
            DepartmentMembership.user_id == member.user_id,
        )
        .update({DepartmentMembership.status: "removed"}, synchronize_session=False)
    )
    db.add(
        AuditEvent(
            tenant_id=tenant.id,
            tenant_key=tenant.tenant_id,
            actor_user_id=actor_user_id,
            target_user_id=member.user_id,
            action="member.removed",
            resource_type="user",
            resource_id=member.user_id,
            metadata_={},
            request_id=request_id,
        )
    )
    db.commit()


def list_audit_events(
    db: Session,
    tenant_id: uuid.UUID,
    page: int,
    size: int,
) -> tuple[list[AuditEvent], int]:
    query = db.query(AuditEvent).filter(AuditEvent.tenant_id == tenant_id)
    total = query.count()
    events = (
        query.order_by(AuditEvent.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return events, total
