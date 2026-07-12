import logging
from typing import Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from agentic_rag.shared.db.models import Role, Tenant, User, UserGroup, UserRole


logger = logging.getLogger(__name__)


def get_tenant_by_identity_organization(
    db: Session,
    identity_provider: str,
    external_organization_id: str,
) -> Optional[Tenant]:
    logger.info(
        f"[DB] Resolving identity organization provider={identity_provider} "
        f"organization={external_organization_id}"
    )
    return (
        db.query(Tenant)
        .filter(
            Tenant.identity_provider == identity_provider,
            Tenant.external_organization_id == external_organization_id,
            Tenant.status == "active",
        )
        .first()
    )


def get_tenant_user_by_subject(
    db: Session,
    tenant_id: str,
    external_subject: str,
) -> Optional[User]:
    logger.info(
        f"[DB] Fetching tenant user tenant={tenant_id} subject={external_subject}"
    )
    return (
        db.query(User)
        .options(
            selectinload(User.role_links).selectinload(UserRole.role),
            selectinload(User.group_links).selectinload(UserGroup.group),
        )
        .filter(
            User.tenant_id == tenant_id,
            User.external_subject == external_subject,
        )
        .first()
    )


def get_tenant_user_by_email(
    db: Session,
    tenant_id: str,
    email: str,
) -> Optional[User]:
    normalized_email = email.strip().lower()
    logger.info(f"[DB] Fetching tenant user tenant={tenant_id} by email")
    return (
        db.query(User)
        .options(
            selectinload(User.role_links).selectinload(UserRole.role),
            selectinload(User.group_links).selectinload(UserGroup.group),
        )
        .filter(
            User.tenant_id == tenant_id,
            User.email == normalized_email,
        )
        .first()
    )


def list_tenant_users(
    db: Session,
    tenant_id: str,
    page: int = 1,
    size: int = 50,
) -> Tuple[list[User], int]:
    logger.info(
        f"[DB] Listing tenant users tenant={tenant_id} page={page} size={size}"
    )
    query = (
        db.query(User)
        .options(
            selectinload(User.role_links).selectinload(UserRole.role),
            selectinload(User.group_links).selectinload(UserGroup.group),
        )
        .filter(User.tenant_id == tenant_id)
    )
    total = query.count()
    users = (
        query.order_by(User.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return users, total


def create_invited_tenant_user(
    db: Session,
    tenant_id: str,
    external_subject: str,
    email: str,
    display_name: str | None,
    role_name: str,
    workspace_id: str | None,
    invited_by: str,
    identity_invitation_id: str,
) -> User:
    normalized_email = email.strip().lower()
    tenant = (
        db.query(Tenant)
        .filter(
            Tenant.tenant_id == tenant_id,
            Tenant.status == "active",
        )
        .first()
    )
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The current tenant is not provisioned or active.",
        )

    existing_user = (
        db.query(User)
        .filter(
            User.tenant_id == tenant_id,
            (
                (User.external_subject == external_subject)
                | (User.email == normalized_email)
            ),
        )
        .first()
    )
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email is already a member of the tenant.",
        )

    try:
        logger.info(
            f"[DB] Creating invited tenant user tenant={tenant_id} "
            f"role={role_name} invited_by={invited_by}"
        )
        role = (
            db.query(Role)
            .filter(
                Role.tenant_id == tenant_id,
                Role.name == role_name,
            )
            .first()
        )
        if role is None:
            role = Role(
                tenant_id=tenant_id,
                name=role_name,
                description=f"Built-in {role_name} tenant role.",
                is_system=True,
            )
            db.add(role)
            db.flush()

        metadata = {
            "invitation_source": "admin_portal",
            "invited_by": invited_by,
            "identity_invitation_id": identity_invitation_id,
        }
        if workspace_id:
            metadata["workspace_id"] = workspace_id

        user = User(
            tenant_id=tenant_id,
            external_subject=external_subject,
            email=normalized_email,
            display_name=display_name.strip() if display_name else None,
            status="invited",
            acl_version=1,
            metadata_=metadata,
        )
        db.add(user)
        db.flush()
        db.add(
            UserRole(
                tenant_id=tenant_id,
                user_id=user.id,
                role_id=role.id,
            )
        )
        db.commit()

        invited_user = get_tenant_user_by_subject(
            db,
            tenant_id=tenant_id,
            external_subject=external_subject,
        )
        if invited_user is None:
            raise RuntimeError("Invited user could not be loaded after creation.")

        logger.info(
            f"[DB] Created invited tenant user user={invited_user.id} "
            f"tenant={tenant_id} role={role_name}"
        )
        return invited_user

    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to create invited tenant user tenant={tenant_id}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The invited user or tenant role already exists.",
        ) from error
    except Exception as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to create invited tenant user tenant={tenant_id}: {error}"
        )
        raise


def activate_tenant_user(db: Session, user: User) -> User:
    if user.status == "active":
        return user

    if user.status != "invited":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant membership is not active.",
        )

    try:
        logger.info(
            f"[DB] Activating invited tenant user user={user.id} "
            f"tenant={user.tenant_id}"
        )
        user.status = "active"
        db.commit()
        db.refresh(user)
        _ = user.role_links
        return user

    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to activate tenant user user={user.id}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tenant membership could not be activated.",
        ) from error


def bind_invited_tenant_user_identity(
    db: Session,
    user: User,
    external_subject: str,
    display_name: str | None,
) -> User:
    if user.status != "invited":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant membership is not awaiting invitation acceptance.",
        )

    existing_subject = (
        db.query(User)
        .filter(
            User.tenant_id == user.tenant_id,
            User.external_subject == external_subject,
            User.id != user.id,
        )
        .first()
    )
    if existing_subject is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This identity is already assigned to another tenant membership.",
        )

    try:
        logger.info(
            f"[DB] Binding invited tenant user user={user.id} "
            f"tenant={user.tenant_id} subject={external_subject}"
        )
        user.external_subject = external_subject
        user.status = "active"
        if display_name and not user.display_name:
            user.display_name = display_name
        metadata = dict(user.metadata_)
        metadata["invitation_accepted"] = True
        user.metadata_ = metadata
        db.commit()

        activated_user = get_tenant_user_by_subject(
            db,
            tenant_id=user.tenant_id,
            external_subject=external_subject,
        )
        if activated_user is None:
            raise RuntimeError("Activated tenant membership could not be loaded.")
        return activated_user

    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to bind invited identity user={user.id} "
            f"tenant={user.tenant_id}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The signed-in identity could not be assigned to this invitation.",
        ) from error


def delete_incomplete_tenant_user(db: Session, user: User) -> None:
    if user.status not in {"invited", "pending"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only an incomplete tenant membership can be removed by rollback.",
        )

    try:
        logger.warning(
            f"[DB] Removing incomplete tenant user user={user.id} "
            f"tenant={user.tenant_id}"
        )
        db.delete(user)
        db.commit()
    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to remove incomplete tenant user user={user.id}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Incomplete tenant membership could not be removed.",
        ) from error
