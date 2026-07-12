import logging
from uuid import UUID

from fastapi import HTTPException

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.models import Document
from agentic_rag.shared.schemas.auth import AclDecision, PermissionAction, Visibility


logger = logging.getLogger(__name__)


TENANT_PERMISSION_CODES = frozenset(
    {
        "tenant.view",
        "tenant.update",
        "tenant.archive",
        "tenant.delete",
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
        "tenant.billing.manage",
    }
)

DEPARTMENT_PERMISSION_CODES = frozenset(
    {
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
    }
)

DEFAULT_ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "tenant-owner": TENANT_PERMISSION_CODES,
    "tenant-admin": frozenset(
        {
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
        }
    ),
    "tenant-member": frozenset({"tenant.view", "tenant.departments.view"}),
    "tenant-auditor": frozenset(
        {
            "tenant.view",
            "tenant.members.view",
            "tenant.departments.view",
            "tenant.roles.view",
            "tenant.audit.view",
        }
    ),
    "department-admin": DEPARTMENT_PERMISSION_CODES,
    "editor": frozenset(
        {
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
        }
    ),
    "contributor": frozenset(
        {
            "department.view",
            "workspaces.view",
            "documents.view",
            "documents.upload",
            "collections.view",
            "rag.query",
            "conversations.view",
            "conversations.create",
        }
    ),
    "viewer": frozenset(
        {
            "department.view",
            "workspaces.view",
            "documents.view",
            "collections.view",
            "rag.query",
            "conversations.view",
        }
    ),
    "chat-only": frozenset({"department.view", "rag.query", "conversations.create"}),
}


def require_tenant_permission(
    user_context: UserContext,
    tenant_id: UUID,
    permission_code: str,
) -> UserContext:
    if user_context.must_change_password:
        raise HTTPException(
            status_code=403,
            detail="Change the temporary password before accessing company data.",
        )
    if user_context.tenant_uuid != tenant_id:
        logger.warning(
            f"[AuthZ] Tenant boundary denied user={user_context.id} "
            f"active_tenant={user_context.tenant_uuid} requested_tenant={tenant_id}"
        )
        raise HTTPException(status_code=404, detail="Company not found.")
    if permission_code not in user_context.tenant_permissions:
        logger.warning(
            f"[AuthZ] Tenant permission denied user={user_context.id} "
            f"tenant={tenant_id} permission={permission_code}"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Missing required permission: {permission_code}",
        )
    return user_context


def require_department_permission(
    user_context: UserContext,
    tenant_id: UUID,
    department_id: UUID,
    permission_code: str,
) -> UserContext:
    if user_context.must_change_password:
        raise HTTPException(
            status_code=403,
            detail="Change the temporary password before accessing company data.",
        )
    if user_context.tenant_uuid != tenant_id:
        logger.warning(
            f"[AuthZ] Department tenant boundary denied user={user_context.id} "
            f"active_tenant={user_context.tenant_uuid} requested_tenant={tenant_id}"
        )
        raise HTTPException(status_code=404, detail="Department not found.")
    if user_context.has_department_permission(department_id, permission_code):
        return user_context

    logger.warning(
        f"[AuthZ] Department permission denied user={user_context.id} "
        f"tenant={tenant_id} department={department_id} permission={permission_code}"
    )
    raise HTTPException(status_code=404, detail="Department not found.")


def get_accessible_department_ids(
    user_context: UserContext,
    tenant_id: UUID,
    permission_code: str,
) -> frozenset[UUID]:
    if user_context.tenant_uuid != tenant_id or user_context.must_change_password:
        return frozenset()
    if "tenant.data.manage_all" in user_context.tenant_permissions:
        return frozenset(user_context.accessible_department_ids)
    if (
        permission_code in {"department.view", "documents.view", "rag.query"}
        and "tenant.data.view_all" in user_context.tenant_permissions
    ):
        return frozenset(user_context.accessible_department_ids)
    return frozenset(
        department_id
        for department_id, permissions in user_context.department_permissions.items()
        if permission_code in permissions
    )


def resolve_authorized_department_ids(
    user_context: UserContext,
    requested_department_ids: list[UUID] | None,
    permission_code: str,
) -> frozenset[UUID]:
    requested_ids = frozenset(requested_department_ids or [])
    if user_context.tenant_uuid is None:
        return requested_ids

    allowed_ids = get_accessible_department_ids(
        user_context,
        user_context.tenant_uuid,
        permission_code,
    )
    if requested_ids and not requested_ids.issubset(allowed_ids):
        logger.warning(
            f"[AuthZ] Department filter denied user={user_context.id} "
            f"tenant={user_context.tenant_uuid} permission={permission_code}"
        )
        raise HTTPException(status_code=404, detail="Department not found.")
    return requested_ids or allowed_ids


def get_document_acl_decision(
    user_context: UserContext,
    document: Document,
    action: PermissionAction = PermissionAction.READ,
) -> AclDecision:
    if document.tenant_id != user_context.tenant_id:
        logger.warning(
            f"[AuthZ] Denied document {document.id} action={action} "
            f"user={user_context.id} tenant={user_context.tenant_id} "
            f"document_tenant={document.tenant_id}"
        )
        return AclDecision(
            allowed=False,
            reason="Document belongs to a different tenant.",
            acl_version=document.acl_version,
            denied_by="tenant",
        )

    if user_context.tenant_uuid is not None:
        permission_by_action = {
            PermissionAction.READ: "documents.view",
            PermissionAction.WRITE: "documents.update",
            PermissionAction.DELETE: "documents.delete",
            PermissionAction.ADMIN: "documents.update",
        }
        department_permission = permission_by_action[action]
        if document.department_id is None or not user_context.has_department_permission(
            document.department_id,
            department_permission,
        ):
            logger.warning(
                f"[AuthZ] Denied document {document.id} action={action} "
                f"user={user_context.id} department={document.department_id}"
            )
            return AclDecision(
                allowed=False,
                reason="Document department is not accessible.",
                acl_version=document.acl_version,
                denied_by="department",
            )

    if document.is_deleted and action == PermissionAction.READ:
        logger.warning(
            f"[AuthZ] Denied deleted document {document.id} read "
            f"user={user_context.id} tenant={user_context.tenant_id}"
        )
        return AclDecision(
            allowed=False,
            reason="Document is deleted.",
            acl_version=document.acl_version,
            denied_by="document_status",
        )

    if (
        user_context.workspace_id
        and document.workspace_id
        and user_context.workspace_id != document.workspace_id
    ):
        logger.warning(
            f"[AuthZ] Denied document {document.id} action={action} "
            f"user={user_context.id} workspace={user_context.workspace_id} "
            f"document_workspace={document.workspace_id}"
        )
        return AclDecision(
            allowed=False,
            reason="Document belongs to a different workspace.",
            acl_version=document.acl_version,
            denied_by="workspace",
        )

    document_acl = document.acl
    acl_version = document_acl.acl_version if document_acl else document.acl_version

    if user_context.acl_version < acl_version:
        logger.warning(
            f"[AuthZ] Denied document {document.id} action={action} "
            f"user={user_context.id} stale_acl={user_context.acl_version} "
            f"required_acl={acl_version}"
        )
        return AclDecision(
            allowed=False,
            reason="User ACL context is older than the document ACL version.",
            acl_version=acl_version,
            denied_by="acl_version",
        )

    user_roles = set(user_context.roles or [])
    user_groups = set(user_context.group_ids or [])

    if document_acl:
        if user_context.id in document_acl.denied_user_ids:
            logger.warning(
                f"[AuthZ] Denied document {document.id} action={action} "
                f"user={user_context.id} by user deny ACL"
            )
            return AclDecision(
                allowed=False,
                reason="User is explicitly denied by document ACL.",
                acl_version=acl_version,
                denied_by="user",
            )

        if user_groups.intersection(document_acl.denied_group_ids):
            logger.warning(
                f"[AuthZ] Denied document {document.id} action={action} "
                f"user={user_context.id} by group deny ACL"
            )
            return AclDecision(
                allowed=False,
                reason="User group is explicitly denied by document ACL.",
                acl_version=acl_version,
                denied_by="group",
            )

    if "admin" in user_roles:
        logger.info(
            f"[AuthZ] Allowed document {document.id} action={action} "
            f"user={user_context.id} by admin role"
        )
        return AclDecision(
            allowed=True,
            reason="User has admin role for this tenant.",
            acl_version=acl_version,
        )

    if document.owner_user_id == user_context.id:
        logger.info(
            f"[AuthZ] Allowed document {document.id} action={action} "
            f"user={user_context.id} by ownership"
        )
        return AclDecision(
            allowed=True,
            reason="User owns the document.",
            acl_version=acl_version,
        )

    if action in (PermissionAction.DELETE, PermissionAction.ADMIN):
        logger.warning(
            f"[AuthZ] Denied document {document.id} action={action} "
            f"user={user_context.id}; owner/admin required"
        )
        return AclDecision(
            allowed=False,
            reason="Only the owner or an admin can delete or administer the document.",
            acl_version=acl_version,
            denied_by="action",
        )

    if not document_acl:
        logger.warning(
            f"[AuthZ] Denied document {document.id} action={action} "
            f"user={user_context.id}; missing ACL"
        )
        return AclDecision(
            allowed=False,
            reason="Document has no ACL and user is not the owner.",
            acl_version=acl_version,
            denied_by="acl",
        )

    allowed_by_user = user_context.id in document_acl.allowed_user_ids
    allowed_by_group = bool(user_groups.intersection(document_acl.allowed_group_ids))
    allowed_by_role = bool(user_roles.intersection(document_acl.allowed_roles))

    if allowed_by_user or allowed_by_group or allowed_by_role:
        logger.info(
            f"[AuthZ] Allowed document {document.id} action={action} "
            f"user={user_context.id} by explicit ACL"
        )
        return AclDecision(
            allowed=True,
            reason="User is explicitly allowed by document ACL.",
            acl_version=acl_version,
        )

    visibility = str(document_acl.visibility)

    if action == PermissionAction.READ:
        if visibility in (Visibility.PUBLIC.value, Visibility.TENANT.value):
            logger.info(
                f"[AuthZ] Allowed document {document.id} action={action} "
                f"user={user_context.id} by visibility={visibility}"
            )
            return AclDecision(
                allowed=True,
                reason="Document visibility allows tenant read access.",
                acl_version=acl_version,
            )

        if visibility == Visibility.GROUP.value and document_acl.allowed_group_ids:
            logger.warning(
                f"[AuthZ] Denied document {document.id} action={action} "
                f"user={user_context.id}; group visibility mismatch"
            )
            return AclDecision(
                allowed=False,
                reason="Document is group-visible but user is not in an allowed group.",
                acl_version=acl_version,
                denied_by="group",
            )

    logger.warning(
        f"[AuthZ] Denied document {document.id} action={action} "
        f"user={user_context.id}; no matching ACL rule"
    )
    return AclDecision(
        allowed=False,
        reason="User is not allowed by document ACL.",
        acl_version=acl_version,
        denied_by="acl",
    )


def can_read_document(user_context: UserContext, document: Document) -> bool:
    return get_document_acl_decision(
        user_context=user_context,
        document=document,
        action=PermissionAction.READ,
    ).allowed


def can_write_document(user_context: UserContext, document: Document) -> bool:
    return get_document_acl_decision(
        user_context=user_context,
        document=document,
        action=PermissionAction.WRITE,
    ).allowed


def can_delete_document(user_context: UserContext, document: Document) -> bool:
    return get_document_acl_decision(
        user_context=user_context,
        document=document,
        action=PermissionAction.DELETE,
    ).allowed


def require_document_permission(
    user_context: UserContext,
    document: Document,
    action: PermissionAction = PermissionAction.READ,
) -> Document:
    decision = get_document_acl_decision(
        user_context=user_context,
        document=document,
        action=action,
    )

    if not decision.allowed:
        logger.warning(
            f"[AuthZ] Permission check failed document={document.id} "
            f"action={action} user={user_context.id}: {decision.reason}"
        )
        raise HTTPException(status_code=403, detail=decision.reason)

    logger.info(
        f"[AuthZ] Permission check passed document={document.id} "
        f"action={action} user={user_context.id}"
    )
    return document
