from fastapi import HTTPException

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.models import Document
from agentic_rag.shared.schemas.auth import AclDecision, PermissionAction, Visibility


def get_document_acl_decision(
    user_context: UserContext,
    document: Document,
    action: PermissionAction = PermissionAction.READ,
) -> AclDecision:
    if document.tenant_id != user_context.tenant_id:
        return AclDecision(
            allowed=False,
            reason="Document belongs to a different tenant.",
            acl_version=document.acl_version,
            denied_by="tenant",
        )

    if document.is_deleted:
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
        return AclDecision(
            allowed=False,
            reason="Document belongs to a different workspace.",
            acl_version=document.acl_version,
            denied_by="workspace",
        )

    document_acl = document.acl
    acl_version = document_acl.acl_version if document_acl else document.acl_version

    if user_context.acl_version < acl_version:
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
            return AclDecision(
                allowed=False,
                reason="User is explicitly denied by document ACL.",
                acl_version=acl_version,
                denied_by="user",
            )

        if user_groups.intersection(document_acl.denied_group_ids):
            return AclDecision(
                allowed=False,
                reason="User group is explicitly denied by document ACL.",
                acl_version=acl_version,
                denied_by="group",
            )

    if "admin" in user_roles:
        return AclDecision(
            allowed=True,
            reason="User has admin role for this tenant.",
            acl_version=acl_version,
        )

    if document.owner_user_id == user_context.id:
        return AclDecision(
            allowed=True,
            reason="User owns the document.",
            acl_version=acl_version,
        )

    if action in (PermissionAction.DELETE, PermissionAction.ADMIN):
        return AclDecision(
            allowed=False,
            reason="Only the owner or an admin can delete or administer the document.",
            acl_version=acl_version,
            denied_by="action",
        )

    if not document_acl:
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
        return AclDecision(
            allowed=True,
            reason="User is explicitly allowed by document ACL.",
            acl_version=acl_version,
        )

    visibility = str(document_acl.visibility)

    if action == PermissionAction.READ:
        if visibility in (Visibility.PUBLIC.value, Visibility.TENANT.value):
            return AclDecision(
                allowed=True,
                reason="Document visibility allows tenant read access.",
                acl_version=acl_version,
            )

        if visibility == Visibility.GROUP.value and document_acl.allowed_group_ids:
            return AclDecision(
                allowed=False,
                reason="Document is group-visible but user is not in an allowed group.",
                acl_version=acl_version,
                denied_by="group",
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
        raise HTTPException(status_code=403, detail=decision.reason)

    return document
