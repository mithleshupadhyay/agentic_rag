import pytest
from fastapi import HTTPException

from agentic_rag.core.authorization import (
    can_delete_document,
    can_read_document,
    can_write_document,
    get_document_acl_decision,
    require_document_permission,
)
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.models import Document, DocumentAcl
from agentic_rag.shared.schemas.auth import PermissionAction, Visibility


def build_user_context(
    user_id: str = "user-1",
    tenant_id: str = "tenant-a",
    workspace_id: str | None = None,
    roles: list[str] | None = None,
    group_ids: list[str] | None = None,
    acl_version: int = 1,
) -> UserContext:
    return UserContext(
        id=user_id,
        customer_id=tenant_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        roles=roles or [],
        group_ids=group_ids or [],
        scopes=[],
        acl_version=acl_version,
    )


def build_document(
    tenant_id: str = "tenant-a",
    workspace_id: str | None = None,
    owner_user_id: str = "owner-1",
    visibility: Visibility = Visibility.PRIVATE,
    allowed_user_ids: list[str] | None = None,
    allowed_group_ids: list[str] | None = None,
    allowed_roles: list[str] | None = None,
    denied_user_ids: list[str] | None = None,
    denied_group_ids: list[str] | None = None,
    acl_version: int = 1,
    is_deleted: bool = False,
) -> Document:
    document = Document(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_type="upload",
        title="Security policy",
        status="ready",
        owner_user_id=owner_user_id,
        acl_version=acl_version,
        classification_level="internal",
        metadata_={},
        is_deleted=is_deleted,
    )
    document.acl = DocumentAcl(
        tenant_id=tenant_id,
        document=document,
        visibility=visibility.value,
        allowed_user_ids=allowed_user_ids or [],
        allowed_group_ids=allowed_group_ids or [],
        allowed_roles=allowed_roles or [],
        denied_user_ids=denied_user_ids or [],
        denied_group_ids=denied_group_ids or [],
        acl_version=acl_version,
    )
    return document


def test_document_acl_denies_cross_tenant_access() -> None:
    user_context = build_user_context(tenant_id="tenant-b")
    document = build_document(tenant_id="tenant-a", visibility=Visibility.PUBLIC)

    decision = get_document_acl_decision(user_context, document)

    assert decision.allowed is False
    assert decision.denied_by == "tenant"


def test_document_owner_can_read_write_and_delete_private_document() -> None:
    user_context = build_user_context(user_id="owner-1")
    document = build_document(owner_user_id="owner-1")

    assert can_read_document(user_context, document) is True
    assert can_write_document(user_context, document) is True
    assert can_delete_document(user_context, document) is True


def test_document_acl_allows_user_group_and_role_for_read_and_write() -> None:
    document = build_document(
        allowed_user_ids=["user-1"],
        allowed_group_ids=["security"],
        allowed_roles=["manager"],
    )

    user_allowed_by_id = build_user_context(user_id="user-1")
    user_allowed_by_group = build_user_context(
        user_id="user-2",
        group_ids=["security"],
    )
    user_allowed_by_role = build_user_context(
        user_id="user-3",
        roles=["manager"],
    )

    assert can_read_document(user_allowed_by_id, document) is True
    assert can_write_document(user_allowed_by_id, document) is True
    assert can_read_document(user_allowed_by_group, document) is True
    assert can_write_document(user_allowed_by_group, document) is True
    assert can_read_document(user_allowed_by_role, document) is True
    assert can_write_document(user_allowed_by_role, document) is True


def test_document_acl_denies_user_and_group_before_admin_or_allow_rules() -> None:
    denied_user_context = build_user_context(
        user_id="user-1",
        roles=["admin"],
        group_ids=["security"],
    )
    denied_group_context = build_user_context(
        user_id="user-2",
        group_ids=["blocked"],
    )
    document = build_document(
        allowed_group_ids=["security", "blocked"],
        denied_user_ids=["user-1"],
        denied_group_ids=["blocked"],
    )

    user_decision = get_document_acl_decision(denied_user_context, document)
    group_decision = get_document_acl_decision(denied_group_context, document)

    assert user_decision.allowed is False
    assert user_decision.denied_by == "user"
    assert group_decision.allowed is False
    assert group_decision.denied_by == "group"


def test_tenant_visibility_allows_read_but_not_write_or_delete() -> None:
    user_context = build_user_context(user_id="user-1")
    document = build_document(visibility=Visibility.TENANT)

    assert can_read_document(user_context, document) is True
    assert can_write_document(user_context, document) is False
    assert can_delete_document(user_context, document) is False


def test_admin_can_read_write_and_delete_inside_tenant() -> None:
    user_context = build_user_context(user_id="admin-1", roles=["admin"])
    document = build_document(owner_user_id="owner-1")

    assert can_read_document(user_context, document) is True
    assert can_write_document(user_context, document) is True
    assert can_delete_document(user_context, document) is True


def test_workspace_mismatch_is_denied() -> None:
    user_context = build_user_context(workspace_id="workspace-b")
    document = build_document(workspace_id="workspace-a", visibility=Visibility.TENANT)

    decision = get_document_acl_decision(user_context, document)

    assert decision.allowed is False
    assert decision.denied_by == "workspace"


def test_stale_acl_context_is_denied() -> None:
    user_context = build_user_context(acl_version=1, roles=["admin"])
    document = build_document(acl_version=2)

    decision = get_document_acl_decision(user_context, document)

    assert decision.allowed is False
    assert decision.denied_by == "acl_version"


def test_require_document_permission_returns_document_or_raises() -> None:
    allowed_context = build_user_context(user_id="owner-1")
    denied_context = build_user_context(user_id="user-2")
    document = build_document(owner_user_id="owner-1")

    assert (
        require_document_permission(
            allowed_context,
            document,
            PermissionAction.READ,
        )
        is document
    )

    with pytest.raises(HTTPException) as exc_info:
        require_document_permission(
            denied_context,
            document,
            PermissionAction.READ,
        )

    assert exc_info.value.status_code == 403
