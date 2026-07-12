from collections.abc import Iterator
from unittest.mock import AsyncMock

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from agentic_rag.api import admin
from agentic_rag.core.auth import get_current_user
from agentic_rag.core.auth0_management import Auth0OrganizationInvitation
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.models import Tenant, User
from agentic_rag.shared.db.session import get_session


def create_test_db() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add_all(
        [
            Tenant(
                tenant_id="tenant-a",
                name="Tenant A",
                slug="tenant-a",
                status="active",
                identity_provider="auth0",
                external_organization_id="org_tenant_a",
                metadata_={},
            ),
            Tenant(
                tenant_id="tenant-b",
                name="Tenant B",
                slug="tenant-b",
                status="active",
                identity_provider="auth0",
                external_organization_id="org_tenant_b",
                metadata_={},
            ),
        ]
    )
    session.commit()
    return session


def client_with_user(user_context: UserContext, db: Session) -> Iterator[TestClient]:
    async def override_get_current_user() -> UserContext:
        return user_context

    def override_get_session() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_session] = override_get_session
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def build_user_context(role: str = "admin") -> UserContext:
    return UserContext(
        id="auth0|admin-subject",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        identity_organization_id="org_tenant_a",
        roles=[role],
        scopes=["documents:read"],
        acl_version=1,
    )


def test_admin_invites_tenant_user_and_persists_role(monkeypatch) -> None:
    db = create_test_db()
    create_invitation = AsyncMock(
        return_value=Auth0OrganizationInvitation(
            id="uinv_123",
            organization_id="org_tenant_a",
            email="new.user@example.com",
            email_sent=True,
        )
    )
    delete_invitation = AsyncMock(return_value=None)
    monkeypatch.setattr(
        admin,
        "create_auth0_organization_invitation",
        create_invitation,
    )
    monkeypatch.setattr(
        admin,
        "delete_auth0_organization_invitation",
        delete_invitation,
    )

    try:
        for client in client_with_user(build_user_context(), db):
            response = client.post(
                "/admin/users/invitations",
                headers={"Authorization": "Bearer test-token"},
                json={
                    "email": "New.User@Example.com",
                    "display_name": "New User",
                    "role": "user",
                    "workspace_id": "workspace-a",
                },
            )

        assert response.status_code == 201
        payload = response.json()
        assert payload["invitation_email_sent"] is True
        assert payload["identity_invitation_id"] == "uinv_123"
        assert payload["user"]["tenant_id"] == "tenant-a"
        assert payload["user"]["email"] == "new.user@example.com"
        assert payload["user"]["roles"] == ["user"]
        assert payload["user"]["workspace_id"] == "workspace-a"
        assert payload["user"]["status"] == "invited"
        create_invitation.assert_awaited_once_with(
            organization_id="org_tenant_a",
            email="new.user@example.com",
            display_name="New User",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            role_name="user",
        )
        delete_invitation.assert_not_awaited()

        stored_user = (
            db.query(User)
            .filter(
                User.tenant_id == "tenant-a",
                User.external_subject == "auth0-invitation:uinv_123",
            )
            .one()
        )
        assert stored_user.email == "new.user@example.com"
        assert stored_user.status == "invited"
        assert stored_user.metadata_["workspace_id"] == "workspace-a"
        assert stored_user.metadata_["identity_invitation_id"] == "uinv_123"
        assert stored_user.role_links[0].role.name == "user"
    finally:
        db.close()


def test_admin_user_list_is_tenant_scoped(monkeypatch) -> None:
    db = create_test_db()
    db.add(
        User(
            tenant_id="tenant-b",
            external_subject="auth0|identity-b",
            email="member-b@example.com",
            status="active",
            acl_version=1,
            metadata_={},
        )
    )
    db.commit()
    monkeypatch.setattr(
        admin,
        "create_auth0_organization_invitation",
        AsyncMock(
            return_value=Auth0OrganizationInvitation(
                id="uinv_a",
                organization_id="org_tenant_a",
                email="member-a@example.com",
                email_sent=True,
            )
        ),
    )
    monkeypatch.setattr(
        admin,
        "delete_auth0_organization_invitation",
        AsyncMock(return_value=None),
    )

    try:
        for client in client_with_user(build_user_context(), db):
            invite_response = client.post(
                "/admin/users/invitations",
                headers={"Authorization": "Bearer test-token"},
                json={"email": "member-a@example.com", "role": "viewer"},
            )
            list_response = client.get(
                "/admin/users",
                headers={"Authorization": "Bearer test-token"},
            )

        assert invite_response.status_code == 201
        assert list_response.status_code == 200
        assert list_response.json()["page"]["total"] == 1
        assert db.query(User).count() == 2
        assert [item["email"] for item in list_response.json()["items"]] == [
            "member-a@example.com"
        ]
    finally:
        db.close()


def test_non_admin_cannot_list_or_invite_users() -> None:
    db = create_test_db()
    try:
        for client in client_with_user(build_user_context(role="user"), db):
            list_response = client.get(
                "/admin/users",
                headers={"Authorization": "Bearer test-token"},
            )
            invite_response = client.post(
                "/admin/users/invitations",
                headers={"Authorization": "Bearer test-token"},
                json={"email": "member@example.com", "role": "user"},
            )

        assert list_response.status_code == 403
        assert invite_response.status_code == 403
    finally:
        db.close()


def test_database_invitation_failure_revokes_auth0_invitation(monkeypatch) -> None:
    db = create_test_db()
    delete_invitation = AsyncMock(return_value=None)

    def fail_membership_creation(*args, **kwargs):
        raise HTTPException(status_code=409, detail="Database conflict")

    monkeypatch.setattr(
        admin,
        "create_auth0_organization_invitation",
        AsyncMock(
            return_value=Auth0OrganizationInvitation(
                id="uinv_rollback",
                organization_id="org_tenant_a",
                email="rollback@example.com",
                email_sent=True,
            )
        ),
    )
    monkeypatch.setattr(
        admin,
        "create_invited_tenant_user",
        fail_membership_creation,
    )
    monkeypatch.setattr(
        admin,
        "delete_auth0_organization_invitation",
        delete_invitation,
    )

    try:
        for client in client_with_user(build_user_context(), db):
            response = client.post(
                "/admin/users/invitations",
                headers={"Authorization": "Bearer test-token"},
                json={"email": "rollback@example.com", "role": "admin"},
            )

        assert response.status_code == 409
        assert db.query(User).filter(User.tenant_id == "tenant-a").count() == 0
        delete_invitation.assert_awaited_once_with(
            "org_tenant_a",
            "uinv_rollback",
        )
    finally:
        db.close()
