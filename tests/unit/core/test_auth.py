import asyncio

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from agentic_rag.core.auth import _verify_auth0_token, get_current_user
from agentic_rag.core.dependencies import require_role, require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.models import (
    Group,
    Role,
    Tenant,
    User,
    UserGroup,
    UserRole,
)


AUTH0_SIGNING_KEY = {
    "kid": "test-key",
    "kty": "RSA",
    "use": "sig",
    "n": "modulus",
    "e": "AQAB",
}


def _build_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/me")
    async def read_me(user_context: UserContext = Depends(get_current_user)) -> dict:
        return {
            "user_id": user_context.id,
            "tenant_id": user_context.tenant_id,
            "roles": user_context.roles,
            "scopes": user_context.scopes,
        }

    @app.get("/admin")
    async def read_admin(
        user_context: UserContext = Depends(require_role("admin")),
    ) -> dict:
        return {"user_id": user_context.id}

    @app.get("/documents")
    async def read_documents(
        user_context: UserContext = Depends(require_scope("documents:read")),
    ) -> dict:
        return {"tenant_id": user_context.tenant_id}

    @app.get("/blocked")
    async def read_blocked(
        user_context: UserContext = Depends(require_scope("blocked:scope")),
    ) -> dict:
        return {"tenant_id": user_context.tenant_id}

    return app


def configure_auth0_token_validation(monkeypatch, payload: dict) -> None:
    async def load_signing_keys(force_refresh: bool = False) -> list[dict]:
        return [AUTH0_SIGNING_KEY]

    monkeypatch.setattr(settings, "auth_provider", "auth0")
    monkeypatch.setattr(settings, "auth0_domain", "identity.example.com")
    monkeypatch.setattr(settings, "auth0_api_audience", "https://api.example.com/")
    monkeypatch.setattr(
        "agentic_rag.core.auth._load_auth0_jwks",
        load_signing_keys,
    )
    monkeypatch.setattr(
        "agentic_rag.core.auth.jwt.get_unverified_header",
        lambda token: {"kid": "test-key", "alg": "RS256"},
    )
    monkeypatch.setattr(
        "agentic_rag.core.auth.jwt.decode",
        lambda token, **kwargs: payload,
    )


def create_auth_test_db() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine)


@pytest.fixture
def local_auth(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_provider", "local")


def test_missing_token_is_rejected(local_auth) -> None:
    response = TestClient(_build_test_app()).get("/me")

    assert response.status_code == 401


def test_local_token_builds_user_context(local_auth) -> None:
    response = TestClient(_build_test_app()).get(
        "/me",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == "local-user"
    assert response.json()["tenant_id"] == "local-tenant"
    assert "admin" in response.json()["roles"]


def test_required_role_allows_matching_user(local_auth) -> None:
    response = TestClient(_build_test_app()).get(
        "/admin",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 200


def test_required_scope_allows_matching_user(local_auth) -> None:
    response = TestClient(_build_test_app()).get(
        "/documents",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "local-tenant"


def test_required_scope_rejects_missing_scope(local_auth) -> None:
    response = TestClient(_build_test_app()).get(
        "/blocked",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 403


def test_auth0_token_returns_identity_without_authorization_claims(monkeypatch) -> None:
    configure_auth0_token_validation(
        monkeypatch,
        {
            "sub": "auth0|identity-user-1",
            "org_id": "org_tenant_a",
            "email": "Member@Example.com",
            "email_verified": True,
            "https://agentic-rag.ai/roles": ["untrusted-token-role"],
            "https://agentic-rag.ai/groups": ["engineering"],
            "permissions": ["untrusted:permission"],
            "https://agentic-rag.ai/acl_version": 4,
        },
    )

    user_context = asyncio.run(
        _verify_auth0_token("header.payload.signature")
    )

    assert user_context.id == "auth0|identity-user-1"
    assert user_context.identity_organization_id == "org_tenant_a"
    assert user_context.email == "member@example.com"
    assert user_context.email_verified is True
    assert user_context.group_ids == []
    assert user_context.roles == []
    assert user_context.scopes == []
    assert user_context.acl_version == 1


def test_auth0_token_without_organization_is_rejected(monkeypatch) -> None:
    configure_auth0_token_validation(
        monkeypatch,
        {"sub": "auth0|identity-user-1"},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_verify_auth0_token("header.payload.signature"))

    assert exc_info.value.status_code == 401
    assert "organization" in str(exc_info.value.detail).lower()


def test_auth0_membership_role_is_authoritative_and_invitation_activates(
    monkeypatch,
) -> None:
    db = create_auth_test_db()
    tenant = Tenant(
        tenant_id="tenant-a",
        name="Tenant A",
        slug="tenant-a",
        status="active",
        identity_provider="auth0",
        external_organization_id="org_tenant_a",
        metadata_={},
    )
    role = Role(
        tenant_id="tenant-a",
        name="viewer",
        is_system=True,
    )
    group = Group(
        tenant_id="tenant-a",
        name="security-reviewers",
    )
    user = User(
        tenant_id="tenant-a",
        external_subject="auth0|identity-user-1",
        email="viewer@example.com",
        status="invited",
        acl_version=3,
        metadata_={"workspace_id": "workspace-a"},
    )
    db.add_all([tenant, role, group, user])
    db.flush()
    db.add(
        UserRole(
            tenant_id="tenant-a",
            user_id=user.id,
            role_id=role.id,
        )
    )
    db.add(
        UserGroup(
            tenant_id="tenant-a",
            user_id=user.id,
            group_id=group.id,
        )
    )
    db.commit()
    configure_auth0_token_validation(
        monkeypatch,
        {
            "sub": "auth0|identity-user-1",
            "org_id": "org_tenant_a",
            "https://agentic-rag.ai/roles": ["admin"],
            "https://agentic-rag.ai/groups": ["untrusted-token-group"],
            "permissions": ["documents:delete"],
        },
    )

    try:
        user_context = asyncio.run(
            get_current_user(
                request=Request({"type": "http", "headers": []}),
                token="header.payload.signature",
                db=db,
            )
        )

        assert user_context.tenant_id == "tenant-a"
        assert user_context.identity_organization_id == "org_tenant_a"
        assert user_context.roles == ["viewer"]
        assert user_context.group_ids == ["security-reviewers"]
        assert user_context.scopes == ["documents:read", "query:run"]
        assert user_context.workspace_id == "workspace-a"
        assert user_context.acl_version == 3
        assert "documents:delete" not in user_context.scopes
        db.refresh(user)
        assert user.status == "active"
    finally:
        db.close()


def test_auth0_first_login_binds_verified_invited_email(monkeypatch) -> None:
    db = create_auth_test_db()
    tenant = Tenant(
        tenant_id="tenant-a",
        name="Tenant A",
        slug="tenant-a",
        status="active",
        identity_provider="auth0",
        external_organization_id="org_tenant_a",
        metadata_={},
    )
    role = Role(tenant_id="tenant-a", name="user", is_system=True)
    invited_user = User(
        tenant_id="tenant-a",
        external_subject="auth0-invitation:uinv_123",
        email="member@example.com",
        status="invited",
        acl_version=1,
        metadata_={"identity_invitation_id": "uinv_123"},
    )
    db.add_all([tenant, role, invited_user])
    db.flush()
    db.add(
        UserRole(
            tenant_id="tenant-a",
            user_id=invited_user.id,
            role_id=role.id,
        )
    )
    db.commit()
    configure_auth0_token_validation(
        monkeypatch,
        {
            "sub": "github|accepted-user",
            "org_id": "org_tenant_a",
            "email": "Member@Example.com",
            "email_verified": True,
        },
    )

    try:
        user_context = asyncio.run(
            get_current_user(
                Request({"type": "http", "headers": []}),
                "header.payload.signature",
                db,
            )
        )

        assert user_context.id == "github|accepted-user"
        assert user_context.tenant_id == "tenant-a"
        assert user_context.roles == ["user"]
        db.refresh(invited_user)
        assert invited_user.external_subject == "github|accepted-user"
        assert invited_user.status == "active"
        assert invited_user.metadata_["invitation_accepted"] is True
    finally:
        db.close()


def test_auth0_unknown_organization_is_rejected(monkeypatch) -> None:
    db = create_auth_test_db()
    configure_auth0_token_validation(
        monkeypatch,
        {
            "sub": "auth0|identity-user-1",
            "org_id": "org_unknown",
        },
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                get_current_user(
                    Request({"type": "http", "headers": []}),
                    "header.payload.signature",
                    db,
                )
            )
        assert exc_info.value.status_code == 403
        assert "not provisioned" in str(exc_info.value.detail)
    finally:
        db.close()
