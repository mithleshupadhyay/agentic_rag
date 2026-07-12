from collections.abc import Iterator

from fastapi.testclient import TestClient

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.config import settings


def client_with_user(user_context: UserContext) -> Iterator[TestClient]:
    async def override_get_current_user() -> UserContext:
        return user_context

    app.dependency_overrides[get_current_user] = override_get_current_user
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_auth_config_returns_local_mode_without_exposing_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_provider", "local")

    response = TestClient(app).get("/auth/config")

    assert response.status_code == 200
    assert response.json() == {
        "mode": "local",
        "provider": "local",
        "issuer_url": None,
        "audience": None,
        "client_id": None,
        "scope": "openid profile email",
        "identity_connections": [],
        "api_base_path": "/auth",
        "website_base_path": "/auth",
        "public_tenant_signup_enabled": False,
        "social_providers": [],
    }
    assert settings.local_auth_token not in response.text


def test_auth_config_returns_public_auth0_configuration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_provider", "auth0")
    monkeypatch.setattr(settings, "auth0_domain", "tenant.eu.auth0.com")
    monkeypatch.setattr(
        settings,
        "auth0_api_audience",
        "https://agentic-rag.example.com/api",
    )
    monkeypatch.setattr(settings, "auth0_frontend_client_id", "spa-client-id")
    monkeypatch.setattr(settings, "auth0_frontend_scope", "openid profile email")
    monkeypatch.setattr(
        settings,
        "auth0_identity_connections_csv",
        "google-oauth2, github",
    )
    monkeypatch.setattr(
        settings,
        "auth0_management_client_secret",
        "must-not-be-public",
    )

    response = TestClient(app).get("/auth/config")

    assert response.status_code == 200
    assert response.json() == {
        "mode": "auth0",
        "provider": "auth0",
        "issuer_url": "https://tenant.eu.auth0.com/",
        "audience": "https://agentic-rag.example.com/api",
        "client_id": "spa-client-id",
        "scope": "openid profile email",
        "identity_connections": ["google-oauth2", "github"],
        "api_base_path": "/auth",
        "website_base_path": "/auth",
        "public_tenant_signup_enabled": False,
        "social_providers": [],
    }
    assert "must-not-be-public" not in response.text


def test_auth_config_rejects_incomplete_auth0_configuration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_provider", "auth0")
    monkeypatch.setattr(settings, "auth0_domain", "")
    monkeypatch.setattr(settings, "auth0_api_audience", "")
    monkeypatch.setattr(settings, "auth0_frontend_client_id", "")

    response = TestClient(app).get("/auth/config")

    assert response.status_code == 503
    assert response.json()["detail"] == "Auth0 frontend configuration is incomplete."


def test_auth_session_returns_authoritative_tenant_context() -> None:
    user_context = UserContext(
        id="auth0|user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        email="member@example.com",
        email_verified=True,
        roles=["user", "admin", "user"],
        group_ids=["engineering", "engineering"],
        scopes=["query:run", "documents:read", "query:run"],
        acl_version=3,
    )

    for client in client_with_user(user_context):
        response = client.get(
            "/auth/session",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "auth0|user-1",
        "tenant_id": "tenant-a",
        "tenant_uuid": None,
        "department_id": None,
        "workspace_id": "workspace-a",
        "email": "member@example.com",
        "roles": ["admin", "user"],
        "group_ids": ["engineering"],
        "scopes": ["documents:read", "query:run"],
        "tenant_permissions": [],
        "department_permissions": {},
        "must_change_password": False,
        "acl_version": 3,
        "auth_provider": settings.auth_provider,
    }
