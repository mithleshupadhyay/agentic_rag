import asyncio
from typing import Any

import httpx
import pytest

from agentic_rag.core import auth0_management
from agentic_rag.shared.config import settings


class FakeAuth0Client:
    requests: list[dict[str, Any]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.requests = FakeAuth0Client.requests

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def post(self, url: str, **kwargs) -> httpx.Response:
        self.requests.append({"method": "POST", "url": url, **kwargs})
        if url.endswith("/oauth/token"):
            return httpx.Response(
                200,
                json={"access_token": "management-token", "expires_in": 3600},
            )
        if url.endswith("/api/v2/organizations"):
            return httpx.Response(
                201,
                json={"id": "org_tenant_a", "name": "tenant-a"},
            )
        if url.endswith("/organizations/org_tenant_a/invitations"):
            return httpx.Response(
                200,
                json={
                    "id": "uinv_123",
                    "organization_id": "org_tenant_a",
                    "invitee": {"email": "member@example.com"},
                },
            )
        raise AssertionError(f"Unexpected POST request: {url}")

    async def get(self, url: str, **kwargs) -> httpx.Response:
        self.requests.append({"method": "GET", "url": url, **kwargs})
        if url.endswith("/api/v2/users/github%7Cuser-1"):
            return httpx.Response(
                200,
                json={
                    "user_id": "github|user-1",
                    "email": "Member@Example.com",
                    "email_verified": True,
                    "name": "Tenant Member",
                },
            )
        raise AssertionError(f"Unexpected GET request: {url}")

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        self.requests.append({"method": "DELETE", "url": url, **kwargs})
        return httpx.Response(204)


def configure_auth0_management(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_provider", "auth0")
    monkeypatch.setattr(settings, "auth0_domain", "identity.example.com")
    monkeypatch.setattr(settings, "auth0_frontend_client_id", "spa-client-id")
    monkeypatch.setattr(settings, "auth0_management_client_id", "m2m-client-id")
    monkeypatch.setattr(
        settings,
        "auth0_management_client_secret",
        "management-secret",
    )
    monkeypatch.setattr(settings, "auth0_database_connection_id", "con_database")
    monkeypatch.setattr(
        settings,
        "auth0_organization_connection_ids_csv",
        "con_google,con_github",
    )
    monkeypatch.setattr(settings, "auth0_inviter_name", "Agentic RAG Admin")
    monkeypatch.setattr(settings, "auth0_invitation_ttl_seconds", 604800)
    monkeypatch.setattr(
        auth0_management.httpx,
        "AsyncClient",
        FakeAuth0Client,
    )
    auth0_management._management_access_token = None
    auth0_management._management_access_token_expires_at = 0.0
    FakeAuth0Client.requests = []


def test_auth0_organization_enables_configured_connections(monkeypatch) -> None:
    configure_auth0_management(monkeypatch)

    organization = asyncio.run(
        auth0_management.create_auth0_organization(
            tenant_id="tenant-a",
            name="tenant-a",
            display_name="Tenant A",
        )
    )

    assert organization.id == "org_tenant_a"
    token_request = next(
        request
        for request in FakeAuth0Client.requests
        if request["url"].endswith("/oauth/token")
    )
    assert token_request["data"] == {
        "grant_type": "client_credentials",
        "client_id": "m2m-client-id",
        "client_secret": "management-secret",
        "audience": "https://identity.example.com/api/v2/",
    }
    organization_request = next(
        request
        for request in FakeAuth0Client.requests
        if request["url"].endswith("/api/v2/organizations")
    )
    assert organization_request["json"]["name"] == "tenant-a"
    assert organization_request["json"]["metadata"] == {
        "agentic_rag_tenant_id": "tenant-a"
    }
    assert organization_request["json"]["enabled_connections"] == [
        {"connection_id": "con_google", "assign_membership_on_login": False},
        {"connection_id": "con_github", "assign_membership_on_login": False},
        {"connection_id": "con_database", "assign_membership_on_login": False},
    ]


def test_auth0_invitation_is_tenant_scoped_and_sends_email(monkeypatch) -> None:
    configure_auth0_management(monkeypatch)

    invitation = asyncio.run(
        auth0_management.create_auth0_organization_invitation(
            organization_id="org_tenant_a",
            email="Member@Example.com",
            display_name="Tenant Member",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            role_name="user",
        )
    )

    assert invitation.id == "uinv_123"
    assert invitation.organization_id == "org_tenant_a"
    assert invitation.email == "member@example.com"
    assert invitation.email_sent is True
    request = next(
        request
        for request in FakeAuth0Client.requests
        if request["url"].endswith("/organizations/org_tenant_a/invitations")
    )
    assert request["json"] == {
        "inviter": {"name": "Agentic RAG Admin"},
        "invitee": {"email": "member@example.com"},
        "client_id": "spa-client-id",
        "app_metadata": {
            "agentic_rag_tenant_id": "tenant-a",
            "agentic_rag_role": "user",
            "agentic_rag_workspace_id": "workspace-a",
        },
        "user_metadata": {"display_name": "Tenant Member"},
        "ttl_sec": 604800,
        "send_invitation_email": True,
        "connection_id": "con_database",
    }


def test_auth0_invitation_and_organization_can_be_removed(monkeypatch) -> None:
    configure_auth0_management(monkeypatch)

    asyncio.run(
        auth0_management.delete_auth0_organization_invitation(
            "org_tenant_a",
            "uinv_123",
        )
    )
    asyncio.run(auth0_management.delete_auth0_organization("org_tenant_a"))

    delete_urls = [
        request["url"]
        for request in FakeAuth0Client.requests
        if request["method"] == "DELETE"
    ]
    assert delete_urls == [
        "https://identity.example.com/api/v2/organizations/"
        "org_tenant_a/invitations/uinv_123",
        "https://identity.example.com/api/v2/organizations/org_tenant_a",
    ]


def test_auth0_user_lookup_returns_verified_normalized_identity(monkeypatch) -> None:
    configure_auth0_management(monkeypatch)

    identity = asyncio.run(
        auth0_management.get_auth0_user_identity("github|user-1")
    )

    assert identity.subject == "github|user-1"
    assert identity.email == "member@example.com"
    assert identity.email_verified is True
    assert identity.display_name == "Tenant Member"


def test_auth0_management_rejects_missing_credentials(monkeypatch) -> None:
    configure_auth0_management(monkeypatch)
    monkeypatch.setattr(settings, "auth0_management_client_secret", "")

    with pytest.raises(auth0_management.Auth0ManagementError) as exc_info:
        asyncio.run(
            auth0_management.create_auth0_organization_invitation(
                organization_id="org_tenant_a",
                email="member@example.com",
                display_name=None,
                tenant_id="tenant-a",
                workspace_id=None,
                role_name="viewer",
            )
        )

    assert exc_info.value.status_code == 503
    assert "credentials" in str(exc_info.value)
