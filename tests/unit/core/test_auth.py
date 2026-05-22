from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.dependencies import require_role, require_scope
from agentic_rag.core.models.user_context import UserContext


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


def test_missing_token_is_rejected() -> None:
    client = TestClient(_build_test_app())

    response = client.get("/me")

    assert response.status_code == 401


def test_local_token_builds_user_context() -> None:
    client = TestClient(_build_test_app())

    response = client.get(
        "/me",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == "local-user"
    assert response.json()["tenant_id"] == "local-tenant"
    assert "admin" in response.json()["roles"]


def test_required_role_allows_matching_user() -> None:
    client = TestClient(_build_test_app())

    response = client.get(
        "/admin",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 200


def test_required_scope_allows_matching_user() -> None:
    client = TestClient(_build_test_app())

    response = client.get(
        "/documents",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "local-tenant"


def test_required_scope_rejects_missing_scope() -> None:
    client = TestClient(_build_test_app())

    response = client.get(
        "/blocked",
        headers={"Authorization": "Bearer local-dev-token"},
    )

    assert response.status_code == 403
