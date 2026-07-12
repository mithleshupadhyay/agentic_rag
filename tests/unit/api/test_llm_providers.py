from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from agentic_rag.api import admin
from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.models import LLMProvider, Tenant
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.llm import LLMResponse


@pytest.fixture()
def db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    monkeypatch.setattr(
        settings,
        "llm_provider_encryption_key",
        Fernet.generate_key().decode("utf-8"),
    )
    monkeypatch.setattr(settings, "embedding_dimension", 768)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                Tenant(
                    tenant_id="tenant-a",
                    name="Tenant A",
                    slug="tenant-a",
                    status="active",
                    metadata_={},
                ),
                Tenant(
                    tenant_id="tenant-b",
                    name="Tenant B",
                    slug="tenant-b",
                    status="active",
                    metadata_={},
                ),
            ]
        )
        session.commit()
        yield session


def client_with_user(
    user_context: UserContext,
    db: Session,
) -> Iterator[TestClient]:
    async def override_get_current_user() -> UserContext:
        return user_context

    def override_get_session() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_session] = override_get_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_admin_manages_tenant_provider_without_exposing_key(db: Session) -> None:
    admin_context = UserContext(
        id="auth0|admin-a",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        roles=["admin"],
    )

    for client in client_with_user(admin_context, db):
        create_response = client.post(
            "/admin/llm-providers",
            headers={"Authorization": "Bearer test-token"},
            json={
                "name": "Gemini production",
                "provider_type": "google",
                "chat_model": "gemini-2.5-flash",
                "embedding_model": "gemini-embedding-001",
                "embedding_dimension": 768,
                "api_key": "tenant-gemini-secret",
                "config": {"temperature": 0.1},
                "is_active": True,
                "is_default_chat": True,
                "is_default_embedding": True,
            },
        )
        list_response = client.get(
            "/admin/llm-providers",
            headers={"Authorization": "Bearer test-token"},
        )

    assert create_response.status_code == 201
    payload = create_response.json()
    assert payload["tenant_id"] == "tenant-a"
    assert payload["has_api_key"] is True
    assert "api_key" not in payload
    assert "encrypted_api_key" not in payload
    assert "tenant-gemini-secret" not in create_response.text
    assert list_response.status_code == 200
    assert list_response.json()["page"]["total"] == 1

    stored_provider = db.query(LLMProvider).one()
    assert stored_provider.tenant_id == "tenant-a"
    assert stored_provider.encrypted_api_key is not None
    assert "tenant-gemini-secret" not in stored_provider.encrypted_api_key


def test_provider_api_is_admin_only_and_tenant_scoped(db: Session) -> None:
    db.add(
        LLMProvider(
            tenant_id="tenant-b",
            name="Tenant B provider",
            provider_type="openai",
            chat_model="gpt-4.1-mini",
            config={},
            is_active=True,
            is_default_chat=True,
            is_default_embedding=False,
            created_by="admin-b",
            updated_by="admin-b",
            is_deleted=False,
        )
    )
    db.commit()
    user_context = UserContext(
        id="auth0|member-a",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        roles=["user"],
    )

    for client in client_with_user(user_context, db):
        response = client.get(
            "/admin/llm-providers",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 403

    admin_context = user_context.model_copy(update={"roles": ["admin"]})
    for client in client_with_user(admin_context, db):
        response = client.get(
            "/admin/llm-providers",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_admin_validates_selected_chat_provider(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LLMProvider(
        tenant_id="tenant-a",
        name="Gemini production",
        provider_type="google",
        chat_model="gemini-2.5-flash",
        config={},
        is_active=True,
        is_default_chat=True,
        is_default_embedding=False,
        created_by="admin-a",
        updated_by="admin-a",
        is_deleted=False,
    )
    db.add(provider)
    db.commit()
    db.refresh(provider)
    captured = {}

    def generate_validation_response(request):
        captured["request"] = request
        return LLMResponse(
            text="healthy",
            model="gemini/gemini-2.5-flash",
            provider="google",
            latency_ms=12,
        )

    monkeypatch.setattr(
        admin,
        "generate_chat_completion",
        generate_validation_response,
    )
    admin_context = UserContext(
        id="auth0|admin-a",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        roles=["admin"],
    )

    for client in client_with_user(admin_context, db):
        response = client.post(
            f"/admin/llm-providers/{provider.id}/validate",
            headers={"Authorization": "Bearer test-token"},
            json={"capability": "chat"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "capability": "chat",
        "provider": "google",
        "model": "gemini/gemini-2.5-flash",
        "latency_ms": 12,
    }
    assert captured["request"].provider_id == provider.id
    assert captured["request"].auth.tenant_id == "tenant-a"
