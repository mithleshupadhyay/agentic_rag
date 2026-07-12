from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.llm.manager import LLMManager, LLMProviderResolutionError
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.llm_providers import create_llm_provider
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.llm import (
    ChatCompletionRequest,
    EmbeddingRequest,
    LLMMessage,
    LLMProviderCreate,
    LLMProviderType,
)


@pytest.fixture()
def db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    monkeypatch.setattr(settings, "llm_provider_database_enabled", True)
    monkeypatch.setattr(settings, "llm_provider_env_fallback_enabled", True)
    monkeypatch.setattr(
        settings,
        "llm_provider_encryption_key",
        Fernet.generate_key().decode("utf-8"),
    )
    monkeypatch.setattr(settings, "embedding_dimension", 768)

    engine = create_engine("sqlite:///:memory:")
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


def test_manager_resolves_tenant_chat_and_embedding_routes(db: Session) -> None:
    provider = create_llm_provider(
        db,
        tenant_id="tenant-a",
        created_by="admin-a",
        data=LLMProviderCreate(
            name="Gemini production",
            provider_type=LLMProviderType.GOOGLE,
            chat_model="gemini-2.5-flash",
            embedding_model="gemini-embedding-001",
            embedding_dimension=768,
            api_key="tenant-gemini-key",
            config={
                "temperature": 0.2,
                "max_tokens": 1200,
                "timeout_seconds": 45,
                "vertex_project": "tenant-a-project",
            },
        ),
    )
    manager = LLMManager()
    auth = AuthContext(user_id="user-a", tenant_id="tenant-a")

    chat_route = manager.resolve_chat_provider(
        ChatCompletionRequest(
            auth=auth,
            messages=[LLMMessage(role="user", content="Summarize the policy.")],
        ),
        db=db,
    )
    embedding_route = manager.resolve_embedding_provider(
        EmbeddingRequest(auth=auth, texts=["policy text"]),
        db=db,
    )

    assert chat_route.provider_id == provider.id
    assert chat_route.provider == "google"
    assert chat_route.model == "gemini/gemini-2.5-flash"
    assert chat_route.api_key == "tenant-gemini-key"
    assert chat_route.temperature == 0.2
    assert chat_route.max_tokens == 1200
    assert chat_route.timeout_seconds == 45
    assert chat_route.options == {"vertex_project": "tenant-a-project"}
    assert embedding_route.provider_id == provider.id
    assert embedding_route.model == "gemini/gemini-embedding-001"
    assert embedding_route.embedding_dimension == 768


def test_manager_rejects_cross_tenant_provider_id(db: Session) -> None:
    provider = create_llm_provider(
        db,
        tenant_id="tenant-b",
        created_by="admin-b",
        data=LLMProviderCreate(
            name="Tenant B provider",
            provider_type=LLMProviderType.OPENAI,
            chat_model="gpt-4.1-mini",
        ),
    )
    manager = LLMManager()

    with pytest.raises(LLMProviderResolutionError) as exc_info:
        manager.resolve_chat_provider(
            ChatCompletionRequest(
                auth=AuthContext(user_id="user-a", tenant_id="tenant-a"),
                provider_id=provider.id,
                messages=[LLMMessage(role="user", content="Tenant A question")],
            ),
            db=db,
        )

    assert "not found in this tenant" in str(exc_info.value)


def test_manager_uses_environment_fallback_without_tenant_default(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "litellm")
    monkeypatch.setattr(
        settings,
        "embedding_model_name",
        "gemini/gemini-embedding-001",
    )
    monkeypatch.setattr(settings, "gemini_api_key", "environment-gemini-key")
    monkeypatch.setattr(settings, "embedding_timeout_seconds", 20)
    manager = LLMManager()

    route = manager.resolve_embedding_provider(
        EmbeddingRequest(
            auth=AuthContext(user_id="user-b", tenant_id="tenant-b"),
            texts=["tenant B content"],
        ),
        db=db,
    )

    assert route.provider_id is None
    assert route.provider == "litellm"
    assert route.model == "gemini/gemini-embedding-001"
    assert route.api_key == "environment-gemini-key"
    assert route.embedding_dimension == 768
    assert route.timeout_seconds == 20


def test_manager_can_require_database_provider(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_provider_env_fallback_enabled", False)
    manager = LLMManager()

    with pytest.raises(LLMProviderResolutionError) as exc_info:
        manager.resolve_chat_provider(
            ChatCompletionRequest(
                auth=AuthContext(user_id="user-b", tenant_id="tenant-b"),
                messages=[LLMMessage(role="user", content="Tenant B question")],
            ),
            db=db,
        )

    assert "No active default chat provider" in str(exc_info.value)
