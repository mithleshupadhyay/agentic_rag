from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.llm_providers import (
    create_llm_provider,
    decrypt_provider_api_key,
    delete_llm_provider,
    get_default_llm_provider,
    get_llm_provider,
    list_llm_providers,
    update_llm_provider,
)
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.schemas.llm import (
    LLMProviderCreate,
    LLMProviderType,
    LLMProviderUpdate,
)


@pytest.fixture()
def db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
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


def test_create_provider_encrypts_key_and_never_crosses_tenant(db: Session) -> None:
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
            api_key="tenant-secret-key",
        ),
    )

    assert provider.encrypted_api_key is not None
    assert provider.encrypted_api_key.startswith("v1:")
    assert "tenant-secret-key" not in provider.encrypted_api_key
    assert decrypt_provider_api_key(provider.encrypted_api_key) == "tenant-secret-key"
    assert get_llm_provider(db, "tenant-a", provider.id) is not None
    assert get_llm_provider(db, "tenant-b", provider.id) is None
    assert get_default_llm_provider(db, "tenant-a", "chat") == provider
    assert get_default_llm_provider(db, "tenant-a", "embedding") == provider


def test_create_provider_requires_credential_encryption(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_provider_encryption_key", "")

    with pytest.raises(HTTPException) as exc_info:
        create_llm_provider(
            db,
            tenant_id="tenant-a",
            created_by="admin-a",
            data=LLMProviderCreate(
                name="OpenAI production",
                provider_type=LLMProviderType.OPENAI,
                chat_model="gpt-4.1-mini",
                api_key="provider-secret",
            ),
        )

    assert exc_info.value.status_code == 503
    providers, total = list_llm_providers(db, "tenant-a")
    assert providers == []
    assert total == 0


def test_update_provider_moves_default_routes(db: Session) -> None:
    first_provider = create_llm_provider(
        db,
        tenant_id="tenant-a",
        created_by="admin-a",
        data=LLMProviderCreate(
            name="Gemini primary",
            provider_type=LLMProviderType.GOOGLE,
            chat_model="gemini-2.5-flash",
            embedding_model="gemini-embedding-001",
            embedding_dimension=768,
        ),
    )
    second_provider = create_llm_provider(
        db,
        tenant_id="tenant-a",
        created_by="admin-a",
        data=LLMProviderCreate(
            name="OpenAI secondary",
            provider_type=LLMProviderType.OPENAI,
            chat_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
            embedding_dimension=768,
        ),
    )

    updated_provider = update_llm_provider(
        db,
        provider=second_provider,
        updated_by="admin-a",
        data=LLMProviderUpdate(
            is_default_chat=True,
            is_default_embedding=True,
        ),
    )
    db.refresh(first_provider)

    assert updated_provider.is_default_chat is True
    assert updated_provider.is_default_embedding is True
    assert first_provider.is_default_chat is False
    assert first_provider.is_default_embedding is False
    assert get_default_llm_provider(db, "tenant-a", "chat") == second_provider
    assert get_default_llm_provider(db, "tenant-a", "embedding") == second_provider


def test_delete_default_provider_promotes_active_replacement(db: Session) -> None:
    first_provider = create_llm_provider(
        db,
        tenant_id="tenant-a",
        created_by="admin-a",
        data=LLMProviderCreate(
            name="Gemini primary",
            provider_type=LLMProviderType.GOOGLE,
            chat_model="gemini-2.5-flash",
        ),
    )
    second_provider = create_llm_provider(
        db,
        tenant_id="tenant-a",
        created_by="admin-a",
        data=LLMProviderCreate(
            name="OpenAI replacement",
            provider_type=LLMProviderType.OPENAI,
            chat_model="gpt-4.1-mini",
        ),
    )

    delete_llm_provider(db, first_provider, deleted_by="admin-a")
    db.refresh(second_provider)

    assert get_llm_provider(db, "tenant-a", first_provider.id) is None
    assert second_provider.is_default_chat is True
    assert get_default_llm_provider(db, "tenant-a", "chat") == second_provider


def test_provider_config_rejects_embedded_secrets(db: Session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        create_llm_provider(
            db,
            tenant_id="tenant-a",
            created_by="admin-a",
            data=LLMProviderCreate(
                name="Unsafe provider",
                provider_type=LLMProviderType.OPENAI_COMPATIBLE,
                chat_model="custom-chat",
                config={"headers": {"authorization_token": "secret"}},
            ),
        )

    assert exc_info.value.status_code == 422
    assert "api_key field" in str(exc_info.value.detail)
