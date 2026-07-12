import logging
from datetime import datetime, timezone
from typing import Any, Optional, Tuple
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.models import LLMProvider, Tenant
from agentic_rag.shared.schemas.llm import (
    LLMProviderCreate,
    LLMProviderUpdate,
)


logger = logging.getLogger(__name__)


def encrypt_provider_api_key(api_key: str) -> str:
    encryption_key = settings.llm_provider_encryption_key.strip()
    if not encryption_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider credential encryption is not configured.",
        )

    try:
        cipher = Fernet(encryption_key.encode("utf-8"))
    except (TypeError, ValueError) as error:
        logger.exception(
            f"[LLMProvider] Invalid credential encryption key: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider credential encryption key is invalid.",
        ) from error

    encrypted_value = cipher.encrypt(api_key.encode("utf-8")).decode("utf-8")
    return f"v1:{encrypted_value}"


def decrypt_provider_api_key(encrypted_api_key: str | None) -> str:
    if not encrypted_api_key:
        return ""

    encryption_key = settings.llm_provider_encryption_key.strip()
    if not encryption_key:
        raise RuntimeError("LLM provider credential encryption is not configured.")
    if not encrypted_api_key.startswith("v1:"):
        raise RuntimeError("LLM provider credential has an unsupported format.")

    try:
        cipher = Fernet(encryption_key.encode("utf-8"))
        return cipher.decrypt(
            encrypted_api_key.removeprefix("v1:").encode("utf-8")
        ).decode("utf-8")
    except (InvalidToken, TypeError, ValueError) as error:
        logger.exception(
            f"[LLMProvider] Credential decryption failed: {type(error).__name__}"
        )
        raise RuntimeError("LLM provider credential could not be decrypted.") from error


def validate_provider_config_has_no_secrets(
    config: dict[str, Any],
    path: str = "config",
) -> None:
    secret_field_names = {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
    secret_field_suffixes = (
        "_api_key",
        "_credential",
        "_credentials",
        "_password",
        "_secret",
        "_token",
    )
    for key, value in config.items():
        normalized_key = key.lower().replace("-", "_")
        field_path = f"{path}.{key}"
        if (
            normalized_key in secret_field_names
            or normalized_key.endswith(secret_field_suffixes)
            or normalized_key.startswith("authorization_")
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"{field_path} looks like a credential. Use the encrypted "
                    "api_key field instead."
                ),
            )
        if isinstance(value, dict):
            validate_provider_config_has_no_secrets(value, field_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    validate_provider_config_has_no_secrets(
                        item,
                        f"{field_path}[{index}]",
                    )


def get_llm_provider(
    db: Session,
    tenant_id: str,
    provider_id: UUID,
    include_deleted: bool = False,
) -> Optional[LLMProvider]:
    query = db.query(LLMProvider).filter(
        LLMProvider.id == provider_id,
        LLMProvider.tenant_id == tenant_id,
    )
    if not include_deleted:
        query = query.filter(LLMProvider.is_deleted.is_(False))
    return query.first()


def list_llm_providers(
    db: Session,
    tenant_id: str,
    page: int = 1,
    size: int = 50,
) -> Tuple[list[LLMProvider], int]:
    logger.info(
        f"[DB] Listing LLM providers tenant={tenant_id} page={page} size={size}"
    )
    query = db.query(LLMProvider).filter(
        LLMProvider.tenant_id == tenant_id,
        LLMProvider.is_deleted.is_(False),
    )
    total = query.count()
    providers = (
        query.order_by(
            LLMProvider.is_default_chat.desc(),
            LLMProvider.is_default_embedding.desc(),
            LLMProvider.created_at.asc(),
        )
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return providers, total


def get_default_llm_provider(
    db: Session,
    tenant_id: str,
    capability: str,
) -> Optional[LLMProvider]:
    query = db.query(LLMProvider).filter(
        LLMProvider.tenant_id == tenant_id,
        LLMProvider.is_active.is_(True),
        LLMProvider.is_deleted.is_(False),
    )
    if capability == "chat":
        query = query.filter(
            LLMProvider.is_default_chat.is_(True),
            LLMProvider.chat_model.is_not(None),
        )
    elif capability == "embedding":
        query = query.filter(
            LLMProvider.is_default_embedding.is_(True),
            LLMProvider.embedding_model.is_not(None),
        )
    else:
        raise ValueError(f"Unsupported LLM capability: {capability}")
    return query.first()


def create_llm_provider(
    db: Session,
    tenant_id: str,
    created_by: str,
    data: LLMProviderCreate,
) -> LLMProvider:
    tenant = (
        db.query(Tenant)
        .filter(Tenant.tenant_id == tenant_id, Tenant.status == "active")
        .first()
    )
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The current tenant is not provisioned or active.",
        )

    validate_provider_config_has_no_secrets(data.config)
    if (
        data.embedding_dimension is not None
        and data.embedding_dimension != settings.embedding_dimension
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Embedding dimension must match the configured vector index "
                f"dimension ({settings.embedding_dimension})."
            ),
        )

    existing_chat_default = get_default_llm_provider(db, tenant_id, "chat")
    existing_embedding_default = get_default_llm_provider(
        db,
        tenant_id,
        "embedding",
    )
    make_default_chat = bool(
        data.chat_model and (data.is_default_chat or existing_chat_default is None)
    )
    make_default_embedding = bool(
        data.embedding_model
        and (data.is_default_embedding or existing_embedding_default is None)
    )

    try:
        if make_default_chat:
            db.query(LLMProvider).filter(
                LLMProvider.tenant_id == tenant_id,
                LLMProvider.is_deleted.is_(False),
            ).update({LLMProvider.is_default_chat: False}, synchronize_session=False)
        if make_default_embedding:
            db.query(LLMProvider).filter(
                LLMProvider.tenant_id == tenant_id,
                LLMProvider.is_deleted.is_(False),
            ).update(
                {LLMProvider.is_default_embedding: False},
                synchronize_session=False,
            )

        encrypted_api_key = None
        if data.api_key and data.api_key.get_secret_value():
            encrypted_api_key = encrypt_provider_api_key(
                data.api_key.get_secret_value()
            )

        provider = LLMProvider(
            tenant_id=tenant_id,
            name=data.name.strip(),
            provider_type=data.provider_type.value,
            chat_model=data.chat_model.strip() if data.chat_model else None,
            embedding_model=(
                data.embedding_model.strip() if data.embedding_model else None
            ),
            embedding_dimension=data.embedding_dimension,
            base_url=data.base_url.strip().rstrip("/") if data.base_url else None,
            encrypted_api_key=encrypted_api_key,
            config=dict(data.config),
            is_active=data.is_active,
            is_default_chat=make_default_chat and data.is_active,
            is_default_embedding=make_default_embedding and data.is_active,
            created_by=created_by,
            updated_by=created_by,
            is_deleted=False,
        )
        db.add(provider)
        db.commit()
        db.refresh(provider)
        logger.info(
            f"[DB] Created LLM provider provider={provider.id} tenant={tenant_id} "
            f"type={provider.provider_type} chat_default={provider.is_default_chat} "
            f"embedding_default={provider.is_default_embedding}"
        )
        return provider

    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to create LLM provider tenant={tenant_id}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An active provider with this name or default route already exists.",
        ) from error


def update_llm_provider(
    db: Session,
    provider: LLMProvider,
    updated_by: str,
    data: LLMProviderUpdate,
) -> LLMProvider:
    fields_set = data.model_fields_set
    merged_chat_model = (
        data.chat_model if "chat_model" in fields_set else provider.chat_model
    )
    merged_embedding_model = (
        data.embedding_model
        if "embedding_model" in fields_set
        else provider.embedding_model
    )
    merged_embedding_dimension = (
        data.embedding_dimension
        if "embedding_dimension" in fields_set
        else provider.embedding_dimension
    )
    merged_active = data.is_active if data.is_active is not None else provider.is_active
    merged_default_chat = (
        data.is_default_chat
        if data.is_default_chat is not None
        else provider.is_default_chat
    )
    merged_default_embedding = (
        data.is_default_embedding
        if data.is_default_embedding is not None
        else provider.is_default_embedding
    )

    if not merged_chat_model and not merged_embedding_model:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="At least one chat or embedding model is required.",
        )
    if merged_embedding_model and merged_embedding_dimension is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Embedding dimension is required for an embedding model.",
        )
    if not merged_embedding_model:
        merged_embedding_dimension = None
        merged_default_embedding = False
    if merged_embedding_dimension not in {None, settings.embedding_dimension}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Embedding dimension must match the configured vector index "
                f"dimension ({settings.embedding_dimension})."
            ),
        )
    if not merged_chat_model:
        merged_default_chat = False
    if not merged_active:
        merged_default_chat = False
        merged_default_embedding = False
    if data.config is not None:
        validate_provider_config_has_no_secrets(data.config)

    try:
        if merged_default_chat:
            db.query(LLMProvider).filter(
                LLMProvider.tenant_id == provider.tenant_id,
                LLMProvider.id != provider.id,
                LLMProvider.is_deleted.is_(False),
            ).update({LLMProvider.is_default_chat: False}, synchronize_session=False)
        if merged_default_embedding:
            db.query(LLMProvider).filter(
                LLMProvider.tenant_id == provider.tenant_id,
                LLMProvider.id != provider.id,
                LLMProvider.is_deleted.is_(False),
            ).update(
                {LLMProvider.is_default_embedding: False},
                synchronize_session=False,
            )

        if data.name is not None:
            provider.name = data.name.strip()
        provider.chat_model = (
            merged_chat_model.strip() if merged_chat_model else None
        )
        provider.embedding_model = (
            merged_embedding_model.strip() if merged_embedding_model else None
        )
        provider.embedding_dimension = merged_embedding_dimension
        if "base_url" in fields_set:
            provider.base_url = (
                data.base_url.strip().rstrip("/") if data.base_url else None
            )
        if data.api_key and data.api_key.get_secret_value():
            provider.encrypted_api_key = encrypt_provider_api_key(
                data.api_key.get_secret_value()
            )
        elif data.clear_api_key:
            provider.encrypted_api_key = None
        if data.config is not None:
            provider.config = dict(data.config)
        provider.is_active = merged_active
        provider.is_default_chat = merged_default_chat
        provider.is_default_embedding = merged_default_embedding
        provider.updated_by = updated_by

        db.commit()
        db.refresh(provider)
        logger.info(
            f"[DB] Updated LLM provider provider={provider.id} "
            f"tenant={provider.tenant_id} active={provider.is_active} "
            f"chat_default={provider.is_default_chat} "
            f"embedding_default={provider.is_default_embedding}"
        )
        return provider

    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to update LLM provider provider={provider.id}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The provider name or default route conflicts with another provider.",
        ) from error


def delete_llm_provider(
    db: Session,
    provider: LLMProvider,
    deleted_by: str,
) -> None:
    was_default_chat = provider.is_default_chat
    was_default_embedding = provider.is_default_embedding

    try:
        provider.is_default_chat = False
        provider.is_default_embedding = False
        provider.is_active = False
        provider.is_deleted = True
        provider.deleted_at = datetime.now(timezone.utc)
        provider.updated_by = deleted_by
        db.flush()

        if was_default_chat:
            replacement_chat = (
                db.query(LLMProvider)
                .filter(
                    LLMProvider.tenant_id == provider.tenant_id,
                    LLMProvider.id != provider.id,
                    LLMProvider.chat_model.is_not(None),
                    LLMProvider.is_active.is_(True),
                    LLMProvider.is_deleted.is_(False),
                )
                .order_by(LLMProvider.created_at.asc())
                .first()
            )
            if replacement_chat is not None:
                replacement_chat.is_default_chat = True
                replacement_chat.updated_by = deleted_by

        if was_default_embedding:
            replacement_embedding = (
                db.query(LLMProvider)
                .filter(
                    LLMProvider.tenant_id == provider.tenant_id,
                    LLMProvider.id != provider.id,
                    LLMProvider.embedding_model.is_not(None),
                    LLMProvider.is_active.is_(True),
                    LLMProvider.is_deleted.is_(False),
                )
                .order_by(LLMProvider.created_at.asc())
                .first()
            )
            if replacement_embedding is not None:
                replacement_embedding.is_default_embedding = True
                replacement_embedding.updated_by = deleted_by

        db.commit()
        logger.info(
            f"[DB] Soft-deleted LLM provider provider={provider.id} "
            f"tenant={provider.tenant_id} deleted_by={deleted_by}"
        )
    except IntegrityError as error:
        db.rollback()
        logger.exception(
            f"[DB] Failed to delete LLM provider provider={provider.id}: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The provider could not be removed.",
        ) from error
