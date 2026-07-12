import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.llm_providers import (
    decrypt_provider_api_key,
    get_default_llm_provider,
    get_llm_provider,
)
from agentic_rag.shared.db.models import LLMProvider
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.shared.schemas.llm import (
    ChatCompletionRequest,
    EmbeddingRequest,
)


logger = logging.getLogger(__name__)


class LLMProviderResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedLLMProvider:
    provider_id: UUID | None
    provider: str
    model: str
    api_key: str
    base_url: str | None
    temperature: float | None
    max_tokens: int | None
    timeout_seconds: int
    embedding_dimension: int | None
    options: dict[str, Any]


class LLMManager:
    def resolve_chat_provider(
        self,
        request: ChatCompletionRequest,
        db: Session | None = None,
    ) -> ResolvedLLMProvider:
        tenant_id = request.auth.tenant_id if request.auth else None
        database_provider = self._resolve_database_provider(
            tenant_id=tenant_id,
            provider_id=request.provider_id,
            capability="chat",
            db=db,
        )
        if database_provider is not None:
            if not database_provider.chat_model:
                raise LLMProviderResolutionError(
                    "The selected tenant provider does not have a chat model."
                )
            config = database_provider.config or {}
            return ResolvedLLMProvider(
                provider_id=database_provider.id,
                provider=database_provider.provider_type,
                model=self._normalize_model_name(
                    database_provider.provider_type,
                    database_provider.chat_model,
                ),
                api_key=decrypt_provider_api_key(
                    database_provider.encrypted_api_key
                ),
                base_url=database_provider.base_url,
                temperature=self._read_float_option(
                    config,
                    "temperature",
                    minimum=0.0,
                    maximum=2.0,
                ),
                max_tokens=self._read_integer_option(
                    config,
                    "max_tokens",
                    minimum=1,
                    maximum=settings.llm_max_output_tokens,
                ),
                timeout_seconds=(
                    self._read_integer_option(
                        config,
                        "timeout_seconds",
                        minimum=1,
                        maximum=300,
                    )
                    or settings.llm_timeout_seconds
                ),
                embedding_dimension=None,
                options=self._read_provider_options(config),
            )

        if tenant_id and not settings.llm_provider_env_fallback_enabled:
            raise LLMProviderResolutionError(
                "No active default chat provider is configured for this tenant."
            )

        model = request.model or settings.default_llm_model
        provider = request.provider or settings.llm_provider
        return ResolvedLLMProvider(
            provider_id=None,
            provider=provider,
            model=model,
            api_key=self._resolve_environment_api_key(model),
            base_url=self._resolve_environment_base_url(model),
            temperature=None,
            max_tokens=None,
            timeout_seconds=settings.llm_timeout_seconds,
            embedding_dimension=None,
            options={},
        )

    def resolve_embedding_provider(
        self,
        request: EmbeddingRequest,
        db: Session | None = None,
    ) -> ResolvedLLMProvider:
        database_provider = self._resolve_database_provider(
            tenant_id=request.auth.tenant_id,
            provider_id=request.provider_id,
            capability="embedding",
            db=db,
        )
        if database_provider is not None:
            if not database_provider.embedding_model:
                raise LLMProviderResolutionError(
                    "The selected tenant provider does not have an embedding model."
                )
            if not database_provider.embedding_dimension:
                raise LLMProviderResolutionError(
                    "The selected tenant provider has no embedding dimension."
                )
            config = database_provider.config or {}
            return ResolvedLLMProvider(
                provider_id=database_provider.id,
                provider=database_provider.provider_type,
                model=self._normalize_model_name(
                    database_provider.provider_type,
                    database_provider.embedding_model,
                ),
                api_key=decrypt_provider_api_key(
                    database_provider.encrypted_api_key
                ),
                base_url=database_provider.base_url,
                temperature=None,
                max_tokens=None,
                timeout_seconds=(
                    self._read_integer_option(
                        config,
                        "timeout_seconds",
                        minimum=1,
                        maximum=300,
                    )
                    or settings.embedding_timeout_seconds
                ),
                embedding_dimension=database_provider.embedding_dimension,
                options=self._read_provider_options(config),
            )

        if not settings.llm_provider_env_fallback_enabled:
            raise LLMProviderResolutionError(
                "No active default embedding provider is configured for this tenant."
            )

        model = request.model or settings.embedding_model_name
        provider = request.provider or settings.embedding_provider
        return ResolvedLLMProvider(
            provider_id=None,
            provider=provider,
            model=model,
            api_key=self._resolve_environment_api_key(model),
            base_url=self._resolve_environment_base_url(model),
            temperature=None,
            max_tokens=None,
            timeout_seconds=settings.embedding_timeout_seconds,
            embedding_dimension=settings.embedding_dimension,
            options={},
        )

    def _resolve_database_provider(
        self,
        tenant_id: str | None,
        provider_id: UUID | None,
        capability: str,
        db: Session | None,
    ) -> LLMProvider | None:
        if not settings.llm_provider_database_enabled or not tenant_id:
            return None

        owns_session = db is None
        if db is None:
            SessionLocal = get_sync_session_factory()
            db = SessionLocal()

        try:
            if provider_id is not None:
                provider = get_llm_provider(
                    db,
                    tenant_id=tenant_id,
                    provider_id=provider_id,
                )
                if provider is None:
                    raise LLMProviderResolutionError(
                        "The requested LLM provider was not found in this tenant."
                    )
            else:
                provider = get_default_llm_provider(
                    db,
                    tenant_id=tenant_id,
                    capability=capability,
                )

            if provider is None:
                return None
            if not provider.is_active or provider.is_deleted:
                raise LLMProviderResolutionError(
                    "The selected LLM provider is not active."
                )

            logger.info(
                f"[LLMManager] Resolved tenant provider tenant={tenant_id} "
                f"provider={provider.id} capability={capability} "
                f"type={provider.provider_type}"
            )
            return provider
        finally:
            if owns_session:
                db.close()

    def _normalize_model_name(self, provider_type: str, model: str) -> str:
        model_name = model.strip()
        provider_prefixes = {
            "google": "gemini/",
            "anthropic": "anthropic/",
            "azure": "azure/",
            "ollama": "ollama/",
            "openai_compatible": "openai/",
        }
        prefix = provider_prefixes.get(provider_type)
        if not prefix or model_name.startswith(prefix):
            return model_name
        return f"{prefix}{model_name}"

    def _resolve_environment_api_key(self, model: str) -> str:
        if settings.llm_api_key:
            return settings.llm_api_key
        if settings.litellm_api_key:
            return settings.litellm_api_key
        if model.startswith("gemini/"):
            return settings.gemini_api_key
        return ""

    def _resolve_environment_base_url(self, model: str) -> str | None:
        if settings.litellm_base_url:
            return settings.litellm_base_url.rstrip("/")
        if model.startswith("ollama/") and settings.ollama_base_url:
            return settings.ollama_base_url.rstrip("/")
        return None

    def _read_float_option(
        self,
        config: dict[str, Any],
        name: str,
        minimum: float,
        maximum: float,
    ) -> float | None:
        value = config.get(name)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LLMProviderResolutionError(
                f"Provider option {name} must be a number."
            )
        parsed_value = float(value)
        if parsed_value < minimum or parsed_value > maximum:
            raise LLMProviderResolutionError(
                f"Provider option {name} must be between {minimum} and {maximum}."
            )
        return parsed_value

    def _read_integer_option(
        self,
        config: dict[str, Any],
        name: str,
        minimum: int,
        maximum: int,
    ) -> int | None:
        value = config.get(name)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise LLMProviderResolutionError(
                f"Provider option {name} must be an integer."
            )
        if value < minimum or value > maximum:
            raise LLMProviderResolutionError(
                f"Provider option {name} must be between {minimum} and {maximum}."
            )
        return value

    def _read_provider_options(self, config: dict[str, Any]) -> dict[str, Any]:
        supported_options = {
            "api_version",
            "deployment_id",
            "organization",
            "project",
            "region_name",
            "vertex_location",
            "vertex_project",
        }
        options: dict[str, Any] = {}
        for name in supported_options:
            value = config.get(name)
            if isinstance(value, (str, int, float, bool)):
                options[name] = value
        return options


llm_manager = LLMManager()
