from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, SecretStr, model_validator

from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.common import APIModel, JsonObject, PageResponse


class ModelTier(StrEnum):
    TEXT_SMALL = "text_small"
    TEXT_LARGE = "text_large"
    EMBEDDING_SMALL = "embedding_small"
    EMBEDDING_LARGE = "embedding_large"
    RERANKER = "reranker"


class LLMTask(StrEnum):
    CLASSIFY = "classify"
    REWRITE = "rewrite"
    GENERATE = "generate"
    VERIFY = "verify"
    EMBED = "embed"
    RERANK = "rerank"


class ProviderName(StrEnum):
    LITELLM = "litellm"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class LLMProviderType(StrEnum):
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    OPENAI_COMPATIBLE = "openai_compatible"


class ModelConfig(APIModel):
    provider: ProviderName | str
    model: str = Field(..., min_length=1)
    tier: ModelTier
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    api_base: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    extra: JsonObject = Field(default_factory=dict)


class LLMRequest(APIModel):
    auth: AuthContext
    task: LLMTask
    prompt: str = Field(..., min_length=1)
    model_tier: ModelTier
    metadata: JsonObject = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1)


class LLMResponse(APIModel):
    text: str
    model: str
    provider: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_estimate: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(..., ge=0)
    metadata: JsonObject = Field(default_factory=dict)


class LLMStreamEventType(StrEnum):
    TOKEN = "token"
    COMPLETED = "completed"


class LLMStreamEvent(APIModel):
    event: LLMStreamEventType
    text_delta: str | None = None
    text: str | None = None
    model: str
    provider: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_estimate: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)
    metadata: JsonObject = Field(default_factory=dict)


class LLMMessage(APIModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(APIModel):
    auth: AuthContext | None = None
    messages: list[LLMMessage] = Field(..., min_length=1)
    provider_id: UUID | None = None
    model: str | None = None
    provider: str | None = None
    model_tier: ModelTier = ModelTier.TEXT_SMALL
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    metadata: JsonObject = Field(default_factory=dict)


class EmbeddingRequest(APIModel):
    auth: AuthContext
    texts: list[str] = Field(..., min_length=1)
    provider_id: UUID | None = None
    model: str | None = None
    provider: str | None = None
    model_tier: ModelTier = ModelTier.EMBEDDING_SMALL
    timeout_seconds: int | None = Field(default=None, ge=1)
    metadata: JsonObject = Field(default_factory=dict)


class EmbeddingResponse(APIModel):
    embeddings: list[list[float]]
    model: str
    provider: str
    dimension: int = Field(..., ge=1)
    latency_ms: int = Field(..., ge=0)


class BudgetDecision(APIModel):
    allowed: bool
    tenant_id: str
    reason: str
    remaining_budget: float | None = Field(default=None, ge=0.0)
    reset_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMProviderCreate(APIModel):
    name: str = Field(..., min_length=1, max_length=128)
    provider_type: LLMProviderType
    chat_model: str | None = Field(default=None, min_length=1, max_length=256)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=256)
    embedding_dimension: int | None = Field(default=None, ge=1, le=65536)
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    api_key: SecretStr | None = None
    config: JsonObject = Field(default_factory=dict)
    is_active: bool = True
    is_default_chat: bool = False
    is_default_embedding: bool = False

    @model_validator(mode="after")
    def validate_capabilities(self) -> "LLMProviderCreate":
        if not self.chat_model and not self.embedding_model:
            raise ValueError("At least one chat or embedding model is required.")
        if self.embedding_model and self.embedding_dimension is None:
            raise ValueError(
                "Embedding dimension is required when an embedding model is configured."
            )
        if not self.embedding_model and self.embedding_dimension is not None:
            raise ValueError(
                "Embedding model is required when an embedding dimension is configured."
            )
        if self.is_default_chat and not self.chat_model:
            raise ValueError("A default chat provider requires a chat model.")
        if self.is_default_embedding and not self.embedding_model:
            raise ValueError(
                "A default embedding provider requires an embedding model."
            )
        if not self.is_active and (
            self.is_default_chat or self.is_default_embedding
        ):
            raise ValueError("An inactive provider cannot be a default provider.")
        return self


class LLMProviderUpdate(APIModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    chat_model: str | None = Field(default=None, min_length=1, max_length=256)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=256)
    embedding_dimension: int | None = Field(default=None, ge=1, le=65536)
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    api_key: SecretStr | None = None
    clear_api_key: bool = False
    config: JsonObject | None = None
    is_active: bool | None = None
    is_default_chat: bool | None = None
    is_default_embedding: bool | None = None


class LLMProviderRead(APIModel):
    id: UUID
    tenant_id: str
    name: str
    provider_type: LLMProviderType
    chat_model: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    base_url: str | None = None
    has_api_key: bool
    config: JsonObject = Field(default_factory=dict)
    is_active: bool
    is_default_chat: bool
    is_default_embedding: bool
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime


class LLMProviderListResponse(APIModel):
    items: list[LLMProviderRead] = Field(default_factory=list)
    page: PageResponse


class LLMProviderValidationRequest(APIModel):
    capability: Literal["chat", "embedding"]


class LLMProviderValidationResponse(APIModel):
    status: Literal["healthy"]
    capability: Literal["chat", "embedding"]
    provider: str
    model: str
    latency_ms: int = Field(..., ge=0)
