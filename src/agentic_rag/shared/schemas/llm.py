from enum import StrEnum
from typing import Any

from pydantic import Field

from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.common import APIModel, JsonObject


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


class LLMMessage(APIModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(APIModel):
    messages: list[LLMMessage] = Field(..., min_length=1)
    model: str | None = None
    provider: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    metadata: JsonObject = Field(default_factory=dict)


class EmbeddingRequest(APIModel):
    auth: AuthContext
    texts: list[str] = Field(..., min_length=1)
    model_tier: ModelTier = ModelTier.EMBEDDING_SMALL
    metadata: JsonObject = Field(default_factory=dict)


class EmbeddingResponse(APIModel):
    embeddings: list[list[float]]
    model: str
    dimension: int = Field(..., ge=1)
    latency_ms: int = Field(..., ge=0)


class BudgetDecision(APIModel):
    allowed: bool
    tenant_id: str
    reason: str
    remaining_budget: float | None = Field(default=None, ge=0.0)
    reset_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
