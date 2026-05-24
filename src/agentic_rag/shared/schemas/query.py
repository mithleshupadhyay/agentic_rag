from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, Citation
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    ContextChunk,
    RetrievalFilters,
    RetrievalStrategy,
)


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationMessage(APIModel):
    role: MessageRole
    content: str = Field(..., min_length=1)
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(APIModel):
    query: str = Field(..., min_length=1)
    workspace_id: str | None = None
    conversation_id: str | None = None
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    history: list[ConversationMessage] = Field(default_factory=list)
    stream: bool = False
    retrieval_limit: int = Field(default=20, ge=1, le=200)
    max_context_chunks: int = Field(default=12, ge=1, le=50)
    max_context_tokens: int = Field(default=6000, ge=500)


class QueryResponse(APIModel):
    agent_run_id: UUID
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    candidates: list[CandidateChunk] = Field(default_factory=list)
    context: list[ContextChunk] = Field(default_factory=list)
    context_token_count: int = Field(default=0, ge=0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_strategy: RetrievalStrategy
    latency_ms: int = Field(..., ge=0)
    synthesis_enabled: bool = False


class QueryRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueryRunRead(APIModel):
    agent_run_id: UUID
    status: QueryRunStatus
    created_at: datetime
    completed_at: datetime | None = None
    response: QueryResponse | None = None


class QueryStreamEvent(APIModel):
    event: str = Field(..., min_length=1)
    agent_run_id: UUID
    data: dict[str, Any] = Field(default_factory=dict)
