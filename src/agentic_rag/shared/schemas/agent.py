from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.common import APIModel, Citation, JsonObject, ORMModel
from agentic_rag.shared.schemas.retrieval import CandidateChunk, ContextChunk, RetrievalStrategy


class AgentRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    HANDOFF_REQUIRED = "handoff_required"
    CANCELLED = "cancelled"


class AgentNodeName(StrEnum):
    CLASSIFY_INTENT = "classify_intent"
    REWRITE_QUERY = "rewrite_query"
    PLAN_FILTERS = "plan_filters"
    SELECT_RETRIEVAL_STRATEGY = "select_retrieval_strategy"
    METADATA_SEARCH = "metadata_search"
    BM25_SEARCH = "bm25_search"
    VECTOR_SEARCH = "vector_search"
    MERGE_CANDIDATES = "merge_candidates"
    FILTER_AUTHORIZED_CHUNKS = "filter_authorized_chunks"
    RERANK = "rerank"
    BUILD_CONTEXT = "build_context"
    GENERATE_ANSWER = "generate_answer"
    VERIFY_GROUNDING = "verify_grounding"
    FINALIZE = "finalize"
    FALLBACK = "fallback"
    HUMAN_HANDOFF = "human_handoff"


class AgentLimits(APIModel):
    max_steps: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    step_timeout_seconds: int = Field(default=20, ge=1)
    total_timeout_seconds: int = Field(default=90, ge=1)
    max_same_tool_repeat: int = Field(default=2, ge=1)
    max_retries_per_node: int = Field(default=2, ge=0)


class ToolCallRecord(APIModel):
    tool_name: str = Field(..., min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    result_hash: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)


class AgentStateModel(APIModel):
    agent_run_id: UUID
    auth: AuthContext
    query: str
    rewritten_query: str | None = None
    intent: str | None = None
    filters: JsonObject = Field(default_factory=dict)
    retrieval_strategy: RetrievalStrategy | None = None
    retrieved_candidates: list[CandidateChunk] = Field(default_factory=list)
    authorized_chunks: list[CandidateChunk] = Field(default_factory=list)
    reranked_chunks: list[CandidateChunk] = Field(default_factory=list)
    context: list[ContextChunk] = Field(default_factory=list)
    draft_answer: str | None = None
    final_answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    step_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    visited_nodes: list[str] = Field(default_factory=list)
    last_tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    last_results_hash: str | None = None
    deadline_at: datetime
    handoff_required: bool = False


class AgentRunRead(ORMModel):
    id: UUID
    tenant_id: str
    workspace_id: str | None = None
    user_id: str
    query: str
    status: AgentRunStatus
    retrieval_strategy: RetrievalStrategy | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    started_at: datetime
    completed_at: datetime | None = None
    total_steps: int = Field(default=0, ge=0)
    total_tool_calls: int = Field(default=0, ge=0)
    timeout_at: datetime


class AgentStepRead(ORMModel):
    id: UUID
    agent_run_id: UUID
    tenant_id: str
    node_name: AgentNodeName | str
    step_number: int = Field(..., ge=0)
    tool_name: str | None = None
    tool_input: JsonObject = Field(default_factory=dict)
    tool_output_summary: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    status: str
    error_type: str | None = None
    created_at: datetime


class AgentCheckpoint(APIModel):
    agent_run_id: UUID
    checkpoint_key: str = Field(..., min_length=1)
    state: dict[str, Any]
    created_at: datetime

