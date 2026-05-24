from enum import StrEnum
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.common import APIModel, Citation, JsonObject


class RetrievalStrategy(StrEnum):
    METADATA = "metadata"
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    AGENTIC = "agentic"


class RetrievalTool(StrEnum):
    METADATA_SEARCH = "metadata_search"
    BM25_SEARCH = "bm25_search"
    VECTOR_SEARCH = "vector_search"
    DOCUMENT_FETCH = "document_fetch"
    RERANK = "rerank"


class RetrievalFilters(APIModel):
    workspace_id: str | None = None
    document_ids: list[UUID] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)
    date_range: JsonObject = Field(default_factory=dict)


class RetrievalRequest(APIModel):
    auth: AuthContext
    query: str = Field(..., min_length=1)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    limit: int = Field(default=20, ge=1, le=200)
    deadline_ms: int = Field(default=1500, ge=100)


class BM25SearchRequest(APIModel):
    query: str = Field(..., min_length=1)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    limit: int = Field(default=20, ge=1, le=200)
    deadline_ms: int = Field(default=1500, ge=100)


class CandidateDocument(APIModel):
    document_id: UUID
    score: float = Field(..., ge=0.0)
    source: RetrievalTool | str
    metadata: JsonObject = Field(default_factory=dict)


class CandidateChunk(APIModel):
    chunk_id: UUID
    document_id: UUID
    content: str | None = None
    score: float = Field(..., ge=0.0)
    source: RetrievalTool | str
    metadata: JsonObject = Field(default_factory=dict)
    citation: Citation | None = None


class RetrievalResponse(APIModel):
    strategy: RetrievalStrategy
    candidates: list[CandidateChunk] = Field(default_factory=list)
    latency_ms: int = Field(..., ge=0)


class RerankRequest(APIModel):
    query: str = Field(..., min_length=1)
    candidates: list[CandidateChunk]
    top_k: int = Field(default=12, ge=1, le=100)


class RerankResponse(APIModel):
    chunks: list[CandidateChunk]
    latency_ms: int = Field(..., ge=0)


class ContextChunk(APIModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    token_count: int = Field(..., ge=1)
    citation: Citation
    metadata: JsonObject = Field(default_factory=dict)


class ContextBuildRequest(APIModel):
    query: str = Field(..., min_length=1)
    chunks: list[CandidateChunk]
    max_context_chunks: int = Field(default=12, ge=1, le=50)
    max_tokens: int = Field(default=6000, ge=500)


class ContextBuildResponse(APIModel):
    context: list[ContextChunk]
    token_count: int = Field(..., ge=0)
