from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, JsonObject


class EventType(StrEnum):
    DOCUMENT_PARSE_REQUESTED = "document.parse_requested"
    DOCUMENT_METADATA_REQUESTED = "document.metadata_requested"
    DOCUMENT_CHUNK_REQUESTED = "document.chunk_requested"
    DOCUMENT_EMBED_REQUESTED = "document.embed_requested"
    DOCUMENT_INDEX_REQUESTED = "document.index_requested"
    RAG_LONG_QUERY_REQUESTED = "rag.long_query_requested"
    EVALUATION_BATCH_REQUESTED = "evaluation.batch_requested"


class EventEnvelope(APIModel):
    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType | str
    event_version: int = Field(default=1, ge=1)
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str | None = None
    correlation_id: str = Field(..., min_length=1)
    causation_id: str | None = None
    idempotency_key: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any]


class ParseDocumentPayload(APIModel):
    job_id: UUID
    document_id: UUID
    object_key: str = Field(..., min_length=1)
    mime_type: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)


class ExtractMetadataPayload(APIModel):
    job_id: UUID
    document_id: UUID
    extracted_text_key: str = Field(..., min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class ChunkDocumentPayload(APIModel):
    job_id: UUID
    document_id: UUID
    extracted_text_key: str = Field(..., min_length=1)
    metadata: JsonObject = Field(default_factory=dict)
    acl_version: int = Field(default=1, ge=1)


class EmbedChunksPayload(APIModel):
    job_id: UUID
    document_id: UUID
    chunk_ids: list[UUID] = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    vector_version: int = Field(default=1, ge=1)


class IndexChunksPayload(APIModel):
    job_id: UUID
    document_id: UUID
    chunk_ids: list[UUID] = Field(..., min_length=1)
    index_name: str = Field(..., min_length=1)

