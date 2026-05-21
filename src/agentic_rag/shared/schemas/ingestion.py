from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, JsonObject, ORMModel, PageRequest, PageResponse
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngestionStage(StrEnum):
    CREATED = "created"
    PARSE = "parse"
    METADATA = "metadata"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    COMPLETE = "complete"


class IngestionJobCreate(APIModel):
    workspace_id: str | None = None
    source_type: DocumentSourceType
    source_uri: str | None = None
    object_key: str | None = None
    idempotency_key: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class BatchDocumentCreateRequest(APIModel):
    documents: list[DocumentCreateRequest] = Field(..., min_length=1, max_length=100)


class IngestionJobRead(ORMModel):
    id: UUID
    tenant_id: str
    workspace_id: str | None = None
    document_id: UUID | None = None
    source_type: DocumentSourceType
    source_uri: str | None = None
    object_key: str | None = None
    status: IngestionJobStatus
    current_stage: IngestionStage
    retry_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    idempotency_key: str | None = None
    created_by: str | None = None
    created_at: datetime
    updated_at: datetime


class IngestionJobResponse(APIModel):
    job_id: UUID
    document_id: UUID | None = None
    status: IngestionJobStatus
    current_stage: IngestionStage


class IngestionJobSearchRequest(APIModel):
    page: PageRequest = Field(default_factory=PageRequest)
    workspace_id: str | None = None
    status: IngestionJobStatus | None = None
    current_stage: IngestionStage | None = None


class IngestionJobSearchResponse(APIModel):
    items: list[IngestionJobRead]
    page: PageResponse

