from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, JsonObject, ORMModel, PageRequest, PageResponse
from agentic_rag.shared.schemas.documents import ClassificationLevel


class ChunkStatus(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"


class ChunkCreate(APIModel):
    document_id: UUID
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    content_hash: str
    token_count: int = Field(..., ge=1)
    section_path: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    metadata: JsonObject = Field(default_factory=dict)
    acl_version: int = Field(default=1, ge=1)
    classification_level: ClassificationLevel = ClassificationLevel.INTERNAL


class ChunkUpdate(APIModel):
    metadata: JsonObject | None = None
    acl_version: int | None = Field(default=None, ge=1)
    classification_level: ClassificationLevel | None = None


class ChunkRead(ORMModel):
    id: UUID
    tenant_id: str
    workspace_id: str | None = None
    document_id: UUID
    chunk_index: int
    content: str
    content_hash: str
    token_count: int
    section_path: str | None = None
    page_number: int | None = None
    metadata: JsonObject = Field(default_factory=dict)
    acl_version: int
    classification_level: ClassificationLevel
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False
    deleted_at: datetime | None = None


class ChunkEmbeddingCreate(APIModel):
    chunk_id: UUID
    document_id: UUID
    embedding: list[float]
    embedding_model: str = Field(..., min_length=1)
    embedding_dimension: int = Field(..., ge=1)
    content_hash: str
    vector_version: int = Field(default=1, ge=1)
    metadata: JsonObject = Field(default_factory=dict)


class ChunkEmbeddingRead(ORMModel):
    id: UUID
    tenant_id: str
    workspace_id: str | None = None
    document_id: UUID
    chunk_id: UUID
    embedding_model: str
    embedding_dimension: int
    content_hash: str
    vector_version: int
    created_at: datetime
    updated_at: datetime


class ChunkSearchRequest(APIModel):
    document_id: UUID | None = None
    page: PageRequest = Field(default_factory=PageRequest)
    metadata_filters: JsonObject = Field(default_factory=dict)


class ChunkSearchResponse(APIModel):
    items: list[ChunkRead]
    page: PageResponse
