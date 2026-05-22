from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.auth import AclPolicy
from agentic_rag.shared.schemas.common import APIModel, JsonObject, ORMModel, PageRequest, PageResponse


class DocumentSourceType(StrEnum):
    UPLOAD = "upload"
    S3 = "s3"
    URL = "url"
    CONNECTOR = "connector"


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class ClassificationLevel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class FileMetadata(APIModel):
    file_name: str | None = None
    mime_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    content_hash: str | None = None


class DocumentCreateRequest(APIModel):
    workspace_id: str | None = None
    source_type: DocumentSourceType
    source_uri: str | None = None
    title: str | None = None
    file: FileMetadata | None = None
    metadata: JsonObject = Field(default_factory=dict)
    acl: AclPolicy
    idempotency_key: str | None = None


class DocumentUpdateRequest(APIModel):
    title: str | None = None
    metadata: JsonObject | None = None
    acl: AclPolicy | None = None
    classification_level: ClassificationLevel | None = None


class DocumentRead(ORMModel):
    id: UUID
    tenant_id: str
    workspace_id: str | None = None
    source_type: DocumentSourceType
    source_uri: str | None = None
    object_key: str | None = None
    title: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    content_hash: str | None = None
    status: DocumentStatus
    owner_user_id: str | None = None
    acl_version: int = Field(..., ge=1)
    classification_level: ClassificationLevel = ClassificationLevel.INTERNAL
    metadata: JsonObject = Field(default_factory=dict, validation_alias="metadata_")
    created_by: str | None = None
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False
    deleted_at: datetime | None = None


class DocumentListItem(ORMModel):
    id: UUID
    tenant_id: str
    workspace_id: str | None = None
    title: str | None = None
    file_name: str | None = None
    source_type: DocumentSourceType
    status: DocumentStatus
    classification_level: ClassificationLevel
    created_at: datetime
    updated_at: datetime


class DocumentSearchRequest(APIModel):
    page: PageRequest = Field(default_factory=PageRequest)
    workspace_id: str | None = None
    source_type: DocumentSourceType | None = None
    status: DocumentStatus | None = None
    owner_user_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata_filters: JsonObject = Field(default_factory=dict)


class DocumentSearchResponse(APIModel):
    items: list[DocumentListItem]
    page: PageResponse


class DocumentActionResponse(APIModel):
    id: UUID
    status: str = Field(..., min_length=1)
