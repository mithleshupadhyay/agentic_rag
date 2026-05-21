from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


JsonObject = dict[str, Any]
HealthState = Literal["healthy", "degraded", "unhealthy"]


class APIModel(BaseModel):
    """Base schema for strict API, worker, and event contracts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ORMModel(APIModel):
    """Base schema for SQLAlchemy model serialization."""

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
    )


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class DependencyStatus(APIModel):
    name: str = Field(..., min_length=1)
    status: HealthState
    detail: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)


class HealthResponse(APIModel):
    service: str = Field(..., min_length=1)
    status: HealthState
    version: str = Field(..., min_length=1)
    dependencies: dict[str, DependencyStatus] = Field(default_factory=dict)


class ErrorDetail(APIModel):
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    field: str | None = None


class ErrorResponse(APIModel):
    request_id: str | None = None
    errors: list[ErrorDetail] = Field(default_factory=list)


class PageRequest(APIModel):
    page: int = Field(default=1, ge=1)
    size: int = Field(default=50, ge=1, le=500)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PageResponse(APIModel):
    page: int = Field(..., ge=1)
    size: int = Field(..., ge=1)
    total: int = Field(..., ge=0)


class SortSpec(APIModel):
    field: str = Field(..., min_length=1)
    direction: SortDirection = SortDirection.ASC


class DateRange(APIModel):
    start: datetime | None = None
    end: datetime | None = None


class TenantScopedRead(ORMModel):
    id: UUID
    tenant_id: str
    workspace_id: str | None = None
    created_at: datetime
    updated_at: datetime


class Citation(APIModel):
    document_id: UUID
    chunk_id: UUID
    title: str | None = None
    source_uri: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_path: str | None = None
    quote: str | None = None
    score: float | None = Field(default=None, ge=0.0)

