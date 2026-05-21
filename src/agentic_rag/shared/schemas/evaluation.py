from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from agentic_rag.shared.schemas.common import APIModel, Citation, JsonObject, ORMModel


class FeedbackRating(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EvaluationStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GoldenQuestion(APIModel):
    question: str = Field(..., min_length=1)
    expected_answer: str | None = None
    expected_document_ids: list[UUID] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)


class FeedbackEventCreate(APIModel):
    agent_run_id: UUID
    rating: FeedbackRating
    feedback_text: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class FeedbackEventRead(ORMModel):
    id: UUID
    tenant_id: str
    user_id: str
    agent_run_id: UUID
    rating: FeedbackRating
    feedback_text: str | None = None
    created_at: datetime


class EvaluationMetric(APIModel):
    name: str = Field(..., min_length=1)
    value: float
    threshold: float | None = None
    passed: bool | None = None


class EvaluationRunCreate(APIModel):
    dataset_name: str = Field(..., min_length=1)
    questions: list[GoldenQuestion] = Field(..., min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class EvaluationRunRead(ORMModel):
    id: UUID
    tenant_id: str
    dataset_name: str
    status: EvaluationStatus
    metrics: list[EvaluationMetric] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None


class AnswerEvaluation(APIModel):
    agent_run_id: UUID
    grounded: bool
    citation_coverage: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

