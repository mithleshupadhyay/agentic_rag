from collections.abc import Iterator
from uuid import uuid4

from fastapi import HTTPException
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.query_runs import (
    cancel_query_run,
    create_query_run,
    get_query_run,
    list_query_runs,
    mark_query_run_completed,
    mark_query_run_failed,
)
from agentic_rag.shared.db.models import QueryRun, Tenant
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.query import (
    AnswerVerificationStatus,
    QueryRequest,
    QueryResponse,
    QueryRunStatus,
)
from agentic_rag.shared.schemas.retrieval import RetrievalStrategy


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def add_tenant(db: Session, tenant_id: str) -> None:
    db.add(
        Tenant(
            tenant_id=tenant_id,
            name=tenant_id.title(),
            slug=tenant_id,
            status="active",
            metadata_={},
        )
    )
    db.commit()


def test_create_and_complete_query_run(db: Session) -> None:
    add_tenant(db, "tenant-a")
    document_id = uuid4()
    chunk_id = uuid4()
    agent_run_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )
    request = QueryRequest(
        query="security policy",
        workspace_id="workspace-a",
        retrieval_limit=5,
        max_context_chunks=2,
        max_context_tokens=500,
    )

    query_run = create_query_run(
        user_context=user_context,
        db=db,
        request=request,
        agent_run_id=agent_run_id,
        request_id="request-id-1",
    )
    citation = Citation(
        document_id=document_id,
        chunk_id=chunk_id,
        title="Security Policy",
        quote="Security policy content.",
        score=1.2,
    )
    completed = mark_query_run_completed(
        db=db,
        query_run=query_run,
        response=QueryResponse(
            agent_run_id=agent_run_id,
            answer="Security policy content [1].",
            citations=[citation],
            context_token_count=3,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=25,
            synthesis_enabled=True,
            llm_provider="litellm",
            llm_model="gemini/gemini-2.0-flash",
            llm_input_tokens=100,
            llm_output_tokens=12,
            llm_cost_estimate=0.001,
            verification_status=AnswerVerificationStatus.PASSED,
            verification_reason="Answer citations match retrieved context.",
        ),
    )

    assert completed.id == agent_run_id
    assert completed.status == QueryRunStatus.COMPLETED
    assert completed.tenant_id == "tenant-a"
    assert completed.workspace_id == "workspace-a"
    assert completed.user_id == "user-1"
    assert completed.request_id == "request-id-1"
    assert completed.query_text == "security policy"
    assert completed.retrieval_limit == 5
    assert completed.answer == "Security policy content [1]."
    assert completed.citations["items"][0]["title"] == "Security Policy"
    assert completed.response_payload["agent_run_id"] == str(agent_run_id)
    assert completed.latency_ms == 25
    assert completed.context_token_count == 3
    assert completed.llm_input_tokens == 100
    assert completed.llm_output_tokens == 12
    assert completed.llm_cost_estimate == 0.001
    assert completed.llm_model == "gemini/gemini-2.0-flash"
    assert completed.verification_status == "passed"
    assert completed.verification_reason == "Answer citations match retrieved context."
    assert completed.response_payload["verification_status"] == "passed"
    assert completed.completed_at is not None


def test_mark_query_run_failed(db: Session) -> None:
    add_tenant(db, "tenant-a")
    agent_run_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    query_run = create_query_run(
        user_context=user_context,
        db=db,
        request=QueryRequest(query="security policy"),
        agent_run_id=agent_run_id,
    )

    failed = mark_query_run_failed(
        db=db,
        query_run=query_run,
        error_type="RuntimeError",
        error_message="retrieval failed",
        latency_ms=13,
    )

    assert failed.status == QueryRunStatus.FAILED
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "retrieval failed"
    assert failed.latency_ms == 13
    assert failed.completed_at is not None


def test_mark_query_run_failed_defaults_missing_latency_to_zero(db: Session) -> None:
    add_tenant(db, "tenant-a")
    agent_run_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    query_run = create_query_run(
        user_context=user_context,
        db=db,
        request=QueryRequest(query="security policy"),
        agent_run_id=agent_run_id,
    )

    failed = mark_query_run_failed(
        db=db,
        query_run=query_run,
        error_type="RuntimeError",
        error_message="retrieval failed",
    )

    assert failed.status == QueryRunStatus.FAILED
    assert failed.latency_ms == 0
    assert failed.llm_input_tokens == 0
    assert failed.llm_output_tokens == 0
    assert failed.llm_cost_estimate == 0.0


def test_cancel_query_run_marks_running_run_cancelled(db: Session) -> None:
    add_tenant(db, "tenant-a")
    agent_run_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    create_query_run(
        user_context=user_context,
        db=db,
        request=QueryRequest(query="security policy"),
        agent_run_id=agent_run_id,
    )

    cancelled = cancel_query_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
    )

    assert cancelled is not None
    assert cancelled.id == agent_run_id
    assert cancelled.status == QueryRunStatus.CANCELLED
    assert cancelled.completed_at is not None


def test_cancel_query_run_is_tenant_scoped(db: Session) -> None:
    add_tenant(db, "tenant-a")
    add_tenant(db, "tenant-b")
    agent_run_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    create_query_run(
        user_context=user_context,
        db=db,
        request=QueryRequest(query="security policy"),
        agent_run_id=agent_run_id,
    )

    cancelled = cancel_query_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-b",
    )
    stored = get_query_run(db, agent_run_id, "tenant-a")

    assert cancelled is None
    assert stored is not None
    assert stored.status == QueryRunStatus.RUNNING
    assert stored.completed_at is None


def test_cancel_query_run_rejects_completed_run(db: Session) -> None:
    add_tenant(db, "tenant-a")
    document_id = uuid4()
    chunk_id = uuid4()
    agent_run_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    query_run = create_query_run(
        user_context=user_context,
        db=db,
        request=QueryRequest(query="security policy"),
        agent_run_id=agent_run_id,
    )
    citation = Citation(
        document_id=document_id,
        chunk_id=chunk_id,
        title="Security Policy",
        quote="Security policy content.",
        score=1.2,
    )
    mark_query_run_completed(
        db=db,
        query_run=query_run,
        response=QueryResponse(
            agent_run_id=agent_run_id,
            answer="Security policy content [1].",
            citations=[citation],
            context_token_count=3,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=25,
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        cancel_query_run(
            db=db,
            agent_run_id=agent_run_id,
            tenant_id="tenant-a",
        )

    assert exc_info.value.status_code == 409
    assert "completed" in exc_info.value.detail


def test_get_and_list_query_runs_are_tenant_scoped(db: Session) -> None:
    add_tenant(db, "tenant-a")
    add_tenant(db, "tenant-b")
    tenant_a_context = UserContext(
        id="user-a",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    tenant_b_context = UserContext(
        id="user-b",
        customer_id="tenant-b",
        tenant_id="tenant-b",
    )
    tenant_a_run = create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="tenant a query"),
        agent_run_id=uuid4(),
        request_id="tenant-a-request",
    )
    create_query_run(
        user_context=tenant_b_context,
        db=db,
        request=QueryRequest(query="tenant b query"),
        agent_run_id=uuid4(),
        request_id="tenant-b-request",
    )

    assert get_query_run(db, tenant_a_run.id, "tenant-a") is not None
    assert get_query_run(db, tenant_a_run.id, "tenant-b") is None

    tenant_a_runs, tenant_a_total = list_query_runs(db, "tenant-a")
    tenant_b_runs, tenant_b_total = list_query_runs(db, "tenant-b")

    assert tenant_a_total == 1
    assert tenant_b_total == 1
    assert [query_run.query_text for query_run in tenant_a_runs] == ["tenant a query"]
    assert [query_run.query_text for query_run in tenant_b_runs] == ["tenant b query"]

    filtered_runs, filtered_total = list_query_runs(
        db,
        "tenant-a",
        request_id="tenant-a-request",
    )

    assert filtered_total == 1
    assert filtered_runs[0].id == tenant_a_run.id


def test_list_query_runs_filters_by_status(db: Session) -> None:
    add_tenant(db, "tenant-a")
    add_tenant(db, "tenant-b")
    document_id = uuid4()
    chunk_id = uuid4()
    tenant_a_context = UserContext(
        id="user-a",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    tenant_b_context = UserContext(
        id="user-b",
        customer_id="tenant-b",
        tenant_id="tenant-b",
    )
    completed_run_id = uuid4()
    failed_run_id = uuid4()
    running_run_id = uuid4()
    tenant_b_run_id = uuid4()

    completed_run = create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="completed query"),
        agent_run_id=completed_run_id,
    )
    failed_run = create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="failed query"),
        agent_run_id=failed_run_id,
    )
    create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="running query"),
        agent_run_id=running_run_id,
    )
    tenant_b_run = create_query_run(
        user_context=tenant_b_context,
        db=db,
        request=QueryRequest(query="tenant b query"),
        agent_run_id=tenant_b_run_id,
    )

    mark_query_run_completed(
        db=db,
        query_run=completed_run,
        response=QueryResponse(
            agent_run_id=completed_run_id,
            answer="Security policy content [1].",
            citations=[
                Citation(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    title="Security Policy",
                    quote="Security policy content.",
                    score=1.2,
                )
            ],
            context_token_count=3,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=25,
        ),
    )
    mark_query_run_failed(
        db=db,
        query_run=failed_run,
        error_type="RuntimeError",
        error_message="retrieval failed",
        latency_ms=13,
    )
    mark_query_run_failed(
        db=db,
        query_run=tenant_b_run,
        error_type="RuntimeError",
        error_message="tenant b retrieval failed",
        latency_ms=21,
    )

    completed_runs, completed_total = list_query_runs(
        db,
        "tenant-a",
        status=QueryRunStatus.COMPLETED,
    )
    failed_runs, failed_total = list_query_runs(
        db,
        "tenant-a",
        status=QueryRunStatus.FAILED,
    )
    running_runs, running_total = list_query_runs(
        db,
        "tenant-a",
        status=QueryRunStatus.RUNNING,
    )

    assert completed_total == 1
    assert [query_run.id for query_run in completed_runs] == [completed_run_id]
    assert failed_total == 1
    assert [query_run.id for query_run in failed_runs] == [failed_run_id]
    assert running_total == 1
    assert [query_run.id for query_run in running_runs] == [running_run_id]


def test_list_query_runs_filters_by_verification_status(db: Session) -> None:
    add_tenant(db, "tenant-a")
    add_tenant(db, "tenant-b")
    document_id = uuid4()
    chunk_id = uuid4()
    tenant_a_context = UserContext(
        id="user-a",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    tenant_b_context = UserContext(
        id="user-b",
        customer_id="tenant-b",
        tenant_id="tenant-b",
    )
    passed_run_id = uuid4()
    failed_run_id = uuid4()
    skipped_run_id = uuid4()
    not_required_run_id = uuid4()
    tenant_b_run_id = uuid4()

    passed_run = create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="passed query"),
        agent_run_id=passed_run_id,
    )
    failed_run = create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="failed verification query"),
        agent_run_id=failed_run_id,
    )
    skipped_run = create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="skipped verification query"),
        agent_run_id=skipped_run_id,
    )
    not_required_run = create_query_run(
        user_context=tenant_a_context,
        db=db,
        request=QueryRequest(query="no synthesis query"),
        agent_run_id=not_required_run_id,
    )
    tenant_b_run = create_query_run(
        user_context=tenant_b_context,
        db=db,
        request=QueryRequest(query="tenant b query"),
        agent_run_id=tenant_b_run_id,
    )

    mark_query_run_completed(
        db=db,
        query_run=passed_run,
        response=QueryResponse(
            agent_run_id=passed_run_id,
            answer="Security policy content [1].",
            citations=[
                Citation(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    title="Security Policy",
                    quote="Security policy content.",
                    score=1.2,
                )
            ],
            context_token_count=3,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=25,
            verification_status=AnswerVerificationStatus.PASSED,
            verification_reason="Answer citations match retrieved context.",
        ),
    )
    mark_query_run_completed(
        db=db,
        query_run=failed_run,
        response=QueryResponse(
            agent_run_id=failed_run_id,
            answer="Security policy content without citation.",
            citations=[],
            context_token_count=3,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=31,
            verification_status=AnswerVerificationStatus.FAILED,
            verification_reason="Answer did not cite retrieved context.",
        ),
    )
    mark_query_run_completed(
        db=db,
        query_run=skipped_run,
        response=QueryResponse(
            agent_run_id=skipped_run_id,
            answer="Synthesis failed before verification.",
            citations=[],
            context_token_count=0,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=19,
            verification_status=AnswerVerificationStatus.SKIPPED,
            verification_reason="LLM synthesis failed before verification.",
        ),
    )
    mark_query_run_completed(
        db=db,
        query_run=not_required_run,
        response=QueryResponse(
            agent_run_id=not_required_run_id,
            answer="Security policy content.",
            citations=[],
            context_token_count=0,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=11,
        ),
    )
    mark_query_run_completed(
        db=db,
        query_run=tenant_b_run,
        response=QueryResponse(
            agent_run_id=tenant_b_run_id,
            answer="Tenant B security policy content [1].",
            citations=[
                Citation(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    title="Tenant B Security Policy",
                    quote="Tenant B security policy content.",
                    score=1.1,
                )
            ],
            context_token_count=3,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=23,
            verification_status=AnswerVerificationStatus.PASSED,
            verification_reason="Answer citations match retrieved context.",
        ),
    )

    passed_runs, passed_total = list_query_runs(
        db,
        "tenant-a",
        verification_status=AnswerVerificationStatus.PASSED,
    )
    failed_runs, failed_total = list_query_runs(
        db,
        "tenant-a",
        verification_status=AnswerVerificationStatus.FAILED,
    )
    skipped_runs, skipped_total = list_query_runs(
        db,
        "tenant-a",
        verification_status=AnswerVerificationStatus.SKIPPED,
    )
    not_required_runs, not_required_total = list_query_runs(
        db,
        "tenant-a",
        verification_status=AnswerVerificationStatus.NOT_REQUIRED,
    )

    assert passed_total == 1
    assert [query_run.id for query_run in passed_runs] == [passed_run_id]
    assert failed_total == 1
    assert [query_run.id for query_run in failed_runs] == [failed_run_id]
    assert skipped_total == 1
    assert [query_run.id for query_run in skipped_runs] == [skipped_run_id]
    assert not_required_total == 1
    assert [query_run.id for query_run in not_required_runs] == [not_required_run_id]


def test_create_query_run_rolls_back_on_integrity_error(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    agent_run_id = uuid4()
    create_query_run(
        user_context=user_context,
        db=db,
        request=QueryRequest(query="security policy"),
        agent_run_id=agent_run_id,
    )

    with pytest.raises(HTTPException) as exc_info:
        create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="security policy"),
            agent_run_id=agent_run_id,
        )

    assert exc_info.value.status_code == 400
    assert db.query(QueryRun).count() == 1
