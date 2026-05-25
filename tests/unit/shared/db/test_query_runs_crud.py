from collections.abc import Iterator
from uuid import uuid4

from fastapi import HTTPException
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.query_runs import (
    create_query_run,
    get_query_run,
    list_query_runs,
    mark_query_run_completed,
    mark_query_run_failed,
)
from agentic_rag.shared.db.models import QueryRun, Tenant
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.query import QueryRequest, QueryResponse, QueryRunStatus
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
    assert completed.llm_model == "gemini/gemini-2.0-flash"
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
