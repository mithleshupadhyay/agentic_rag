import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.query_runs import (
    create_query_run,
    get_query_run,
    mark_query_run_completed,
    mark_query_run_failed,
)
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.llm import LLMResponse
from agentic_rag.shared.schemas.agent import (
    AgentRunStatus,
    AgentStreamEvent,
    AgentStreamEventType,
)
from agentic_rag.shared.schemas.query import (
    AnswerVerificationStatus,
    QueryRequest,
    QueryResponse,
)
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    ContextChunk,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTool,
)


def client_with_user(
    user_context: UserContext,
    db: Session | None = None,
) -> Iterator[TestClient]:
    async def override_get_current_user() -> UserContext:
        return user_context

    app.dependency_overrides[get_current_user] = override_get_current_user
    if db is not None:
        def override_get_session():
            yield db

        app.dependency_overrides[get_session] = override_get_session

    test_client = TestClient(app)

    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


def create_test_db() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(
        Tenant(
            tenant_id="tenant-a",
            name="Tenant A",
            slug="tenant-a",
            status="active",
            metadata_={},
        )
    )
    session.commit()
    return session


def test_query_endpoint_returns_grounded_retrieval_output(monkeypatch, caplog) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        scopes=["query:run"],
    )
    captured = {}

    def fake_run_bm25_query(user_context, request, db, request_id):
        captured["user_context"] = user_context
        captured["request"] = request
        captured["db"] = db
        captured["request_id"] = request_id
        citation = Citation(
            document_id=document_id,
            chunk_id=chunk_id,
            title="Security Policy",
            quote="Security policy content.",
            score=2.1,
        )
        return QueryResponse(
            agent_run_id=uuid4(),
            answer="LLM synthesis is not enabled yet. Retrieved 1 context chunks for this query.",
            citations=[citation],
            context=[
                ContextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content="Security policy content.",
                    token_count=3,
                    citation=citation,
                )
            ],
            context_token_count=3,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=15,
            synthesis_enabled=False,
        )

    monkeypatch.setattr(
        "agentic_rag.api.query.run_bm25_query",
        fake_run_bm25_query,
    )

    caplog.set_level(logging.INFO, logger="agentic_rag.api.query")
    for client in client_with_user(user_context):
        response = client.post(
            "/query",
            headers={"X-Request-ID": "query-request-id"},
            json={
                "query": "security policy",
                "workspace_id": "workspace-a",
                "retrieval_limit": 8,
                "max_context_chunks": 3,
                "max_context_tokens": 500,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_strategy"] == "bm25"
    assert body["synthesis_enabled"] is False
    assert body["context_token_count"] == 3
    assert body["context"][0]["content"] == "Security policy content."
    assert body["citations"][0]["title"] == "Security Policy"
    assert captured["user_context"].id == "user-1"
    assert captured["request"].query == "security policy"
    assert captured["request"].workspace_id == "workspace-a"
    assert captured["request"].retrieval_limit == 8
    assert captured["db"] is not None
    assert captured["request_id"] == "query-request-id"
    assert response.headers["X-Request-ID"] == "query-request-id"
    assert "request_id=query-request-id" in caplog.text


def test_query_endpoint_persists_synthesized_answer(monkeypatch) -> None:
    db = create_test_db()
    try:
        document_id = uuid4()
        chunk_id = uuid4()
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        captured = {}

        def fake_search_bm25_chunks(user_context, query, filters, limit):
            captured["query"] = query
            captured["filters"] = filters
            captured["limit"] = limit
            return RetrievalResponse(
                strategy=RetrievalStrategy.BM25,
                candidates=[
                    CandidateChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content="Security policy content.",
                        score=2.3,
                        source=RetrievalTool.BM25_SEARCH,
                        metadata={"token_count": 3},
                        citation=Citation(
                            document_id=document_id,
                            chunk_id=chunk_id,
                            title="Security Policy",
                            source_uri="upload://security-policy.txt",
                            quote="Security policy content.",
                            score=2.3,
                        ),
                    )
                ],
                latency_ms=11,
            )

        def fake_generate_chat_completion(request):
            captured["llm_request"] = request
            return LLMResponse(
                text="Security policy content is available in the retrieved document [1].",
                model="gemini/gemini-2.0-flash",
                provider="litellm",
                input_tokens=128,
                output_tokens=14,
                cost_estimate=0.001,
                latency_ms=20,
            )

        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.search_bm25_chunks",
            fake_search_bm25_chunks,
        )
        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.generate_chat_completion",
            fake_generate_chat_completion,
        )
        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.settings.llm_synthesis_enabled",
            True,
        )
        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.settings.query_cache_enabled",
            False,
        )

        for client in client_with_user(user_context, db):
            response = client.post(
                "/query",
                headers={"X-Request-ID": "synthesis-request-id"},
                json={
                    "query": "security policy",
                    "workspace_id": "workspace-a",
                    "retrieval_limit": 8,
                    "max_context_chunks": 3,
                    "max_context_tokens": 500,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert (
            body["answer"]
            == "Security policy content is available in the retrieved document [1]."
        )
        assert body["synthesis_enabled"] is True
        assert body["llm_provider"] == "litellm"
        assert body["llm_model"] == "gemini/gemini-2.0-flash"
        assert body["llm_input_tokens"] == 128
        assert body["llm_output_tokens"] == 14
        assert body["llm_cost_estimate"] == 0.001
        assert body["verification_status"] == "passed"
        assert body["verification_reason"] == "Answer citations match retrieved context."
        assert body["context_token_count"] == 3
        assert body["citations"][0]["title"] == "Security Policy"
        assert response.headers["X-Request-ID"] == "synthesis-request-id"
        assert captured["query"] == "security policy"
        assert captured["filters"].workspace_id == "workspace-a"
        assert captured["limit"] == 8
        assert "Use only the authorized context" in captured["llm_request"].messages[0].content
        assert "Security policy content." in captured["llm_request"].messages[1].content

        query_run = get_query_run(
            db=db,
            agent_run_id=UUID(body["agent_run_id"]),
            tenant_id="tenant-a",
        )
        assert query_run is not None
        assert query_run.request_id == "synthesis-request-id"
        assert query_run.status == "completed"
        assert query_run.synthesis_enabled is True
        assert query_run.llm_provider == "litellm"
        assert query_run.llm_model == "gemini/gemini-2.0-flash"
        assert query_run.llm_input_tokens == 128
        assert query_run.llm_output_tokens == 14
        assert query_run.llm_cost_estimate == 0.001
        assert query_run.verification_status == "passed"
        assert query_run.verification_reason == "Answer citations match retrieved context."
        assert query_run.response_payload["verification_status"] == "passed"
        assert query_run.response_payload["synthesis_enabled"] is True
    finally:
        db.close()


def test_query_endpoint_persists_synthesis_fallback(monkeypatch) -> None:
    db = create_test_db()
    try:
        document_id = uuid4()
        chunk_id = uuid4()
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )

        def fake_search_bm25_chunks(user_context, query, filters, limit):
            return RetrievalResponse(
                strategy=RetrievalStrategy.BM25,
                candidates=[
                    CandidateChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content="Security policy content.",
                        score=2.3,
                        source=RetrievalTool.BM25_SEARCH,
                        metadata={"token_count": 3},
                        citation=Citation(
                            document_id=document_id,
                            chunk_id=chunk_id,
                            title="Security Policy",
                            source_uri="upload://security-policy.txt",
                            quote="Security policy content.",
                            score=2.3,
                        ),
                    )
                ],
                latency_ms=11,
            )

        def fake_generate_chat_completion(request):
            raise RuntimeError("provider timeout")

        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.search_bm25_chunks",
            fake_search_bm25_chunks,
        )
        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.generate_chat_completion",
            fake_generate_chat_completion,
        )
        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.settings.llm_synthesis_enabled",
            True,
        )
        monkeypatch.setattr(
            "agentic_rag.query.bm25_query.settings.query_cache_enabled",
            False,
        )

        for client in client_with_user(user_context, db):
            response = client.post(
                "/query",
                headers={"X-Request-ID": "fallback-request-id"},
                json={
                    "query": "security policy",
                    "workspace_id": "workspace-a",
                    "retrieval_limit": 8,
                    "max_context_chunks": 3,
                    "max_context_tokens": 500,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == (
            "LLM synthesis is temporarily unavailable, so I am returning "
            "the best authorized retrieved excerpts instead.\n\n"
            "I found these relevant excerpts in the selected documents:\n"
            "- Security policy content. [1] Source: Security Policy."
        )
        assert body["synthesis_enabled"] is False
        assert body["synthesis_error"] == "LLM synthesis failed"
        assert body["verification_status"] == "skipped"
        assert body["verification_reason"] == (
            "LLM synthesis failed before verification. error_type=RuntimeError"
        )
        assert body["llm_provider"] is None
        assert body["llm_model"] is None
        assert body["llm_input_tokens"] == 0
        assert body["llm_output_tokens"] == 0
        assert body["llm_cost_estimate"] == 0.0
        assert body["context_token_count"] == 3
        assert body["context"][0]["content"] == "Security policy content."
        assert body["citations"][0]["title"] == "Security Policy"

        query_run = get_query_run(
            db=db,
            agent_run_id=UUID(body["agent_run_id"]),
            tenant_id="tenant-a",
        )
        assert query_run is not None
        assert query_run.status == "completed"
        assert query_run.request_id == "fallback-request-id"
        assert query_run.answer == body["answer"]
        assert query_run.synthesis_enabled is False
        assert query_run.error_type is None
        assert query_run.error_message == "LLM synthesis failed"
        assert query_run.verification_status == "skipped"
        assert query_run.verification_reason == (
            "LLM synthesis failed before verification. error_type=RuntimeError"
        )
        assert query_run.llm_provider is None
        assert query_run.llm_model is None
        assert query_run.llm_input_tokens == 0
        assert query_run.llm_output_tokens == 0
        assert query_run.llm_cost_estimate == 0.0
        assert query_run.response_payload["synthesis_error"] == "LLM synthesis failed"
        assert query_run.response_payload["verification_status"] == "skipped"
    finally:
        db.close()


def test_query_endpoint_requires_query_scope() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["documents:read"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/query",
            json={"query": "security policy"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Missing required scope: query:run"


def test_query_endpoint_records_completed_metrics(monkeypatch) -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["query:run"],
    )
    started_labels = {
        "status": "started",
        "retrieval_strategy": "bm25",
        "synthesis_enabled": "false",
    }
    completed_labels = {
        "status": "completed",
        "retrieval_strategy": "bm25",
        "synthesis_enabled": "false",
    }
    started_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_query_lifecycle_total",
            started_labels,
        )
        or 0
    )
    completed_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_query_lifecycle_total",
            completed_labels,
        )
        or 0
    )
    latency_count_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_query_latency_seconds_count",
            completed_labels,
        )
        or 0
    )

    def fake_run_bm25_query(user_context, request, db, request_id):
        return QueryResponse(
            agent_run_id=uuid4(),
            answer="LLM synthesis is not enabled yet. Retrieved 0 context chunks for this query.",
            context_token_count=0,
            confidence_score=0.0,
            retrieval_strategy=RetrievalStrategy.BM25,
            latency_ms=25,
            synthesis_enabled=False,
        )

    monkeypatch.setattr(
        "agentic_rag.api.query.run_bm25_query",
        fake_run_bm25_query,
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/query",
            json={"query": "security policy"},
        )

    assert response.status_code == 200
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_query_lifecycle_total",
            started_labels,
        )
        == started_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_query_lifecycle_total",
            completed_labels,
        )
        == completed_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_query_latency_seconds_count",
            completed_labels,
        )
        == latency_count_before + 1
    )


def test_query_endpoint_validates_request_body() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["query:run"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/query",
            json={
                "query": "security policy",
                "max_context_tokens": 50,
            },
        )

    assert response.status_code == 422


def test_query_stream_endpoint_returns_runtime_events(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        scopes=["query:run"],
    )
    captured = {}

    def fake_stream_agent_runtime_graph(**kwargs):
        captured.update(kwargs)
        citation = Citation(
            document_id=document_id,
            chunk_id=chunk_id,
            title="Security Policy",
            quote="Security policy content.",
            score=2.1,
        )
        context_chunk = ContextChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content="Security policy content.",
            token_count=3,
            citation=citation,
        )
        yield AgentStreamEvent(
            event=AgentStreamEventType.AGENT_STARTED,
            agent_run_id=kwargs["agent_run_id"],
            status=AgentRunStatus.RUNNING,
            created_at=now,
        )
        yield AgentStreamEvent(
            event=AgentStreamEventType.AGENT_STEP_COMPLETED,
            agent_run_id=kwargs["agent_run_id"],
            node_name="bm25_search",
            status=AgentRunStatus.RUNNING,
            step_number=5,
            data={"checkpoint_key": "step-0005-bm25_search"},
            created_at=now,
        )
        yield AgentStreamEvent(
            event=AgentStreamEventType.ANSWER_TOKEN,
            agent_run_id=kwargs["agent_run_id"],
            node_name="generate_answer",
            status=AgentRunStatus.RUNNING,
            text_delta="Security policy content",
            data={"token_index": 1, "model": "test-model", "provider": "test-provider"},
            created_at=now,
        )
        yield AgentStreamEvent(
            event=AgentStreamEventType.AGENT_COMPLETED,
            agent_run_id=kwargs["agent_run_id"],
            status=AgentRunStatus.COMPLETED,
            data={
                "answer": "Security policy content [1].",
                "citations": [citation.model_dump(mode="json")],
                "context": [context_chunk.model_dump(mode="json")],
                "context_token_count": 3,
                "confidence_score": 1.0,
                "retrieval_strategy": "bm25",
                "checkpoint_count": 8,
                "context_chunks": 1,
                "citation_count": 1,
            },
            created_at=now,
        )

    monkeypatch.setattr(
        "agentic_rag.api.query.stream_agent_runtime_graph",
        fake_stream_agent_runtime_graph,
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/query/stream",
            headers={"X-Request-ID": "stream-request-id"},
            json={
                "query": "security policy",
                "workspace_id": "workspace-a",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    event_blocks = [
        block
        for block in response.text.strip().split("\n\n")
        if block
    ]
    assert len(event_blocks) == 4
    assert event_blocks[0].splitlines()[0] == "event: query_started"
    assert event_blocks[1].splitlines()[0] == "event: agent_step_completed"
    assert event_blocks[2].splitlines()[0] == "event: answer_token"
    assert event_blocks[3].splitlines()[0] == "event: query_completed"

    started_payload = json.loads(event_blocks[0].splitlines()[1].removeprefix("data: "))
    step_payload = json.loads(event_blocks[1].splitlines()[1].removeprefix("data: "))
    token_payload = json.loads(event_blocks[2].splitlines()[1].removeprefix("data: "))
    completed_payload = json.loads(event_blocks[3].splitlines()[1].removeprefix("data: "))

    assert started_payload["agent_run_id"] == str(captured["agent_run_id"])
    assert started_payload["data"]["request_id"] == "stream-request-id"
    assert started_payload["data"]["workspace_id"] == "workspace-a"
    assert step_payload["data"]["node_name"] == "bm25_search"
    assert step_payload["data"]["step_number"] == 5
    assert token_payload["data"]["text_delta"] == "Security policy content"
    assert token_payload["data"]["token_index"] == 1
    assert completed_payload["agent_run_id"] == started_payload["agent_run_id"]
    assert completed_payload["data"]["response"]["agent_run_id"] == started_payload["agent_run_id"]
    assert completed_payload["data"]["response"]["answer"] == "Security policy content [1]."
    assert completed_payload["data"]["response"]["retrieval_strategy"] == "bm25"
    assert completed_payload["data"]["response"]["context_token_count"] == 3
    assert completed_payload["data"]["response"]["synthesis_enabled"] is True
    assert captured["query"] == "security policy"
    assert captured["auth"].user_id == "user-1"
    assert captured["auth"].tenant_id == "tenant-a"
    assert captured["auth"].workspace_id == "workspace-a"
    assert captured["retrieval_filters"].workspace_id == "workspace-a"
    assert captured["retrieval_limit"] == 20
    assert captured["db"] is not None


def test_query_stream_endpoint_returns_failed_event(monkeypatch) -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["query:run"],
    )
    failed_labels = {
        "status": "failed",
        "retrieval_strategy": "bm25",
        "synthesis_enabled": "false",
    }
    failed_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_query_lifecycle_total",
            failed_labels,
        )
        or 0
    )
    latency_count_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_query_latency_seconds_count",
            failed_labels,
        )
        or 0
    )

    def fake_stream_agent_runtime_graph(**kwargs):
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(
        "agentic_rag.api.query.stream_agent_runtime_graph",
        fake_stream_agent_runtime_graph,
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/query/stream",
            headers={"X-Request-ID": "stream-request-id"},
            json={"query": "security policy"},
        )

    assert response.status_code == 200
    event_blocks = [
        block
        for block in response.text.strip().split("\n\n")
        if block
    ]
    assert len(event_blocks) == 2
    assert event_blocks[0].splitlines()[0] == "event: query_started"
    assert event_blocks[1].splitlines()[0] == "event: query_failed"

    failed_payload = json.loads(event_blocks[1].splitlines()[1].removeprefix("data: "))

    assert failed_payload["data"]["error_type"] == "RuntimeError"
    assert failed_payload["data"]["error_message"] == "retrieval unavailable"
    assert failed_payload["data"]["request_id"] == "stream-request-id"
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_query_lifecycle_total",
            failed_labels,
        )
        == failed_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_query_latency_seconds_count",
            failed_labels,
        )
        == latency_count_before + 1
    )


def test_query_stream_endpoint_requires_query_scope() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["documents:read"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/query/stream",
            json={"query": "security policy"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Missing required scope: query:run"


def test_get_query_run_endpoint_returns_persisted_response() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        document_id = uuid4()
        chunk_id = uuid4()
        agent_run_id = uuid4()
        query_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="security policy", workspace_id="workspace-a"),
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
                latency_ms=20,
                synthesis_enabled=False,
                verification_status=AnswerVerificationStatus.PASSED,
                verification_reason="Answer citations match retrieved context.",
            ),
        )

        for client in client_with_user(user_context, db):
            response = client.get(f"/query/{agent_run_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["agent_run_id"] == str(agent_run_id)
        assert body["status"] == "completed"
        assert body["tenant_id"] == "tenant-a"
        assert body["workspace_id"] == "workspace-a"
        assert body["request_id"] == "request-id-1"
        assert body["answer"] == "Security policy content [1]."
        assert body["response"]["answer"] == "Security policy content [1]."
        assert body["verification_status"] == "passed"
        assert body["verification_reason"] == "Answer citations match retrieved context."
        assert body["response"]["verification_status"] == "passed"
        assert body["citations"][0]["title"] == "Security Policy"
    finally:
        db.close()


def test_get_query_run_endpoint_rejects_other_user() -> None:
    db = create_test_db()
    try:
        owner_context = UserContext(
            id="owner",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            scopes=["query:run"],
        )
        requester_context = UserContext(
            id="requester",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            scopes=["query:run"],
        )
        agent_run_id = uuid4()
        create_query_run(
            user_context=owner_context,
            db=db,
            request=QueryRequest(query="security policy"),
            agent_run_id=agent_run_id,
        )

        for client in client_with_user(requester_context, db):
            response = client.get(f"/query/{agent_run_id}")

        assert response.status_code == 403
        assert response.json()["detail"] == "Query run access denied."
    finally:
        db.close()


def test_cancel_query_run_endpoint_cancels_owner_run() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        cancelled_labels = {
            "status": "cancelled",
            "retrieval_strategy": "bm25",
            "synthesis_enabled": "false",
        }
        cancelled_before = (
            REGISTRY.get_sample_value(
                "agentic_rag_query_lifecycle_total",
                cancelled_labels,
            )
            or 0
        )
        agent_run_id = uuid4()
        create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="security policy", workspace_id="workspace-a"),
            agent_run_id=agent_run_id,
            request_id="request-id-1",
        )

        for client in client_with_user(user_context, db):
            response = client.post(f"/query/{agent_run_id}/cancel")

        assert response.status_code == 200
        body = response.json()
        assert body["agent_run_id"] == str(agent_run_id)
        assert body["status"] == "cancelled"
        assert body["tenant_id"] == "tenant-a"
        assert body["workspace_id"] == "workspace-a"
        assert body["completed_at"] is not None
        assert (
            REGISTRY.get_sample_value(
                "agentic_rag_query_lifecycle_total",
                cancelled_labels,
            )
            == cancelled_before + 1
        )
    finally:
        db.close()


def test_cancel_query_run_endpoint_allows_admin_to_cancel_other_user_run() -> None:
    db = create_test_db()
    try:
        owner_context = UserContext(
            id="owner",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            scopes=["query:run"],
        )
        admin_context = UserContext(
            id="admin-user",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            roles=["admin"],
            scopes=["query:run"],
        )
        agent_run_id = uuid4()
        create_query_run(
            user_context=owner_context,
            db=db,
            request=QueryRequest(query="security policy"),
            agent_run_id=agent_run_id,
        )

        for client in client_with_user(admin_context, db):
            response = client.post(f"/query/{agent_run_id}/cancel")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "cancelled"
        assert body["user_id"] == "owner"
    finally:
        db.close()


def test_cancel_query_run_endpoint_rejects_other_user() -> None:
    db = create_test_db()
    try:
        owner_context = UserContext(
            id="owner",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            scopes=["query:run"],
        )
        requester_context = UserContext(
            id="requester",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            scopes=["query:run"],
        )
        agent_run_id = uuid4()
        create_query_run(
            user_context=owner_context,
            db=db,
            request=QueryRequest(query="security policy"),
            agent_run_id=agent_run_id,
        )

        for client in client_with_user(requester_context, db):
            response = client.post(f"/query/{agent_run_id}/cancel")

        assert response.status_code == 403
        assert response.json()["detail"] == "Query run access denied."
    finally:
        db.close()


def test_cancel_query_run_endpoint_is_tenant_scoped() -> None:
    db = create_test_db()
    try:
        owner_context = UserContext(
            id="owner",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            scopes=["query:run"],
        )
        tenant_b_context = UserContext(
            id="owner",
            customer_id="tenant-b",
            tenant_id="tenant-b",
            scopes=["query:run"],
        )
        agent_run_id = uuid4()
        create_query_run(
            user_context=owner_context,
            db=db,
            request=QueryRequest(query="security policy"),
            agent_run_id=agent_run_id,
        )

        for client in client_with_user(tenant_b_context, db):
            response = client.post(f"/query/{agent_run_id}/cancel")

        assert response.status_code == 404
        assert response.json()["detail"] == "Query run not found."
    finally:
        db.close()


def test_cancel_query_run_endpoint_rejects_completed_run() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            scopes=["query:run"],
        )
        document_id = uuid4()
        chunk_id = uuid4()
        agent_run_id = uuid4()
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
                latency_ms=20,
                synthesis_enabled=False,
            ),
        )

        for client in client_with_user(user_context, db):
            response = client.post(f"/query/{agent_run_id}/cancel")

        assert response.status_code == 409
        assert "completed" in response.json()["detail"]
    finally:
        db.close()


def test_list_query_run_endpoint_returns_only_current_user_runs() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        other_context = UserContext(
            id="user-2",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="my query", workspace_id="workspace-a"),
            agent_run_id=uuid4(),
            request_id="request-id-1",
        )
        create_query_run(
            user_context=other_context,
            db=db,
            request=QueryRequest(query="other query", workspace_id="workspace-a"),
            agent_run_id=uuid4(),
            request_id="request-id-2",
        )

        for client in client_with_user(user_context, db):
            response = client.get("/query?page=1&size=20&request_id=request-id-1")

        assert response.status_code == 200
        body = response.json()
        assert body["page"]["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["query"] == "my query"
        assert body["items"][0]["user_id"] == "user-1"
        assert body["items"][0]["request_id"] == "request-id-1"
    finally:
        db.close()


def test_list_query_run_endpoint_admin_lists_same_tenant_runs_with_filters() -> None:
    db = create_test_db()
    try:
        db.add(
            Tenant(
                tenant_id="tenant-b",
                name="Tenant B",
                slug="tenant-b",
                status="active",
                metadata_={},
            )
        )
        db.commit()
        admin_context = UserContext(
            id="admin-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            roles=["admin"],
            scopes=["query:run"],
        )
        owner_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        other_context = UserContext(
            id="user-2",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-b",
            scopes=["query:run"],
        )
        tenant_b_context = UserContext(
            id="tenant-b-user",
            customer_id="tenant-b",
            tenant_id="tenant-b",
            workspace_id="workspace-b",
            scopes=["query:run"],
        )
        passed_run_id = uuid4()
        failed_verification_run_id = uuid4()
        failed_status_run_id = uuid4()
        tenant_b_run_id = uuid4()
        document_id = uuid4()
        chunk_id = uuid4()

        passed_run = create_query_run(
            user_context=owner_context,
            db=db,
            request=QueryRequest(query="passed query", workspace_id="workspace-a"),
            agent_run_id=passed_run_id,
            request_id="request-id-passed",
        )
        failed_verification_run = create_query_run(
            user_context=other_context,
            db=db,
            request=QueryRequest(
                query="failed verification query",
                workspace_id="workspace-b",
            ),
            agent_run_id=failed_verification_run_id,
            request_id="request-id-failed-verification",
        )
        failed_status_run = create_query_run(
            user_context=other_context,
            db=db,
            request=QueryRequest(query="failed query", workspace_id="workspace-b"),
            agent_run_id=failed_status_run_id,
            request_id="request-id-failed",
        )
        tenant_b_run = create_query_run(
            user_context=tenant_b_context,
            db=db,
            request=QueryRequest(query="tenant b query", workspace_id="workspace-b"),
            agent_run_id=tenant_b_run_id,
            request_id="request-id-tenant-b",
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
            query_run=failed_verification_run,
            response=QueryResponse(
                agent_run_id=failed_verification_run_id,
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
        mark_query_run_failed(
            db=db,
            query_run=failed_status_run,
            error_type="RuntimeError",
            error_message="retrieval failed",
            latency_ms=13,
        )
        mark_query_run_completed(
            db=db,
            query_run=tenant_b_run,
            response=QueryResponse(
                agent_run_id=tenant_b_run_id,
                answer="Tenant B answer.",
                citations=[],
                context_token_count=0,
                confidence_score=0.0,
                retrieval_strategy=RetrievalStrategy.BM25,
                latency_ms=18,
            ),
        )

        passed_run.created_at = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
        failed_verification_run.created_at = datetime(
            2026,
            1,
            20,
            9,
            0,
            tzinfo=timezone.utc,
        )
        failed_status_run.created_at = datetime(2026, 1, 30, 9, 0, tzinfo=timezone.utc)
        tenant_b_run.created_at = datetime(2026, 1, 30, 9, 0, tzinfo=timezone.utc)
        db.commit()

        for client in client_with_user(admin_context, db):
            all_response = client.get("/query?page=1&size=20")
            user_response = client.get("/query?page=1&size=20&user_id=user-2")
            workspace_response = client.get(
                "/query?page=1&size=20&workspace_id=workspace-b"
            )
            status_response = client.get("/query?page=1&size=20&status=failed")
            verification_response = client.get(
                "/query?page=1&size=20&verification_status=failed"
            )
            date_response = client.get(
                "/query?page=1&size=20"
                "&created_from=2026-01-15T00:00:00Z"
                "&created_to=2026-01-25T23:59:00Z"
            )

        all_body = all_response.json()
        user_body = user_response.json()
        workspace_body = workspace_response.json()
        status_body = status_response.json()
        verification_body = verification_response.json()
        date_body = date_response.json()

        assert all_response.status_code == 200
        assert all_body["page"]["total"] == 3
        assert [item["agent_run_id"] for item in all_body["items"]] == [
            str(failed_status_run_id),
            str(failed_verification_run_id),
            str(passed_run_id),
        ]
        assert {item["user_id"] for item in all_body["items"]} == {
            "user-1",
            "user-2",
        }
        assert user_response.status_code == 200
        assert user_body["page"]["total"] == 2
        assert [item["agent_run_id"] for item in user_body["items"]] == [
            str(failed_status_run_id),
            str(failed_verification_run_id),
        ]
        assert workspace_response.status_code == 200
        assert workspace_body["page"]["total"] == 2
        assert [item["workspace_id"] for item in workspace_body["items"]] == [
            "workspace-b",
            "workspace-b",
        ]
        assert status_response.status_code == 200
        assert status_body["page"]["total"] == 1
        assert status_body["items"][0]["agent_run_id"] == str(failed_status_run_id)
        assert status_body["items"][0]["status"] == "failed"
        assert verification_response.status_code == 200
        assert verification_body["page"]["total"] == 1
        assert verification_body["items"][0]["agent_run_id"] == str(
            failed_verification_run_id
        )
        assert verification_body["items"][0]["verification_status"] == "failed"
        assert date_response.status_code == 200
        assert date_body["page"]["total"] == 1
        assert date_body["items"][0]["agent_run_id"] == str(
            failed_verification_run_id
        )
    finally:
        db.close()


def test_list_query_run_endpoint_filters_by_status() -> None:
    db = create_test_db()
    try:
        document_id = uuid4()
        chunk_id = uuid4()
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        completed_run_id = uuid4()
        failed_run_id = uuid4()
        running_run_id = uuid4()
        completed_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="completed query", workspace_id="workspace-a"),
            agent_run_id=completed_run_id,
            request_id="request-id-completed",
        )
        failed_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="failed query", workspace_id="workspace-a"),
            agent_run_id=failed_run_id,
            request_id="request-id-failed",
        )
        create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="running query", workspace_id="workspace-a"),
            agent_run_id=running_run_id,
            request_id="request-id-running",
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
                verification_status=AnswerVerificationStatus.PASSED,
                verification_reason="Answer citations match retrieved context.",
            ),
        )
        mark_query_run_failed(
            db=db,
            query_run=failed_run,
            error_type="RuntimeError",
            error_message="retrieval failed",
            latency_ms=13,
        )

        for client in client_with_user(user_context, db):
            failed_response = client.get("/query?page=1&size=20&status=failed")
            completed_response = client.get("/query?page=1&size=20&status=completed")

        failed_body = failed_response.json()
        completed_body = completed_response.json()

        assert failed_response.status_code == 200
        assert failed_body["page"]["total"] == 1
        assert failed_body["items"][0]["agent_run_id"] == str(failed_run_id)
        assert failed_body["items"][0]["status"] == "failed"
        assert completed_response.status_code == 200
        assert completed_body["page"]["total"] == 1
        assert completed_body["items"][0]["agent_run_id"] == str(completed_run_id)
        assert completed_body["items"][0]["status"] == "completed"
        assert completed_body["items"][0]["verification_status"] == "passed"
    finally:
        db.close()


def test_list_query_run_endpoint_filters_by_verification_status() -> None:
    db = create_test_db()
    try:
        document_id = uuid4()
        chunk_id = uuid4()
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        passed_run_id = uuid4()
        failed_run_id = uuid4()
        skipped_run_id = uuid4()
        not_required_run_id = uuid4()
        passed_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="passed query", workspace_id="workspace-a"),
            agent_run_id=passed_run_id,
            request_id="request-id-passed",
        )
        failed_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(
                query="failed verification query",
                workspace_id="workspace-a",
            ),
            agent_run_id=failed_run_id,
            request_id="request-id-failed",
        )
        skipped_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(
                query="skipped verification query",
                workspace_id="workspace-a",
            ),
            agent_run_id=skipped_run_id,
            request_id="request-id-skipped",
        )
        not_required_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(
                query="no synthesis query",
                workspace_id="workspace-a",
            ),
            agent_run_id=not_required_run_id,
            request_id="request-id-not-required",
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

        for client in client_with_user(user_context, db):
            passed_response = client.get(
                "/query?page=1&size=20&verification_status=passed"
            )
            failed_response = client.get(
                "/query?page=1&size=20&verification_status=failed"
            )
            skipped_response = client.get(
                "/query?page=1&size=20&verification_status=skipped"
            )
            not_required_response = client.get(
                "/query?page=1&size=20&verification_status=not_required"
            )

        passed_body = passed_response.json()
        failed_body = failed_response.json()
        skipped_body = skipped_response.json()
        not_required_body = not_required_response.json()

        assert passed_response.status_code == 200
        assert passed_body["page"]["total"] == 1
        assert passed_body["items"][0]["agent_run_id"] == str(passed_run_id)
        assert passed_body["items"][0]["verification_status"] == "passed"
        assert failed_response.status_code == 200
        assert failed_body["page"]["total"] == 1
        assert failed_body["items"][0]["agent_run_id"] == str(failed_run_id)
        assert failed_body["items"][0]["verification_status"] == "failed"
        assert skipped_response.status_code == 200
        assert skipped_body["page"]["total"] == 1
        assert skipped_body["items"][0]["agent_run_id"] == str(skipped_run_id)
        assert skipped_body["items"][0]["verification_status"] == "skipped"
        assert not_required_response.status_code == 200
        assert not_required_body["page"]["total"] == 1
        assert not_required_body["items"][0]["agent_run_id"] == str(
            not_required_run_id
        )
        assert (
            not_required_body["items"][0]["verification_status"] == "not_required"
        )
    finally:
        db.close()


def test_list_query_run_endpoint_filters_by_created_at() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )
        old_run_id = uuid4()
        middle_run_id = uuid4()
        new_run_id = uuid4()
        old_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="old query", workspace_id="workspace-a"),
            agent_run_id=old_run_id,
            request_id="request-id-old",
        )
        middle_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="middle query", workspace_id="workspace-a"),
            agent_run_id=middle_run_id,
            request_id="request-id-middle",
        )
        new_run = create_query_run(
            user_context=user_context,
            db=db,
            request=QueryRequest(query="new query", workspace_id="workspace-a"),
            agent_run_id=new_run_id,
            request_id="request-id-new",
        )

        old_run.created_at = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
        middle_run.created_at = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
        new_run.created_at = datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)
        db.commit()

        for client in client_with_user(user_context, db):
            range_response = client.get(
                "/query?page=1&size=20"
                "&created_from=2026-01-05T00:00:00Z"
                "&created_to=2026-01-15T23:59:00Z"
            )
            from_response = client.get(
                "/query?page=1&size=20&created_from=2026-01-05T00:00:00Z"
            )
            to_response = client.get(
                "/query?page=1&size=20&created_to=2026-01-15T23:59:00Z"
            )

        range_body = range_response.json()
        from_body = from_response.json()
        to_body = to_response.json()

        assert range_response.status_code == 200
        assert range_body["page"]["total"] == 1
        assert range_body["items"][0]["agent_run_id"] == str(middle_run_id)
        assert from_response.status_code == 200
        assert from_body["page"]["total"] == 2
        assert [item["agent_run_id"] for item in from_body["items"]] == [
            str(new_run_id),
            str(middle_run_id),
        ]
        assert to_response.status_code == 200
        assert to_body["page"]["total"] == 2
        assert [item["agent_run_id"] for item in to_body["items"]] == [
            str(middle_run_id),
            str(old_run_id),
        ]
    finally:
        db.close()


def test_list_query_run_endpoint_rejects_invalid_status_filter() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )

        for client in client_with_user(user_context, db):
            response = client.get("/query?page=1&size=20&status=unknown")

        assert response.status_code == 422
    finally:
        db.close()


def test_list_query_run_endpoint_rejects_invalid_created_at_filter() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )

        for client in client_with_user(user_context, db):
            response = client.get("/query?page=1&size=20&created_from=not-a-date")

        assert response.status_code == 422
    finally:
        db.close()


def test_list_query_run_endpoint_rejects_reversed_created_at_filter() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )

        for client in client_with_user(user_context, db):
            response = client.get(
                "/query?page=1&size=20"
                "&created_from=2026-01-15T00:00:00Z"
                "&created_to=2026-01-05T00:00:00Z"
            )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "created_from must be before or equal to created_to"
        )
    finally:
        db.close()


def test_list_query_run_endpoint_rejects_invalid_verification_status_filter() -> None:
    db = create_test_db()
    try:
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            scopes=["query:run"],
        )

        for client in client_with_user(user_context, db):
            response = client.get(
                "/query?page=1&size=20&verification_status=unknown"
            )

        assert response.status_code == 422
    finally:
        db.close()
