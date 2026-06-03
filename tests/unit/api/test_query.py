import json
import logging
from collections.abc import Iterator
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.query_runs import (
    create_query_run,
    mark_query_run_completed,
)
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.query import QueryRequest, QueryResponse
from agentic_rag.shared.schemas.retrieval import (
    ContextChunk,
    RetrievalStrategy,
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


def test_query_stream_endpoint_returns_started_and_completed_events(monkeypatch) -> None:
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

    def fake_run_bm25_query(user_context, request, db, request_id, agent_run_id):
        captured["user_context"] = user_context
        captured["request"] = request
        captured["db"] = db
        captured["request_id"] = request_id
        captured["agent_run_id"] = agent_run_id
        citation = Citation(
            document_id=document_id,
            chunk_id=chunk_id,
            title="Security Policy",
            quote="Security policy content.",
            score=2.1,
        )
        return QueryResponse(
            agent_run_id=agent_run_id,
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
    assert len(event_blocks) == 2
    assert event_blocks[0].splitlines()[0] == "event: query_started"
    assert event_blocks[1].splitlines()[0] == "event: query_completed"

    started_payload = json.loads(event_blocks[0].splitlines()[1].removeprefix("data: "))
    completed_payload = json.loads(event_blocks[1].splitlines()[1].removeprefix("data: "))

    assert started_payload["agent_run_id"] == str(captured["agent_run_id"])
    assert started_payload["data"]["request_id"] == "stream-request-id"
    assert completed_payload["agent_run_id"] == started_payload["agent_run_id"]
    assert completed_payload["data"]["response"]["agent_run_id"] == started_payload["agent_run_id"]
    assert completed_payload["data"]["response"]["retrieval_strategy"] == "bm25"
    assert completed_payload["data"]["response"]["context_token_count"] == 3
    assert captured["request"].stream is True
    assert captured["request"].query == "security policy"
    assert captured["request_id"] == "stream-request-id"
    assert captured["db"] is not None


def test_query_stream_endpoint_returns_failed_event(monkeypatch) -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["query:run"],
    )

    def fake_run_bm25_query(user_context, request, db, request_id, agent_run_id):
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(
        "agentic_rag.api.query.run_bm25_query",
        fake_run_bm25_query,
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
