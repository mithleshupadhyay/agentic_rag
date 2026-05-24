from collections.abc import Iterator
from uuid import uuid4

from fastapi.testclient import TestClient

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.query import QueryResponse
from agentic_rag.shared.schemas.retrieval import (
    ContextChunk,
    RetrievalStrategy,
)


def client_with_user(user_context: UserContext) -> Iterator[TestClient]:
    async def override_get_current_user() -> UserContext:
        return user_context

    app.dependency_overrides[get_current_user] = override_get_current_user
    test_client = TestClient(app)

    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_returns_grounded_retrieval_output(monkeypatch) -> None:
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

    def fake_run_bm25_query(user_context, request):
        captured["user_context"] = user_context
        captured["request"] = request
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

    for client in client_with_user(user_context):
        response = client.post(
            "/query",
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
