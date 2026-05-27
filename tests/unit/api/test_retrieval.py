from collections.abc import Iterator
from uuid import uuid4

from fastapi.testclient import TestClient

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTool,
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


def client_with_user_and_db(user_context: UserContext, db):
    async def override_get_current_user() -> UserContext:
        return user_context

    def override_get_session():
        yield db

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_session] = override_get_session
    test_client = TestClient(app)

    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


def test_bm25_search_endpoint_returns_authorized_candidates(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        roles=["analyst"],
        group_ids=["security"],
        scopes=["query:run"],
        acl_version=5,
    )
    captured = {}

    def fake_search_bm25_chunks(user_context, query, filters, limit):
        captured["user_context"] = user_context
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
                    score=2.4,
                    source=RetrievalTool.BM25_SEARCH,
                    citation=Citation(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        title="Security Policy",
                        quote="Security policy content.",
                        score=2.4,
                    ),
                )
            ],
            latency_ms=12,
        )

    monkeypatch.setattr(
        "agentic_rag.api.retrieval.search_bm25_chunks",
        fake_search_bm25_chunks,
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/retrieval/bm25-search",
            json={
                "query": "security policy",
                "filters": {
                    "workspace_id": "workspace-a",
                    "document_ids": [str(document_id)],
                    "source_types": ["upload"],
                },
                "limit": 5,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "bm25"
    assert body["latency_ms"] == 12
    assert body["candidates"][0]["chunk_id"] == str(chunk_id)
    assert body["candidates"][0]["source"] == "bm25_search"
    assert body["candidates"][0]["citation"]["title"] == "Security Policy"
    assert captured["user_context"].id == "user-1"
    assert captured["query"] == "security policy"
    assert captured["filters"].workspace_id == "workspace-a"
    assert captured["filters"].document_ids == [document_id]
    assert captured["filters"].source_types == ["upload"]
    assert captured["limit"] == 5


def test_bm25_search_endpoint_requires_query_scope() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["documents:read"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/retrieval/bm25-search",
            json={"query": "security policy"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Missing required scope: query:run"


def test_bm25_search_endpoint_validates_request_body() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["query:run"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/retrieval/bm25-search",
            json={
                "query": "security policy",
                "limit": 500,
            },
        )

    assert response.status_code == 422


def test_vector_search_endpoint_returns_authorized_candidates(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    db = object()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        roles=["analyst"],
        group_ids=["security"],
        scopes=["query:run"],
        acl_version=5,
    )
    captured = {}

    def fake_search_vector_chunks(
        db,
        user_context,
        query,
        filters,
        limit,
        min_similarity,
    ):
        captured["db"] = db
        captured["user_context"] = user_context
        captured["query"] = query
        captured["filters"] = filters
        captured["limit"] = limit
        captured["min_similarity"] = min_similarity
        return RetrievalResponse(
            strategy=RetrievalStrategy.VECTOR,
            candidates=[
                CandidateChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content="Security policy vector content.",
                    score=0.91,
                    source=RetrievalTool.VECTOR_SEARCH,
                    metadata={
                        "embedding_model": "openai/text-embedding-3-small",
                        "distance": 0.09,
                    },
                    citation=Citation(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        title="Security Policy",
                        quote="Security policy vector content.",
                        score=0.91,
                    ),
                )
            ],
            latency_ms=18,
        )

    monkeypatch.setattr(
        "agentic_rag.api.retrieval.search_vector_chunks",
        fake_search_vector_chunks,
    )

    for client in client_with_user_and_db(user_context, db):
        response = client.post(
            "/retrieval/vector-search",
            json={
                "query": "security policy",
                "filters": {
                    "workspace_id": "workspace-a",
                    "document_ids": [str(document_id)],
                },
                "limit": 6,
                "min_similarity": 0.75,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "vector"
    assert body["latency_ms"] == 18
    assert body["candidates"][0]["chunk_id"] == str(chunk_id)
    assert body["candidates"][0]["source"] == "vector_search"
    assert body["candidates"][0]["score"] == 0.91
    assert body["candidates"][0]["metadata"]["distance"] == 0.09
    assert body["candidates"][0]["citation"]["title"] == "Security Policy"
    assert captured["db"] is db
    assert captured["user_context"].id == "user-1"
    assert captured["query"] == "security policy"
    assert captured["filters"].workspace_id == "workspace-a"
    assert captured["filters"].document_ids == [document_id]
    assert captured["limit"] == 6
    assert captured["min_similarity"] == 0.75


def test_vector_search_endpoint_requires_query_scope() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["documents:read"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/retrieval/vector-search",
            json={"query": "security policy"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Missing required scope: query:run"


def test_vector_search_endpoint_validates_request_body() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["query:run"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/retrieval/vector-search",
            json={
                "query": "security policy",
                "min_similarity": 1.5,
            },
        )

    assert response.status_code == 422


def test_hybrid_search_endpoint_returns_authorized_candidates(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    db = object()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        roles=["analyst"],
        group_ids=["security"],
        scopes=["query:run"],
        acl_version=5,
    )
    captured = {}

    def fake_search_hybrid_chunks(
        db,
        user_context,
        query,
        filters,
        limit,
        min_similarity,
    ):
        captured["db"] = db
        captured["user_context"] = user_context
        captured["query"] = query
        captured["filters"] = filters
        captured["limit"] = limit
        captured["min_similarity"] = min_similarity
        return RetrievalResponse(
            strategy=RetrievalStrategy.HYBRID,
            candidates=[
                CandidateChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content="Hybrid security policy content.",
                    score=1.0,
                    source=RetrievalStrategy.HYBRID.value,
                    metadata={
                        "retrieval_sources": ["bm25_search", "vector_search"],
                        "bm25_score": 12.0,
                        "vector_score": 0.92,
                    },
                    citation=Citation(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        title="Security Policy",
                        quote="Hybrid security policy content.",
                        score=1.0,
                    ),
                )
            ],
            latency_ms=24,
        )

    monkeypatch.setattr(
        "agentic_rag.api.retrieval.search_hybrid_chunks",
        fake_search_hybrid_chunks,
    )

    for client in client_with_user_and_db(user_context, db):
        response = client.post(
            "/retrieval/hybrid-search",
            json={
                "query": "security policy",
                "filters": {
                    "workspace_id": "workspace-a",
                    "document_ids": [str(document_id)],
                },
                "limit": 7,
                "min_similarity": 0.65,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "hybrid"
    assert body["latency_ms"] == 24
    assert body["candidates"][0]["chunk_id"] == str(chunk_id)
    assert body["candidates"][0]["source"] == "hybrid"
    assert body["candidates"][0]["score"] == 1.0
    assert body["candidates"][0]["metadata"]["retrieval_sources"] == [
        "bm25_search",
        "vector_search",
    ]
    assert body["candidates"][0]["citation"]["title"] == "Security Policy"
    assert captured["db"] is db
    assert captured["user_context"].id == "user-1"
    assert captured["query"] == "security policy"
    assert captured["filters"].workspace_id == "workspace-a"
    assert captured["filters"].document_ids == [document_id]
    assert captured["limit"] == 7
    assert captured["min_similarity"] == 0.65


def test_hybrid_search_endpoint_requires_query_scope() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["documents:read"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/retrieval/hybrid-search",
            json={"query": "security policy"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Missing required scope: query:run"


def test_hybrid_search_endpoint_validates_request_body() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        scopes=["query:run"],
    )

    for client in client_with_user(user_context):
        response = client.post(
            "/retrieval/hybrid-search",
            json={
                "query": "security policy",
                "limit": 500,
            },
        )

    assert response.status_code == 422
