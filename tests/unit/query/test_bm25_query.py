from uuid import uuid4

from fastapi import HTTPException
import pytest

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.query.bm25_query import run_bm25_query
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.query import QueryRequest
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTool,
)


def test_run_bm25_query_retrieves_and_builds_context(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
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
                    content="Security <em>policy</em> content.",
                    score=2.3,
                    source=RetrievalTool.BM25_SEARCH,
                    metadata={"token_count": 4},
                    citation=Citation(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        title="Security Policy",
                        quote="Security <em>policy</em> content.",
                        score=2.3,
                    ),
                )
            ],
            latency_ms=11,
        )

    monkeypatch.setattr(
        "agentic_rag.query.bm25_query.search_bm25_chunks",
        fake_search_bm25_chunks,
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        scopes=["query:run"],
    )

    response = run_bm25_query(
        user_context=user_context,
        request=QueryRequest(
            query=" security policy ",
            workspace_id="workspace-a",
            retrieval_limit=7,
            max_context_chunks=3,
            max_context_tokens=500,
        ),
    )

    assert response.retrieval_strategy == RetrievalStrategy.BM25
    assert response.synthesis_enabled is False
    assert response.confidence_score == 0.0
    assert response.candidates[0].chunk_id == chunk_id
    assert response.context[0].content == "Security policy content."
    assert response.citations[0].title == "Security Policy"
    assert response.context_token_count == 4
    assert "LLM synthesis is not enabled yet" in response.answer
    assert captured["user_context"].id == "user-1"
    assert captured["query"] == "security policy"
    assert captured["filters"].workspace_id == "workspace-a"
    assert captured["limit"] == 7


def test_run_bm25_query_rejects_workspace_conflict() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        run_bm25_query(
            user_context=user_context,
            request=QueryRequest(
                query="security policy",
                workspace_id="workspace-a",
                filters={"workspace_id": "workspace-b"},
            ),
        )

    assert exc_info.value.status_code == 400


def test_run_bm25_query_handles_empty_retrieval(monkeypatch) -> None:
    def fake_search_bm25_chunks(user_context, query, filters, limit):
        return RetrievalResponse(
            strategy=RetrievalStrategy.BM25,
            candidates=[],
            latency_ms=3,
        )

    monkeypatch.setattr(
        "agentic_rag.query.bm25_query.search_bm25_chunks",
        fake_search_bm25_chunks,
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )

    response = run_bm25_query(
        user_context=user_context,
        request=QueryRequest(query="unknown policy"),
    )

    assert response.candidates == []
    assert response.context == []
    assert response.citations == []
    assert response.context_token_count == 0
    assert "Retrieved 0 context chunks" in response.answer
