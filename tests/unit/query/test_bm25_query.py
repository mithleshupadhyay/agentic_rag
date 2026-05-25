from uuid import uuid4

from fastapi import HTTPException
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.query.bm25_query import run_bm25_query
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.models import QueryRun, Tenant
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.llm import LLMResponse
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


def test_run_bm25_query_synthesizes_answer_when_enabled(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    captured = {}

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
        captured["request"] = request
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
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    response = run_bm25_query(
        user_context=user_context,
        request=QueryRequest(query="security policy", workspace_id="workspace-a"),
    )

    assert response.answer == "Security policy content is available in the retrieved document [1]."
    assert response.synthesis_enabled is True
    assert response.llm_provider == "litellm"
    assert response.llm_model == "gemini/gemini-2.0-flash"
    assert response.llm_input_tokens == 128
    assert response.llm_output_tokens == 14
    assert response.llm_cost_estimate == 0.001
    assert response.synthesis_error is None
    assert "Use only the authorized context" in captured["request"].messages[0].content
    assert "Security policy content." in captured["request"].messages[1].content


def test_run_bm25_query_returns_context_when_synthesis_fails(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()

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
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )

    response = run_bm25_query(
        user_context=user_context,
        request=QueryRequest(query="security policy"),
    )

    assert response.synthesis_enabled is False
    assert response.synthesis_error == "LLM synthesis failed"
    assert response.context[0].content == "Security policy content."
    assert response.citations[0].title == "Security Policy"
    assert "answer synthesis failed" in response.answer


def test_run_bm25_query_persists_completed_query_run(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = uuid4()
    chunk_id = uuid4()

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
                        quote="Security policy content.",
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

    with Session(engine) as db:
        db.add(
            Tenant(
                tenant_id="tenant-a",
                name="Tenant A",
                slug="tenant-a",
                status="active",
                metadata_={},
            )
        )
        db.commit()
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
        )

        response = run_bm25_query(
            user_context=user_context,
            request=QueryRequest(query="security policy", workspace_id="workspace-a"),
            db=db,
        )
        query_run = db.query(QueryRun).filter(QueryRun.id == response.agent_run_id).one()

        assert query_run.status == "completed"
        assert query_run.tenant_id == "tenant-a"
        assert query_run.workspace_id == "workspace-a"
        assert query_run.user_id == "user-1"
        assert query_run.answer == response.answer
        assert query_run.response_payload["agent_run_id"] == str(response.agent_run_id)
        assert query_run.citations["items"][0]["title"] == "Security Policy"


def test_run_bm25_query_marks_query_run_failed(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    def fake_search_bm25_chunks(user_context, query, filters, limit):
        raise RuntimeError("search backend unavailable")

    monkeypatch.setattr(
        "agentic_rag.query.bm25_query.search_bm25_chunks",
        fake_search_bm25_chunks,
    )

    with Session(engine) as db:
        db.add(
            Tenant(
                tenant_id="tenant-a",
                name="Tenant A",
                slug="tenant-a",
                status="active",
                metadata_={},
            )
        )
        db.commit()
        user_context = UserContext(
            id="user-1",
            customer_id="tenant-a",
            tenant_id="tenant-a",
        )

        with pytest.raises(RuntimeError):
            run_bm25_query(
                user_context=user_context,
                request=QueryRequest(query="security policy"),
                db=db,
            )

        query_run = db.query(QueryRun).one()
        assert query_run.status == "failed"
        assert query_run.error_type == "RuntimeError"
        assert query_run.error_message == "search backend unavailable"
        assert query_run.completed_at is not None
