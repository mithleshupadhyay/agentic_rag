from uuid import uuid4

from fastapi import HTTPException
import pytest

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval import hybrid_search
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RerankResponse,
    RetrievalFilters,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTool,
)


def test_search_hybrid_chunks_merges_bm25_and_vector_candidates(monkeypatch) -> None:
    shared_document_id = uuid4()
    shared_chunk_id = uuid4()
    bm25_only_chunk_id = uuid4()
    vector_only_chunk_id = uuid4()
    db = object()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        roles=["analyst"],
        group_ids=["security"],
        acl_version=5,
    )
    filters = RetrievalFilters(
        workspace_id="workspace-a",
        document_ids=[shared_document_id],
    )
    captured = {}

    def fake_search_bm25_chunks(user_context, query, filters, limit):
        captured["bm25_user_context"] = user_context
        captured["bm25_query"] = query
        captured["bm25_filters"] = filters
        captured["bm25_limit"] = limit
        return RetrievalResponse(
            strategy=RetrievalStrategy.BM25,
            candidates=[
                CandidateChunk(
                    chunk_id=shared_chunk_id,
                    document_id=shared_document_id,
                    content="BM25 highlighted shared content.",
                    score=12.0,
                    source=RetrievalTool.BM25_SEARCH,
                    metadata={"chunk_index": 1},
                    citation=Citation(
                        document_id=shared_document_id,
                        chunk_id=shared_chunk_id,
                        title="Security Policy",
                        quote="BM25 highlighted shared content.",
                        score=12.0,
                    ),
                ),
                CandidateChunk(
                    chunk_id=bm25_only_chunk_id,
                    document_id=shared_document_id,
                    content="BM25 only content.",
                    score=7.0,
                    source=RetrievalTool.BM25_SEARCH,
                    metadata={"chunk_index": 2},
                    citation=Citation(
                        document_id=shared_document_id,
                        chunk_id=bm25_only_chunk_id,
                        title="Security Policy",
                        quote="BM25 only content.",
                        score=7.0,
                    ),
                ),
            ],
            latency_ms=9,
        )

    def fake_search_vector_chunks(
        db,
        user_context,
        query,
        filters,
        limit,
        min_similarity,
    ):
        captured["vector_db"] = db
        captured["vector_user_context"] = user_context
        captured["vector_query"] = query
        captured["vector_filters"] = filters
        captured["vector_limit"] = limit
        captured["vector_min_similarity"] = min_similarity
        return RetrievalResponse(
            strategy=RetrievalStrategy.VECTOR,
            candidates=[
                CandidateChunk(
                    chunk_id=shared_chunk_id,
                    document_id=shared_document_id,
                    content="Vector shared content.",
                    score=0.92,
                    source=RetrievalTool.VECTOR_SEARCH,
                    metadata={"distance": 0.08},
                    citation=Citation(
                        document_id=shared_document_id,
                        chunk_id=shared_chunk_id,
                        title="Security Policy",
                        quote="Vector shared content.",
                        score=0.92,
                    ),
                ),
                CandidateChunk(
                    chunk_id=vector_only_chunk_id,
                    document_id=shared_document_id,
                    content="Vector only content.",
                    score=0.81,
                    source=RetrievalTool.VECTOR_SEARCH,
                    metadata={"distance": 0.19},
                    citation=Citation(
                        document_id=shared_document_id,
                        chunk_id=vector_only_chunk_id,
                        title="Security Policy",
                        quote="Vector only content.",
                        score=0.81,
                    ),
                ),
            ],
            latency_ms=11,
        )

    def fake_rerank_chunks(query, candidates, top_k):
        captured["rerank_query"] = query
        captured["rerank_candidates"] = candidates
        captured["rerank_top_k"] = top_k
        reranked_candidates = []
        for index, candidate in enumerate(candidates, start=1):
            reranked_candidate = candidate.model_copy(deep=True)
            rerank_score = 1 / index
            reranked_candidate.score = rerank_score
            reranked_candidate.source = RetrievalTool.RERANK.value
            reranked_candidate.metadata = {
                **candidate.metadata,
                "original_score": candidate.score,
                "original_source": candidate.source,
                "rerank_score": rerank_score,
                "rerank_rank": index,
            }
            if reranked_candidate.citation:
                reranked_candidate.citation.score = rerank_score
            reranked_candidates.append(reranked_candidate)
        return RerankResponse(chunks=reranked_candidates, latency_ms=2)

    monkeypatch.setattr(
        hybrid_search,
        "search_bm25_chunks",
        fake_search_bm25_chunks,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_vector_chunks",
        fake_search_vector_chunks,
    )
    monkeypatch.setattr(
        hybrid_search,
        "rerank_chunks",
        fake_rerank_chunks,
    )

    response = hybrid_search.search_hybrid_chunks(
        db=db,
        user_context=user_context,
        query=" security policy ",
        filters=filters,
        limit=5,
        min_similarity=0.7,
    )

    assert response.strategy == RetrievalStrategy.HYBRID
    assert [candidate.chunk_id for candidate in response.candidates] == [
        shared_chunk_id,
        bm25_only_chunk_id,
        vector_only_chunk_id,
    ]
    assert response.candidates[0].source == RetrievalTool.RERANK.value
    assert response.candidates[0].score == 1.0
    assert response.candidates[0].content == "BM25 highlighted shared content."
    assert response.candidates[0].citation.score == 1.0
    assert response.candidates[0].metadata["retrieval_sources"] == [
        RetrievalTool.BM25_SEARCH.value,
        RetrievalTool.VECTOR_SEARCH.value,
    ]
    assert response.candidates[0].metadata["bm25_score"] == 12.0
    assert response.candidates[0].metadata["vector_score"] == 0.92
    assert response.candidates[0].metadata["hybrid_rank_score"] == 1.0
    assert response.candidates[0].metadata["pre_rerank_score"] == 1.0
    assert response.candidates[0].metadata["pre_rerank_source"] == (
        RetrievalStrategy.HYBRID.value
    )
    assert response.candidates[0].metadata["original_score"] == 1.0
    assert response.candidates[0].metadata["original_source"] == (
        RetrievalStrategy.HYBRID.value
    )
    assert response.candidates[0].metadata["rerank_score"] == 1.0
    assert response.candidates[0].metadata["rerank_rank"] == 1
    assert response.candidates[1].metadata["retrieval_sources"] == [
        RetrievalTool.BM25_SEARCH.value
    ]
    assert response.candidates[2].metadata["retrieval_sources"] == [
        RetrievalTool.VECTOR_SEARCH.value
    ]
    assert captured["bm25_query"] == "security policy"
    assert captured["bm25_filters"] == filters
    assert captured["bm25_limit"] == 5
    assert captured["vector_db"] is db
    assert captured["vector_query"] == "security policy"
    assert captured["vector_filters"] == filters
    assert captured["vector_limit"] == 5
    assert captured["vector_min_similarity"] == 0.7
    assert captured["rerank_query"] == "security policy"
    assert captured["rerank_top_k"] == 5
    assert [candidate.chunk_id for candidate in captured["rerank_candidates"]] == [
        shared_chunk_id,
        bm25_only_chunk_id,
        vector_only_chunk_id,
    ]
    assert captured["rerank_candidates"][0].metadata["pre_rerank_score"] == 1.0


def test_search_hybrid_chunks_truncates_to_limit(monkeypatch) -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    first_chunk_id = uuid4()
    second_chunk_id = uuid4()
    document_id = uuid4()

    def fake_search_bm25_chunks(user_context, query, filters, limit):
        return RetrievalResponse(
            strategy=RetrievalStrategy.BM25,
            candidates=[
                CandidateChunk(
                    chunk_id=first_chunk_id,
                    document_id=document_id,
                    content="First content.",
                    score=9.0,
                    source=RetrievalTool.BM25_SEARCH,
                ),
                CandidateChunk(
                    chunk_id=second_chunk_id,
                    document_id=document_id,
                    content="Second content.",
                    score=8.0,
                    source=RetrievalTool.BM25_SEARCH,
                ),
            ],
            latency_ms=3,
        )

    def fake_search_vector_chunks(
        db,
        user_context,
        query,
        filters,
        limit,
        min_similarity,
    ):
        return RetrievalResponse(
            strategy=RetrievalStrategy.VECTOR,
            candidates=[],
            latency_ms=4,
        )

    monkeypatch.setattr(
        hybrid_search,
        "search_bm25_chunks",
        fake_search_bm25_chunks,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_vector_chunks",
        fake_search_vector_chunks,
    )

    response = hybrid_search.search_hybrid_chunks(
        db=None,
        user_context=user_context,
        query="policy",
        limit=1,
    )

    assert [candidate.chunk_id for candidate in response.candidates] == [first_chunk_id]


def test_search_hybrid_chunks_rejects_unsupported_filters() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        hybrid_search.search_hybrid_chunks(
            db=None,
            user_context=user_context,
            query="security policy",
            filters=RetrievalFilters(source_types=["upload"]),
        )

    assert exc_info.value.status_code == 400
    assert "workspace_id and document_ids" in exc_info.value.detail


def test_search_hybrid_chunks_rejects_blank_query() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        hybrid_search.search_hybrid_chunks(
            db=None,
            user_context=user_context,
            query="   ",
        )

    assert exc_info.value.status_code == 400
