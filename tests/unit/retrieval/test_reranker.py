from uuid import uuid4

from fastapi import HTTPException
import pytest

from agentic_rag.retrieval.reranker import rerank_chunks
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RetrievalStrategy,
    RetrievalTool,
)


def test_rerank_chunks_returns_empty_response_for_no_candidates() -> None:
    response = rerank_chunks(
        query="security policy",
        candidates=[],
        top_k=3,
    )

    assert response.chunks == []
    assert response.latency_ms >= 0


def test_rerank_chunks_prioritizes_query_match_and_limits_results() -> None:
    document_id = uuid4()
    unrelated_chunk_id = uuid4()
    matching_chunk_id = uuid4()
    partial_chunk_id = uuid4()
    unrelated_candidate = CandidateChunk(
        chunk_id=unrelated_chunk_id,
        document_id=document_id,
        content="Employee onboarding checklist and office access process.",
        score=0.99,
        source=RetrievalStrategy.HYBRID.value,
        metadata={
            "retrieval_sources": [
                RetrievalTool.BM25_SEARCH.value,
                RetrievalTool.VECTOR_SEARCH.value,
            ],
        },
        citation=Citation(
            document_id=document_id,
            chunk_id=unrelated_chunk_id,
            title="Onboarding",
            quote="Employee onboarding checklist.",
            score=0.99,
        ),
    )
    matching_candidate = CandidateChunk(
        chunk_id=matching_chunk_id,
        document_id=document_id,
        content="Security policy requires PCI encryption controls.",
        score=0.20,
        source=RetrievalTool.BM25_SEARCH,
        metadata={"retrieval_sources": [RetrievalTool.BM25_SEARCH.value]},
        citation=Citation(
            document_id=document_id,
            chunk_id=matching_chunk_id,
            title="Security Policy",
            quote="Security policy requires PCI encryption controls.",
            score=0.20,
        ),
    )
    partial_candidate = CandidateChunk(
        chunk_id=partial_chunk_id,
        document_id=document_id,
        content="Security policy overview.",
        score=0.35,
        source=RetrievalTool.VECTOR_SEARCH,
        metadata={"retrieval_sources": [RetrievalTool.VECTOR_SEARCH.value]},
        citation=Citation(
            document_id=document_id,
            chunk_id=partial_chunk_id,
            title="Security Policy",
            quote="Security policy overview.",
            score=0.35,
        ),
    )

    response = rerank_chunks(
        query="security policy encryption",
        candidates=[
            unrelated_candidate,
            matching_candidate,
            partial_candidate,
        ],
        top_k=2,
    )

    assert [chunk.chunk_id for chunk in response.chunks] == [
        matching_chunk_id,
        partial_chunk_id,
    ]
    assert response.chunks[0].source == RetrievalTool.RERANK.value
    assert response.chunks[0].score > response.chunks[1].score
    assert response.chunks[0].metadata["original_score"] == 0.20
    assert response.chunks[0].metadata["original_source"] == (
        RetrievalTool.BM25_SEARCH.value
    )
    assert response.chunks[0].metadata["rerank_score"] == response.chunks[0].score
    assert response.chunks[0].metadata["rerank_rank"] == 1
    assert response.chunks[0].citation.score == response.chunks[0].score


def test_rerank_chunks_uses_citation_and_metadata_text() -> None:
    document_id = uuid4()
    metadata_chunk_id = uuid4()
    content_chunk_id = uuid4()
    metadata_candidate = CandidateChunk(
        chunk_id=metadata_chunk_id,
        document_id=document_id,
        content=None,
        score=0.10,
        source=RetrievalTool.VECTOR_SEARCH,
        metadata={
            "retrieval_sources": [RetrievalTool.VECTOR_SEARCH.value],
            "section": "security policy encryption",
        },
        citation=Citation(
            document_id=document_id,
            chunk_id=metadata_chunk_id,
            title="Security Policy",
            quote=None,
            score=0.10,
        ),
    )
    content_candidate = CandidateChunk(
        chunk_id=content_chunk_id,
        document_id=document_id,
        content="General policy content.",
        score=0.50,
        source=RetrievalTool.BM25_SEARCH,
        metadata={"retrieval_sources": [RetrievalTool.BM25_SEARCH.value]},
    )

    response = rerank_chunks(
        query="security policy encryption",
        candidates=[content_candidate, metadata_candidate],
        top_k=2,
    )

    assert response.chunks[0].chunk_id == metadata_chunk_id


def test_rerank_chunks_does_not_mutate_original_candidates() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    candidate = CandidateChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content="Security policy content.",
        score=0.42,
        source=RetrievalTool.BM25_SEARCH,
        metadata={"retrieval_sources": [RetrievalTool.BM25_SEARCH.value]},
        citation=Citation(
            document_id=document_id,
            chunk_id=chunk_id,
            title="Security Policy",
            quote="Security policy content.",
            score=0.42,
        ),
    )

    response = rerank_chunks(
        query="security policy",
        candidates=[candidate],
        top_k=1,
    )

    assert response.chunks[0].score != candidate.score
    assert response.chunks[0].source == RetrievalTool.RERANK.value
    assert candidate.score == 0.42
    assert candidate.source == RetrievalTool.BM25_SEARCH
    assert candidate.metadata == {
        "retrieval_sources": [RetrievalTool.BM25_SEARCH.value]
    }
    assert candidate.citation.score == 0.42


def test_rerank_chunks_rejects_blank_query() -> None:
    with pytest.raises(HTTPException) as exc_info:
        rerank_chunks(
            query="   ",
            candidates=[],
            top_k=3,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Rerank query is required."


def test_rerank_chunks_rejects_invalid_top_k() -> None:
    with pytest.raises(HTTPException) as exc_info:
        rerank_chunks(
            query="security policy",
            candidates=[],
            top_k=0,
        )

    assert exc_info.value.status_code == 400
    assert "between 1 and 100" in exc_info.value.detail
