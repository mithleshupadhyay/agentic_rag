from uuid import uuid4

from fastapi import HTTPException
import pytest

from agentic_rag.retrieval.context_builder import build_context
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    ContextBuildRequest,
    RetrievalTool,
)


def make_candidate(
    content: str | None,
    score: float = 1.0,
    token_count: int | None = None,
    chunk_id=None,
    document_id=None,
    quote: str | None = None,
) -> CandidateChunk:
    chunk_id = chunk_id or uuid4()
    document_id = document_id or uuid4()
    metadata = {}
    if token_count is not None:
        metadata["token_count"] = token_count

    return CandidateChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        score=score,
        source=RetrievalTool.BM25_SEARCH,
        metadata=metadata,
        citation=Citation(
            document_id=document_id,
            chunk_id=chunk_id,
            title="Security Policy",
            quote=quote,
            score=score,
        ),
    )


def test_build_context_strips_highlight_html_and_preserves_citation() -> None:
    candidate = make_candidate(
        content="Security <em>policy</em> content &amp; controls.",
        token_count=5,
    )

    response = build_context(
        ContextBuildRequest(
            query="security policy",
            chunks=[candidate],
            max_context_chunks=3,
            max_tokens=500,
        )
    )

    assert response.token_count == 5
    assert len(response.context) == 1
    assert response.context[0].content == "Security policy content & controls."
    assert response.context[0].citation.quote == "Security policy content & controls."
    assert response.context[0].citation.title == "Security Policy"
    assert response.context[0].metadata["score"] == 1.0


def test_build_context_deduplicates_by_chunk_id_and_content() -> None:
    chunk_id = uuid4()
    first = make_candidate(
        content="Same chunk content.",
        score=2.0,
        token_count=3,
        chunk_id=chunk_id,
    )
    same_chunk = make_candidate(
        content="Same chunk content with lower score.",
        score=1.0,
        token_count=6,
        chunk_id=chunk_id,
    )
    repeated_content = make_candidate(
        content="Same chunk content.",
        score=0.5,
        token_count=3,
    )
    unique = make_candidate(
        content="Different useful content.",
        score=0.4,
        token_count=3,
    )

    response = build_context(
        ContextBuildRequest(
            query="policy",
            chunks=[first, same_chunk, repeated_content, unique],
            max_context_chunks=10,
            max_tokens=500,
        )
    )

    assert [chunk.content for chunk in response.context] == [
        "Same chunk content.",
        "Different useful content.",
    ]
    assert response.token_count == 6


def test_build_context_enforces_token_budget_with_truncation() -> None:
    first_content = " ".join(f"first-{index}" for index in range(400))
    second_content = " ".join(f"second-{index}" for index in range(200))
    first = make_candidate(
        content=first_content,
        token_count=400,
    )
    second = make_candidate(
        content=second_content,
        token_count=200,
    )

    response = build_context(
        ContextBuildRequest(
            query="numbers",
            chunks=[first, second],
            max_context_chunks=5,
            max_tokens=700,
        )
    )

    assert response.token_count == 600

    response = build_context(
        ContextBuildRequest(
            query="numbers",
            chunks=[first, second],
            max_context_chunks=5,
            max_tokens=500,
        )
    )

    assert response.token_count == 500
    assert response.context[0].content == first_content
    assert response.context[1].content == " ".join(f"second-{index}" for index in range(100))
    assert response.context[1].token_count == 100


def test_build_context_uses_citation_quote_when_content_missing() -> None:
    candidate = make_candidate(
        content=None,
        quote="Citation quote content.",
    )

    response = build_context(
        ContextBuildRequest(
            query="quote",
            chunks=[candidate],
            max_context_chunks=3,
            max_tokens=500,
        )
    )

    assert response.context[0].content == "Citation quote content."
    assert response.context[0].token_count == 3


def test_build_context_rejects_blank_query() -> None:
    with pytest.raises(HTTPException) as exc_info:
        build_context(
            ContextBuildRequest(
                query="   ",
                chunks=[],
                max_context_chunks=3,
                max_tokens=500,
            )
        )

    assert exc_info.value.status_code == 400
