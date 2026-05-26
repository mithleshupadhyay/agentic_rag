from uuid import uuid4

from agentic_rag.query.answer_verifier import verify_answer_support
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import ContextChunk


def test_verify_answer_support_passes_with_valid_citation() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    context = [
        ContextChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content="Security policy content.",
            token_count=3,
            citation=Citation(
                document_id=document_id,
                chunk_id=chunk_id,
                title="Security Policy",
            ),
        )
    ]

    result = verify_answer_support(
        answer="Security policy content is available [1].",
        context=context,
    )

    assert result.passed is True
    assert result.cited_source_numbers == [1]


def test_verify_answer_support_fails_without_citation_when_context_exists() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    context = [
        ContextChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content="Security policy content.",
            token_count=3,
            citation=Citation(
                document_id=document_id,
                chunk_id=chunk_id,
                title="Security Policy",
            ),
        )
    ]

    result = verify_answer_support(
        answer="Security policy content is available.",
        context=context,
    )

    assert result.passed is False
    assert result.reason == "Answer did not cite retrieved context."


def test_verify_answer_support_fails_with_unknown_citation_number() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    context = [
        ContextChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content="Security policy content.",
            token_count=3,
            citation=Citation(
                document_id=document_id,
                chunk_id=chunk_id,
                title="Security Policy",
            ),
        )
    ]

    result = verify_answer_support(
        answer="Security policy content is available [2].",
        context=context,
    )

    assert result.passed is False
    assert result.cited_source_numbers == [2]
    assert "not present" in result.reason


def test_verify_answer_support_passes_when_context_is_empty() -> None:
    result = verify_answer_support(
        answer="No relevant context was found.",
        context=[],
    )

    assert result.passed is True
