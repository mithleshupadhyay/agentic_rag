from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException
import pytest

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval import vector_search
from agentic_rag.shared.db.crud.embeddings import ChunkVectorSearchResult
from agentic_rag.shared.db.models import Document, DocumentChunk
from agentic_rag.shared.schemas.llm import EmbeddingResponse
from agentic_rag.shared.schemas.retrieval import (
    RetrievalFilters,
    RetrievalStrategy,
    RetrievalTool,
)


def test_search_vector_chunks_generates_embedding_and_returns_candidates(
    monkeypatch,
) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    document = Document(
        id=document_id,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_type="upload",
        source_uri="upload://policy.md",
        title="Security Policy",
        file_name="policy.md",
        status="ready",
        owner_user_id="user-1",
        metadata_={},
    )
    chunk = DocumentChunk(
        id=chunk_id,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id=document_id,
        chunk_index=2,
        content="Password rotation policy content.",
        content_hash="hash-1",
        token_count=4,
        section_path="Security / Passwords",
        page_number=3,
        start_offset=10,
        end_offset=42,
        classification_level="internal",
        metadata_={},
    )
    chunk.document = document
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

    monkeypatch.setattr(
        vector_search.settings,
        "embedding_provider",
        "litellm",
    )
    monkeypatch.setattr(
        vector_search.settings,
        "embedding_model_name",
        "openai/text-embedding-3-small",
    )
    monkeypatch.setattr(vector_search.settings, "embedding_dimension", 768)
    monkeypatch.setattr(vector_search.settings, "embedding_vector_version", 7)

    def fake_embedding_client(request):
        captured["embedding_request"] = request
        return EmbeddingResponse(
            embeddings=[[1.0] + [0.0] * 767],
            model="openai/text-embedding-3-small",
            provider="litellm",
            dimension=768,
            latency_ms=4,
        )

    def fake_search_similar_chunks_by_embedding(**kwargs):
        captured["search_kwargs"] = kwargs
        return [
            ChunkVectorSearchResult(
                chunk=chunk,
                similarity=0.92,
                distance=0.08,
            )
        ]

    monkeypatch.setattr(
        vector_search,
        "search_similar_chunks_by_embedding",
        fake_search_similar_chunks_by_embedding,
    )

    response = vector_search.search_vector_chunks(
        db=None,
        user_context=user_context,
        query=" password policy ",
        filters=RetrievalFilters(
            workspace_id="workspace-a",
            document_ids=[document_id],
            source_types=["upload"],
            tags=["security"],
            metadata={"department": "security"},
            date_range={"created_at": {"gte": "2026-01-01T00:00:00Z"}},
        ),
        limit=5,
        min_similarity=0.7,
        embedding_client=fake_embedding_client,
    )

    embedding_request = captured["embedding_request"]
    search_kwargs = captured["search_kwargs"]

    assert response.strategy == RetrievalStrategy.VECTOR
    assert len(response.candidates) == 1
    assert response.candidates[0].source == RetrievalTool.VECTOR_SEARCH
    assert response.candidates[0].score == 0.92
    assert response.candidates[0].content == "Password rotation policy content."
    assert response.candidates[0].citation.title == "Security Policy"
    assert response.candidates[0].citation.page_number == 3
    assert response.candidates[0].metadata["embedding_model"] == (
        "openai/text-embedding-3-small"
    )
    assert response.candidates[0].metadata["vector_version"] == 7
    assert response.candidates[0].metadata["distance"] == 0.08
    assert embedding_request.texts == ["password policy"]
    assert embedding_request.provider == "litellm"
    assert embedding_request.model == "openai/text-embedding-3-small"
    assert embedding_request.auth.tenant_id == "tenant-a"
    assert embedding_request.auth.group_ids == ["security"]
    assert search_kwargs["tenant_id"] == "tenant-a"
    assert search_kwargs["embedding_model"] == "openai/text-embedding-3-small"
    assert search_kwargs["vector_version"] == 7
    assert search_kwargs["embedding_dimension"] == 768
    assert search_kwargs["workspace_id"] == "workspace-a"
    assert search_kwargs["document_ids"] == [document_id]
    assert search_kwargs["source_types"] == ["upload"]
    assert search_kwargs["tags"] == ["security"]
    assert search_kwargs["metadata_filters"] == {"department": "security"}
    assert search_kwargs["date_range"] == {
        "created_at": {"gte": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    }
    assert search_kwargs["user_context"] == user_context
    assert search_kwargs["min_similarity"] == 0.7


def test_search_vector_chunks_returns_empty_for_workspace_mismatch() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    def fake_embedding_client(_request):
        raise AssertionError("Embedding client should not be called.")

    response = vector_search.search_vector_chunks(
        db=None,
        user_context=user_context,
        query="security policy",
        filters=RetrievalFilters(workspace_id="workspace-b"),
        embedding_client=fake_embedding_client,
    )

    assert response.strategy == RetrievalStrategy.VECTOR
    assert response.candidates == []


def test_search_vector_chunks_deduplicates_results_before_limit(
    monkeypatch,
) -> None:
    document_id = uuid4()
    first_chunk_id = uuid4()
    second_chunk_id = uuid4()
    third_chunk_id = uuid4()
    document = Document(
        id=document_id,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_type="upload",
        source_uri="upload://guide.md",
        title="Security Guide",
        file_name="guide.md",
        status="ready",
        owner_user_id="user-1",
        metadata_={},
    )
    first_chunk = DocumentChunk(
        id=first_chunk_id,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id=document_id,
        chunk_index=1,
        content="First password policy chunk.",
        content_hash="hash-1",
        token_count=4,
        metadata_={},
    )
    second_chunk = DocumentChunk(
        id=second_chunk_id,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id=document_id,
        chunk_index=2,
        content="Second password policy chunk.",
        content_hash="hash-2",
        token_count=4,
        metadata_={},
    )
    third_chunk = DocumentChunk(
        id=third_chunk_id,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id=document_id,
        chunk_index=3,
        content="Third password policy chunk.",
        content_hash="hash-3",
        token_count=4,
        metadata_={},
    )
    first_chunk.document = document
    second_chunk.document = document
    third_chunk.document = document
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        scopes=["query:run"],
    )
    captured = {}

    monkeypatch.setattr(
        vector_search.settings,
        "embedding_provider",
        "litellm",
    )
    monkeypatch.setattr(
        vector_search.settings,
        "embedding_model_name",
        "openai/text-embedding-3-small",
    )
    monkeypatch.setattr(vector_search.settings, "embedding_dimension", 3)
    monkeypatch.setattr(vector_search.settings, "embedding_vector_version", 1)

    def fake_embedding_client(_request):
        return EmbeddingResponse(
            embeddings=[[1.0, 0.0, 0.0]],
            model="openai/text-embedding-3-small",
            provider="litellm",
            dimension=3,
            latency_ms=4,
        )

    def fake_search_similar_chunks_by_embedding(**kwargs):
        captured["search_kwargs"] = kwargs
        return [
            ChunkVectorSearchResult(
                chunk=first_chunk,
                similarity=0.70,
                distance=0.30,
            ),
            ChunkVectorSearchResult(
                chunk=second_chunk,
                similarity=0.90,
                distance=0.10,
            ),
            ChunkVectorSearchResult(
                chunk=first_chunk,
                similarity=0.95,
                distance=0.05,
            ),
            ChunkVectorSearchResult(
                chunk=third_chunk,
                similarity=0.80,
                distance=0.20,
            ),
        ]

    monkeypatch.setattr(
        vector_search,
        "search_similar_chunks_by_embedding",
        fake_search_similar_chunks_by_embedding,
    )

    response = vector_search.search_vector_chunks(
        db=None,
        user_context=user_context,
        query="password policy",
        limit=2,
        embedding_client=fake_embedding_client,
    )

    assert captured["search_kwargs"]["limit"] == 6
    assert response.strategy == RetrievalStrategy.VECTOR
    assert [candidate.chunk_id for candidate in response.candidates] == [
        first_chunk_id,
        second_chunk_id,
    ]
    assert response.candidates[0].score == 0.95
    assert response.candidates[0].metadata["distance"] == 0.05
    assert response.candidates[0].content == "First password policy chunk."
    assert response.candidates[1].score == 0.90


def test_search_vector_chunks_rejects_invalid_metadata_filter() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )

    def fake_embedding_client(_request):
        raise AssertionError("Embedding client should not be called.")

    with pytest.raises(HTTPException) as exc_info:
        vector_search.search_vector_chunks(
            db=None,
            user_context=user_context,
            query="security policy",
            filters=RetrievalFilters(metadata={"department.name": "security"}),
            embedding_client=fake_embedding_client,
        )

    assert exc_info.value.status_code == 400
    assert "Metadata filter keys" in exc_info.value.detail


def test_search_vector_chunks_rejects_embedding_dimension_mismatch() -> None:
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )

    def fake_embedding_client(_request):
        return EmbeddingResponse(
            embeddings=[[1.0, 0.0]],
            model="custom/embedding-model",
            provider="litellm",
            dimension=2,
            latency_ms=4,
        )

    with pytest.raises(RuntimeError) as exc_info:
        vector_search.search_vector_chunks(
            db=None,
            user_context=user_context,
            query="security policy",
            embedding_client=fake_embedding_client,
        )

    assert "Embedding dimension does not match" in str(exc_info.value)
