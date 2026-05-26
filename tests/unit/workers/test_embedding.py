from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import ChunkEmbedding, DocumentChunk, Tenant
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType
from agentic_rag.shared.schemas.llm import EmbeddingRequest, EmbeddingResponse
from agentic_rag.workers.embedding import (
    process_embedding_batch,
    process_embedding_batches,
)


class FakeEmbeddingClient:
    def __init__(self, vector_value: float = 0.01, fail: bool = False):
        self.vector_value = vector_value
        self.fail = fail
        self.requests: list[EmbeddingRequest] = []

    def __call__(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("Embedding provider unavailable")
        return EmbeddingResponse(
            embeddings=[[self.vector_value] * 768 for _ in request.texts],
            model=request.model or "text-embedding-test",
            provider=request.provider or "litellm",
            dimension=768,
            latency_ms=15,
        )


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def add_tenant(db: Session, tenant_id: str) -> None:
    existing_tenant = db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
    if existing_tenant:
        return

    db.add(
        Tenant(
            tenant_id=tenant_id,
            name=tenant_id.title(),
            slug=tenant_id,
            status="active",
            metadata_={},
        )
    )
    db.commit()


def add_ready_chunk(
    db: Session,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-a",
    content: str = "Only authorized users can read this policy.",
    content_hash: str = "hash-1",
) -> DocumentChunk:
    add_tenant(db, tenant_id)
    user_context = UserContext(
        id="user-1",
        customer_id=tenant_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )
    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=DocumentCreateRequest(
            workspace_id=workspace_id,
            source_type=DocumentSourceType.UPLOAD,
            source_uri=f"upload://{tenant_id}/policy.txt",
            title=f"{tenant_id} Policy",
            metadata={},
            acl=AclPolicy(
                visibility=Visibility.TENANT,
                acl_version=1,
            ),
        ),
    )
    document.status = "ready"
    db.commit()
    db.refresh(document)
    return replace_document_chunks(
        db=db,
        document=document,
        chunks=[
            {
                "chunk_index": 0,
                "content": content,
                "content_hash": content_hash,
                "token_count": max(1, len(content.split())),
                "metadata": {},
            }
        ],
    )[0]


def test_process_embedding_batch_writes_missing_embedding(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = add_ready_chunk(db)
    embedding_client = FakeEmbeddingClient()
    monkeypatch.setattr(
        "agentic_rag.workers.embedding.settings.embedding_model_name",
        "text-embedding-test",
    )
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_provider", "litellm")
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_dimension", 768)
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_vector_version", 1)

    written_count = process_embedding_batch(
        db=db,
        tenant_id="tenant-a",
        embedding_client=embedding_client,
        limit=10,
    )
    stored_embedding = db.scalars(select(ChunkEmbedding)).one()

    assert written_count == 1
    assert len(embedding_client.requests) == 1
    assert embedding_client.requests[0].auth.tenant_id == "tenant-a"
    assert embedding_client.requests[0].texts == [chunk.content]
    assert stored_embedding.tenant_id == "tenant-a"
    assert stored_embedding.chunk_id == chunk.id
    assert stored_embedding.embedding_model == "text-embedding-test"
    assert stored_embedding.content_hash == "hash-1"
    assert stored_embedding.metadata_["source"] == "embedding-worker"


def test_process_embedding_batch_reembeds_stale_chunk_hash(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = add_ready_chunk(db)
    monkeypatch.setattr(
        "agentic_rag.workers.embedding.settings.embedding_model_name",
        "text-embedding-test",
    )
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_provider", "litellm")
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_dimension", 768)
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_vector_version", 1)

    process_embedding_batch(
        db=db,
        tenant_id="tenant-a",
        embedding_client=FakeEmbeddingClient(vector_value=0.01),
        limit=10,
    )
    chunk.content = "Updated policy text for semantic retrieval."
    chunk.content_hash = "hash-2"
    db.commit()
    db.refresh(chunk)

    written_count = process_embedding_batch(
        db=db,
        tenant_id="tenant-a",
        embedding_client=FakeEmbeddingClient(vector_value=0.02),
        limit=10,
    )
    stored_embeddings = db.scalars(select(ChunkEmbedding)).all()

    assert written_count == 1
    assert len(stored_embeddings) == 1
    assert stored_embeddings[0].content_hash == "hash-2"


def test_process_embedding_batches_runs_per_active_tenant(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_ready_chunk(db, tenant_id="tenant-a", content_hash="hash-a")
    add_ready_chunk(
        db,
        tenant_id="tenant-b",
        workspace_id="workspace-b",
        content="Tenant B policy content.",
        content_hash="hash-b",
    )
    embedding_client = FakeEmbeddingClient()
    monkeypatch.setattr(
        "agentic_rag.workers.embedding.settings.embedding_model_name",
        "text-embedding-test",
    )
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_provider", "litellm")
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_dimension", 768)
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_vector_version", 1)
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_batch_size", 10)

    written_count = process_embedding_batches(
        db=db,
        embedding_client=embedding_client,
        max_chunks=10,
    )
    stored_embeddings = db.scalars(select(ChunkEmbedding)).all()

    assert written_count == 2
    assert len(embedding_client.requests) == 2
    assert {request.auth.tenant_id for request in embedding_client.requests} == {
        "tenant-a",
        "tenant-b",
    }
    assert len(stored_embeddings) == 2


def test_process_embedding_batch_keeps_chunks_pending_on_provider_failure(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_ready_chunk(db)
    monkeypatch.setattr(
        "agentic_rag.workers.embedding.settings.embedding_model_name",
        "text-embedding-test",
    )
    monkeypatch.setattr("agentic_rag.workers.embedding.settings.embedding_vector_version", 1)

    written_count = process_embedding_batch(
        db=db,
        tenant_id="tenant-a",
        embedding_client=FakeEmbeddingClient(fail=True),
        limit=10,
    )
    stored_embeddings = db.scalars(select(ChunkEmbedding)).all()

    assert written_count == 0
    assert stored_embeddings == []
