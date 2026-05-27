from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.embeddings import (
    bulk_create_chunk_embeddings,
    create_chunk_embedding,
    embedding_exists,
    get_chunks_missing_embedding,
    search_similar_chunks_by_embedding,
)
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import ChunkEmbedding, DocumentChunk, Tenant
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.chunks import ChunkEmbeddingCreate
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def add_tenant(db: Session, tenant_id: str) -> None:
    if db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first():
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


def create_ready_document_with_chunk(
    db: Session,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-a",
    content: str = "Only authorized users can read this policy.",
    content_hash: str = "hash-1",
):
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
            title=f"{tenant_id} Security Policy",
            metadata={"department": "security"},
            acl=AclPolicy(
                visibility=Visibility.GROUP,
                allowed_group_ids=["security"],
                acl_version=2,
            ),
        ),
    )
    document.status = "ready"
    db.commit()
    db.refresh(document)
    chunks = replace_document_chunks(
        db=db,
        document=document,
        chunks=[
            {
                "chunk_index": 0,
                "content": content,
                "content_hash": content_hash,
                "token_count": 7,
                "start_offset": 0,
                "end_offset": len(content),
                "metadata": {"section": "access"},
            }
        ],
    )
    return document, chunks[0]


def embedding_payload(
    chunk: DocumentChunk,
    embedding: list[float] | None = None,
    content_hash: str | None = None,
) -> ChunkEmbeddingCreate:
    vector = embedding or [0.01] * 768
    return ChunkEmbeddingCreate(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        embedding=vector,
        embedding_model="text-embedding-test",
        embedding_dimension=768,
        content_hash=content_hash or chunk.content_hash,
        vector_version=1,
        metadata={"source": "unit-test"},
    )


def vector_at(index: int) -> list[float]:
    vector = [0.0] * 768
    vector[index] = 1.0
    return vector


def test_create_chunk_embedding_persists_tenant_scoped_embedding(
    db: Session,
) -> None:
    document, chunk = create_ready_document_with_chunk(db)

    embedding = create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(chunk),
    )

    assert embedding.tenant_id == "tenant-a"
    assert embedding.workspace_id == document.workspace_id
    assert embedding.document_id == document.id
    assert embedding.chunk_id == chunk.id
    assert embedding.embedding_model == "text-embedding-test"
    assert embedding.embedding_dimension == 768
    assert embedding.content_hash == "hash-1"
    assert embedding.metadata_ == {"source": "unit-test"}
    assert embedding_exists(
        db=db,
        tenant_id="tenant-a",
        chunk_id=chunk.id,
        embedding_model="text-embedding-test",
        content_hash="hash-1",
    )


def test_create_chunk_embedding_rejects_cross_tenant_chunk(db: Session) -> None:
    _, chunk = create_ready_document_with_chunk(db)

    with pytest.raises(HTTPException) as exc_info:
        create_chunk_embedding(
            db=db,
            tenant_id="tenant-b",
            obj_in=embedding_payload(chunk),
        )

    assert exc_info.value.status_code == 404


def test_create_chunk_embedding_skips_existing_same_hash(db: Session) -> None:
    _, chunk = create_ready_document_with_chunk(db)

    first_embedding = create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(chunk),
    )
    second_embedding = create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(chunk),
    )

    stored_embeddings = db.scalars(select(ChunkEmbedding)).all()

    assert first_embedding.id == second_embedding.id
    assert len(stored_embeddings) == 1


def test_bulk_create_chunk_embeddings_updates_stale_embedding(db: Session) -> None:
    _, chunk = create_ready_document_with_chunk(db)
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(chunk),
    )
    chunk.content = "Updated policy text."
    chunk.content_hash = "hash-2"
    db.commit()
    db.refresh(chunk)

    written_count = bulk_create_chunk_embeddings(
        db=db,
        tenant_id="tenant-a",
        embeddings=[
            embedding_payload(
                chunk,
                embedding=[0.02] * 768,
            )
        ],
    )
    skipped_count = bulk_create_chunk_embeddings(
        db=db,
        tenant_id="tenant-a",
        embeddings=[
            embedding_payload(
                chunk,
                embedding=[0.02] * 768,
            )
        ],
    )
    stored_embeddings = db.scalars(select(ChunkEmbedding)).all()

    assert written_count == 1
    assert skipped_count == 0
    assert len(stored_embeddings) == 1
    assert stored_embeddings[0].content_hash == "hash-2"


def test_get_chunks_missing_embedding_selects_only_missing_ready_chunks(
    db: Session,
) -> None:
    _, embedded_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content_hash="hash-1",
    )
    _, missing_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-b",
        workspace_id="workspace-b",
        content="Security incidents must be reported immediately.",
        content_hash="hash-2",
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(embedded_chunk),
    )

    tenant_a_missing_chunks = get_chunks_missing_embedding(
        db=db,
        tenant_id="tenant-a",
        embedding_model="text-embedding-test",
    )
    tenant_b_missing_chunks = get_chunks_missing_embedding(
        db=db,
        tenant_id="tenant-b",
        embedding_model="text-embedding-test",
    )

    assert tenant_a_missing_chunks == []
    assert [chunk.id for chunk in tenant_b_missing_chunks] == [missing_chunk.id]

    embedded_chunk.content_hash = "hash-3"
    db.commit()

    tenant_a_stale_chunks = get_chunks_missing_embedding(
        db=db,
        tenant_id="tenant-a",
        embedding_model="text-embedding-test",
    )

    assert [chunk.id for chunk in tenant_a_stale_chunks] == [embedded_chunk.id]


def test_bulk_create_chunk_embeddings_rejects_dimension_mismatch(
    db: Session,
) -> None:
    _, chunk = create_ready_document_with_chunk(db)

    with pytest.raises(HTTPException) as exc_info:
        bulk_create_chunk_embeddings(
            db=db,
            tenant_id="tenant-a",
            embeddings=[
                embedding_payload(
                    chunk,
                    embedding=[0.01, 0.02],
                )
            ],
        )

    assert exc_info.value.status_code == 400
    assert db.scalars(select(ChunkEmbedding)).all() == []


def test_search_similar_chunks_by_embedding_returns_ranked_tenant_results(
    db: Session,
) -> None:
    _, closest_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Password rotation policy.",
        content_hash="hash-vector-1",
    )
    _, farther_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Office lunch schedule.",
        content_hash="hash-vector-2",
    )
    _, other_tenant_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-b",
        workspace_id="workspace-b",
        content="Password rotation policy from another tenant.",
        content_hash="hash-vector-3",
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(closest_chunk, embedding=vector_at(0)),
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(farther_chunk, embedding=vector_at(1)),
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-b",
        obj_in=embedding_payload(other_tenant_chunk, embedding=vector_at(0)),
    )

    results = search_similar_chunks_by_embedding(
        db=db,
        tenant_id="tenant-a",
        query_embedding=vector_at(0),
        embedding_model="text-embedding-test",
        limit=10,
    )

    assert [result.chunk.id for result in results] == [
        closest_chunk.id,
        farther_chunk.id,
    ]
    assert results[0].similarity > results[1].similarity
    assert results[0].distance < results[1].distance


def test_search_similar_chunks_by_embedding_filters_model_and_version(
    db: Session,
) -> None:
    _, current_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Current embedding model chunk.",
        content_hash="hash-current-model",
    )
    _, wrong_model_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Old embedding model chunk.",
        content_hash="hash-wrong-model",
    )
    _, wrong_version_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Old vector version chunk.",
        content_hash="hash-wrong-version",
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(current_chunk, embedding=vector_at(0)),
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=ChunkEmbeddingCreate(
            chunk_id=wrong_model_chunk.id,
            document_id=wrong_model_chunk.document_id,
            embedding=vector_at(0),
            embedding_model="old-embedding-model",
            embedding_dimension=768,
            content_hash=wrong_model_chunk.content_hash,
            vector_version=1,
            metadata={"source": "unit-test"},
        ),
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=ChunkEmbeddingCreate(
            chunk_id=wrong_version_chunk.id,
            document_id=wrong_version_chunk.document_id,
            embedding=vector_at(0),
            embedding_model="text-embedding-test",
            embedding_dimension=768,
            content_hash=wrong_version_chunk.content_hash,
            vector_version=2,
            metadata={"source": "unit-test"},
        ),
    )

    results = search_similar_chunks_by_embedding(
        db=db,
        tenant_id="tenant-a",
        query_embedding=vector_at(0),
        embedding_model="text-embedding-test",
        vector_version=1,
    )

    assert [result.chunk.id for result in results] == [current_chunk.id]


def test_search_similar_chunks_by_embedding_excludes_deleted_records(
    db: Session,
) -> None:
    _, deleted_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Deleted chunk.",
        content_hash="hash-deleted-chunk",
    )
    deleted_document, document_chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Deleted document chunk.",
        content_hash="hash-deleted-document",
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(deleted_chunk, embedding=vector_at(0)),
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(document_chunk, embedding=vector_at(0)),
    )
    deleted_chunk.is_deleted = True
    deleted_document.is_deleted = True
    db.commit()

    results = search_similar_chunks_by_embedding(
        db=db,
        tenant_id="tenant-a",
        query_embedding=vector_at(0),
        embedding_model="text-embedding-test",
    )

    assert results == []


def test_search_similar_chunks_by_embedding_applies_user_acl(
    db: Session,
) -> None:
    _, chunk = create_ready_document_with_chunk(
        db,
        tenant_id="tenant-a",
        content="Security group only chunk.",
        content_hash="hash-acl-vector",
    )
    create_chunk_embedding(
        db=db,
        tenant_id="tenant-a",
        obj_in=embedding_payload(chunk, embedding=vector_at(0)),
    )
    denied_context = UserContext(
        id="user-2",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        group_ids=[],
        acl_version=2,
    )
    allowed_context = UserContext(
        id="user-2",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        group_ids=["security"],
        acl_version=2,
    )

    denied_results = search_similar_chunks_by_embedding(
        db=db,
        tenant_id="tenant-a",
        query_embedding=vector_at(0),
        embedding_model="text-embedding-test",
        user_context=denied_context,
    )
    allowed_results = search_similar_chunks_by_embedding(
        db=db,
        tenant_id="tenant-a",
        query_embedding=vector_at(0),
        embedding_model="text-embedding-test",
        user_context=allowed_context,
    )

    assert denied_results == []
    assert [result.chunk.id for result in allowed_results] == [chunk.id]
