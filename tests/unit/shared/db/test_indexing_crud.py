from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.indexing import (
    list_chunks_pending_bm25_index,
    mark_chunk_bm25_failed,
    mark_chunk_bm25_indexed,
)
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def add_tenant(db: Session, tenant_id: str) -> None:
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


def create_ready_document_with_chunk(db: Session, tenant_id: str = "tenant-a"):
    add_tenant(db, tenant_id)
    user_context = UserContext(
        id="user-1",
        customer_id=tenant_id,
        tenant_id=tenant_id,
        workspace_id="workspace-a",
    )
    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=DocumentCreateRequest(
            workspace_id="workspace-a",
            source_type=DocumentSourceType.UPLOAD,
            source_uri="upload://policy.txt",
            title="Security Policy",
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
                "content": "Only authorized users can read this policy.",
                "content_hash": "hash-1",
                "token_count": 7,
                "start_offset": 0,
                "end_offset": 44,
                "metadata": {"section": "access"},
            }
        ],
    )
    return document, chunks[0]


def test_list_chunks_pending_bm25_index_finds_ready_chunk(db: Session) -> None:
    _, chunk = create_ready_document_with_chunk(db)

    chunks = list_chunks_pending_bm25_index(
        db=db,
        index_name="chunks-test",
    )

    assert [item.id for item in chunks] == [chunk.id]
    assert chunks[0].document.title == "Security Policy"
    assert chunks[0].acl.allowed_group_ids == ["security"]


def test_mark_chunk_bm25_indexed_skips_unchanged_chunk(db: Session) -> None:
    _, chunk = create_ready_document_with_chunk(db)

    mark_chunk_bm25_indexed(db, chunk, "chunks-test")
    chunks = list_chunks_pending_bm25_index(
        db=db,
        index_name="chunks-test",
    )

    assert chunks == []


def test_changed_chunk_hash_is_selected_for_reindex(db: Session) -> None:
    _, chunk = create_ready_document_with_chunk(db)

    chunk = mark_chunk_bm25_indexed(db, chunk, "chunks-test")
    chunk.content = "Updated policy text."
    chunk.content_hash = "hash-2"
    db.commit()

    chunks = list_chunks_pending_bm25_index(
        db=db,
        index_name="chunks-test",
    )

    assert [item.id for item in chunks] == [chunk.id]


def test_mark_chunk_bm25_failed_records_error(db: Session) -> None:
    _, chunk = create_ready_document_with_chunk(db)

    failed_chunk = mark_chunk_bm25_failed(
        db=db,
        chunk=chunk,
        error_message="OpenSearch unavailable",
    )

    assert failed_chunk.bm25_index_status == "failed"
    assert failed_chunk.bm25_index_error == "OpenSearch unavailable"
