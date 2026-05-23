from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import DocumentChunk, Tenant
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType
from agentic_rag.workers.indexing import process_bm25_index_batch


class FakeSearchClient:
    def __init__(self, fail: bool = False):
        self.index_name = "chunks-test"
        self.fail = fail
        self.ensured = False
        self.indexed_chunks = []

    def ensure_chunk_index(self, index_name: str) -> None:
        self.ensured = True
        assert index_name == self.index_name

    def bulk_index_chunks(
        self,
        chunks: list[DocumentChunk],
        index_name: str,
    ) -> int:
        if self.fail:
            raise RuntimeError("OpenSearch unavailable")
        self.indexed_chunks.extend(chunks)
        assert index_name == self.index_name
        return len(chunks)


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def add_ready_chunk(db: Session) -> DocumentChunk:
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
    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=DocumentCreateRequest(
            workspace_id="workspace-a",
            source_type=DocumentSourceType.UPLOAD,
            source_uri="upload://policy.txt",
            title="Policy",
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
                "content": "Policy content for full text search.",
                "content_hash": "hash-1",
                "token_count": 6,
                "metadata": {},
            }
        ],
    )[0]


def test_process_bm25_index_batch_indexes_pending_chunk(db: Session) -> None:
    chunk = add_ready_chunk(db)
    search_client = FakeSearchClient()

    indexed_count = process_bm25_index_batch(
        db=db,
        search_client=search_client,
        limit=10,
    )
    stored_chunk = db.get(DocumentChunk, chunk.id)

    assert indexed_count == 1
    assert search_client.ensured is True
    assert [item.id for item in search_client.indexed_chunks] == [chunk.id]
    assert stored_chunk.bm25_index_status == "indexed"
    assert stored_chunk.bm25_index_name == "chunks-test"
    assert stored_chunk.bm25_index_content_hash == "hash-1"


def test_process_bm25_index_batch_marks_chunks_failed_on_error(db: Session) -> None:
    chunk = add_ready_chunk(db)
    search_client = FakeSearchClient(fail=True)

    indexed_count = process_bm25_index_batch(
        db=db,
        search_client=search_client,
        limit=10,
    )
    stored_chunk = db.get(DocumentChunk, chunk.id)

    assert indexed_count == 0
    assert stored_chunk.bm25_index_status == "failed"
    assert "OpenSearch unavailable" in stored_chunk.bm25_index_error
