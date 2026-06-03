from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import agentic_rag.workers.indexing as indexing_worker_module
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import DocumentChunk, Tenant
from agentic_rag.shared.kafka.events import EventEnvelope, EventType
from agentic_rag.shared.kafka.topics import INGESTION_INDEX
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType
from agentic_rag.workers.indexing import handle_indexing_event, process_bm25_index_batch


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


def add_ready_chunk(db: Session) -> DocumentChunk:
    add_tenant(db, "tenant-a")
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


def test_process_bm25_index_batch_filters_event_chunk_ids(db: Session) -> None:
    initial_chunk = add_ready_chunk(db)
    document = initial_chunk.document
    db.delete(initial_chunk)
    db.commit()
    db.refresh(document)
    first_chunk, second_chunk = replace_document_chunks(
        db=db,
        document=document,
        chunks=[
            {
                "chunk_index": 0,
                "content": "First policy section.",
                "content_hash": "hash-1",
                "token_count": 3,
                "metadata": {},
            },
            {
                "chunk_index": 1,
                "content": "Second policy section.",
                "content_hash": "hash-2",
                "token_count": 3,
                "metadata": {},
            },
        ],
    )
    search_client = FakeSearchClient()

    indexed_count = process_bm25_index_batch(
        db=db,
        search_client=search_client,
        limit=10,
        tenant_id="tenant-a",
        document_id=document.id,
        chunk_ids=[second_chunk.id],
    )

    assert indexed_count == 1
    assert [chunk.id for chunk in search_client.indexed_chunks] == [second_chunk.id]
    assert first_chunk.bm25_index_status == "pending"
    assert db.get(DocumentChunk, second_chunk.id).bm25_index_status == "indexed"


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


def test_handle_indexing_event_processes_scoped_chunks(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = add_ready_chunk(db)
    search_client = FakeSearchClient()

    class ExistingSessionContext:
        def __enter__(self):
            return db

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def get_session_factory():
        return ExistingSessionContext

    monkeypatch.setattr(
        indexing_worker_module,
        "get_sync_session_factory",
        get_session_factory,
    )
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_INDEX_REQUESTED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id="job-1",
        payload={
            "job_id": str(uuid4()),
            "document_id": str(chunk.document_id),
            "chunk_ids": [str(chunk.id)],
            "index_name": "chunks-test",
        },
    )

    handled = handle_indexing_event(
        envelope=envelope,
        search_client=search_client,
    )
    stored_chunk = db.get(DocumentChunk, chunk.id)

    assert handled is True
    assert [item.id for item in search_client.indexed_chunks] == [chunk.id]
    assert stored_chunk.bm25_index_status == "indexed"
    assert stored_chunk.bm25_index_name == "chunks-test"


def test_handle_indexing_event_skips_invalid_payload() -> None:
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_INDEX_REQUESTED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id="job-1",
        payload={"document_id": "not-a-uuid"},
    )

    handled = handle_indexing_event(envelope)

    assert handled is False


def test_handle_indexing_event_skips_unexpected_index_name() -> None:
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_INDEX_REQUESTED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id="job-1",
        payload={
            "job_id": str(uuid4()),
            "document_id": str(uuid4()),
            "chunk_ids": [str(uuid4())],
            "index_name": "unexpected-index",
        },
    )

    handled = handle_indexing_event(
        envelope=envelope,
        search_client=FakeSearchClient(),
    )

    assert handled is False


def test_indexing_worker_loop_uses_configured_indexing_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_consumers = []

    class RuntimeConsumer:
        def __init__(self) -> None:
            self.closed = False
            self.ran = False

        def run_forever(self) -> None:
            self.ran = True
            raise KeyboardInterrupt

        def close(self) -> None:
            self.closed = True

    runtime_consumer = RuntimeConsumer()
    monkeypatch.setattr(
        indexing_worker_module.settings,
        "kafka_consuming_enabled",
        True,
    )
    monkeypatch.setattr(
        indexing_worker_module.settings,
        "kafka_indexing_consumer_group",
        "agentic-rag-indexing-test",
    )

    def create_consumer_spy(**kwargs):
        created_consumers.append(kwargs)
        return runtime_consumer

    monkeypatch.setattr(
        indexing_worker_module,
        "create_kafka_event_consumer",
        create_consumer_spy,
    )
    monkeypatch.setattr(
        indexing_worker_module,
        "run_indexing_worker_once",
        lambda **kwargs: pytest.fail("DB polling should not run in consume mode"),
    )

    indexing_worker_module.run_indexing_worker_loop()

    assert created_consumers[0]["topics"] == [INGESTION_INDEX]
    assert created_consumers[0]["group_id"] == "agentic-rag-indexing-test"
    assert created_consumers[0]["client_id"] == "indexing-worker-consumer"
    assert callable(created_consumers[0]["handler"])
    assert runtime_consumer.ran is True
    assert runtime_consumer.closed is True
