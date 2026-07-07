from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import agentic_rag.workers.ingestion as ingestion_worker_module
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.monitoring.metrics import (
    WORKER_ITEM_TOTAL,
    WORKER_JOB_LATENCY_SECONDS,
    WORKER_JOB_LIFECYCLE_TOTAL,
    WORKER_QUEUE_LAG_SECONDS,
    WORKER_RETRY_LIFECYCLE_TOTAL,
)
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import (
    attach_document_object,
    create_document,
    create_ingestion_job_for_document,
)
from agentic_rag.shared.db.crud.ingestion import (
    renew_ingestion_job_lease as real_renew_ingestion_job_lease,
)
from agentic_rag.shared.db.models import Document, DocumentChunk, IngestionJob, Tenant
from agentic_rag.shared.kafka.events import EventEnvelope, EventType
from agentic_rag.shared.kafka.topics import (
    DLQ_INGESTION,
    INGESTION_EMBED,
    INGESTION_INDEX,
    INGESTION_PARSE,
    RETRY_INGESTION,
)
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSourceType,
    FileMetadata,
)
from agentic_rag.workers.ingestion import (
    decode_text_document,
    handle_ingestion_parse_event,
    handle_ingestion_retry_event,
    process_ingestion_job,
    run_ingestion_worker_once,
    split_text_into_chunks,
)


class FakeObjectStore:
    def __init__(self, data: bytes):
        self.data = data
        self.read_keys = []

    def get_bytes(self, object_key: str) -> bytes:
        self.read_keys.append(object_key)
        return self.data


class FakeEventPublisher:
    def __init__(self):
        self.published = []

    def __call__(self, topic: str, envelope: EventEnvelope) -> None:
        self.published.append((topic, envelope))


def counter_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


def histogram_count(histogram, **labels) -> float:
    return sum(bucket.get() for bucket in histogram.labels(**labels)._buckets)


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


def create_job(db: Session, file_name: str = "policy.md", mime_type: str = "text/markdown"):
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
            source_uri=f"upload://{file_name}",
            title="Policy",
            file=FileMetadata(
                file_name=file_name,
                mime_type=mime_type,
                byte_size=80,
                content_hash="document-hash",
            ),
            metadata={},
            acl=AclPolicy(
                visibility=Visibility.PRIVATE,
                allowed_user_ids=["user-1"],
                acl_version=3,
            ),
        ),
    )
    document = attach_document_object(
        db=db,
        db_obj=document,
        object_key=f"tenants/tenant-a/workspaces/workspace-a/documents/{document.id}/raw/{file_name}",
    )
    job = create_ingestion_job_for_document(
        user_context=user_context,
        db=db,
        document=document,
    )
    return document, job


def test_decode_text_document_accepts_text_like_files() -> None:
    text = decode_text_document(
        data=b"# Policy\nOnly authorized users can read this.",
        file_name="policy.md",
        mime_type="text/markdown",
    )

    assert text.startswith("# Policy")


def test_decode_text_document_rejects_unsupported_file_type() -> None:
    with pytest.raises(ValueError):
        decode_text_document(
            data=b"%PDF-1.7",
            file_name="policy.pdf",
            mime_type="application/pdf",
        )


def test_split_text_into_chunks_uses_overlap_and_offsets() -> None:
    chunks = split_text_into_chunks(
        text="alpha beta gamma delta epsilon zeta eta theta",
        chunk_size=24,
        chunk_overlap=5,
    )

    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset > chunks[0].start_offset
    assert chunks[1].start_offset < chunks[0].end_offset
    assert chunks[0].content_hash


def test_process_ingestion_job_reads_object_and_stores_chunks(db: Session) -> None:
    document, job = create_job(db)
    object_store = FakeObjectStore(
        b"# Security Policy\n\nOnly authorized users can read this policy.\n"
        b"Every document must be tenant scoped."
    )
    lifecycle_labels = {
        "worker": "ingestion-worker",
        "job_type": "document_ingestion",
    }
    completed_before = counter_value(
        WORKER_JOB_LIFECYCLE_TOTAL,
        **lifecycle_labels,
        status="completed",
    )
    latency_before = histogram_count(
        WORKER_JOB_LATENCY_SECONDS,
        **lifecycle_labels,
        status="completed",
    )
    queue_lag_before = histogram_count(
        WORKER_QUEUE_LAG_SECONDS,
        **lifecycle_labels,
        source="direct",
    )
    chunks_before = counter_value(
        WORKER_ITEM_TOTAL,
        **lifecycle_labels,
        item_type="chunk",
        status="created",
    )

    processed_job = process_ingestion_job(
        db=db,
        job=job,
        object_store=object_store,
    )
    stored_document = db.get(Document, document.id)
    stored_job = db.get(IngestionJob, job.id)
    stored_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    assert processed_job.status == "completed"
    assert stored_job.status == "completed"
    assert stored_job.current_stage == "complete"
    assert stored_document.status == "ready"
    assert len(stored_chunks) == 1
    assert stored_chunks[0].content.startswith("# Security Policy")
    assert stored_chunks[0].acl.allowed_user_ids == ["user-1"]
    assert object_store.read_keys == [document.object_key]
    assert (
        counter_value(
            WORKER_JOB_LIFECYCLE_TOTAL,
            **lifecycle_labels,
            status="completed",
        )
        == completed_before + 1
    )
    assert (
        histogram_count(
            WORKER_JOB_LATENCY_SECONDS,
            **lifecycle_labels,
            status="completed",
        )
        == latency_before + 1
    )
    assert (
        histogram_count(
            WORKER_QUEUE_LAG_SECONDS,
            **lifecycle_labels,
            source="direct",
        )
        == queue_lag_before + 1
    )
    assert (
        counter_value(
            WORKER_ITEM_TOTAL,
            **lifecycle_labels,
            item_type="chunk",
            status="created",
        )
        == chunks_before + len(stored_chunks)
    )


def test_process_ingestion_job_publishes_embedding_and_indexing_events(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, job = create_job(db)
    event_publisher = FakeEventPublisher()
    object_store = FakeObjectStore(
        b"# Security Policy\n\nOnly authorized users can read this policy."
    )
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "embedding_model_name",
        "text-embedding-test",
    )
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "embedding_vector_version",
        2,
    )
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "opensearch_chunk_index",
        "chunks-test",
    )

    process_ingestion_job(
        db=db,
        job=job,
        object_store=object_store,
        event_publisher=event_publisher,
    )
    stored_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    assert len(event_publisher.published) == 2
    published_by_topic = {
        topic: envelope
        for topic, envelope in event_publisher.published
    }

    embed_envelope = published_by_topic[INGESTION_EMBED]
    assert embed_envelope.event_type == EventType.DOCUMENT_EMBED_REQUESTED
    assert embed_envelope.tenant_id == "tenant-a"
    assert embed_envelope.workspace_id == "workspace-a"
    assert embed_envelope.correlation_id == str(job.id)
    assert embed_envelope.idempotency_key == f"ingestion-embed:{job.id}"
    assert embed_envelope.payload["job_id"] == str(job.id)
    assert embed_envelope.payload["document_id"] == str(document.id)
    assert embed_envelope.payload["chunk_ids"] == [
        str(chunk.id)
        for chunk in stored_chunks
    ]
    assert embed_envelope.payload["embedding_model"] == "text-embedding-test"
    assert embed_envelope.payload["vector_version"] == 2

    index_envelope = published_by_topic[INGESTION_INDEX]
    assert index_envelope.event_type == EventType.DOCUMENT_INDEX_REQUESTED
    assert index_envelope.tenant_id == "tenant-a"
    assert index_envelope.workspace_id == "workspace-a"
    assert index_envelope.correlation_id == str(job.id)
    assert index_envelope.idempotency_key == f"ingestion-index:{job.id}"
    assert index_envelope.payload["job_id"] == str(job.id)
    assert index_envelope.payload["document_id"] == str(document.id)
    assert index_envelope.payload["chunk_ids"] == [
        str(chunk.id)
        for chunk in stored_chunks
    ]
    assert index_envelope.payload["index_name"] == "chunks-test"


def test_run_ingestion_worker_once_claims_and_processes_job(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, job = create_job(db)
    object_store = FakeObjectStore(
        b"# Access Policy\n\nTenant scoped documents must stay isolated."
    )

    class ExistingSessionContext:
        def __enter__(self):
            return db

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def get_session_factory():
        return ExistingSessionContext

    monkeypatch.setattr(
        "agentic_rag.workers.ingestion.get_sync_session_factory",
        get_session_factory,
    )

    processed = run_ingestion_worker_once(object_store=object_store)
    stored_job = db.get(IngestionJob, job.id)
    stored_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .all()
    )

    assert processed is True
    assert stored_job.status == "completed"
    assert stored_job.locked_by is None
    assert stored_job.lease_expires_at is None
    assert len(stored_chunks) == 1
    assert object_store.read_keys == [document.object_key]


def test_process_ingestion_job_renews_lease_between_steps(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, job = create_job(db)
    object_store = FakeObjectStore(
        b"# Security Policy\n\nOnly authorized users can read this policy.\n"
        b"Every document must be tenant scoped."
    )
    renewed_stages = []

    def renew_lease_spy(
        db: Session,
        job: IngestionJob,
        worker_id: str,
        lease_seconds: int,
    ) -> IngestionJob:
        renewed_stages.append(job.current_stage)
        return real_renew_ingestion_job_lease(
            db=db,
            job=job,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )

    monkeypatch.setattr(
        "agentic_rag.workers.ingestion.renew_ingestion_job_lease",
        renew_lease_spy,
    )

    process_ingestion_job(
        db=db,
        job=job,
        object_store=object_store,
    )

    assert renewed_stages == ["parse", "parse", "chunk", "chunk"]


def test_process_ingestion_job_does_not_fail_job_after_lost_lease(
    db: Session,
) -> None:
    _, job = create_job(db)
    event_publisher = FakeEventPublisher()
    job.status = "running"
    job.locked_by = "other-worker"
    db.commit()

    process_ingestion_job(
        db=db,
        job=job,
        object_store=FakeObjectStore(b"# Security Policy"),
        event_publisher=event_publisher,
    )
    stored_job = db.get(IngestionJob, job.id)

    assert stored_job.status == "running"
    assert stored_job.locked_by == "other-worker"
    assert stored_job.error_type is None
    assert event_publisher.published == []


def test_process_ingestion_job_publishes_retry_event_for_retryable_failure(
    db: Session,
) -> None:
    document, job = create_job(
        db=db,
        file_name="policy.pdf",
        mime_type="application/pdf",
    )
    object_store = FakeObjectStore(b"%PDF-1.7")
    event_publisher = FakeEventPublisher()
    lifecycle_labels = {
        "worker": "ingestion-worker",
        "job_type": "document_ingestion",
    }
    failed_before = counter_value(
        WORKER_JOB_LIFECYCLE_TOTAL,
        **lifecycle_labels,
        status="failed",
    )
    retry_before = counter_value(
        WORKER_RETRY_LIFECYCLE_TOTAL,
        **lifecycle_labels,
        status="retry_scheduled",
        topic=RETRY_INGESTION,
    )

    process_ingestion_job(
        db=db,
        job=job,
        object_store=object_store,
        event_publisher=event_publisher,
    )
    stored_document = db.get(Document, document.id)
    stored_job = db.get(IngestionJob, job.id)

    assert stored_document.status == "failed"
    assert stored_job.status == "failed"
    assert stored_job.error_type == "ValueError"
    assert "Unsupported ingestion file type" in stored_job.error_message
    assert stored_job.next_retry_at is not None
    assert len(event_publisher.published) == 1

    topic, envelope = event_publisher.published[0]
    assert topic == RETRY_INGESTION
    assert envelope.event_type == EventType.INGESTION_RETRY_SCHEDULED
    assert envelope.tenant_id == "tenant-a"
    assert envelope.workspace_id == "workspace-a"
    assert envelope.correlation_id == str(job.id)
    assert envelope.idempotency_key == f"ingestion-retry:{job.id}:1"
    assert envelope.payload["job_id"] == str(job.id)
    assert envelope.payload["document_id"] == str(document.id)
    assert envelope.payload["retry_topic"] == RETRY_INGESTION
    assert envelope.payload["attempt"] == 1
    assert envelope.payload["max_attempts"] == stored_job.max_retries
    assert envelope.payload["error_type"] == "ValueError"
    assert envelope.payload["metadata"]["worker_id"] == "ingestion-worker"
    assert (
        counter_value(
            WORKER_JOB_LIFECYCLE_TOTAL,
            **lifecycle_labels,
            status="failed",
        )
        == failed_before + 1
    )
    assert (
        counter_value(
            WORKER_RETRY_LIFECYCLE_TOTAL,
            **lifecycle_labels,
            status="retry_scheduled",
            topic=RETRY_INGESTION,
        )
        == retry_before + 1
    )


def test_process_ingestion_job_publishes_dlq_event_when_retries_exhausted(
    db: Session,
) -> None:
    document, job = create_job(
        db=db,
        file_name="policy.pdf",
        mime_type="application/pdf",
    )
    job.retry_count = job.max_retries - 1
    db.commit()
    object_store = FakeObjectStore(b"%PDF-1.7")
    event_publisher = FakeEventPublisher()
    lifecycle_labels = {
        "worker": "ingestion-worker",
        "job_type": "document_ingestion",
    }
    dlq_before = counter_value(
        WORKER_RETRY_LIFECYCLE_TOTAL,
        **lifecycle_labels,
        status="dlq_recorded",
        topic=DLQ_INGESTION,
    )

    process_ingestion_job(
        db=db,
        job=job,
        object_store=object_store,
        event_publisher=event_publisher,
    )
    stored_job = db.get(IngestionJob, job.id)

    assert stored_job.status == "failed"
    assert stored_job.retry_count == stored_job.max_retries
    assert stored_job.next_retry_at is None
    assert len(event_publisher.published) == 1

    topic, envelope = event_publisher.published[0]
    assert topic == DLQ_INGESTION
    assert envelope.event_type == EventType.INGESTION_DLQ_RECORDED
    assert envelope.idempotency_key == f"ingestion-dlq:{job.id}:{stored_job.max_retries}"
    assert envelope.payload["job_id"] == str(job.id)
    assert envelope.payload["document_id"] == str(document.id)
    assert envelope.payload["dlq_topic"] == DLQ_INGESTION
    assert envelope.payload["attempt"] == stored_job.max_retries
    assert envelope.payload["terminal_reason"] == "max_retries_exhausted"
    assert (
        counter_value(
            WORKER_RETRY_LIFECYCLE_TOTAL,
            **lifecycle_labels,
            status="dlq_recorded",
            topic=DLQ_INGESTION,
        )
        == dlq_before + 1
    )


def test_handle_ingestion_parse_event_claims_and_processes_queued_job(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, job = create_job(db)
    processed_jobs = []
    event_publisher = FakeEventPublisher()

    class ExistingSessionContext:
        def __enter__(self):
            return db

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def get_session_factory():
        return ExistingSessionContext

    def process_spy(
        db: Session,
        job: IngestionJob,
        object_store=None,
        event_publisher=None,
        source: str = "direct",
    ) -> IngestionJob:
        processed_jobs.append((job.id, event_publisher))
        return job

    monkeypatch.setattr(
        ingestion_worker_module,
        "get_sync_session_factory",
        get_session_factory,
    )
    monkeypatch.setattr(
        ingestion_worker_module,
        "process_ingestion_job",
        process_spy,
    )
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_PARSE_REQUESTED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id=str(job.id),
        payload={
            "job_id": str(job.id),
            "document_id": str(job.document_id),
            "object_key": job.object_key,
            "mime_type": "text/markdown",
            "source_type": "upload",
        },
    )

    processed = handle_ingestion_parse_event(
        envelope=envelope,
        event_publisher=event_publisher,
    )
    stored_job = db.get(IngestionJob, job.id)

    assert processed is True
    assert processed_jobs == [(job.id, event_publisher)]
    assert stored_job.status == "running"
    assert stored_job.locked_by == "ingestion-worker"
    assert stored_job.next_retry_at is None


def test_handle_ingestion_parse_event_skips_invalid_payload() -> None:
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_PARSE_REQUESTED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id="request-1",
        payload={"job_id": str(uuid4())},
    )

    processed = handle_ingestion_parse_event(envelope)

    assert processed is False


def test_handle_ingestion_retry_event_claims_and_processes_retry_job(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, job = create_job(db)
    job.status = "failed"
    job.retry_count = 1
    job.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()
    processed_jobs = []
    event_publisher = FakeEventPublisher()

    class ExistingSessionContext:
        def __enter__(self):
            return db

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def get_session_factory():
        return ExistingSessionContext

    def process_spy(
        db: Session,
        job: IngestionJob,
        object_store=None,
        event_publisher=None,
        source: str = "direct",
    ) -> IngestionJob:
        processed_jobs.append((job.id, event_publisher))
        return job

    monkeypatch.setattr(
        ingestion_worker_module,
        "get_sync_session_factory",
        get_session_factory,
    )
    monkeypatch.setattr(
        ingestion_worker_module,
        "process_ingestion_job",
        process_spy,
    )
    envelope = EventEnvelope(
        event_type=EventType.INGESTION_RETRY_SCHEDULED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id=str(job.id),
        payload={
            "job_id": str(job.id),
            "document_id": str(job.document_id),
            "failed_stage": "parse",
            "source_topic": "ingestion.parse",
            "retry_topic": RETRY_INGESTION,
            "attempt": 1,
            "max_attempts": job.max_retries,
            "error_type": "ValueError",
            "error_message": "Temporary parser failure",
            "failed_at": datetime.now(timezone.utc),
            "next_retry_at": datetime.now(timezone.utc) - timedelta(seconds=1),
            "metadata": {},
        },
    )

    processed = handle_ingestion_retry_event(
        envelope=envelope,
        event_publisher=event_publisher,
    )
    stored_job = db.get(IngestionJob, job.id)

    assert processed is True
    assert processed_jobs == [(job.id, event_publisher)]
    assert stored_job.status == "running"
    assert stored_job.locked_by == "ingestion-worker"
    assert stored_job.next_retry_at is None


def test_handle_ingestion_retry_event_skips_missing_job(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExistingSessionContext:
        def __enter__(self):
            return db

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def get_session_factory():
        return ExistingSessionContext

    def process_spy(
        db: Session,
        job: IngestionJob,
        object_store=None,
        event_publisher=None,
    ) -> IngestionJob:
        raise AssertionError("missing retry job should not be processed")

    monkeypatch.setattr(
        ingestion_worker_module,
        "get_sync_session_factory",
        get_session_factory,
    )
    monkeypatch.setattr(
        ingestion_worker_module,
        "process_ingestion_job",
        process_spy,
    )
    job_id = uuid4()
    envelope = EventEnvelope(
        event_type=EventType.INGESTION_RETRY_SCHEDULED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id=str(job_id),
        payload={
            "job_id": str(job_id),
            "failed_stage": "parse",
            "source_topic": "ingestion.parse",
            "retry_topic": RETRY_INGESTION,
            "attempt": 1,
            "max_attempts": 3,
            "error_type": "ValueError",
            "failed_at": datetime.now(timezone.utc),
            "next_retry_at": datetime.now(timezone.utc) - timedelta(seconds=1),
            "metadata": {},
        },
    )

    processed = handle_ingestion_retry_event(envelope)

    assert processed is False


def test_handle_ingestion_retry_event_skips_invalid_payload() -> None:
    envelope = EventEnvelope(
        event_type=EventType.INGESTION_RETRY_SCHEDULED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id="request-1",
        payload={"job_id": str(uuid4())},
    )

    processed = handle_ingestion_retry_event(envelope)

    assert processed is False


def test_ingestion_worker_loop_keeps_kafka_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passed_publishers = []
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "kafka_publishing_enabled",
        False,
    )
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "kafka_consuming_enabled",
        False,
    )

    def run_once_spy(
        object_store=None,
        event_publisher=None,
    ) -> bool:
        passed_publishers.append(event_publisher)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        ingestion_worker_module,
        "run_ingestion_worker_once",
        run_once_spy,
    )
    monkeypatch.setattr(
        ingestion_worker_module,
        "create_kafka_event_consumer",
        lambda **kwargs: pytest.fail("Kafka consumer should be disabled"),
    )

    ingestion_worker_module.run_ingestion_worker_loop()

    assert passed_publishers == [None]


def test_ingestion_worker_loop_uses_configured_kafka_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passed_publishers = []
    created_client_ids = []

    class RuntimePublisher:
        def __init__(self) -> None:
            self.closed = False

        def __call__(self, topic: str, envelope: EventEnvelope) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    runtime_publisher = RuntimePublisher()
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "kafka_publishing_enabled",
        True,
    )
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "kafka_consuming_enabled",
        False,
    )

    def create_publisher_spy(client_id: str):
        created_client_ids.append(client_id)
        return runtime_publisher

    def run_once_spy(
        object_store=None,
        event_publisher=None,
    ) -> bool:
        passed_publishers.append(event_publisher)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        ingestion_worker_module,
        "create_kafka_event_publisher",
        create_publisher_spy,
    )
    monkeypatch.setattr(
        ingestion_worker_module,
        "run_ingestion_worker_once",
        run_once_spy,
    )

    ingestion_worker_module.run_ingestion_worker_loop()

    assert created_client_ids == ["ingestion-worker"]
    assert passed_publishers == [runtime_publisher]
    assert runtime_publisher.closed is True


def test_ingestion_worker_loop_uses_configured_ingestion_consumer(
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
        ingestion_worker_module.settings,
        "kafka_publishing_enabled",
        False,
    )
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "kafka_consuming_enabled",
        True,
    )
    monkeypatch.setattr(
        ingestion_worker_module.settings,
        "kafka_ingestion_consumer_group",
        "agentic-rag-ingestion-test",
    )

    def create_consumer_spy(**kwargs):
        created_consumers.append(kwargs)
        return runtime_consumer

    monkeypatch.setattr(
        ingestion_worker_module,
        "create_kafka_event_consumer",
        create_consumer_spy,
    )
    monkeypatch.setattr(
        ingestion_worker_module,
        "run_ingestion_worker_once",
        lambda **kwargs: pytest.fail("DB polling should not run in consume mode"),
    )

    ingestion_worker_module.run_ingestion_worker_loop()

    assert created_consumers[0]["topics"] == [INGESTION_PARSE, RETRY_INGESTION]
    assert created_consumers[0]["group_id"] == "agentic-rag-ingestion-test"
    assert created_consumers[0]["client_id"] == "ingestion-worker-consumer"
    assert callable(created_consumers[0]["handler"])
    assert runtime_consumer.ran is True
    assert runtime_consumer.closed is True
