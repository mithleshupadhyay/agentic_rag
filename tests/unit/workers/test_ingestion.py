from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
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
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSourceType,
    FileMetadata,
)
from agentic_rag.workers.ingestion import (
    decode_text_document,
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
    job.status = "running"
    job.locked_by = "other-worker"
    db.commit()

    process_ingestion_job(
        db=db,
        job=job,
        object_store=FakeObjectStore(b"# Security Policy"),
    )
    stored_job = db.get(IngestionJob, job.id)

    assert stored_job.status == "running"
    assert stored_job.locked_by == "other-worker"
    assert stored_job.error_type is None


def test_process_ingestion_job_marks_failed_for_unsupported_file(db: Session) -> None:
    document, job = create_job(
        db=db,
        file_name="policy.pdf",
        mime_type="application/pdf",
    )
    object_store = FakeObjectStore(b"%PDF-1.7")

    process_ingestion_job(
        db=db,
        job=job,
        object_store=object_store,
    )
    stored_document = db.get(Document, document.id)
    stored_job = db.get(IngestionJob, job.id)

    assert stored_document.status == "failed"
    assert stored_job.status == "failed"
    assert stored_job.error_type == "ValueError"
    assert "Unsupported ingestion file type" in stored_job.error_message
