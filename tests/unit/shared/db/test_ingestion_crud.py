from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

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
    claim_next_ingestion_job,
    get_next_queued_ingestion_job,
    mark_ingestion_job_completed,
    mark_ingestion_job_failed,
    mark_ingestion_job_running,
    replace_document_chunks,
    update_ingestion_job_stage,
)
from agentic_rag.shared.db.models import ChunkAcl, DocumentChunk, Tenant
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSourceType,
    FileMetadata,
)


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


def create_uploaded_document(db: Session):
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
            file=FileMetadata(
                file_name="policy.txt",
                mime_type="text/plain",
                byte_size=50,
                content_hash="document-hash",
            ),
            metadata={},
            acl=AclPolicy(
                visibility=Visibility.GROUP,
                allowed_group_ids=["security"],
                acl_version=2,
            ),
        ),
    )
    document = attach_document_object(
        db=db,
        db_obj=document,
        object_key="tenants/tenant-a/workspaces/workspace-a/documents/document/raw/policy.txt",
    )
    job = create_ingestion_job_for_document(
        user_context=user_context,
        db=db,
        document=document,
    )
    return user_context, document, job


def create_uploaded_document_for_user(
    db: Session,
    user_context: UserContext,
    title: str,
):
    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=DocumentCreateRequest(
            workspace_id="workspace-a",
            source_type=DocumentSourceType.UPLOAD,
            source_uri=f"upload://{title}.txt",
            title=title,
            file=FileMetadata(
                file_name=f"{title}.txt",
                mime_type="text/plain",
                byte_size=50,
                content_hash=f"{title}-hash",
            ),
            metadata={},
            acl=AclPolicy(
                visibility=Visibility.GROUP,
                allowed_group_ids=["security"],
                acl_version=2,
            ),
        ),
    )
    document = attach_document_object(
        db=db,
        db_obj=document,
        object_key=(
            "tenants/tenant-a/workspaces/workspace-a/documents/"
            f"{document.id}/raw/{title}.txt"
        ),
    )
    job = create_ingestion_job_for_document(
        user_context=user_context,
        db=db,
        document=document,
    )
    return document, job


def test_ingestion_job_status_transitions(db: Session) -> None:
    _, _, job = create_uploaded_document(db)

    queued_job = get_next_queued_ingestion_job(db)
    running_job = mark_ingestion_job_running(db, queued_job)
    assert queued_job.id == job.id
    assert running_job.status == "running"

    chunk_job = update_ingestion_job_stage(db, running_job, "chunk")
    assert chunk_job.current_stage == "chunk"

    completed_job = mark_ingestion_job_completed(db, chunk_job)

    assert completed_job.status == "completed"
    assert completed_job.current_stage == "complete"
    assert completed_job.completed_at is not None
    assert completed_job.locked_by is None
    assert completed_job.lease_expires_at is None


def test_claim_next_ingestion_job_claims_oldest_queued_job(db: Session) -> None:
    _, _, job = create_uploaded_document(db)

    claimed_job = claim_next_ingestion_job(
        db=db,
        worker_id="worker-1",
        lease_seconds=300,
    )

    assert claimed_job.id == job.id
    assert claimed_job.status == "running"
    assert claimed_job.current_stage == "parse"
    assert claimed_job.locked_by == "worker-1"
    assert claimed_job.locked_at is not None
    assert claimed_job.lease_expires_at is not None
    assert claimed_job.next_retry_at is None


def test_claim_next_ingestion_job_skips_active_lease(db: Session) -> None:
    user_context, _, first_job = create_uploaded_document(db)
    _, second_job = create_uploaded_document_for_user(
        db=db,
        user_context=user_context,
        title="second-policy",
    )
    first_job.status = "running"
    first_job.locked_by = "worker-1"
    first_job.locked_at = datetime.now(timezone.utc)
    first_job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db.commit()

    claimed_job = claim_next_ingestion_job(
        db=db,
        worker_id="worker-2",
        lease_seconds=300,
    )

    assert claimed_job.id == second_job.id
    assert claimed_job.locked_by == "worker-2"


def test_claim_next_ingestion_job_reclaims_expired_lease(db: Session) -> None:
    _, _, job = create_uploaded_document(db)
    job.status = "running"
    job.locked_by = "worker-1"
    job.locked_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    job.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.commit()

    claimed_job = claim_next_ingestion_job(
        db=db,
        worker_id="worker-2",
        lease_seconds=300,
    )

    assert claimed_job.id == job.id
    assert claimed_job.status == "running"
    assert claimed_job.locked_by == "worker-2"
    assert claimed_job.lease_expires_at is not None


def test_claim_next_ingestion_job_retries_failed_job_before_max_retries(
    db: Session,
) -> None:
    _, _, job = create_uploaded_document(db)
    failed_job = mark_ingestion_job_failed(
        db=db,
        job=job,
        error_type="ValueError",
        error_message="Temporary parser failure",
    )

    claimed_job = claim_next_ingestion_job(
        db=db,
        worker_id="worker-1",
        lease_seconds=300,
    )

    assert failed_job.retry_count == 1
    assert claimed_job.id == job.id
    assert claimed_job.status == "running"
    assert claimed_job.error_type is None
    assert claimed_job.next_retry_at is None


def test_claim_next_ingestion_job_skips_failed_job_after_max_retries(
    db: Session,
) -> None:
    _, _, job = create_uploaded_document(db)
    job.retry_count = job.max_retries
    job.status = "failed"
    job.next_retry_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    claimed_job = claim_next_ingestion_job(
        db=db,
        worker_id="worker-1",
        lease_seconds=300,
    )

    assert claimed_job is None


def test_ingestion_job_failure_records_error(db: Session) -> None:
    _, _, job = create_uploaded_document(db)

    failed_job = mark_ingestion_job_failed(
        db=db,
        job=job,
        error_type="ValueError",
        error_message="Unsupported file type",
    )

    assert failed_job.status == "failed"
    assert failed_job.error_type == "ValueError"
    assert failed_job.error_message == "Unsupported file type"
    assert failed_job.retry_count == 1
    assert failed_job.locked_by is None
    assert failed_job.lease_expires_at is None
    assert failed_job.next_retry_at is not None


def test_replace_document_chunks_persists_chunks_and_copies_acl(db: Session) -> None:
    _, document, _ = create_uploaded_document(db)

    chunks = replace_document_chunks(
        db=db,
        document=document,
        chunks=[
            {
                "chunk_index": 0,
                "content": "First policy chunk",
                "content_hash": "hash-1",
                "token_count": 3,
                "start_offset": 0,
                "end_offset": 18,
                "metadata": {"splitter": "character"},
            },
            {
                "chunk_index": 1,
                "content": "Second policy chunk",
                "content_hash": "hash-2",
                "token_count": 3,
                "start_offset": 19,
                "end_offset": 38,
                "metadata": {"splitter": "character"},
            },
        ],
    )

    stored_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )
    chunk_acls = db.query(ChunkAcl).order_by(ChunkAcl.created_at.asc()).all()

    assert [chunk.content for chunk in chunks] == [
        "First policy chunk",
        "Second policy chunk",
    ]
    assert [chunk.chunk_index for chunk in stored_chunks] == [0, 1]
    assert stored_chunks[0].tenant_id == "tenant-a"
    assert stored_chunks[0].workspace_id == "workspace-a"
    assert stored_chunks[0].acl_version == 2
    assert chunk_acls[0].visibility == Visibility.GROUP
    assert chunk_acls[0].allowed_group_ids == ["security"]
