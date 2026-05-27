import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from agentic_rag.shared.db.models import ChunkAcl, Document, DocumentChunk, IngestionJob


logger = logging.getLogger(__name__)


def claim_next_ingestion_job(
    db: Session,
    worker_id: str,
    lease_seconds: int,
) -> Optional[IngestionJob]:
    logger.info(f"[DB] Claiming next ingestion job worker={worker_id}")
    now = datetime.now(timezone.utc)
    lease_expires_at = now + timedelta(seconds=lease_seconds)
    query = (
        db.query(IngestionJob)
        .options(
            selectinload(IngestionJob.document).selectinload(Document.acl),
        )
        .filter(
            or_(
                IngestionJob.status == "queued",
                and_(
                    IngestionJob.status == "running",
                    IngestionJob.lease_expires_at.is_not(None),
                    IngestionJob.lease_expires_at <= now,
                ),
                and_(
                    IngestionJob.status == "failed",
                    IngestionJob.retry_count < IngestionJob.max_retries,
                    or_(
                        IngestionJob.next_retry_at.is_(None),
                        IngestionJob.next_retry_at <= now,
                    ),
                ),
            )
        )
        .order_by(IngestionJob.created_at.asc())
    )

    bind = db.get_bind()
    dialect_name = bind.dialect.name if bind else ""
    if dialect_name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    job = query.first()
    if not job:
        logger.info("[DB] No claimable ingestion job found")
        return None

    logger.info(f"[DB] Claimed ingestion job {job.id} worker={worker_id}")
    job.status = "running"
    job.current_stage = "parse"
    job.error_type = None
    job.error_message = None
    job.locked_by = worker_id
    job.locked_at = now
    job.lease_expires_at = lease_expires_at
    job.next_retry_at = None
    job.started_at = now

    try:
        db.commit()
        db.refresh(job)
        _ = job.document
        logger.info(
            f"[DB] Ingestion job {job.id} lease_expires_at={lease_expires_at}"
        )
        return job

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to claim ingestion job {job.id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during ingestion job claim.",
        )


def get_next_queued_ingestion_job(db: Session) -> Optional[IngestionJob]:
    logger.info("[DB] Fetching next queued ingestion job")
    query = (
        db.query(IngestionJob)
        .options(
            selectinload(IngestionJob.document).selectinload(Document.acl),
        )
        .filter(IngestionJob.status == "queued")
        .order_by(IngestionJob.created_at.asc())
    )

    bind = db.get_bind()
    dialect_name = bind.dialect.name if bind else ""
    if dialect_name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    job = query.first()
    if job:
        logger.info(f"[DB] Found queued ingestion job {job.id}")
    else:
        logger.info("[DB] No queued ingestion job found")
    return job


def mark_ingestion_job_running(
    db: Session,
    job: IngestionJob,
    worker_id: str | None = None,
    lease_seconds: int | None = None,
) -> IngestionJob:
    logger.info(f"[DB] Marking ingestion job {job.id} as running")
    now = datetime.now(timezone.utc)
    job.status = "running"
    job.current_stage = "parse"
    job.error_type = None
    job.error_message = None
    job.started_at = now
    job.next_retry_at = None
    if worker_id and lease_seconds:
        job.locked_by = worker_id
        job.locked_at = now
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)

    try:
        db.commit()
        db.refresh(job)
        logger.info(f"[DB] Ingestion job {job.id} is running")
        return job

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to mark ingestion job {job.id} running: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during ingestion job status update.",
        )


def renew_ingestion_job_lease(
    db: Session,
    job: IngestionJob,
    worker_id: str,
    lease_seconds: int,
) -> IngestionJob:
    logger.info(f"[DB] Renewing ingestion job {job.id} lease worker={worker_id}")
    query = db.query(IngestionJob).populate_existing().filter(IngestionJob.id == job.id)

    bind = db.get_bind()
    dialect_name = bind.dialect.name if bind else ""
    if dialect_name == "postgresql":
        query = query.with_for_update()

    db_job = query.first()
    if not db_job:
        logger.warning(f"[DB] Cannot renew missing ingestion job {job.id}")
        raise HTTPException(
            status_code=404,
            detail="Ingestion job not found.",
        )
    if db_job.status != "running" or db_job.locked_by != worker_id:
        logger.warning(
            f"[DB] Cannot renew ingestion job {job.id}; "
            f"status={db_job.status} locked_by={db_job.locked_by}"
        )
        raise HTTPException(
            status_code=409,
            detail="Ingestion job lease is no longer owned by this worker.",
        )

    now = datetime.now(timezone.utc)
    db_job.locked_at = now
    db_job.lease_expires_at = now + timedelta(seconds=lease_seconds)

    try:
        db.commit()
        db.refresh(db_job)
        logger.info(
            f"[DB] Renewed ingestion job {db_job.id} "
            f"lease_expires_at={db_job.lease_expires_at}"
        )
        return db_job

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to renew ingestion job {job.id} lease: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during ingestion job lease renewal.",
        )


def update_ingestion_job_stage(
    db: Session,
    job: IngestionJob,
    stage: str,
) -> IngestionJob:
    logger.info(f"[DB] Updating ingestion job {job.id} stage={stage}")
    job.current_stage = stage

    try:
        db.commit()
        db.refresh(job)
        logger.info(f"[DB] Updated ingestion job {job.id} stage={stage}")
        return job

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to update ingestion job {job.id} stage: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during ingestion job stage update.",
        )


def mark_ingestion_job_completed(db: Session, job: IngestionJob) -> IngestionJob:
    logger.info(f"[DB] Marking ingestion job {job.id} completed")
    job.status = "completed"
    job.current_stage = "complete"
    job.completed_at = datetime.now(timezone.utc)
    job.error_type = None
    job.error_message = None
    job.locked_by = None
    job.locked_at = None
    job.lease_expires_at = None
    job.next_retry_at = None

    try:
        db.commit()
        db.refresh(job)
        logger.info(f"[DB] Ingestion job {job.id} completed")
        return job

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to complete ingestion job {job.id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during ingestion job completion.",
        )


def mark_ingestion_job_failed(
    db: Session,
    job: IngestionJob,
    error_type: str,
    error_message: str,
) -> IngestionJob:
    logger.warning(f"[DB] Marking ingestion job {job.id} failed: {error_message}")
    job.status = "failed"
    job.error_type = error_type[:128]
    job.error_message = error_message
    job.retry_count = (job.retry_count or 0) + 1
    job.completed_at = datetime.now(timezone.utc)
    job.locked_by = None
    job.locked_at = None
    job.lease_expires_at = None
    job.next_retry_at = (
        datetime.now(timezone.utc) if job.retry_count < job.max_retries else None
    )

    try:
        db.commit()
        db.refresh(job)
        logger.info(f"[DB] Ingestion job {job.id} failed")
        return job

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to mark ingestion job {job.id} failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during ingestion job failure update.",
        )


def update_document_ingestion_status(
    db: Session,
    document: Document,
    status: str,
) -> Document:
    logger.info(f"[DB] Updating document {document.id} ingestion status={status}")
    document.status = status

    try:
        db.commit()
        db.refresh(document)
        _ = document.acl
        logger.info(f"[DB] Updated document {document.id} ingestion status={status}")
        return document

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to update document {document.id} status: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during document status update.",
        )


def replace_document_chunks(
    db: Session,
    document: Document,
    chunks: list[dict],
) -> list[DocumentChunk]:
    logger.info(f"[DB] Replacing chunks for document {document.id}")

    try:
        existing_chunks = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.tenant_id == document.tenant_id,
                DocumentChunk.document_id == document.id,
            )
            .all()
        )
        for chunk in existing_chunks:
            db.delete(chunk)

        created_chunks = []
        for chunk_payload in chunks:
            db_chunk = DocumentChunk(
                tenant_id=document.tenant_id,
                workspace_id=document.workspace_id,
                document_id=document.id,
                chunk_index=chunk_payload["chunk_index"],
                content=chunk_payload["content"],
                content_hash=chunk_payload["content_hash"],
                token_count=chunk_payload["token_count"],
                section_path=chunk_payload.get("section_path"),
                page_number=chunk_payload.get("page_number"),
                start_offset=chunk_payload.get("start_offset"),
                end_offset=chunk_payload.get("end_offset"),
                metadata_=chunk_payload.get("metadata", {}),
                acl_version=document.acl_version,
                classification_level=document.classification_level,
            )
            if document.acl:
                db_chunk.acl = ChunkAcl(
                    tenant_id=document.tenant_id,
                    chunk=db_chunk,
                    visibility=document.acl.visibility,
                    allowed_user_ids=document.acl.allowed_user_ids,
                    allowed_group_ids=document.acl.allowed_group_ids,
                    allowed_roles=document.acl.allowed_roles,
                    denied_user_ids=document.acl.denied_user_ids,
                    denied_group_ids=document.acl.denied_group_ids,
                    acl_version=document.acl.acl_version,
                )
            db.add(db_chunk)
            created_chunks.append(db_chunk)

        document.status = "ready"
        db.commit()
        for db_chunk in created_chunks:
            db.refresh(db_chunk)
            _ = db_chunk.acl
        db.refresh(document)
        logger.info(
            f"[DB] Replaced chunks for document {document.id}, "
            f"count={len(created_chunks)}"
        )
        return created_chunks

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to replace chunks for document {document.id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during document chunk storage.",
        )
