import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.ingestion import (
    claim_next_ingestion_job,
    mark_ingestion_job_completed,
    mark_ingestion_job_failed,
    mark_ingestion_job_running,
    replace_document_chunks,
    renew_ingestion_job_lease,
    update_document_ingestion_status,
    update_ingestion_job_stage,
)
from agentic_rag.shared.db.models import Document, IngestionJob
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.storage.object_store import ObjectStoreClient


logger = logging.getLogger(__name__)


INGESTION_WORKER_ID = "ingestion-worker"


TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".ndjson",
    ".csv",
    ".tsv",
}

TEXT_MIME_TYPES = {
    "application/json",
    "application/jsonl",
    "application/ndjson",
    "application/x-ndjson",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/tab-separated-values",
}


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    content_hash: str
    token_count: int
    start_offset: int
    end_offset: int
    metadata: dict[str, Any]


def decode_text_document(
    data: bytes,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> str:
    extension = os.path.splitext(file_name or "")[1].lower()
    normalized_mime_type = (mime_type or "").split(";")[0].strip().lower()
    is_text_type = (
        normalized_mime_type.startswith("text/")
        or normalized_mime_type in TEXT_MIME_TYPES
        or extension in TEXT_FILE_EXTENSIONS
    )

    if not is_text_type:
        raise ValueError(
            f"Unsupported ingestion file type: file_name={file_name}, "
            f"mime_type={mime_type}"
        )

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("Uploaded document is not valid UTF-8 text") from e

    if not text.strip():
        raise ValueError("Uploaded document has no extractable text")

    return text


def split_text_into_chunks(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[TextChunk]:
    size = chunk_size or settings.ingestion_chunk_size
    overlap = settings.ingestion_chunk_overlap if chunk_overlap is None else chunk_overlap

    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + size, text_length)
        if end < text_length:
            newline_index = text.rfind("\n", start, end)
            if newline_index > start:
                end = newline_index + 1
            else:
                space_index = text.rfind(" ", start, end)
                if space_index > start:
                    end = space_index + 1

        content = text[start:end].strip()
        if content:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    content=content,
                    content_hash=content_hash,
                    token_count=max(1, len(content.split())),
                    start_offset=start,
                    end_offset=end,
                    metadata={
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "splitter": "character",
                    },
                )
            )
            chunk_index += 1

        if end >= text_length:
            break
        start = max(0, end - overlap)

    if not chunks:
        raise ValueError("Document did not produce any chunks")

    return chunks


def process_ingestion_job(
    db: Session,
    job: IngestionJob,
    object_store: Optional[ObjectStoreClient] = None,
) -> IngestionJob:
    logger.info(f"[IngestionWorker] Processing ingestion job {job.id}")
    object_store = object_store or ObjectStoreClient()
    document = job.document

    try:
        if not document:
            raise ValueError(f"Ingestion job {job.id} has no document")
        if not job.object_key:
            raise ValueError(f"Ingestion job {job.id} has no object key")
        object_key = job.object_key

        if job.status != "running":
            job = mark_ingestion_job_running(
                db=db,
                job=job,
                worker_id=INGESTION_WORKER_ID,
                lease_seconds=settings.ingestion_worker_lease_seconds,
            )
        job = renew_ingestion_job_lease(
            db=db,
            job=job,
            worker_id=INGESTION_WORKER_ID,
            lease_seconds=settings.ingestion_worker_lease_seconds,
        )
        document = update_document_ingestion_status(db, document, "parsing")

        raw_data = object_store.get_bytes(object_key)
        job = renew_ingestion_job_lease(
            db=db,
            job=job,
            worker_id=INGESTION_WORKER_ID,
            lease_seconds=settings.ingestion_worker_lease_seconds,
        )
        text = decode_text_document(
            data=raw_data,
            file_name=document.file_name,
            mime_type=document.mime_type,
        )

        job = update_ingestion_job_stage(db, job, "chunk")
        job = renew_ingestion_job_lease(
            db=db,
            job=job,
            worker_id=INGESTION_WORKER_ID,
            lease_seconds=settings.ingestion_worker_lease_seconds,
        )
        document = update_document_ingestion_status(db, document, "indexing")
        chunks = split_text_into_chunks(text)
        job = renew_ingestion_job_lease(
            db=db,
            job=job,
            worker_id=INGESTION_WORKER_ID,
            lease_seconds=settings.ingestion_worker_lease_seconds,
        )
        chunk_payloads = [
            {
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_hash": chunk.content_hash,
                "token_count": chunk.token_count,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        replace_document_chunks(db, document, chunk_payloads)

        job = mark_ingestion_job_completed(db, job)
        logger.info(
            f"[IngestionWorker] Completed ingestion job {job.id} "
            f"document={document.id} chunks={len(chunk_payloads)}"
        )
        return job

    except HTTPException as e:
        if e.status_code == 409:
            logger.warning(
                f"[IngestionWorker] Lost ingestion job lease {job.id}; "
                "skipping failure update"
            )
            return job

        logger.exception(f"[IngestionWorker] Failed ingestion job {job.id}: {e}")
        mark_ingestion_job_failed(
            db=db,
            job=job,
            error_type=type(e).__name__,
            error_message=str(e.detail),
        )
        if isinstance(document, Document):
            update_document_ingestion_status(db, document, "failed")
        return job

    except Exception as e:
        logger.exception(f"[IngestionWorker] Failed ingestion job {job.id}: {e}")
        mark_ingestion_job_failed(
            db=db,
            job=job,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        if isinstance(document, Document):
            update_document_ingestion_status(db, document, "failed")
        return job


def run_ingestion_worker_once(
    object_store: Optional[ObjectStoreClient] = None,
) -> bool:
    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        job = claim_next_ingestion_job(
            db=db,
            worker_id=INGESTION_WORKER_ID,
            lease_seconds=settings.ingestion_worker_lease_seconds,
        )
        if not job:
            return False

        process_ingestion_job(
            db=db,
            job=job,
            object_store=object_store,
        )
        return True


def run_ingestion_worker_loop() -> None:
    logger.info("[IngestionWorker] Worker loop started")

    while True:
        try:
            processed = run_ingestion_worker_once()
            if not processed:
                time.sleep(settings.ingestion_worker_poll_seconds)
        except KeyboardInterrupt:
            logger.info("[IngestionWorker] Worker loop stopped")
            break
        except Exception as e:
            logger.exception(f"[IngestionWorker] Worker loop error: {e}")
            time.sleep(settings.ingestion_worker_poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )
    run_ingestion_worker_loop()
