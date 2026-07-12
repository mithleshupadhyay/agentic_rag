import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from io import BytesIO
from typing import Any, Optional

from fastapi import HTTPException
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlalchemy.orm import Session

from agentic_rag.llm.manager import llm_manager
from agentic_rag.monitoring.metrics import (
    WORKER_ITEM_TOTAL,
    WORKER_JOB_LATENCY_SECONDS,
    WORKER_JOB_LIFECYCLE_TOTAL,
    WORKER_QUEUE_LAG_SECONDS,
    WORKER_RETRY_LIFECYCLE_TOTAL,
)
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.ingestion import (
    claim_ingestion_job_by_id,
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
from agentic_rag.shared.schemas.auth import AuthContext, TokenType
from agentic_rag.shared.schemas.llm import EmbeddingRequest
from agentic_rag.shared.kafka.events import (
    EmbedChunksPayload,
    EventEnvelope,
    EventType,
    IndexChunksPayload,
    IngestionDLQPayload,
    IngestionRetryPayload,
    ParseDocumentPayload,
)
from agentic_rag.shared.kafka.consumer import create_kafka_event_consumer
from agentic_rag.shared.kafka.producer import create_kafka_event_publisher
from agentic_rag.shared.kafka.topics import (
    DLQ_INGESTION,
    INGESTION_CHUNK,
    INGESTION_EMBED,
    INGESTION_INDEX,
    INGESTION_METADATA,
    INGESTION_PARSE,
    RETRY_INGESTION,
    TOPIC_TO_DLQ_TOPIC,
    TOPIC_TO_RETRY_TOPIC,
)
from agentic_rag.storage.object_store import ObjectStoreClient


logger = logging.getLogger(__name__)


EventPublisher = Callable[[str, EventEnvelope], None]


INGESTION_WORKER_ID = "ingestion-worker"
INGESTION_CONSUMER_ID = "ingestion-worker-consumer"
INGESTION_JOB_TYPE = "document_ingestion"
INGESTION_METRIC_SOURCES = {"direct", "db_poll", "parse_event", "retry_event"}


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

PDF_FILE_EXTENSIONS = {
    ".pdf",
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

PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
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
    is_pdf_type = (
        normalized_mime_type in PDF_MIME_TYPES or extension in PDF_FILE_EXTENSIONS
    )
    is_text_type = (
        normalized_mime_type.startswith("text/")
        or normalized_mime_type in TEXT_MIME_TYPES
        or extension in TEXT_FILE_EXTENSIONS
    )

    if is_pdf_type:
        try:
            reader = PdfReader(BytesIO(data))
        except (PdfReadError, ValueError, TypeError) as e:
            raise ValueError(
                f"Could not read uploaded PDF: file_name={file_name}, "
                f"mime_type={mime_type}"
            ) from e

        page_texts = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                raise ValueError(
                    f"Could not extract text from uploaded PDF page "
                    f"{page_index}: file_name={file_name}"
                ) from e

            clean_page_text = page_text.strip()
            if clean_page_text:
                page_texts.append(f"Page {page_index}\n{clean_page_text}")

        text = "\n\n".join(page_texts).strip()
        if not text:
            raise ValueError("Uploaded PDF has no extractable text")
        return text

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
    overlap = (
        settings.ingestion_chunk_overlap if chunk_overlap is None else chunk_overlap
    )

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
    event_publisher: EventPublisher | None = None,
    source: str = "direct",
) -> IngestionJob:
    logger.info(f"[IngestionWorker] Processing ingestion job {job.id}")
    started_at = time.perf_counter()
    metric_source = source if source in INGESTION_METRIC_SOURCES else "direct"
    WORKER_JOB_LIFECYCLE_TOTAL.labels(
        worker=INGESTION_WORKER_ID,
        job_type=INGESTION_JOB_TYPE,
        status="started",
    ).inc()
    if job.created_at:
        queued_at = job.created_at
        if queued_at.tzinfo is None:
            queued_at = queued_at.replace(tzinfo=timezone.utc)
        queue_lag_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - queued_at).total_seconds(),
        )
        WORKER_QUEUE_LAG_SECONDS.labels(
            worker=INGESTION_WORKER_ID,
            job_type=INGESTION_JOB_TYPE,
            source=metric_source,
        ).observe(queue_lag_seconds)
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
        stored_chunks = replace_document_chunks(db, document, chunk_payloads)

        if event_publisher and stored_chunks:
            embedding_route = llm_manager.resolve_embedding_provider(
                EmbeddingRequest(
                    auth=AuthContext(
                        user_id=INGESTION_WORKER_ID,
                        tenant_id=job.tenant_id,
                        department_id=job.department_id,
                        workspace_id=job.workspace_id,
                        scopes=["embeddings:write"],
                        token_type=TokenType.SERVICE,
                    ),
                    texts=[stored_chunks[0].content],
                    metadata={"source": INGESTION_WORKER_ID},
                ),
                db=db,
            )
            embed_payload = EmbedChunksPayload(
                job_id=job.id,
                document_id=document.id,
                chunk_ids=[chunk.id for chunk in stored_chunks],
                provider_id=embedding_route.provider_id,
                embedding_model=embedding_route.model,
                embedding_dimension=(
                    embedding_route.embedding_dimension or settings.embedding_dimension
                ),
                vector_version=settings.embedding_vector_version,
            )
            embed_envelope = EventEnvelope(
                event_type=EventType.DOCUMENT_EMBED_REQUESTED,
                tenant_id=job.tenant_id,
                department_id=job.department_id,
                workspace_id=job.workspace_id,
                actor_user_id=job.created_by,
                correlation_id=str(job.id),
                idempotency_key=f"ingestion-embed:{job.id}",
                payload=embed_payload.model_dump(mode="json"),
            )
            try:
                event_publisher(INGESTION_EMBED, embed_envelope)
                logger.info(
                    f"[IngestionWorker] Published embedding event job={job.id} "
                    f"topic={INGESTION_EMBED} chunks={len(stored_chunks)}"
                )
            except Exception as publish_error:
                logger.exception(
                    f"[IngestionWorker] Failed to publish embedding event "
                    f"job={job.id} topic={INGESTION_EMBED}: {publish_error}"
                )

            index_payload = IndexChunksPayload(
                job_id=job.id,
                document_id=document.id,
                chunk_ids=[chunk.id for chunk in stored_chunks],
                index_name=settings.opensearch_chunk_index,
            )
            index_envelope = EventEnvelope(
                event_type=EventType.DOCUMENT_INDEX_REQUESTED,
                tenant_id=job.tenant_id,
                department_id=job.department_id,
                workspace_id=job.workspace_id,
                actor_user_id=job.created_by,
                correlation_id=str(job.id),
                idempotency_key=f"ingestion-index:{job.id}",
                payload=index_payload.model_dump(mode="json"),
            )
            try:
                event_publisher(INGESTION_INDEX, index_envelope)
                logger.info(
                    f"[IngestionWorker] Published BM25 indexing event job={job.id} "
                    f"topic={INGESTION_INDEX} chunks={len(stored_chunks)}"
                )
            except Exception as publish_error:
                logger.exception(
                    f"[IngestionWorker] Failed to publish BM25 indexing event "
                    f"job={job.id} topic={INGESTION_INDEX}: {publish_error}"
                )

        job = mark_ingestion_job_completed(db, job)
        latency_seconds = time.perf_counter() - started_at
        WORKER_JOB_LIFECYCLE_TOTAL.labels(
            worker=INGESTION_WORKER_ID,
            job_type=INGESTION_JOB_TYPE,
            status="completed",
        ).inc()
        WORKER_JOB_LATENCY_SECONDS.labels(
            worker=INGESTION_WORKER_ID,
            job_type=INGESTION_JOB_TYPE,
            status="completed",
        ).observe(latency_seconds)
        WORKER_ITEM_TOTAL.labels(
            worker=INGESTION_WORKER_ID,
            job_type=INGESTION_JOB_TYPE,
            item_type="chunk",
            status="created",
        ).inc(len(stored_chunks))
        logger.info(
            f"[IngestionWorker] Completed ingestion job {job.id} "
            f"document={document.id} chunks={len(chunk_payloads)}"
        )
        return job

    except Exception as e:
        if isinstance(e, HTTPException) and e.status_code == 409:
            logger.warning(
                f"[IngestionWorker] Lost ingestion job lease {job.id}; "
                "skipping failure update"
            )
            latency_seconds = time.perf_counter() - started_at
            WORKER_JOB_LIFECYCLE_TOTAL.labels(
                worker=INGESTION_WORKER_ID,
                job_type=INGESTION_JOB_TYPE,
                status="lease_lost",
            ).inc()
            WORKER_JOB_LATENCY_SECONDS.labels(
                worker=INGESTION_WORKER_ID,
                job_type=INGESTION_JOB_TYPE,
                status="lease_lost",
            ).observe(latency_seconds)
            return job

        logger.exception(f"[IngestionWorker] Failed ingestion job {job.id}: {e}")
        error_message = str(e.detail) if isinstance(e, HTTPException) else str(e)
        job = mark_ingestion_job_failed(
            db=db,
            job=job,
            error_type=type(e).__name__,
            error_message=error_message,
        )
        if isinstance(document, Document):
            update_document_ingestion_status(db, document, "failed")
        latency_seconds = time.perf_counter() - started_at
        WORKER_JOB_LIFECYCLE_TOTAL.labels(
            worker=INGESTION_WORKER_ID,
            job_type=INGESTION_JOB_TYPE,
            status="failed",
        ).inc()
        WORKER_JOB_LATENCY_SECONDS.labels(
            worker=INGESTION_WORKER_ID,
            job_type=INGESTION_JOB_TYPE,
            status="failed",
        ).observe(latency_seconds)

        if event_publisher:
            source_topic = INGESTION_PARSE
            if job.current_stage == "metadata":
                source_topic = INGESTION_METADATA
            elif job.current_stage == "chunk":
                source_topic = INGESTION_CHUNK
            elif job.current_stage in {"embed", "embedding"}:
                source_topic = INGESTION_EMBED
            elif job.current_stage in {"index", "indexing"}:
                source_topic = INGESTION_INDEX

            failed_at = job.completed_at or datetime.now(timezone.utc)
            event_type = EventType.INGESTION_DLQ_RECORDED
            topic = TOPIC_TO_DLQ_TOPIC.get(source_topic, DLQ_INGESTION)
            idempotency_key = f"ingestion-dlq:{job.id}:{job.retry_count}"
            failure_payload: IngestionRetryPayload | IngestionDLQPayload
            if job.next_retry_at is not None:
                retry_topic = TOPIC_TO_RETRY_TOPIC.get(source_topic, RETRY_INGESTION)
                retry_delay_seconds = int(
                    (job.next_retry_at - failed_at).total_seconds()
                )
                failure_payload = IngestionRetryPayload(
                    job_id=job.id,
                    document_id=job.document_id,
                    failed_stage=job.current_stage,
                    source_topic=source_topic,
                    retry_topic=retry_topic,
                    attempt=job.retry_count,
                    max_attempts=job.max_retries,
                    error_type=job.error_type or type(e).__name__,
                    error_message=job.error_message,
                    failed_at=failed_at,
                    next_retry_at=job.next_retry_at,
                    metadata={
                        "worker_id": INGESTION_WORKER_ID,
                        "retry_delay_seconds": retry_delay_seconds,
                    },
                )
                event_type = EventType.INGESTION_RETRY_SCHEDULED
                topic = retry_topic
                idempotency_key = f"ingestion-retry:{job.id}:{job.retry_count}"
            else:
                failure_payload = IngestionDLQPayload(
                    job_id=job.id,
                    document_id=job.document_id,
                    failed_stage=job.current_stage,
                    source_topic=source_topic,
                    dlq_topic=topic,
                    attempt=job.retry_count,
                    max_attempts=job.max_retries,
                    error_type=job.error_type or type(e).__name__,
                    error_message=job.error_message,
                    failed_at=failed_at,
                    terminal_reason="max_retries_exhausted",
                    metadata={"worker_id": INGESTION_WORKER_ID},
                )

            envelope = EventEnvelope(
                event_type=event_type,
                tenant_id=job.tenant_id,
                workspace_id=job.workspace_id,
                correlation_id=str(job.id),
                idempotency_key=idempotency_key,
                payload=failure_payload.model_dump(mode="json"),
            )
            try:
                event_publisher(topic, envelope)
                retry_status = (
                    "retry_scheduled"
                    if event_type == EventType.INGESTION_RETRY_SCHEDULED
                    else "dlq_recorded"
                )
                WORKER_RETRY_LIFECYCLE_TOTAL.labels(
                    worker=INGESTION_WORKER_ID,
                    job_type=INGESTION_JOB_TYPE,
                    status=retry_status,
                    topic=topic,
                ).inc()
            except Exception as publish_error:
                WORKER_RETRY_LIFECYCLE_TOTAL.labels(
                    worker=INGESTION_WORKER_ID,
                    job_type=INGESTION_JOB_TYPE,
                    status="publish_failed",
                    topic=topic,
                ).inc()
                logger.exception(
                    f"[IngestionWorker] Failed to publish ingestion failure event "
                    f"job={job.id} topic={topic}: {publish_error}"
                )

        return job


def run_ingestion_worker_once(
    object_store: Optional[ObjectStoreClient] = None,
    event_publisher: EventPublisher | None = None,
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
            event_publisher=event_publisher,
            source="db_poll",
        )
        return True


def handle_ingestion_parse_event(
    envelope: EventEnvelope,
    object_store: Optional[ObjectStoreClient] = None,
    event_publisher: EventPublisher | None = None,
) -> bool:
    if envelope.event_type != EventType.DOCUMENT_PARSE_REQUESTED:
        logger.warning(
            f"[IngestionWorker] Skipping non-parse event "
            f"event_type={envelope.event_type} event_id={envelope.event_id}"
        )
        return False

    try:
        payload = ParseDocumentPayload.model_validate(envelope.payload)
    except ValidationError as e:
        logger.warning(
            f"[IngestionWorker] Skipping invalid parse payload "
            f"event_id={envelope.event_id}: {e}"
        )
        return False

    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        job = claim_ingestion_job_by_id(
            db=db,
            job_id=payload.job_id,
            tenant_id=envelope.tenant_id,
            worker_id=INGESTION_WORKER_ID,
            lease_seconds=settings.ingestion_worker_lease_seconds,
        )
        if not job:
            logger.info(
                f"[IngestionWorker] Parse event did not claim job "
                f"job={payload.job_id} tenant={envelope.tenant_id}"
            )
            return False

        process_ingestion_job(
            db=db,
            job=job,
            object_store=object_store,
            event_publisher=event_publisher,
            source="parse_event",
        )
        return True


def handle_ingestion_retry_event(
    envelope: EventEnvelope,
    object_store: Optional[ObjectStoreClient] = None,
    event_publisher: EventPublisher | None = None,
) -> bool:
    if envelope.event_type != EventType.INGESTION_RETRY_SCHEDULED:
        logger.warning(
            f"[IngestionWorker] Skipping non-retry event "
            f"event_type={envelope.event_type} event_id={envelope.event_id}"
        )
        return False

    try:
        payload = IngestionRetryPayload.model_validate(envelope.payload)
    except ValidationError as e:
        logger.warning(
            f"[IngestionWorker] Skipping invalid retry payload "
            f"event_id={envelope.event_id}: {e}"
        )
        return False

    if payload.retry_topic != RETRY_INGESTION:
        logger.warning(
            f"[IngestionWorker] Skipping retry event for unexpected topic "
            f"event_id={envelope.event_id} retry_topic={payload.retry_topic}"
        )
        return False

    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        job = claim_ingestion_job_by_id(
            db=db,
            job_id=payload.job_id,
            tenant_id=envelope.tenant_id,
            worker_id=INGESTION_WORKER_ID,
            lease_seconds=settings.ingestion_worker_lease_seconds,
        )
        if not job:
            logger.info(
                f"[IngestionWorker] Retry event did not claim job "
                f"job={payload.job_id} tenant={envelope.tenant_id}"
            )
            return False

        process_ingestion_job(
            db=db,
            job=job,
            object_store=object_store,
            event_publisher=event_publisher,
            source="retry_event",
        )
        return True


def handle_ingestion_consumer_event(
    envelope: EventEnvelope,
    object_store: Optional[ObjectStoreClient] = None,
    event_publisher: EventPublisher | None = None,
) -> None:
    if envelope.event_type == EventType.DOCUMENT_PARSE_REQUESTED:
        handle_ingestion_parse_event(
            envelope=envelope,
            object_store=object_store,
            event_publisher=event_publisher,
        )
        return

    if envelope.event_type == EventType.INGESTION_RETRY_SCHEDULED:
        handle_ingestion_retry_event(
            envelope=envelope,
            object_store=object_store,
            event_publisher=event_publisher,
        )
        return

    logger.warning(
        f"[IngestionWorker] Skipping unsupported ingestion event "
        f"event_type={envelope.event_type} event_id={envelope.event_id}"
    )


def run_ingestion_worker_loop(event_publisher: EventPublisher | None = None) -> None:
    logger.info("[IngestionWorker] Worker loop started")
    configured_event_publisher = event_publisher
    if configured_event_publisher is None and settings.kafka_publishing_enabled:
        logger.info("[IngestionWorker] Kafka publishing enabled")
        configured_event_publisher = create_kafka_event_publisher(
            client_id=INGESTION_WORKER_ID,
        )

    ingestion_consumer = None
    try:
        if settings.kafka_consuming_enabled:
            logger.info("[IngestionWorker] Kafka ingestion consumer enabled")
            ingestion_consumer = create_kafka_event_consumer(
                topics=[INGESTION_PARSE, RETRY_INGESTION],
                group_id=settings.kafka_ingestion_consumer_group,
                handler=partial(
                    handle_ingestion_consumer_event,
                    event_publisher=configured_event_publisher,
                ),
                client_id=INGESTION_CONSUMER_ID,
            )
            try:
                ingestion_consumer.run_forever()
            except KeyboardInterrupt:
                logger.info("[IngestionWorker] Kafka ingestion consumer stopped")
            return

        while True:
            try:
                processed = run_ingestion_worker_once(
                    event_publisher=configured_event_publisher,
                )
                if not processed:
                    time.sleep(settings.ingestion_worker_poll_seconds)
            except KeyboardInterrupt:
                logger.info("[IngestionWorker] Worker loop stopped")
                break
            except Exception as e:
                logger.exception(f"[IngestionWorker] Worker loop error: {e}")
                time.sleep(settings.ingestion_worker_poll_seconds)
    finally:
        close_publisher = getattr(configured_event_publisher, "close", None)
        if callable(close_publisher):
            close_publisher()
        if ingestion_consumer is not None:
            ingestion_consumer.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )
    run_ingestion_worker_loop()
