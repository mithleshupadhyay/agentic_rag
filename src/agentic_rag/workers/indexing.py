import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from typing import Optional, cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.orm import Session

from agentic_rag.monitoring.metrics import (
    WORKER_ITEM_TOTAL,
    WORKER_JOB_LATENCY_SECONDS,
    WORKER_JOB_LIFECYCLE_TOTAL,
    WORKER_QUEUE_LAG_SECONDS,
)
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.indexing import (
    list_chunks_pending_bm25_index,
    mark_chunk_bm25_failed,
    mark_chunk_bm25_indexed,
)
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.shared.kafka.consumer import create_kafka_event_consumer
from agentic_rag.shared.kafka.events import EventEnvelope, EventType, IndexChunksPayload
from agentic_rag.shared.kafka.topics import INGESTION_INDEX
from agentic_rag.search.opensearch import OpenSearchClient


logger = logging.getLogger(__name__)


INDEXING_CONSUMER_ID = "indexing-worker-consumer"
INDEXING_WORKER_ID = "indexing-worker"
INDEXING_JOB_TYPE = "bm25_indexing"
INDEXING_METRIC_SOURCES = {"direct", "db_poll", "index_event"}


def process_bm25_index_batch(
    db: Session,
    search_client: Optional[OpenSearchClient] = None,
    limit: Optional[int] = None,
    tenant_id: Optional[str] = None,
    department_id: Optional[UUID] = None,
    document_id: Optional[UUID] = None,
    chunk_ids: Optional[list[UUID]] = None,
    source: str = "direct",
) -> int:
    logger.info(
        f"[IndexingWorker] Processing BM25 index batch tenant={tenant_id} "
        f"document={document_id} chunk_count={len(chunk_ids or [])}"
    )
    started_at = time.perf_counter()
    metric_source = source if source in INDEXING_METRIC_SOURCES else "direct"
    search_client = search_client or OpenSearchClient()
    index_name = search_client.index_name

    chunks = list_chunks_pending_bm25_index(
        db=db,
        limit=limit or settings.bm25_index_batch_size,
        index_name=index_name,
        tenant_id=tenant_id,
        department_id=department_id,
        document_id=document_id,
        chunk_ids=chunk_ids,
    )
    if not chunks:
        logger.info("[IndexingWorker] No chunks pending BM25 index")
        return 0

    WORKER_JOB_LIFECYCLE_TOTAL.labels(
        worker=INDEXING_WORKER_ID,
        job_type=INDEXING_JOB_TYPE,
        status="started",
    ).inc()
    queued_at = min(
        (
            chunk.updated_at or chunk.created_at
            for chunk in chunks
            if chunk.updated_at or chunk.created_at
        ),
        default=None,
    )
    if queued_at:
        if queued_at.tzinfo is None:
            queued_at = queued_at.replace(tzinfo=timezone.utc)
        queue_lag_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - queued_at).total_seconds(),
        )
        WORKER_QUEUE_LAG_SECONDS.labels(
            worker=INDEXING_WORKER_ID,
            job_type=INDEXING_JOB_TYPE,
            source=metric_source,
        ).observe(queue_lag_seconds)
    WORKER_ITEM_TOTAL.labels(
        worker=INDEXING_WORKER_ID,
        job_type=INDEXING_JOB_TYPE,
        item_type="chunk",
        status="selected",
    ).inc(len(chunks))

    try:
        search_client.ensure_chunk_index(index_name)
        indexed_count = search_client.bulk_index_chunks(chunks)
        for chunk in chunks:
            mark_chunk_bm25_indexed(
                db=db,
                chunk=chunk,
                index_name=index_name,
            )

        latency_seconds = time.perf_counter() - started_at
        WORKER_JOB_LIFECYCLE_TOTAL.labels(
            worker=INDEXING_WORKER_ID,
            job_type=INDEXING_JOB_TYPE,
            status="completed",
        ).inc()
        WORKER_JOB_LATENCY_SECONDS.labels(
            worker=INDEXING_WORKER_ID,
            job_type=INDEXING_JOB_TYPE,
            status="completed",
        ).observe(latency_seconds)
        WORKER_ITEM_TOTAL.labels(
            worker=INDEXING_WORKER_ID,
            job_type=INDEXING_JOB_TYPE,
            item_type="chunk",
            status="indexed",
        ).inc(indexed_count)
        logger.info(f"[IndexingWorker] Indexed {indexed_count} chunks into BM25")
        return indexed_count

    except Exception as e:
        logger.exception(f"[IndexingWorker] BM25 indexing batch failed: {e}")
        for chunk in chunks:
            mark_chunk_bm25_failed(
                db=db,
                chunk=chunk,
                error_message=str(e),
            )
        latency_seconds = time.perf_counter() - started_at
        WORKER_JOB_LIFECYCLE_TOTAL.labels(
            worker=INDEXING_WORKER_ID,
            job_type=INDEXING_JOB_TYPE,
            status="failed",
        ).inc()
        WORKER_JOB_LATENCY_SECONDS.labels(
            worker=INDEXING_WORKER_ID,
            job_type=INDEXING_JOB_TYPE,
            status="failed",
        ).observe(latency_seconds)
        WORKER_ITEM_TOTAL.labels(
            worker=INDEXING_WORKER_ID,
            job_type=INDEXING_JOB_TYPE,
            item_type="chunk",
            status="failed",
        ).inc(len(chunks))
        return 0


def run_indexing_worker_once(
    search_client: Optional[OpenSearchClient] = None,
) -> bool:
    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        indexed_count = process_bm25_index_batch(
            db=db,
            search_client=search_client,
            source="db_poll",
        )
        return indexed_count > 0


def handle_indexing_event(
    envelope: EventEnvelope,
    search_client: Optional[OpenSearchClient] = None,
) -> bool:
    if envelope.event_type != EventType.DOCUMENT_INDEX_REQUESTED:
        logger.warning(
            f"[IndexingWorker] Skipping non-indexing event "
            f"event_type={envelope.event_type} event_id={envelope.event_id}"
        )
        return False

    try:
        payload = IndexChunksPayload.model_validate(envelope.payload)
    except ValidationError as e:
        logger.warning(
            f"[IndexingWorker] Skipping invalid indexing payload "
            f"event_id={envelope.event_id}: {e}"
        )
        return False

    active_search_client = search_client or OpenSearchClient()
    if payload.index_name != active_search_client.index_name:
        logger.warning(
            f"[IndexingWorker] Skipping indexing event for unexpected index "
            f"event_id={envelope.event_id} index={payload.index_name}"
        )
        return False

    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        indexed_count = process_bm25_index_batch(
            db=db,
            search_client=active_search_client,
            limit=len(payload.chunk_ids),
            tenant_id=envelope.tenant_id,
            department_id=envelope.department_id,
            document_id=payload.document_id,
            chunk_ids=payload.chunk_ids,
            source="index_event",
        )
        logger.info(
            f"[IndexingWorker] Handled indexing event event_id={envelope.event_id} "
            f"indexed_count={indexed_count}"
        )
        return True


def run_indexing_worker_loop(search_client: Optional[OpenSearchClient] = None) -> None:
    logger.info("[IndexingWorker] Worker loop started")
    indexing_consumer = None

    try:
        if settings.kafka_consuming_enabled:
            logger.info("[IndexingWorker] Kafka indexing consumer enabled")
            indexing_consumer = create_kafka_event_consumer(
                topics=[INGESTION_INDEX],
                group_id=settings.kafka_indexing_consumer_group,
                handler=cast(
                    Callable[[EventEnvelope], None],
                    partial(
                        handle_indexing_event,
                        search_client=search_client,
                    ),
                ),
                client_id=INDEXING_CONSUMER_ID,
            )
            try:
                indexing_consumer.run_forever()
            except KeyboardInterrupt:
                logger.info("[IndexingWorker] Kafka indexing consumer stopped")
            return

        while True:
            try:
                processed = run_indexing_worker_once(search_client=search_client)
                if not processed:
                    time.sleep(settings.indexing_worker_poll_seconds)
            except KeyboardInterrupt:
                logger.info("[IndexingWorker] Worker loop stopped")
                break
            except Exception as e:
                logger.exception(f"[IndexingWorker] Worker loop error: {e}")
                time.sleep(settings.indexing_worker_poll_seconds)
    finally:
        if indexing_consumer is not None:
            indexing_consumer.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )
    run_indexing_worker_loop()
