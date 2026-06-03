import logging
import os
import time
from collections.abc import Callable
from functools import partial
from typing import Optional, cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.orm import Session

from agentic_rag.llm.gateway import generate_embeddings
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.embeddings import (
    bulk_create_chunk_embeddings,
    get_chunks_missing_embedding,
)
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.shared.kafka.consumer import create_kafka_event_consumer
from agentic_rag.shared.kafka.events import EmbedChunksPayload, EventEnvelope, EventType
from agentic_rag.shared.kafka.topics import INGESTION_EMBED
from agentic_rag.shared.schemas.auth import AuthContext, TokenType
from agentic_rag.shared.schemas.chunks import ChunkEmbeddingCreate
from agentic_rag.shared.schemas.llm import EmbeddingRequest, EmbeddingResponse


logger = logging.getLogger(__name__)


EmbeddingClient = Callable[[EmbeddingRequest], EmbeddingResponse]


EMBEDDING_WORKER_ID = "embedding-worker"
EMBEDDING_CONSUMER_ID = "embedding-worker-consumer"


def process_embedding_batch(
    db: Session,
    tenant_id: str,
    embedding_client: Optional[EmbeddingClient] = None,
    limit: Optional[int] = None,
    document_id: Optional[UUID] = None,
    chunk_ids: Optional[list[UUID]] = None,
) -> int:
    logger.info(
        f"[EmbeddingWorker] Processing embedding batch tenant={tenant_id} "
        f"document={document_id} chunk_count={len(chunk_ids or [])}"
    )
    embedding_client = embedding_client or generate_embeddings
    batch_limit = limit or settings.embedding_batch_size

    chunks = get_chunks_missing_embedding(
        db=db,
        tenant_id=tenant_id,
        embedding_model=settings.embedding_model_name,
        vector_version=settings.embedding_vector_version,
        limit=batch_limit,
        document_id=document_id,
        chunk_ids=chunk_ids,
    )
    if not chunks:
        logger.info(f"[EmbeddingWorker] No chunks pending embedding tenant={tenant_id}")
        return 0

    try:
        response = embedding_client(
            EmbeddingRequest(
                auth=AuthContext(
                    user_id="embedding-worker",
                    tenant_id=tenant_id,
                    scopes=["embeddings:write"],
                    token_type=TokenType.SERVICE,
                ),
                texts=[chunk.content for chunk in chunks],
                model=settings.embedding_model_name,
                provider=settings.embedding_provider,
                timeout_seconds=settings.embedding_timeout_seconds,
                metadata={
                    "source": EMBEDDING_WORKER_ID,
                    "chunk_count": len(chunks),
                },
            )
        )
        if len(response.embeddings) != len(chunks):
            raise ValueError(
                "Embedding response count did not match selected chunk count "
                f"({len(response.embeddings)}!={len(chunks)})."
            )
        if response.dimension != settings.embedding_dimension:
            raise ValueError(
                "Embedding response dimension does not match configured dimension "
                f"({response.dimension}!={settings.embedding_dimension})."
            )

        embedding_payloads = []
        for chunk, vector in zip(chunks, response.embeddings):
            embedding_payloads.append(
                ChunkEmbeddingCreate(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    embedding=vector,
                    embedding_model=response.model,
                    embedding_dimension=response.dimension,
                    content_hash=chunk.content_hash,
                    vector_version=settings.embedding_vector_version,
                    metadata={
                        "source": "embedding-worker",
                        "provider": response.provider,
                        "latency_ms": response.latency_ms,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        written_count = bulk_create_chunk_embeddings(
            db=db,
            tenant_id=tenant_id,
            embeddings=embedding_payloads,
        )
        logger.info(
            f"[EmbeddingWorker] Completed embedding batch tenant={tenant_id} "
            f"selected_count={len(chunks)} written_count={written_count}"
        )
        return written_count

    except Exception as e:
        db.rollback()
        logger.exception(
            f"[EmbeddingWorker] Failed embedding batch tenant={tenant_id}: {e}"
        )
        return 0


def process_embedding_batches(
    db: Session,
    tenant_id: Optional[str] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    max_chunks: Optional[int] = None,
) -> int:
    total_written = 0
    max_chunks_per_loop = max_chunks or settings.embedding_worker_max_chunks_per_loop

    if tenant_id:
        return process_embedding_batch(
            db=db,
            tenant_id=tenant_id,
            embedding_client=embedding_client,
            limit=min(settings.embedding_batch_size, max_chunks_per_loop),
        )

    tenants = (
        db.query(Tenant)
        .filter(Tenant.status == "active")
        .order_by(Tenant.tenant_id.asc())
        .all()
    )
    logger.info(f"[EmbeddingWorker] Active tenant count={len(tenants)}")

    for tenant in tenants:
        remaining = max_chunks_per_loop - total_written
        if remaining <= 0:
            break

        written_count = process_embedding_batch(
            db=db,
            tenant_id=tenant.tenant_id,
            embedding_client=embedding_client,
            limit=min(settings.embedding_batch_size, remaining),
        )
        total_written += written_count

    logger.info(f"[EmbeddingWorker] Worker pass wrote embeddings={total_written}")
    return total_written


def run_embedding_worker_once(
    tenant_id: Optional[str] = None,
    embedding_client: Optional[EmbeddingClient] = None,
) -> bool:
    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        written_count = process_embedding_batches(
            db=db,
            tenant_id=tenant_id,
            embedding_client=embedding_client,
        )
        return written_count > 0


def handle_embedding_event(
    envelope: EventEnvelope,
    embedding_client: Optional[EmbeddingClient] = None,
) -> bool:
    if envelope.event_type != EventType.DOCUMENT_EMBED_REQUESTED:
        logger.warning(
            f"[EmbeddingWorker] Skipping non-embedding event "
            f"event_type={envelope.event_type} event_id={envelope.event_id}"
        )
        return False

    try:
        payload = EmbedChunksPayload.model_validate(envelope.payload)
    except ValidationError as e:
        logger.warning(
            f"[EmbeddingWorker] Skipping invalid embedding payload "
            f"event_id={envelope.event_id}: {e}"
        )
        return False

    if payload.embedding_model != settings.embedding_model_name:
        logger.warning(
            f"[EmbeddingWorker] Skipping embedding event for unexpected model "
            f"event_id={envelope.event_id} model={payload.embedding_model}"
        )
        return False
    if payload.vector_version != settings.embedding_vector_version:
        logger.warning(
            f"[EmbeddingWorker] Skipping embedding event for unexpected vector version "
            f"event_id={envelope.event_id} vector_version={payload.vector_version}"
        )
        return False

    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        written_count = process_embedding_batch(
            db=db,
            tenant_id=envelope.tenant_id,
            embedding_client=embedding_client,
            limit=len(payload.chunk_ids),
            document_id=payload.document_id,
            chunk_ids=payload.chunk_ids,
        )
        logger.info(
            f"[EmbeddingWorker] Handled embedding event event_id={envelope.event_id} "
            f"written_count={written_count}"
        )
        return True


def run_embedding_worker_loop(
    tenant_id: Optional[str] = None,
    embedding_client: Optional[EmbeddingClient] = None,
) -> None:
    logger.info("[EmbeddingWorker] Worker loop started")
    embedding_consumer = None

    try:
        if settings.kafka_consuming_enabled:
            logger.info("[EmbeddingWorker] Kafka embedding consumer enabled")
            embedding_consumer = create_kafka_event_consumer(
                topics=[INGESTION_EMBED],
                group_id=settings.kafka_embedding_consumer_group,
                handler=cast(
                    Callable[[EventEnvelope], None],
                    partial(
                        handle_embedding_event,
                        embedding_client=embedding_client,
                    ),
                ),
                client_id=EMBEDDING_CONSUMER_ID,
            )
            try:
                embedding_consumer.run_forever()
            except KeyboardInterrupt:
                logger.info("[EmbeddingWorker] Kafka embedding consumer stopped")
            return

        while True:
            try:
                processed = run_embedding_worker_once(
                    tenant_id=tenant_id,
                    embedding_client=embedding_client,
                )
                if not processed:
                    time.sleep(settings.embedding_worker_poll_seconds)
            except KeyboardInterrupt:
                logger.info("[EmbeddingWorker] Worker loop stopped")
                break
            except Exception as e:
                logger.exception(f"[EmbeddingWorker] Worker loop error: {e}")
                time.sleep(settings.embedding_worker_poll_seconds)
    finally:
        if embedding_consumer is not None:
            embedding_consumer.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )
    run_embedding_worker_loop()
