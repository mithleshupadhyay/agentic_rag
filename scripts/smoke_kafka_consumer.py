import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.documents import attach_document_object, create_document
from agentic_rag.shared.db.models import DocumentChunk, IngestionJob, Tenant
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.shared.kafka.consumer import create_kafka_event_consumer
from agentic_rag.shared.kafka.events import EventEnvelope, EventType, IngestionRetryPayload
from agentic_rag.shared.kafka.producer import create_kafka_event_publisher
from agentic_rag.shared.kafka.topics import INGESTION_PARSE, RETRY_INGESTION
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSourceType,
    FileMetadata,
)
from agentic_rag.storage.object_store import ObjectStoreClient
from agentic_rag.workers.ingestion import handle_ingestion_retry_event


logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )

    if not settings.kafka_consuming_enabled:
        logger.error(
            "[KafkaSmoke] KAFKA_CONSUMING_ENABLED must be true for this smoke check"
        )
        return 1

    tenant_id = settings.local_tenant_id
    workspace_id = settings.local_workspace_id or "local-workspace"
    user_id = settings.local_user_id
    smoke_id = uuid4()
    file_name = f"kafka-consumer-smoke-{smoke_id}.txt"
    file_bytes = (
        b"# Kafka consumer smoke\n\n"
        b"This local document verifies the retry.ingestion consumer path.\n"
        b"It is safe to delete after the smoke check completes.\n"
    )
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    SessionLocal = get_sync_session_factory()

    logger.info(f"[KafkaSmoke] Preparing retryable ingestion job tenant={tenant_id}")
    with SessionLocal() as db:
        tenant = db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
        if tenant is None:
            tenant = Tenant(
                tenant_id=tenant_id,
                name="Local Tenant",
                slug=tenant_id,
                status="active",
                metadata_={"created_by": "kafka_consumer_smoke"},
            )
            db.add(tenant)
            db.commit()

        user_context = UserContext(
            id=user_id,
            customer_id=tenant_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            roles=settings.local_roles,
            group_ids=settings.local_groups,
            scopes=settings.local_scopes,
            acl_version=settings.local_acl_version,
        )
        document = create_document(
            user_context=user_context,
            db=db,
            obj_in=DocumentCreateRequest(
                workspace_id=workspace_id,
                source_type=DocumentSourceType.UPLOAD,
                source_uri=f"upload://{file_name}",
                title="Kafka consumer smoke",
                file=FileMetadata(
                    file_name=file_name,
                    mime_type="text/plain",
                    byte_size=len(file_bytes),
                    content_hash=content_hash,
                ),
                metadata={"smoke": True, "smoke_id": str(smoke_id)},
                acl=AclPolicy(
                    visibility=Visibility.PRIVATE,
                    allowed_user_ids=[user_id],
                    acl_version=settings.local_acl_version,
                ),
            ),
        )

        object_store = ObjectStoreClient()
        object_key = object_store.build_object_key(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            document_id=document.id,
            file_name=file_name,
        )
        object_store.put_bytes(
            object_key=object_key,
            data=file_bytes,
            content_type="text/plain",
            metadata={"smoke_id": str(smoke_id)},
        )
        document = attach_document_object(
            db=db,
            db_obj=document,
            object_key=object_key,
        )

        now = datetime.now(timezone.utc)
        job = IngestionJob(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            document_id=document.id,
            source_type=document.source_type,
            source_uri=document.source_uri,
            object_key=document.object_key,
            status="failed",
            current_stage="parse",
            retry_count=1,
            max_retries=3,
            error_type="KafkaConsumerSmoke",
            error_message="Synthetic retry event for local Kafka consumer smoke check.",
            next_retry_at=now - timedelta(seconds=1),
            idempotency_key=f"kafka-consumer-smoke:{smoke_id}",
            metadata_={"smoke": True, "smoke_id": str(smoke_id)},
            created_by=user_id,
            completed_at=now,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
        document_id = document.id

    handled_events: list[str] = []
    claim_results: list[bool] = []

    def handle_smoke_event(envelope: EventEnvelope) -> None:
        if envelope.correlation_id != str(job_id):
            logger.info(
                f"[KafkaSmoke] Skipping unrelated retry event "
                f"correlation_id={envelope.correlation_id}"
            )
            return

        handled_events.append(str(envelope.event_id))
        handled = handle_ingestion_retry_event(envelope=envelope)
        claim_results.append(handled)

    group_id = f"agentic-rag-smoke-retry-{smoke_id}"
    consumer = create_kafka_event_consumer(
        topics=[RETRY_INGESTION],
        group_id=group_id,
        handler=handle_smoke_event,
        client_id=f"agentic-rag-smoke-consumer-{smoke_id}",
        auto_offset_reset="earliest",
        consumer_timeout_ms=15000,
    )
    publisher = create_kafka_event_publisher(
        client_id=f"agentic-rag-smoke-producer-{smoke_id}",
    )

    try:
        payload = IngestionRetryPayload(
            job_id=job_id,
            document_id=document_id,
            failed_stage="parse",
            source_topic=INGESTION_PARSE,
            retry_topic=RETRY_INGESTION,
            attempt=1,
            max_attempts=3,
            error_type="KafkaConsumerSmoke",
            error_message="Synthetic retry event for local Kafka consumer smoke check.",
            failed_at=datetime.now(timezone.utc),
            next_retry_at=datetime.now(timezone.utc),
            metadata={"smoke": True, "smoke_id": str(smoke_id)},
        )
        envelope = EventEnvelope(
            event_type=EventType.INGESTION_RETRY_SCHEDULED,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            correlation_id=str(job_id),
            idempotency_key=f"kafka-consumer-smoke:{job_id}:1",
            payload=payload.model_dump(mode="json"),
        )
        publisher(RETRY_INGESTION, envelope)
        consumer.run_forever()
    finally:
        publisher.close()
        consumer.close()

    if not handled_events or not any(claim_results):
        logger.error(f"[KafkaSmoke] Retry event was not handled job={job_id}")
        return 1

    with SessionLocal() as db:
        stored_job = db.get(IngestionJob, job_id)
        stored_chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        if stored_job is None:
            logger.error(f"[KafkaSmoke] Ingestion job disappeared job={job_id}")
            return 1
        if stored_job.status != "completed":
            logger.error(
                f"[KafkaSmoke] Ingestion job did not complete "
                f"job={job_id} status={stored_job.status}"
            )
            return 1
        if not stored_chunks:
            logger.error(f"[KafkaSmoke] No chunks created for document={document_id}")
            return 1

    logger.info(
        f"[KafkaSmoke] Kafka consumer smoke ok job={job_id} "
        f"events={len(handled_events)} chunks={len(stored_chunks)}"
    )
    print("kafka consumer smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
