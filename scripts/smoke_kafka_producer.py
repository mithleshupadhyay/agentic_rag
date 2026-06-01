import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from agentic_rag.shared.config import settings
from agentic_rag.shared.kafka.events import EventEnvelope, EventType, IngestionRetryPayload
from agentic_rag.shared.kafka.producer import create_kafka_event_publisher
from agentic_rag.shared.kafka.topics import INGESTION_PARSE, RETRY_INGESTION


logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )

    smoke_id = uuid4()
    job_id = uuid4()
    now = datetime.now(timezone.utc)
    tenant_id = settings.local_tenant_id
    workspace_id = settings.local_workspace_id or "local-workspace"
    payload = IngestionRetryPayload(
        job_id=job_id,
        document_id=None,
        failed_stage="parse",
        source_topic=INGESTION_PARSE,
        retry_topic=RETRY_INGESTION,
        attempt=1,
        max_attempts=3,
        error_type="KafkaProducerSmoke",
        error_message="Synthetic producer event for local Kafka smoke check.",
        failed_at=now,
        next_retry_at=now,
        metadata={"smoke": True, "smoke_id": str(smoke_id)},
    )
    envelope = EventEnvelope(
        event_type=EventType.INGESTION_RETRY_SCHEDULED,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        correlation_id=str(job_id),
        idempotency_key=f"kafka-producer-smoke:{job_id}:1",
        payload=payload.model_dump(mode="json"),
    )

    logger.info(
        f"[KafkaSmoke] Publishing producer smoke event topic={RETRY_INGESTION} "
        f"tenant={tenant_id} correlation_id={job_id}"
    )
    publisher = create_kafka_event_publisher(
        client_id=f"agentic-rag-smoke-producer-{smoke_id}",
    )
    try:
        publisher(RETRY_INGESTION, envelope)
    finally:
        publisher.close()

    logger.info(f"[KafkaSmoke] Kafka producer smoke ok event_id={envelope.event_id}")
    print("kafka producer smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
