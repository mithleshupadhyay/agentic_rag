import json
from typing import Any
from uuid import uuid4

import pytest

from agentic_rag.shared.kafka.events import EventEnvelope, EventType
from agentic_rag.shared.kafka.producer import KafkaEventPublisher
from agentic_rag.shared.kafka.topics import RETRY_INGESTION


class FakeProducer:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self.flush_timeouts: list[float | None] = []

    def send(self, topic: str, value: bytes, key: bytes | None = None) -> object:
        self.sent_messages.append(
            {
                "topic": topic,
                "value": value,
                "key": key,
            }
        )
        return object()

    def flush(self, timeout: float | None = None) -> object:
        self.flush_timeouts.append(timeout)
        return object()


def test_kafka_event_publisher_serializes_event_envelope() -> None:
    producer = FakeProducer()
    job_id = uuid4()
    envelope = EventEnvelope(
        event_type=EventType.INGESTION_RETRY_SCHEDULED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id="request-1",
        idempotency_key=f"retry:{job_id}:2",
        payload={
            "job_id": str(job_id),
            "failed_stage": "parse",
            "attempt": 2,
        },
    )
    publisher = KafkaEventPublisher(
        producer=producer,
        flush_timeout_seconds=1.5,
    )

    publisher(RETRY_INGESTION, envelope)

    assert len(producer.sent_messages) == 1
    message = producer.sent_messages[0]
    body = json.loads(message["value"].decode("utf-8"))
    assert message["topic"] == RETRY_INGESTION
    assert message["key"] == f"retry:{job_id}:2".encode("utf-8")
    assert body["event_type"] == EventType.INGESTION_RETRY_SCHEDULED
    assert body["tenant_id"] == "tenant-a"
    assert body["workspace_id"] == "workspace-a"
    assert body["correlation_id"] == "request-1"
    assert body["payload"]["attempt"] == 2
    assert producer.flush_timeouts == [1.5]


def test_kafka_event_publisher_uses_event_id_when_idempotency_key_is_missing() -> None:
    producer = FakeProducer()
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_PARSE_REQUESTED,
        tenant_id="tenant-a",
        correlation_id="request-1",
        payload={"document_id": str(uuid4())},
    )
    publisher = KafkaEventPublisher(
        producer=producer,
        flush_timeout_seconds=2.0,
    )

    publisher.publish(RETRY_INGESTION, envelope)

    assert producer.sent_messages[0]["key"] == str(envelope.event_id).encode("utf-8")
    assert producer.flush_timeouts == [2.0]


def test_kafka_event_publisher_rejects_empty_topic() -> None:
    producer = FakeProducer()
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_PARSE_REQUESTED,
        tenant_id="tenant-a",
        correlation_id="request-1",
        payload={"document_id": str(uuid4())},
    )
    publisher = KafkaEventPublisher(
        producer=producer,
        flush_timeout_seconds=2.0,
    )

    with pytest.raises(ValueError, match="Kafka topic must not be empty"):
        publisher(" ", envelope)

    assert producer.sent_messages == []
    assert producer.flush_timeouts == []
