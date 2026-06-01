from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest

import agentic_rag.shared.kafka.consumer as kafka_consumer_module
from agentic_rag.shared.kafka.consumer import (
    KafkaEventConsumer,
    create_kafka_event_consumer,
)
from agentic_rag.shared.kafka.events import EventEnvelope, EventType
from agentic_rag.shared.kafka.topics import RETRY_INGESTION


@dataclass(frozen=True)
class FakeMessage:
    value: bytes | str | None
    topic: str = RETRY_INGESTION
    partition: int = 0
    offset: int = 12


class FakeConsumer:
    def __init__(self, messages: list[FakeMessage] | None = None) -> None:
        self.messages = messages or []
        self.commits = 0
        self.closed = False

    def __iter__(self) -> Iterator[FakeMessage]:
        return iter(self.messages)

    def commit(self) -> object:
        self.commits += 1
        return object()

    def close(self) -> object:
        self.closed = True
        return object()


def test_kafka_event_consumer_handles_valid_event_and_commits() -> None:
    handled_events: list[EventEnvelope] = []
    envelope = EventEnvelope(
        event_type=EventType.INGESTION_RETRY_SCHEDULED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        correlation_id="request-1",
        idempotency_key=f"retry:{uuid4()}:2",
        payload={"job_id": str(uuid4()), "attempt": 2},
    )
    consumer = FakeConsumer(
        messages=[
            FakeMessage(
                value=envelope.model_dump_json().encode("utf-8"),
                offset=44,
            )
        ]
    )
    event_consumer = KafkaEventConsumer(
        consumer=consumer,
        handler=handled_events.append,
    )

    event_consumer.run_forever()

    assert len(handled_events) == 1
    assert handled_events[0].event_id == envelope.event_id
    assert handled_events[0].event_type == EventType.INGESTION_RETRY_SCHEDULED
    assert handled_events[0].tenant_id == "tenant-a"
    assert consumer.commits == 1


def test_kafka_event_consumer_skips_invalid_json_and_commits(caplog: pytest.LogCaptureFixture) -> None:
    handled_events: list[EventEnvelope] = []
    consumer = FakeConsumer(messages=[FakeMessage(value=b"{not-json")])
    event_consumer = KafkaEventConsumer(
        consumer=consumer,
        handler=handled_events.append,
    )

    processed = event_consumer.process_message(consumer.messages[0])

    assert processed is False
    assert handled_events == []
    assert consumer.commits == 1
    assert "Skipping invalid event" in caplog.text


def test_kafka_event_consumer_skips_invalid_schema_and_commits() -> None:
    handled_events: list[EventEnvelope] = []
    consumer = FakeConsumer(
        messages=[
            FakeMessage(
                value=b'{"event_type":"document.parse_requested","payload":{}}',
            )
        ]
    )
    event_consumer = KafkaEventConsumer(
        consumer=consumer,
        handler=handled_events.append,
    )

    processed = event_consumer.process_message(consumer.messages[0])

    assert processed is False
    assert handled_events == []
    assert consumer.commits == 1


def test_kafka_event_consumer_does_not_commit_when_handler_fails() -> None:
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_PARSE_REQUESTED,
        tenant_id="tenant-a",
        correlation_id="request-1",
        payload={"document_id": str(uuid4())},
    )
    consumer = FakeConsumer(
        messages=[FakeMessage(value=envelope.model_dump_json())],
    )

    def failing_handler(event: EventEnvelope) -> None:
        raise RuntimeError(f"failed event={event.event_id}")

    event_consumer = KafkaEventConsumer(
        consumer=consumer,
        handler=failing_handler,
    )

    with pytest.raises(RuntimeError, match="failed event="):
        event_consumer.process_message(consumer.messages[0])

    assert consumer.commits == 0


def test_kafka_event_consumer_closes_consumer() -> None:
    consumer = FakeConsumer()
    event_consumer = KafkaEventConsumer(
        consumer=consumer,
        handler=lambda envelope: None,
    )

    event_consumer.close()

    assert consumer.closed is True


def test_create_kafka_event_consumer_builds_concrete_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_topics = ()
    created_kwargs = {}

    class FakeKafkaConsumer(FakeConsumer):
        def __init__(self, *topics: str, **kwargs: Any) -> None:
            super().__init__()
            nonlocal created_topics
            created_topics = topics
            created_kwargs.update(kwargs)

    class FakeKafkaModule:
        KafkaConsumer = FakeKafkaConsumer

    monkeypatch.setattr(
        kafka_consumer_module,
        "import_module",
        lambda module_name: FakeKafkaModule if module_name == "kafka" else None,
    )

    event_consumer = create_kafka_event_consumer(
        topics=[RETRY_INGESTION],
        group_id="agentic-rag-ingestion-retry",
        handler=lambda envelope: None,
        bootstrap_servers="kafka-a:9092, kafka-b:9092",
        client_id="retry-consumer",
        consumer_timeout_ms=100,
    )

    assert isinstance(event_consumer, KafkaEventConsumer)
    assert created_topics == (RETRY_INGESTION,)
    assert created_kwargs["bootstrap_servers"] == ["kafka-a:9092", "kafka-b:9092"]
    assert created_kwargs["group_id"] == "agentic-rag-ingestion-retry"
    assert created_kwargs["client_id"] == "retry-consumer"
    assert created_kwargs["enable_auto_commit"] is False
    assert created_kwargs["auto_offset_reset"] == "earliest"
    assert created_kwargs["consumer_timeout_ms"] == 100


def test_create_kafka_event_consumer_requires_topics() -> None:
    with pytest.raises(ValueError, match="Kafka topics must not be empty"):
        create_kafka_event_consumer(
            topics=[" "],
            group_id="group-a",
            handler=lambda envelope: None,
        )


def test_create_kafka_event_consumer_requires_group_id() -> None:
    with pytest.raises(ValueError, match="Kafka consumer group must not be empty"):
        create_kafka_event_consumer(
            topics=[RETRY_INGESTION],
            group_id=" ",
            handler=lambda envelope: None,
        )


def test_create_kafka_event_consumer_requires_bootstrap_servers() -> None:
    with pytest.raises(ValueError, match="KAFKA_BOOTSTRAP_SERVERS must not be empty"):
        create_kafka_event_consumer(
            topics=[RETRY_INGESTION],
            group_id="group-a",
            handler=lambda envelope: None,
            bootstrap_servers=" ",
        )
