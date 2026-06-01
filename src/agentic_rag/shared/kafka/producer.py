import logging
from importlib import import_module
from typing import Protocol

from agentic_rag.shared.config import settings
from agentic_rag.shared.kafka.events import EventEnvelope


logger = logging.getLogger(__name__)


class KafkaProducerClient(Protocol):
    def send(self, topic: str, value: bytes, key: bytes | None = None) -> object:
        ...

    def flush(self, timeout: float | None = None) -> object:
        ...


class KafkaEventPublisher:
    def __init__(
        self,
        producer: KafkaProducerClient,
        flush_timeout_seconds: float | None = None,
    ) -> None:
        self.producer = producer
        self.flush_timeout_seconds = (
            settings.event_stream_producer_flush_timeout_seconds
            if flush_timeout_seconds is None
            else flush_timeout_seconds
        )

    def __call__(self, topic: str, envelope: EventEnvelope) -> None:
        self.publish(topic=topic, envelope=envelope)

    def publish(self, topic: str, envelope: EventEnvelope) -> None:
        if not topic.strip():
            raise ValueError("Kafka topic must not be empty")

        key = envelope.idempotency_key or str(envelope.event_id)
        value = envelope.model_dump_json().encode("utf-8")

        logger.info(
            f"[KafkaProducer] Publishing event topic={topic} "
            f"event_type={envelope.event_type} tenant={envelope.tenant_id} "
            f"correlation_id={envelope.correlation_id}"
        )
        self.producer.send(topic, value, key.encode("utf-8"))
        self.producer.flush(timeout=self.flush_timeout_seconds)
        logger.info(
            f"[KafkaProducer] Published event topic={topic} "
            f"event_id={envelope.event_id} tenant={envelope.tenant_id}"
        )

    def close(self) -> None:
        close_producer = getattr(self.producer, "close", None)
        if callable(close_producer):
            close_producer(timeout=self.flush_timeout_seconds)


def create_kafka_event_publisher(
    bootstrap_servers: str | None = None,
    client_id: str | None = None,
) -> KafkaEventPublisher:
    configured_bootstrap_servers = bootstrap_servers or settings.event_stream_bootstrap_servers
    bootstrap_server_list = [
        server.strip()
        for server in configured_bootstrap_servers.split(",")
        if server.strip()
    ]
    if not bootstrap_server_list:
        raise ValueError("EVENT_STREAM_BOOTSTRAP_SERVERS must not be empty")

    kafka_module = import_module("kafka")
    kafka_producer_class = getattr(kafka_module, "KafkaProducer")
    logger.info(
        f"[KafkaProducer] Creating Kafka producer "
        f"bootstrap_servers={','.join(bootstrap_server_list)}"
    )
    producer = kafka_producer_class(
        bootstrap_servers=bootstrap_server_list,
        client_id=client_id or settings.event_stream_client_id,
        acks="all",
        retries=3,
    )
    return KafkaEventPublisher(producer=producer)
