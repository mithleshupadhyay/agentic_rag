import logging
from collections.abc import Callable, Iterator, Sequence
from importlib import import_module
from typing import Any, Protocol

from pydantic import ValidationError

from agentic_rag.shared.config import settings
from agentic_rag.shared.kafka.events import EventEnvelope


logger = logging.getLogger(__name__)


class KafkaConsumerMessage(Protocol):
    topic: str
    partition: int
    offset: int
    value: bytes | str | None


class KafkaConsumerClient(Protocol):
    def __iter__(self) -> Iterator[KafkaConsumerMessage]:
        ...

    def commit(self) -> object:
        ...

    def close(self) -> object:
        ...


EventHandler = Callable[[EventEnvelope], None]


class KafkaEventConsumer:
    def __init__(
        self,
        consumer: KafkaConsumerClient,
        handler: EventHandler,
    ) -> None:
        self.consumer = consumer
        self.handler = handler

    def process_message(self, message: KafkaConsumerMessage) -> bool:
        topic = getattr(message, "topic", "")
        partition = getattr(message, "partition", -1)
        offset = getattr(message, "offset", -1)

        raw_value = message.value
        if raw_value is None:
            logger.warning(
                f"[KafkaConsumer] Skipping invalid event topic={topic} "
                f"partition={partition} offset={offset}: empty message value"
            )
            self.consumer.commit()
            return False

        try:
            envelope = EventEnvelope.model_validate_json(raw_value)
        except (TypeError, ValueError, ValidationError) as e:
            logger.warning(
                f"[KafkaConsumer] Skipping invalid event topic={topic} "
                f"partition={partition} offset={offset}: {e}"
            )
            self.consumer.commit()
            return False

        logger.info(
            f"[KafkaConsumer] Handling event topic={topic} "
            f"event_type={envelope.event_type} tenant={envelope.tenant_id} "
            f"correlation_id={envelope.correlation_id}"
        )
        try:
            self.handler(envelope)
        except Exception as e:
            logger.exception(
                f"[KafkaConsumer] Handler failed topic={topic} "
                f"partition={partition} offset={offset}: {e}"
            )
            raise

        self.consumer.commit()
        logger.info(
            f"[KafkaConsumer] Committed event topic={topic} "
            f"partition={partition} offset={offset} event_id={envelope.event_id}"
        )
        return True

    def run_forever(self) -> None:
        logger.info("[KafkaConsumer] Consumer loop started")
        for message in self.consumer:
            self.process_message(message)

    def close(self) -> None:
        self.consumer.close()


def create_kafka_event_consumer(
    topics: Sequence[str],
    group_id: str,
    handler: EventHandler,
    bootstrap_servers: str | None = None,
    client_id: str | None = None,
    auto_offset_reset: str = "earliest",
    consumer_timeout_ms: int | None = None,
) -> KafkaEventConsumer:
    topic_list = [topic.strip() for topic in topics if topic.strip()]
    if not topic_list:
        raise ValueError("Kafka topics must not be empty")
    if not group_id.strip():
        raise ValueError("Kafka consumer group must not be empty")

    configured_bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
    bootstrap_server_list = [
        server.strip()
        for server in configured_bootstrap_servers.split(",")
        if server.strip()
    ]
    if not bootstrap_server_list:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS must not be empty")

    kafka_module = import_module("kafka")
    kafka_consumer_class = getattr(kafka_module, "KafkaConsumer")
    consumer_kwargs: dict[str, Any] = {
        "bootstrap_servers": bootstrap_server_list,
        "group_id": group_id,
        "client_id": client_id or settings.kafka_client_id,
        "enable_auto_commit": False,
        "auto_offset_reset": auto_offset_reset,
    }
    if consumer_timeout_ms is not None:
        consumer_kwargs["consumer_timeout_ms"] = consumer_timeout_ms

    logger.info(
        f"[KafkaConsumer] Creating Kafka consumer topics={','.join(topic_list)} "
        f"group_id={group_id} bootstrap_servers={','.join(bootstrap_server_list)}"
    )
    consumer = kafka_consumer_class(*topic_list, **consumer_kwargs)
    return KafkaEventConsumer(consumer=consumer, handler=handler)
