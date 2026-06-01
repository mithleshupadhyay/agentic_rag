#!/usr/bin/env bash
set -euo pipefail

bootstrap_servers="${EVENT_STREAM_BOOTSTRAP_SERVERS:-}"
topic_partitions="${EVENT_STREAM_TOPIC_PARTITIONS:-3}"
topic_replication_factor="${EVENT_STREAM_TOPIC_REPLICATION_FACTOR:-1}"
kafka_topics_bin="${EVENT_STREAM_TOPICS_BIN:-/opt/kafka/bin/kafka-topics.sh}"

if [[ -z "${bootstrap_servers}" ]]; then
  echo "[Kafka] EVENT_STREAM_BOOTSTRAP_SERVERS must not be empty" >&2
  exit 1
fi

if [[ ! -x "${kafka_topics_bin}" ]]; then
  echo "[Kafka] kafka-topics command not found: ${kafka_topics_bin}" >&2
  exit 1
fi

if [[ ! "${topic_partitions}" =~ ^[0-9]+$ ]] || [[ "${topic_partitions}" -lt 1 ]]; then
  echo "[Kafka] EVENT_STREAM_TOPIC_PARTITIONS must be a positive integer" >&2
  exit 1
fi

if [[ ! "${topic_replication_factor}" =~ ^[0-9]+$ ]] || [[ "${topic_replication_factor}" -lt 1 ]]; then
  echo "[Kafka] EVENT_STREAM_TOPIC_REPLICATION_FACTOR must be a positive integer" >&2
  exit 1
fi

topics=(
  "${EVENT_STREAM_INGESTION_PARSE_TOPIC:-ingestion.parse}"
  "${EVENT_STREAM_INGESTION_METADATA_TOPIC:-ingestion.metadata}"
  "${EVENT_STREAM_INGESTION_CHUNK_TOPIC:-ingestion.chunk}"
  "${EVENT_STREAM_INGESTION_EMBED_TOPIC:-ingestion.embed}"
  "${EVENT_STREAM_INGESTION_INDEX_TOPIC:-ingestion.index}"
  "${EVENT_STREAM_RAG_LONG_QUERY_TOPIC:-rag.long_query}"
  "${EVENT_STREAM_EVAL_BATCH_TOPIC:-eval.batch}"
  "${EVENT_STREAM_RETRY_INGESTION_TOPIC:-retry.ingestion}"
  "${EVENT_STREAM_RETRY_EMBEDDING_TOPIC:-retry.embedding}"
  "${EVENT_STREAM_RETRY_INDEXING_TOPIC:-retry.indexing}"
  "${EVENT_STREAM_DLQ_INGESTION_TOPIC:-dlq.ingestion}"
  "${EVENT_STREAM_DLQ_EMBEDDING_TOPIC:-dlq.embedding}"
  "${EVENT_STREAM_DLQ_INDEXING_TOPIC:-dlq.indexing}"
  "${EVENT_STREAM_DLQ_RAG_TOPIC:-dlq.rag}"
)

existing_topics="$("${kafka_topics_bin}" --bootstrap-server "${bootstrap_servers}" --list)"

for topic in "${topics[@]}"; do
  if [[ -z "${topic}" ]]; then
    echo "[Kafka] Topic name must not be empty" >&2
    exit 1
  fi

  if grep -Fxq -- "${topic}" <<< "${existing_topics}"; then
    echo "[Kafka] Topic already exists: ${topic}"
    continue
  fi

  echo "[Kafka] Creating topic: ${topic}"
  "${kafka_topics_bin}" \
    --bootstrap-server "${bootstrap_servers}" \
    --create \
    --if-not-exists \
    --topic "${topic}" \
    --partitions "${topic_partitions}" \
    --replication-factor "${topic_replication_factor}"
done

echo "[Kafka] Topics ready"
