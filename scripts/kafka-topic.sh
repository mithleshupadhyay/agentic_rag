#!/usr/bin/env bash
set -euo pipefail

bootstrap_servers="${KAFKA_BOOTSTRAP_SERVERS:-}"
topic_partitions="${KAFKA_TOPIC_PARTITIONS:-3}"
topic_replication_factor="${KAFKA_TOPIC_REPLICATION_FACTOR:-1}"
kafka_topics_bin="${KAFKA_TOPICS_BIN:-/opt/kafka/bin/kafka-topics.sh}"

if [[ -z "${bootstrap_servers}" ]]; then
  echo "[Kafka] KAFKA_BOOTSTRAP_SERVERS must not be empty" >&2
  exit 1
fi

if [[ ! -x "${kafka_topics_bin}" ]]; then
  echo "[Kafka] kafka-topics command not found: ${kafka_topics_bin}" >&2
  exit 1
fi

if [[ ! "${topic_partitions}" =~ ^[0-9]+$ ]] || [[ "${topic_partitions}" -lt 1 ]]; then
  echo "[Kafka] KAFKA_TOPIC_PARTITIONS must be a positive integer" >&2
  exit 1
fi

if [[ ! "${topic_replication_factor}" =~ ^[0-9]+$ ]] || [[ "${topic_replication_factor}" -lt 1 ]]; then
  echo "[Kafka] KAFKA_TOPIC_REPLICATION_FACTOR must be a positive integer" >&2
  exit 1
fi

topics=(
  "${KAFKA_INGESTION_PARSE_TOPIC:-ingestion.parse}"
  "${KAFKA_INGESTION_METADATA_TOPIC:-ingestion.metadata}"
  "${KAFKA_INGESTION_CHUNK_TOPIC:-ingestion.chunk}"
  "${KAFKA_INGESTION_EMBED_TOPIC:-ingestion.embed}"
  "${KAFKA_INGESTION_INDEX_TOPIC:-ingestion.index}"
  "${KAFKA_RAG_LONG_QUERY_TOPIC:-rag.long_query}"
  "${KAFKA_EVAL_BATCH_TOPIC:-eval.batch}"
  "${KAFKA_RETRY_INGESTION_TOPIC:-retry.ingestion}"
  "${KAFKA_RETRY_EMBEDDING_TOPIC:-retry.embedding}"
  "${KAFKA_RETRY_INDEXING_TOPIC:-retry.indexing}"
  "${KAFKA_DLQ_INGESTION_TOPIC:-dlq.ingestion}"
  "${KAFKA_DLQ_EMBEDDING_TOPIC:-dlq.embedding}"
  "${KAFKA_DLQ_INDEXING_TOPIC:-dlq.indexing}"
  "${KAFKA_DLQ_RAG_TOPIC:-dlq.rag}"
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
