#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

env_file="${ENV_FILE:-.env}"
docker_compose_file="${DOCKER_COMPOSE_FILE:-docker-compose.yml}"
docker_project="${DOCKER_PROJECT:-agentic-rag}"

if [[ ! -f "${env_file}" ]]; then
  echo "Missing ${env_file}. Run: cp .env.template ${env_file}" >&2
  exit 1
fi

compose=(
  docker compose
  --env-file "${env_file}"
  -f "${docker_compose_file}"
  -p "${docker_project}"
)

"${compose[@]}" exec -T kafka \
  /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:9092 \
  --list >/dev/null

was_running="$("${compose[@]}" ps --services --filter status=running | grep -Fx "ingestion-worker" || true)"

if [[ -n "${was_running}" ]]; then
  "${compose[@]}" stop ingestion-worker >/dev/null
fi

restore_ingestion_worker() {
  if [[ -n "${was_running}" ]]; then
    "${compose[@]}" start ingestion-worker >/dev/null
  fi
}

trap restore_ingestion_worker EXIT

"${compose[@]}" exec -T api \
  env KAFKA_CONSUMING_ENABLED=true \
  python scripts/smoke_kafka_consumer.py
