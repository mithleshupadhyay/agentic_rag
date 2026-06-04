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

echo "[MonitoringSmoke] Checking Prometheus configuration and alert rules"
"${compose[@]}" exec -T prometheus \
  promtool check config /etc/prometheus/prometheus.yml

echo "[MonitoringSmoke] Checking Prometheus API health"
"${compose[@]}" exec -T api \
  python - <<'PY'
from urllib.request import urlopen

with urlopen("http://prometheus:9090/-/ready", timeout=10) as response:
    body = response.read().decode("utf-8", errors="replace")
    if response.status != 200:
        raise SystemExit(f"Prometheus readiness failed with status {response.status}")
    if "Prometheus Server is Ready" not in body:
        raise SystemExit(f"Prometheus readiness returned unexpected body: {body}")
PY

echo "[MonitoringSmoke] Checking Grafana provisioning files"
"${compose[@]}" exec -T grafana \
  /bin/sh -c '
    test -s /etc/grafana/provisioning/datasources/prometheus.yml
    test -s /etc/grafana/provisioning/dashboards/query.yml
    test -s /var/lib/grafana/dashboards/query_dashboard.json
  '

echo "[MonitoringSmoke] Checking Grafana API health"
"${compose[@]}" exec -T api \
  python - <<'PY'
import json
from urllib.request import urlopen

with urlopen("http://grafana:3000/api/health", timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))
    if response.status != 200:
        raise SystemExit(f"Grafana health failed with status {response.status}")
    if payload.get("database") != "ok":
        raise SystemExit(f"Grafana database health is not ok: {payload}")
PY

echo "monitoring smoke ok"
