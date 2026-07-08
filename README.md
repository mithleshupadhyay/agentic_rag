# Agentic RAG

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-dependency%20management-blue.svg)](https://python-poetry.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)

A production-ready, multi-tenant Agentic RAG platform for large datasets, metadata-first retrieval, selective vectorization, document ingestion, authorization-aware search, and grounded query responses.

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- Docker and Docker Compose for the local full stack
- Optional: Ollama or external LLM provider credentials for answer synthesis and embeddings

### Installation

```bash
# Clone the repository
git clone git@github.com:mithleshupadhyay/agentic_rag.git
cd agentic_rag

# Create local configuration
cp .env.template .env

# Install dependencies
poetry install

# Run validation checks
make check

# Build and start the local stack
make docker-up-build

# Check API readiness
curl http://localhost:8100/readiness

# Open the frontend
open http://localhost:5173
```

### Basic Usage

```bash
# Upload a text document
curl -X POST http://localhost:8100/documents/upload \
  -H "Authorization: Bearer local-dev-token" \
  -F "file=@README.md;type=text/plain" \
  -F "workspace_id=local-workspace" \
  -F "title=Agentic RAG README"

# Check document-scoped ingestion jobs
curl http://localhost:8100/documents/{document_id}/ingestion-jobs \
  -H "Authorization: Bearer local-dev-token"

# Run a retrieval-backed query
curl -X POST http://localhost:8100/query \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this document describe?",
    "workspace_id": "local-workspace",
    "filters": {"workspace_id": "local-workspace"},
    "retrieval_limit": 5,
    "max_context_chunks": 3,
    "max_context_tokens": 1000
  }'

# Open interactive API docs
open http://localhost:8100/docs
```

### Frontend Demo

The React frontend gives clients, founders, and hiring managers a browser chat
interface for uploading a PDF or text document, indexing it, asking questions,
and reviewing citations.

Run the full Docker stack:

```bash
make docker-up-build
curl http://localhost:8100/readiness
```

Open:

```text
http://localhost:5173
```

For frontend development:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

More frontend notes are in [frontend/README.md](frontend/README.md).

### Docker Smoke Checks

The local Docker stack includes PostgreSQL with pgvector, Redis, Kafka, MinIO, OpenSearch, API, workers, Prometheus, and Grafana. Use the smoke checks below after `make docker-up-build`.

| Check | Command | What It Validates |
|---|---|---|
| Embedding worker import and DB access | `make docker-smoke-embedding-worker` | Worker container startup path and configured database connectivity |
| pgvector retrieval | `make docker-smoke-vector-retrieval` | Tenant, workspace, ACL, and ordering behavior for vector retrieval |
| OpenSearch BM25 retrieval | `make docker-smoke-bm25-retrieval` | Tenant, workspace, ACL, and BM25 score ordering through OpenSearch |
| Upload to query | `make docker-smoke-upload-query` | API upload, ingestion job status, BM25 indexing, and `/query` retrieval |
| Kafka producer | `make docker-smoke-kafka-producer` | Event envelope serialization and Kafka publishing |
| Kafka consumer | `make docker-smoke-kafka-consumer` | Kafka consuming and ingestion retry handling |
| Monitoring | `make docker-smoke-monitoring` | Prometheus rule loading and Grafana provisioning |

## Features

- Multi-tenant document, chunk, query, and job data model
- FastAPI application with local and OIDC-ready authentication
- Tenant, workspace, owner, role, group, scope, and ACL authorization checks
- Upload API backed by S3-compatible object storage
- DB-backed ingestion job lifecycle with leases, retries, and status APIs
- Text parsing and chunking worker for uploaded documents
- OpenSearch BM25 retrieval with exact metadata, date, source, workspace, tenant, and ACL filters
- PostgreSQL pgvector retrieval with model/version-aware embeddings
- Hybrid retrieval and deterministic reranking services
- Query API with persisted query runs, citations, context, cache metadata, and optional LLM synthesis
- LangGraph-backed agent runtime with checkpoints, cancellation checks, guardrails, and streaming events
- LiteLLM gateway with retries, budget checks, streaming, embeddings, and circuit breaker support
- Prometheus metrics, Grafana dashboard provisioning, and alert rules for query and retrieval behavior
- React frontend for document upload, ingestion polling, document-scoped chat, citations, and context evidence
- Docker Compose stack for local frontend, API, workers, database, object storage, search, queue, cache, and monitoring

## Available Services

| Service | Runtime | Responsibility |
|---|---|---|
| API | `agentic_rag.main:app` | Health, document, retrieval, and query endpoints |
| Frontend | React + Nginx | Browser chat interface for document upload, indexing, query, citations, and context |
| Ingestion worker | `agentic_rag.workers.ingestion` | Claims upload jobs, reads objects, parses text, writes chunks, and schedules downstream work |
| Indexing worker | `agentic_rag.workers.indexing` | Indexes ready chunks into OpenSearch for BM25 retrieval |
| Embedding worker | `agentic_rag.workers.embedding` | Writes missing or stale chunk embeddings into pgvector |
| PostgreSQL | Docker service | Tenant-scoped metadata, documents, chunks, jobs, runs, ACLs, and vectors |
| OpenSearch | Docker service | BM25 chunk search index |
| MinIO | Docker service | Local S3-compatible raw object storage |
| Kafka | Docker service | Local event topics for ingestion, embedding, indexing, retries, and DLQs |
| Redis | Docker service | Cache and shared state backend |
| Prometheus and Grafana | Docker services | Metrics, alerts, and dashboards |

## Runtime States

### Document Status

| Status | Description |
|---|---|
| `queued` | Document was accepted and is waiting for ingestion |
| `parsing` | Ingestion worker is reading and extracting text |
| `indexing` | Chunks have been created and are waiting for retrieval indexes |
| `ready` | Document chunks are stored and available to downstream retrieval workers |
| `failed` | Document ingestion failed |
| `deleted` | Document was soft deleted |

### Ingestion Job Status

| Status | Description |
|---|---|
| `queued` | Job is waiting for a worker |
| `running` | A worker owns the job lease |
| `completed` | Ingestion completed successfully |
| `failed` | Ingestion failed and may retry based on retry settings |
| `cancelled` | Job was cancelled |

## Configuration

### Quick Setup

Create a `.env` file in the project root:

```bash
cp .env.template .env
```

The Docker Compose stack reads `.env` and routes service URLs to Docker service names internally. Host-local defaults in `.env.template` are suitable for local development.

### Key Environment Variables

| Variable | Purpose |
|---|---|
| `AUTH_PROVIDER` | `local`, `keycloak`, `oidc`, or `auth0` |
| `LOCAL_AUTH_TOKEN` | Local development bearer token |
| `LOCAL_TENANT_ID` | Local tenant scope |
| `LOCAL_WORKSPACE_ID` | Optional local workspace scope |
| `DATABASE_URL` | SQLAlchemy async database URL |
| `REDIS_URL` | Redis cache and state URL |
| `KAFKA_PUBLISHING_ENABLED` | Enables Kafka publishing from API or workers |
| `KAFKA_CONSUMING_ENABLED` | Enables Kafka consumer loops |
| `S3_ENDPOINT_URL` | S3-compatible object storage endpoint |
| `S3_BUCKET_NAME` | Raw object bucket |
| `OPENSEARCH_URL` | OpenSearch endpoint |
| `OPENSEARCH_CHUNK_INDEX` | Physical chunk index name |
| `OPENSEARCH_CHUNK_READ_ALIAS` | BM25 search alias |
| `OPENSEARCH_CHUNK_WRITE_ALIAS` | BM25 indexing alias |
| `BM25_MIN_SCORE` | Minimum BM25 candidate score |
| `EMBEDDING_PROVIDER` | Embedding gateway provider |
| `EMBEDDING_MODEL_NAME` | Embedding model name |
| `LLM_SYNTHESIS_ENABLED` | Enables answer generation after retrieval |
| `LLM_PROVIDER` | Chat completion provider |
| `DEFAULT_LLM_MODEL` | Default answer model |
| `OLLAMA_BASE_URL` | Local Ollama endpoint |

### Manual Local API Run

```bash
cp .env.template .env
poetry install
poetry run uvicorn agentic_rag.main:app --reload --host 0.0.0.0 --port 8000
```

Use Docker Compose for the full dependency stack.

## Architecture

```text
Client
  |
  v
FastAPI API
  |-- documents/upload -> object storage -> ingestion_jobs
  |-- documents/{id}/ingestion-jobs -> ingestion status
  |-- retrieval/* -> BM25, vector, hybrid, rerank
  |-- query -> BM25 retrieval -> context -> optional LLM synthesis
  |
  +--> PostgreSQL + pgvector
  +--> Redis
  +--> MinIO
  +--> OpenSearch
  +--> Kafka

Workers
  |-- ingestion -> parse, chunk, schedule embedding/indexing
  |-- indexing -> OpenSearch BM25 index
  |-- embedding -> pgvector embeddings

Observability
  |-- Prometheus metrics
  |-- Grafana dashboards
```

The production retrieval strategy is metadata-first:

```text
store raw data cheaply
-> extract metadata and ACLs
-> search metadata and BM25 first
-> use pgvector only for selected semantic chunks
-> rerank
-> build authorized context
-> generate grounded answers with citations
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Target production architecture and service boundaries
- [Low-Level Design](docs/LOW_LEVEL_DESIGN.md) - Implementation-level data models, API contracts, worker contracts, and test strategy
- [RAG Flow](docs/RAG_FLOW.md) - Upload, ingestion, retrieval, query, and agent flow details
- [Next Sprint](docs/NEXT_SPRINT.md) - Current implementation status and recommended next work
- [Alembic](alembic/README) - Migration folder notes

## Development

### Setup Development Environment

```bash
poetry install
cp .env.template .env
```

### Running Checks

```bash
# Run the full validation suite
make check

# Run unit tests
make test

# Run linting only
make lint

# Run type checking only
make typecheck

# Validate project metadata
make poetry-check
```

### Docker Development

```bash
# Build all local images
make docker-build

# Start the stack
make docker-up

# Build and start the stack
make docker-up-build

# Recreate running services
make docker-up-recreate

# Show service status
make docker-ps

# Follow logs
make docker-logs

# Follow one service
make docker-logs SERVICE=api

# Open a service shell
make docker-exec SERVICE=api

# Stop the stack
make docker-stop

# Stop and remove local volumes
make docker-down
```

### Testing Details

Unit tests use SQLite or mocked dependencies where possible and do not require Docker. Docker smoke checks require the Compose stack and validate service integration against PostgreSQL, pgvector, OpenSearch, Kafka, MinIO, Prometheus, and Grafana.

```bash
# Unit tests only
poetry run pytest -vv

# A focused API test file
poetry run pytest tests/unit/api/test_documents.py -vv

# A focused worker test file
poetry run pytest tests/unit/workers/test_ingestion.py -vv
```

### Troubleshooting

| Issue | Check |
|---|---|
| Missing `.env` | Run `cp .env.template .env` |
| Docker smoke target cannot connect | Start Docker Desktop or the Docker daemon |
| API is not ready | Run `make docker-logs SERVICE=api` and check `/readiness` |
| OpenSearch retrieval returns no chunks | Run `make docker-smoke-bm25-retrieval` and check `indexing-worker` logs |
| Upload does not become queryable | Run `make docker-smoke-upload-query` and check `ingestion-worker` and `indexing-worker` logs |
| LLM synthesis is skipped | Set `LLM_SYNTHESIS_ENABLED=true` and configure the selected provider credentials |

## Contributing

1. Create a feature branch.
2. Keep changes scoped to one complete implementation slice.
3. Add focused tests for new API, CRUD, worker, retrieval, or authorization behavior.
4. Run `make check`.
5. Use a short, clear commit message.

## License

No license file is currently included. Treat this repository as private/internal unless a project license is added.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [PostgreSQL](https://www.postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector) for metadata and vector storage
- [OpenSearch](https://opensearch.org/) for BM25 retrieval
- [LangGraph](https://www.langchain.com/langgraph) for agent workflow orchestration
- [LiteLLM](https://www.litellm.ai/) for model gateway integration
- [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) for local observability

## Support

- Documentation: [docs/](docs/)
- Frontend when running locally: `http://localhost:5173`
- API docs when running locally: `http://localhost:8100/docs`
- Health endpoint: `http://localhost:8100/readiness`

---

Agentic RAG - Production-oriented retrieval, ingestion, authorization, and query orchestration for large document datasets.

Agentic RAG was planned, developed, and is maintained by **Mithlesh Upadhyay** as a solo effort, with a production-oriented focus on very large datasets, low-cost retrieval, and efficient LLM usage.
