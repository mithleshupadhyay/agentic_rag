# Low-Level Design

## Purpose

This document converts the production architecture into an implementation-level
design. It defines the initial folder structure, service boundaries, database
tables, Pydantic schemas, repository layer, API contracts, Kafka events, agent
state, and test strategy.

The design follows these implementation principles:

- Build microservices from day one, managed in one repository.
- Keep each service independently runnable and deployable.
- Use Pydantic schemas for every API and Kafka contract.
- Use SQLAlchemy models for database tables.
- Use repositories for database CRUD/query logic.
- Keep tenant and ACL filtering in every data access path.
- Use LangGraph for the agent runtime.
- Use Kafka for ingestion, indexing, embedding, and long-running async work.
- Keep local development simple with Docker Compose and Makefile targets.

## Reference Patterns Applied

The design borrows these local implementation patterns:

| Pattern | Applied In This Repo |
|---|---|
| FastAPI app with routers | Every API service gets `main.py`, `routes.py`, and versioned endpoints |
| SQLAlchemy model, Pydantic schema, CRUD split | Shared database layer uses `models`, `schemas`, and `repositories` |
| Tenant-scoped access | Every query filters by `tenant_id`, `workspace_id`, and ACL fields |
| Kafka indexing pipeline | Ingestion API publishes events, workers consume topic-specific jobs |
| LangGraph state machine | Agent Runtime owns graph, state, nodes, guards, and checkpoints |
| Agent step tracking | Agent runs and steps are persisted for traceability and loop detection |
| Model tiering | LLM Gateway separates cheap routing models from stronger answer models |
| Document hash and metadata | Ingestion deduplicates by content hash before embedding or indexing |
| Makefile and Docker ergonomics | Common local commands are standardized |

## Repository Layout

```text
src/agentic_rag/
+-- shared/
|   +-- __init__.py
|   +-- config.py
|   +-- logging.py
|   +-- security.py
|   +-- telemetry.py
|   +-- pagination.py
|   +-- exceptions.py
|   |
|   +-- auth/
|   |   +-- context.py
|   |   +-- jwt.py
|   |   +-- dependencies.py
|   |
|   +-- db/
|   |   +-- base.py
|   |   +-- session.py
|   |   +-- models/
|   |   +-- repositories/
|   |
|   +-- schemas/
|   |   +-- common.py
|   |   +-- auth.py
|   |   +-- documents.py
|   |   +-- chunks.py
|   |   +-- ingestion.py
|   |   +-- retrieval.py
|   |   +-- query.py
|   |   +-- agent.py
|   |   +-- llm.py
|   |   +-- evaluation.py
|   |
|   +-- kafka/
|       +-- producer.py
|       +-- consumer.py
|       +-- events.py
|       +-- topics.py
|
+-- services/
|   +-- gateway_api/
|   +-- rag_query_api/
|   +-- agent_runtime/
|   +-- retrieval_service/
|   +-- llm_gateway/
|   +-- ingestion_api/
|   +-- authz_service/
|   +-- evaluation_service/
|   +-- admin_service/
|
+-- workers/
|   +-- parser_worker.py
|   +-- metadata_worker.py
|   +-- chunking_worker.py
|   +-- embedding_worker.py
|   +-- indexing_worker.py
|   +-- evaluation_worker.py
|
+-- ingestion/
|   +-- parser.py
|   +-- chunker.py
|   +-- metadata_extractor.py
|   +-- dedup.py
|   +-- acl_classifier.py
|
+-- retrieval/
|   +-- metadata_search.py
|   +-- bm25_search.py
|   +-- vector_search.py
|   +-- document_fetch.py
|   +-- reranker.py
|   +-- context_builder.py
|
+-- agent/
|   +-- graph.py
|   +-- state.py
|   +-- nodes.py
|   +-- guards.py
|   +-- prompts.py
|   +-- checkpoints.py
|
+-- llm/
|   +-- gateway.py
|   +-- model_router.py
|   +-- budgets.py
|   +-- providers.py
|   +-- retry_policy.py
|
+-- storage/
|   +-- object_store.py
|   +-- opensearch.py
|   +-- pgvector.py
|
+-- monitoring/
    +-- metrics.py
    +-- tracing.py
    +-- audit.py
```

## Shared Conventions

### IDs And Timestamps

- Use UUID primary keys for main business records.
- Use timezone-aware `created_at` and `updated_at`.
- Use `deleted_at` and `is_deleted` for soft-delete records.
- Use `content_hash` for deduplication.
- Use `idempotency_key` for ingestion and write-heavy APIs.

### Tenant Fields

Every important row must include:

```text
tenant_id
workspace_id
created_by
acl_version
data_region
```

Every repository query must receive an `AuthContext` or explicit tenant fields.

### Pydantic Schema Naming

Use these names consistently:

```text
<Entity>Create
<Entity>Update
<Entity>Read
<Entity>ListItem
<Entity>SearchRequest
<Entity>SearchResponse
```

For SQLAlchemy response conversion, use:

```python
model_config = ConfigDict(from_attributes=True)
```

### Repository Naming

Repository files should be entity based:

```text
tenant_repository.py
user_repository.py
document_repository.py
chunk_repository.py
ingestion_job_repository.py
agent_run_repository.py
query_log_repository.py
```

Repository methods should not read from request objects directly. APIs and
services should convert request schemas into explicit method arguments.

## Core Shared Schemas

### AuthContext

```python
class AuthContext(BaseModel):
    user_id: str
    tenant_id: str
    workspace_id: str | None = None
    roles: list[str] = []
    group_ids: list[str] = []
    scopes: list[str] = []
    acl_version: int = 1
    data_region: str | None = None
    request_id: str | None = None
```

### Pagination

```python
class PageRequest(BaseModel):
    page: int = Field(default=1, ge=1)
    size: int = Field(default=50, ge=1, le=500)


class PageResponse(BaseModel):
    page: int
    size: int
    total: int
```

### Health

```python
class HealthResponse(BaseModel):
    service: str
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    dependencies: dict[str, str] = {}
```

## Database Models

### Tenant And User Tables

```text
tenants
- id
- name
- status
- data_region
- created_at
- updated_at

users
- id
- tenant_id
- external_subject
- email
- display_name
- status
- created_at
- updated_at

groups
- id
- tenant_id
- name
- created_at
- updated_at

roles
- id
- tenant_id
- name
- scopes
- created_at
- updated_at

user_groups
- user_id
- group_id
- tenant_id

user_roles
- user_id
- role_id
- tenant_id
```

### Document Tables

```text
documents
- id
- tenant_id
- workspace_id
- source_type
- source_uri
- object_key
- title
- file_name
- mime_type
- byte_size
- content_hash
- status
- owner_user_id
- acl_version
- classification_level
- metadata
- created_by
- created_at
- updated_at
- is_deleted
- deleted_at

document_chunks
- id
- tenant_id
- workspace_id
- document_id
- chunk_index
- content
- content_hash
- token_count
- section_path
- page_number
- start_offset
- end_offset
- metadata
- acl_version
- classification_level
- created_at
- updated_at
- is_deleted
- deleted_at
```

### ACL Tables

```text
document_acl
- id
- tenant_id
- document_id
- allowed_user_ids
- allowed_group_ids
- allowed_roles
- denied_user_ids
- denied_group_ids
- visibility
- acl_version
- created_at
- updated_at

chunk_acl
- id
- tenant_id
- chunk_id
- allowed_user_ids
- allowed_group_ids
- allowed_roles
- denied_user_ids
- denied_group_ids
- visibility
- acl_version
- created_at
- updated_at
```

### Vector Table

Use pgvector inside PostgreSQL.

```text
chunk_embeddings
- id
- tenant_id
- workspace_id
- document_id
- chunk_id
- embedding
- embedding_model
- embedding_dimension
- content_hash
- vector_version
- metadata
- acl_version
- created_at
- updated_at
```

Important indexes:

```text
(tenant_id, workspace_id)
(tenant_id, document_id)
(tenant_id, chunk_id)
(tenant_id, content_hash)
(tenant_id, acl_version)
vector index on embedding
```

### Ingestion And Agent Tables

```text
ingestion_jobs
- id
- tenant_id
- workspace_id
- source_type
- source_uri
- object_key
- status
- idempotency_key
- current_stage
- retry_count
- max_retries
- locked_by
- locked_at
- lease_expires_at
- next_retry_at
- error_type
- error_message
- created_by
- created_at
- updated_at

query_runs
- id
- tenant_id
- workspace_id
- user_id
- request_id
- conversation_id
- query_text
- filters
- status
- retrieval_strategy
- answer
- citations
- candidates
- context
- response_payload
- retrieval_limit
- max_context_chunks
- max_context_tokens
- context_token_count
- confidence_score
- latency_ms
- synthesis_enabled
- llm_provider
- llm_model
- llm_input_tokens
- llm_output_tokens
- llm_cost_estimate
- error_type
- error_message
- completed_at
- created_at
- updated_at

agent_runs
- id
- tenant_id
- workspace_id
- user_id
- query
- status
- retrieval_strategy
- confidence_score
- started_at
- completed_at
- total_steps
- total_tool_calls
- timeout_at

agent_steps
- id
- agent_run_id
- tenant_id
- node_name
- step_number
- tool_name
- tool_input
- tool_output_summary
- latency_ms
- status
- error_type
- created_at

agent_checkpoints
- id
- agent_run_id
- tenant_id
- checkpoint_key
- state
- created_at
```

### Logs, Feedback, And Evaluation Tables

```text
query_logs
- id
- tenant_id
- user_id
- query
- retrieval_strategy
- cache_hit
- latency_ms
- token_count
- cost_estimate
- created_at

retrieval_logs
- id
- tenant_id
- agent_run_id
- tool_name
- filters
- result_count
- selected_chunk_ids
- latency_ms
- created_at

feedback_events
- id
- tenant_id
- user_id
- agent_run_id
- rating
- feedback_text
- created_at

evaluation_runs
- id
- tenant_id
- dataset_name
- status
- metrics
- created_at
- completed_at
```

## Repository Layer

### DocumentRepository

Responsibilities:

- Create document metadata.
- Update document status.
- Soft-delete documents.
- List documents by tenant, source, status, owner, date range.
- Fetch document only if tenant scope matches.
- Check content hash before creating duplicate records.

Core methods:

```python
create_document(ctx, payload, object_key, content_hash) -> Document
get_document(ctx, document_id) -> Document | None
list_documents(ctx, filters, page) -> tuple[list[Document], int]
update_status(ctx, document_id, status) -> Document
soft_delete(ctx, document_id) -> None
content_hash_exists(ctx, content_hash) -> bool
```

### ChunkRepository

Responsibilities:

- Bulk insert chunks.
- Fetch chunks by IDs with ACL scope.
- Search chunk metadata.
- Soft-delete chunks when a document is deleted.

Core methods:

```python
bulk_create_chunks(ctx, document_id, chunks) -> list[DocumentChunk]
get_authorized_chunks(ctx, chunk_ids) -> list[DocumentChunk]
list_chunks_by_document(ctx, document_id, page) -> tuple[list[DocumentChunk], int]
content_hash_exists(ctx, content_hash) -> bool
```

### EmbeddingRepository

Responsibilities:

- Insert pgvector embeddings.
- Search vectors with tenant and ACL filters.
- Avoid re-embedding unchanged chunks.
- Track embedding model and vector version.

Core methods:

```python
create_chunk_embedding(db, tenant_id, obj_in) -> ChunkEmbedding
bulk_create_chunk_embeddings(db, tenant_id, embeddings) -> int
get_chunks_missing_embedding(db, tenant_id, embedding_model, vector_version, limit) -> list[DocumentChunk]
embedding_exists(db, tenant_id, chunk_id, embedding_model, vector_version, content_hash) -> bool
search_similar_chunks_by_embedding(db, tenant_id, query_embedding, embedding_model, vector_version, limit) -> list[ChunkVectorSearchResult]
```

Current implementation covers tenant-scoped embedding writes, idempotent
same-hash writes, stale content-hash updates, dimension checks, chunk selection
for missing embeddings, and a local embedding worker that calls the LLM gateway
embedding contract. It also includes tenant-scoped pgvector similarity search
with model/version filters, deleted-record filtering, optional workspace and
document filters, and optional user ACL filtering. The vector retrieval service
now embeds a user query through the provider-neutral LLM gateway, calls this
pgvector search, and returns authorized vector candidates. The hybrid retrieval
service now calls BM25 and vector retrieval, merges duplicate chunks by ID, and
uses simple rank-based scoring so BM25 and vector scores do not need to share a
numeric scale. The reranker service now applies deterministic, provider-neutral
query-to-candidate scoring, preserves original retrieval score/source metadata,
and returns a bounded reranked candidate list. Hybrid retrieval now passes
merged candidates through the reranker and keeps pre-rerank score/source
metadata for explainability. Model-backed reranking, a protected rerank API
endpoint, and context building are intentionally separate slices.

Core reranker method:

```python
rerank_chunks(query, candidates, top_k) -> RerankResponse
```

### AgentRunRepository

Responsibilities:

- Create agent run.
- Append steps.
- Store checkpoints.
- Mark run completed, failed, timed out, or handed off.

Core methods:

```python
create_run(ctx, query, timeout_at) -> AgentRun
append_step(ctx, run_id, step) -> AgentStep
save_checkpoint(ctx, run_id, checkpoint_key, state) -> AgentCheckpoint
mark_completed(ctx, run_id, result) -> AgentRun
mark_failed(ctx, run_id, error) -> AgentRun
```

## API Service Contracts

### Gateway API

Responsibilities:

- Validate JWT/API key.
- Resolve tenant and user context.
- Apply request size limits, rate limits, and request timeout.
- Route to internal services.

Endpoints:

```text
GET /health
GET /metrics
```

Gateway routes user-facing endpoints to internal services.

### RAG Query API

Endpoints:

```text
POST /query
POST /query/stream
GET  /query
GET  /query/{agent_run_id}
POST /query/{agent_run_id}/cancel
```

The query flow is retrieval-first. It calls BM25 retrieval, builds a safe
authorized context with citations, and optionally sends only that sanitized
context to the LLM gateway when `LLM_SYNTHESIS_ENABLED=true`. Every query run is
persisted tenant-scoped before retrieval starts, then marked completed or failed
with latency, retrieval strategy, context summary, citations, LLM provider/model,
token counts, cost estimate, and the final response payload.

The target agentic query flow is:

```text
user query
-> planner decides what is needed
-> chooses tools
-> retrieval / web / metadata / document fetch / vector search
-> evaluates evidence
-> may retry or reformulate query
-> builds context
-> generates answer
-> verifies answer
-> returns citations
```

The current query path implements this incrementally with BM25 retrieval,
context building, and deterministic answer verification first. Later steps add
planner/tool routing, reformulation, vector search, web search, and full agent
runtime.

Current query metrics are defined in `src/agentic_rag/monitoring/metrics.py`
and exposed through the existing `/metrics` endpoint:

```text
agentic_rag_query_lifecycle_total
- Labels: status, retrieval_strategy, synthesis_enabled
- Status values: started, completed, failed, cancelled

agentic_rag_query_latency_seconds
- Labels: status, retrieval_strategy, synthesis_enabled
- Observed for completed and failed query execution paths

agentic_rag_llm_provider_circuit_state
- Labels: provider, model, state
- State values: closed, open, half_open

agentic_rag_llm_provider_failure_count
- Labels: provider, model
- Tracks consecutive provider/model failures held by the circuit breaker

agentic_rag_llm_provider_retry_after_seconds
- Labels: provider, model
- Tracks remaining retry delay while a provider/model circuit is open
```

The query and LLM provider metrics intentionally exclude tenant IDs, user IDs,
workspace IDs, request IDs, agent run IDs, raw query text, prompts, document
identifiers, chunk identifiers, and provider credentials to avoid
high-cardinality metrics and sensitive data exposure.

The first dashboard and alert artifacts are:

```text
monitoring/query_dashboard.json
monitoring/query_alerts.yml
```

The dashboard covers query lifecycle rate, query failure rate, and p95/p99
query latency. The alert rules cover sustained query failure rate and sustained
p95 query latency.

Local Docker Compose monitoring is provisioned through:

```text
monitoring/prometheus.yml
monitoring/grafana_datasource.yml
monitoring/grafana_dashboard.yml
```

Prometheus scrapes the API at `api:8000` inside the Compose network. Host
browsers still reach Prometheus and Grafana through the published
`PROMETHEUS_PORT` and `GRAFANA_PORT` values.

Current answer verification is deterministic. Synthesized answers must cite
returned context using bracket numbers such as `[1]`; the verifier rejects
answers with missing citations or citations that do not map to the returned
context. When verification fails, the API keeps the retrieved context and
citations in the response but returns a safe fallback answer.

Request:

```python
class QueryRequest(BaseModel):
    query: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    filters: RetrievalFilters = RetrievalFilters()
    stream: bool = False
    retrieval_limit: int = 20
    max_context_chunks: int = 12
    max_context_tokens: int = 6000
```

Response:

```python
class QueryResponse(BaseModel):
    agent_run_id: UUID
    answer: str
    citations: list[Citation]
    candidates: list[CandidateChunk]
    context: list[ContextChunk]
    context_token_count: int
    confidence_score: float
    retrieval_strategy: str
    latency_ms: int
    synthesis_enabled: bool
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_estimate: float = 0.0
    synthesis_error: str | None = None
```

Streaming response:

```text
POST /query/stream
Content-Type: text/event-stream

event: query_started
data: {"event":"query_started","agent_run_id":"...","data":{"request_id":"..."}}

event: query_completed
data: {"event":"query_completed","agent_run_id":"...","data":{"response":{...}}}

event: query_failed
data: {"event":"query_failed","agent_run_id":"...","data":{"error_type":"...","error_message":"..."}}
```

The current streaming endpoint reuses the existing BM25 query flow and streams
query lifecycle events around the final `QueryResponse`. Token-by-token LLM
streaming is intentionally deferred until the LLM and agent runtime streaming
contracts are stable.

Query run endpoints:

```text
GET /query
- Lists persisted query runs for the current tenant.
- Non-admin users only see their own runs.
- Workspace-bound users only see their workspace.

GET /query/{agent_run_id}
- Returns one tenant-scoped query run by run id.
- Enforces workspace and owner access before returning the stored response.

POST /query/{agent_run_id}/cancel
- Cancels a tenant-scoped queued or running query run.
- Enforces the same workspace and owner/admin access rules as query-run fetch.
- Returns `409 Conflict` when the run is already completed, failed, or cancelled.
```

Future query endpoints:

```text
GET  /query/{agent_run_id}/steps
```

### Ingestion API

Endpoints:

```text
POST /v1/documents
POST /v1/documents:batch
GET  /v1/documents/{document_id}
GET  /v1/documents
DELETE /v1/documents/{document_id}
GET  /v1/ingestion/jobs/{job_id}
```

Request:

```python
class DocumentCreateRequest(BaseModel):
    workspace_id: str | None = None
    source_type: Literal["upload", "s3", "url", "connector"]
    source_uri: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = {}
    acl: AclPolicy
    idempotency_key: str | None = None
```

Response:

```python
class IngestionJobResponse(BaseModel):
    job_id: UUID
    document_id: UUID | None = None
    status: Literal["queued", "running", "completed", "failed"]
    current_stage: str
```

### Retrieval Service

Current API endpoint:

```text
POST /retrieval/bm25-search
POST /retrieval/vector-search
POST /retrieval/hybrid-search
POST /retrieval/rerank
```

The API request body must not include tenant or ACL context. The endpoint gets
`UserContext` from auth dependencies. Search endpoints convert it into tenant,
workspace, ACL, deny-rule, and visibility filters before OpenSearch or pgvector
search is called. The rerank endpoint only reorders submitted candidates; it
does not fetch protected chunks from storage.

```python
class BM25SearchRequest(BaseModel):
    query: str
    filters: RetrievalFilters = RetrievalFilters()
    limit: int = 20
    deadline_ms: int = 1500


class VectorSearchRequest(BaseModel):
    query: str
    filters: RetrievalFilters = RetrievalFilters()
    limit: int = 20
    min_similarity: float = 0.0
    deadline_ms: int = 1500


class HybridSearchRequest(BaseModel):
    query: str
    filters: RetrievalFilters = RetrievalFilters()
    limit: int = 20
    min_similarity: float = 0.0
    deadline_ms: int = 1500


class RerankRequest(BaseModel):
    query: str
    candidates: list[CandidateChunk]
    top_k: int = 12
```

Future internal retrieval endpoints:

```text
POST /internal/v1/retrieval/metadata-search
POST /internal/v1/retrieval/context-build
```

### Agent Runtime Service

Internal endpoints:

```text
POST /internal/v1/agent/runs
GET  /internal/v1/agent/runs/{agent_run_id}
GET  /internal/v1/agent/runs/{agent_run_id}/steps
POST /internal/v1/agent/runs/{agent_run_id}/cancel
```

The Agent Runtime owns LangGraph execution and calls Retrieval Service and LLM
Gateway through contracts.

Current local implementation:

```text
src/agentic_rag/agent/runtime.py
```

The local runtime skeleton creates `AgentStateModel` instances from
`AuthContext` and `AgentLimits`, records node progress, increments step and tool
call counts, evaluates guardrails, applies the standard safe fallback answer,
and returns an `AgentCheckpoint` payload after each recorded node. This first
slice intentionally does not wire LangGraph, internal API endpoints, database
persistence, retrieval execution, or LLM calls yet.

### LLM Gateway

Internal endpoints:

```text
POST /internal/v1/llm/classify
POST /internal/v1/llm/rewrite
POST /internal/v1/llm/generate
POST /internal/v1/llm/verify
POST /internal/v1/llm/embed
```

Responsibilities:

- Route cheap tasks to small local models.
- Enforce tenant budgets.
- Count tokens and cost.
- Apply retries and circuit breakers.
- Redact sensitive logs.
- Never receive raw unauthorized chunks; answer generation must use the
  sanitized context returned by retrieval and context building.

Current local implementation:

```text
src/agentic_rag/llm/gateway.py
src/agentic_rag/llm/circuit_breaker.py
```

The local gateway supports chat completion and embedding generation through
LiteLLM. Embedding calls enforce input budget, retry transient provider
failures, reuse circuit-breaker protection, and validate the returned vector
dimension against the configured pgvector dimension before workers persist
vectors. Current local testing uses the Gemini API model
`gemini-embedding-001`, addressed through LiteLLM as
`gemini/gemini-embedding-001`, with 768 output dimensions to keep the existing
pgvector schema. The circuit breaker stores provider/model health in memory by
default for simple local runs. Production and multi-replica deployments can set
`LLM_CIRCUIT_BREAKER_STATE_BACKEND=redis` so API and worker replicas share open,
half-open, and reset state through Redis. If Redis is temporarily unavailable,
the gateway logs the storage failure and falls back to the in-process memory
state instead of dropping circuit-breaker protection.

The circuit breaker exposes a safe provider health snapshot for each checked
provider/model pair. The snapshot contains only operational fields:
`provider`, `model`, `state`, `failure_count`, `retry_after_seconds`,
`opened_until`, `half_open`, `last_error_type`, `last_failure_at`, and
`checked_at`. It does not include tenant IDs, user IDs, request IDs, raw prompts,
documents, chunks, provider credentials, or full exception messages. The same
state updates low-cardinality Prometheus gauges for circuit state, consecutive
failure count, and retry-after seconds through
`src/agentic_rag/monitoring/metrics.py`.

Configuration:

```text
LLM_SYNTHESIS_ENABLED=false
LLM_PROVIDER=litellm
DEFAULT_LLM_MODEL=ollama/llama3.1
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=700
LLM_MAX_INPUT_CHARS=64000
LLM_MAX_OUTPUT_TOKENS=8000
LLM_MAX_RETRIES=2
LLM_RETRY_BACKOFF_SECONDS=0.5
LLM_CIRCUIT_BREAKER_ENABLED=true
LLM_CIRCUIT_BREAKER_STATE_BACKEND=memory
LLM_CIRCUIT_BREAKER_REDIS_KEY_PREFIX=agentic-rag:llm:circuit-breaker
LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS=60
LLM_TIMEOUT_SECONDS=30
REDIS_URL=redis://localhost:6379/0
EMBEDDING_PROVIDER=litellm
EMBEDDING_MODEL_NAME=gemini/gemini-embedding-001
EMBEDDING_DIMENSION=768
GEMINI_API_KEY=
```

### AuthZ Service

Internal endpoints:

```text
POST /internal/v1/authz/documents/filter
POST /internal/v1/authz/chunks/filter
POST /internal/v1/authz/chunks/check
```

The AuthZ Service returns allowed IDs and denial reasons. Retrieval still applies
database-side filters; AuthZ is not a replacement for query-level filtering.

## Kafka Event Contracts

All Kafka events use this envelope:

```python
class EventEnvelope(BaseModel):
    event_id: UUID
    event_type: str
    event_version: int = 1
    tenant_id: str
    workspace_id: str | None = None
    correlation_id: str
    causation_id: str | None = None
    idempotency_key: str | None = None
    occurred_at: datetime
    payload: dict[str, Any]
```

### Topics

```text
ingestion.parse
ingestion.metadata
ingestion.chunk
ingestion.embed
ingestion.index
rag.long_query
eval.batch
retry.ingestion
retry.embedding
retry.indexing
dlq.ingestion
dlq.embedding
dlq.indexing
dlq.rag
```

### Ingestion Parse Event

```python
class ParseDocumentPayload(BaseModel):
    job_id: UUID
    document_id: UUID
    object_key: str
    mime_type: str
    source_type: str
```

### Chunk Event

```python
class ChunkDocumentPayload(BaseModel):
    job_id: UUID
    document_id: UUID
    extracted_text_key: str
    metadata: dict[str, Any]
    acl_version: int
```

### Embedding Event

```python
class EmbedChunksPayload(BaseModel):
    job_id: UUID
    document_id: UUID
    chunk_ids: list[UUID]
    embedding_model: str
    vector_version: int
```

### Indexing Event

```python
class IndexChunksPayload(BaseModel):
    job_id: UUID
    document_id: UUID
    chunk_ids: list[UUID]
    index_name: str
```

### Ingestion Retry Event

```python
class IngestionRetryPayload(BaseModel):
    job_id: UUID
    document_id: UUID | None = None
    failed_stage: str
    source_topic: str
    retry_topic: str
    attempt: int
    max_attempts: int
    error_type: str
    error_message: str | None = None
    failed_at: datetime
    next_retry_at: datetime
    metadata: dict[str, Any]
```

### Ingestion DLQ Event

```python
class IngestionDLQPayload(BaseModel):
    job_id: UUID
    document_id: UUID | None = None
    failed_stage: str
    source_topic: str
    dlq_topic: str
    attempt: int
    max_attempts: int
    error_type: str
    error_message: str | None = None
    failed_at: datetime
    terminal_reason: str
    metadata: dict[str, Any]
```

## Worker Design

Each worker follows the same structure:

```text
load settings
configure logging and tracing
subscribe to topic
validate EventEnvelope
check idempotency
acquire distributed lock
process payload
renew lock before long-running stages finish
write status and outputs
publish next event
commit offset only after success
on retryable failure publish retry event
on terminal failure publish DLQ event
```

The document upload API creates the tenant-scoped document and ingestion job
first. When `KAFKA_PUBLISHING_ENABLED` is true, it then publishes a
`DOCUMENT_PARSE_REQUESTED` envelope with `ParseDocumentPayload` to
`ingestion.parse`. If publishing is disabled or the broker publish fails, the
queued DB job remains available for the polling worker path.

The local ingestion worker accepts an optional event publisher callable. It
writes the database failure state first, then emits an `IngestionRetryPayload`
to `retry.ingestion` when `next_retry_at` is set, or an `IngestionDLQPayload` to
`dlq.ingestion` when retries are exhausted. The Kafka runtime producer is
intentionally separate from this contract so unit tests can use a fake
publisher.

After parsing and chunk replacement succeeds, the ingestion worker emits a
`DOCUMENT_EMBED_REQUESTED` envelope with `EmbedChunksPayload` to
`ingestion.embed` and a `DOCUMENT_INDEX_REQUESTED` envelope with
`IndexChunksPayload` to `ingestion.index` when an event publisher is configured.
If either publish fails, ingestion still completes because the embedding and
indexing workers can discover the same missing chunks through their DB polling
paths.

`KafkaEventPublisher` serializes the validated `EventEnvelope` with
`model_dump_json()`, uses `idempotency_key` as the Kafka message key when one is
provided, falls back to `event_id` otherwise, and delegates delivery to an
injected send-compatible producer client. The document API creates a
`kafka-python` producer only around the upload scheduling publish. The ingestion
worker creates a producer at startup only when `KAFKA_PUBLISHING_ENABLED` is
true and then passes the publisher into the existing success and failure event
hooks. DB job polling remains the default local scheduling mechanism.
`KafkaEventConsumer` mirrors the producer shape for the consume side: it accepts
an injected consumer client and handler callable, validates raw message values
into `EventEnvelope`, commits offsets after handler success, and commits invalid
JSON/schema messages only after logging the validation rejection so a poison
message does not block the partition. The concrete factory creates a
`kafka-python` consumer with automatic commits disabled. When
`KAFKA_CONSUMING_ENABLED` is true, the ingestion worker subscribes to
`ingestion.parse` and `retry.ingestion` with `KAFKA_INGESTION_CONSUMER_GROUP`.
The parse handler validates `ParseDocumentPayload`, claims the referenced
queued job through the existing tenant-scoped DB lease path, and only then calls
the existing ingestion job processor. The retry handler validates
`IngestionRetryPayload`, claims the referenced retryable job through the same
tenant-scoped DB lease path, and then calls the processor.
The embedding worker can subscribe to `ingestion.embed` with
`KAFKA_EMBEDDING_CONSUMER_GROUP`. The embedding handler validates
`EmbedChunksPayload`, checks that the requested model and vector version match
the worker configuration, and then embeds only chunks matching the envelope
tenant, payload document ID, and payload chunk IDs. DB polling remains the
default local embedding scheduling path when Kafka consuming is disabled.
The indexing worker can subscribe to `ingestion.index` with
`KAFKA_INDEXING_CONSUMER_GROUP`. The indexing handler validates
`IndexChunksPayload`, checks that the requested index name matches the worker
OpenSearch client, and then indexes only chunks matching the envelope tenant,
payload document ID, and payload chunk IDs. DB polling remains the default local
BM25 indexing scheduling path when Kafka consuming is disabled.
The local Docker stack includes a single-node Kafka broker and a one-shot topic
provisioning service that runs `scripts/kafka-topic.sh` for ingestion, retry,
DLQ, long-query, and evaluation topics. `make docker-smoke-kafka-producer` runs
`scripts/smoke_kafka_producer.py` through the API container and publishes one
valid `retry.ingestion` envelope to verify the configured producer path.
`make docker-smoke-kafka-consumer` delegates host-side Docker orchestration to
`scripts/docker-smoke-kafka-consumer.sh`, temporarily stops the polling ingestion
worker, and then runs `scripts/smoke_kafka_consumer.py` in the API container with
`KAFKA_CONSUMING_ENABLED=true`. The smoke currently publishes one valid
`retry.ingestion` envelope, consumes it through the concrete Kafka consumer
adapter, and verifies the existing ingestion retry handler completes a
tenant-scoped ingestion job.

Required worker settings:

```text
consumer_group
topic
retry_topic
dlq_topic
max_retries
retry_backoff_base_seconds
retry_backoff_max_seconds
step_timeout_seconds
max_concurrency
lock_ttl_seconds
```

## LangGraph Agent Design

### Agent State

```python
class AgentState(TypedDict):
    agent_run_id: str
    auth: dict
    query: str
    rewritten_query: str | None
    intent: str | None
    filters: dict[str, Any]
    retrieval_strategy: str | None
    retrieved_candidates: list[dict[str, Any]]
    authorized_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    context: list[dict[str, Any]]
    draft_answer: str | None
    final_answer: str | None
    citations: list[dict[str, Any]]
    confidence_score: float
    step_count: int
    tool_call_count: int
    visited_nodes: list[str]
    last_tool_calls: list[dict[str, Any]]
    last_results_hash: str | None
    deadline_at: str
    handoff_required: bool
```

### Nodes

```text
classify_intent
rewrite_query
plan_filters
select_retrieval_strategy
metadata_search
bm25_search
vector_search
merge_candidates
filter_authorized_chunks
rerank
build_context
generate_answer
verify_grounding
finalize
fallback
human_handoff
```

### Guard Rules

- Stop when `step_count > max_steps`.
- Stop when `tool_call_count > max_tool_calls`.
- Stop if deadline is exceeded.
- Stop if a recorded step exceeds `step_timeout_seconds`.
- Stop if the same tool with the same arguments repeats more than twice.
- Do not call generation when no authorized context exists.
- Do not return an answer if citations fail verification.
- Save checkpoint after every node.

## Retrieval Tool Design

### Metadata Search

Input:

```python
class MetadataSearchInput(BaseModel):
    auth: AuthContext
    filters: dict[str, Any]
    limit: int = 50
```

Output:

```python
class CandidateDocument(BaseModel):
    document_id: UUID
    score: float
    source: str = "metadata"
    metadata: dict[str, Any]
```

### BM25 Search

OpenSearch documents must include:

```text
tenant_id
workspace_id
document_id
chunk_id
owner_user_id
content
metadata
acl_version
visibility
allowed_user_ids
allowed_group_ids
allowed_roles
denied_user_ids
denied_group_ids
classification_level
```

Search must apply tenant and ACL filters before returning candidates.

### pgvector Search

The vector query must include:

```text
tenant_id
workspace_id
acl filters
embedding_model
limit
similarity threshold
```

For large tenants, use table partitioning and indexes by tenant and model.

### Context Builder

Responsibilities:

- Remove duplicate chunks by chunk ID and repeated cleaned content.
- Strip OpenSearch highlight markup before building LLM context.
- Work only on already authorized retrieval candidates.
- Preserve citation metadata.
- Respect token budget.
- Truncate oversized chunks when needed.
- Keep the output ready for grounded answer synthesis.

## Testing Design

### Unit Tests

```text
tests/unit/shared/schemas/
tests/unit/shared/repositories/
tests/unit/services/
tests/unit/agent/
tests/unit/workers/
```

Unit tests should cover:

- Pydantic validation.
- Repository tenant filtering.
- Soft-delete behavior.
- Idempotency keys.
- ACL allow/deny behavior.
- Agent guard decisions.
- Kafka event validation.

### Integration Tests

```text
tests/integration/api/
tests/integration/db/
tests/integration/kafka/
tests/integration/retrieval/
```

Integration tests should cover:

- FastAPI routes with dependency overrides.
- PostgreSQL + pgvector migrations.
- OpenSearch indexing and search.
- Kafka publish/consume flow.
- End-to-end ingestion state transitions.

### Smoke Tests

```text
tests/smoke/
```

Smoke tests should verify:

- All service `/health` endpoints.
- One document ingestion job.
- Metadata-only retrieval.
- BM25 retrieval.
- pgvector retrieval.
- Agent answer with citations.


## First Implementation Target

The first code commit after this document should add:

```text
src/agentic_rag/shared/config.py
src/agentic_rag/shared/logging.py
src/agentic_rag/shared/schemas/common.py
src/agentic_rag/services/*/main.py
tests/unit/services/test_health.py
```

Every API service should return:

```json
{
  "service": "gateway-api",
  "status": "healthy",
  "version": "0.1.0",
  "dependencies": {}
}
```

This keeps the first implementation small, testable, and aligned with the
production service boundaries.
