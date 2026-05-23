# Next Sprint

## Purpose

This file tracks the next practical improvements needed to make Agentic RAG ready
for very large datasets, high traffic, tenant isolation, ingestion workers,
retrieval quality, and production operations.

## Future Scope By File

| File | Current Role | Future Scope For Huge Dataset Agentic RAG | Priority |
|---|---|---|---|
| `docs/ARCHITECTURE.md` | Defines the target production architecture. | Keep updated as services become real deployable components; add capacity assumptions for ingestion throughput, retrieval latency, cache hit rate, vectorization percentage, and storage growth. | High |
| `docs/LOW_LEVEL_DESIGN.md` | Converts architecture into implementation details. | Add exact API contracts, worker contracts, retry rules, data retention rules, index ownership, and service-by-service deployment notes. | High |
| `docs/NEXT_SPRINT.md` | Tracks upcoming technical scope. | Keep this file updated after every pushed step so the next development sequence stays clear and explainable. | High |
| `Dockerfile.app` | Builds the shared Python runtime image used by API, migrations, seed, ingestion worker, and indexing worker. | Keep one shared image while API and workers use the same dependencies. Split later into focused Dockerfiles such as `Dockerfile.ingestion-worker`, `Dockerfile.embedding-worker`, or `Dockerfile.browser-worker` only when workers need heavy parser/OCR tools, ML/GPU libraries, browser automation, or different runtime hardening. | Medium |
| `docker-compose.yml` | Runs local API, PostgreSQL/pgvector, MinIO, and bucket initialization. | Add Redis, Kafka, OpenSearch, and worker services only when the code uses them; add profiles for lightweight API-only and full ingestion stacks. | High |
| `.dockerignore` | Keeps local and test artifacts out of Docker images. | Keep updated as new local cache, generated data, model, and artifact directories are added. | Low |
| `Makefile` | Provides repeatable validation and Docker commands. | Add migration, local smoke test, Docker smoke test, and service-specific worker commands as the stack grows. | Medium |
| `src/agentic_rag/main.py` | Main FastAPI app entrypoint. | Add request ID middleware, structured access logging, OpenTelemetry setup, graceful startup/shutdown checks, and explicit router inclusion for ingestion/query/admin endpoints. | High |
| `src/agentic_rag/api/health.py` | Health endpoint. | Expand to readiness checks for PostgreSQL, Redis, Kafka, OpenSearch, object storage, and LLM gateway without leaking secrets. | High |
| `src/agentic_rag/api/documents.py` | Document API endpoints. | Add streaming file upload, object-store write, ingestion job creation, idempotency key handling, upload size limits, MIME validation, and status endpoints. | High |
| `src/agentic_rag/core/auth.py` | Auth token verification. | Add production OIDC hardening, JWKS cache, issuer/audience validation tests, role/group/scope mapping, and clear tenant resolution rules. | High |
| `src/agentic_rag/core/authorization.py` | Tenant, user, group, role, and scope checks. | Extend to chunk-level authorization, workspace policies, document classification checks, deny-by-default rules, and retrieval-time ACL filtering. | High |
| `src/agentic_rag/core/dependencies.py` | Shared FastAPI dependencies. | Add dependencies for object store, request context, pagination, rate limit context, and service-level settings. | Medium |
| `src/agentic_rag/core/models/user_context.py` | Authenticated user context model. | Add data region, request ID, tenant plan, quota context, and normalized permissions once API traffic and tenant plans are implemented. | Medium |
| `src/agentic_rag/shared/config.py` | Central environment configuration. | Add Redis, Kafka, OpenSearch, LLM gateway, embedding model, reranker, object-store upload limits, observability, and worker tuning settings. | High |
| `src/agentic_rag/shared/db/base.py` | SQLAlchemy base and common model helpers. | Add common audit fields if needed, stronger JSON typing helpers, naming conventions, and migration-safe defaults for production databases. | Medium |
| `src/agentic_rag/shared/db/session.py` | Database engine/session management. | Add pool tuning, health check helper, transaction utilities, sync/async separation policy, and production retry guidance. | High |
| `src/agentic_rag/shared/db/models/tenants.py` | Tenant model. | Add tenant status transitions, plan/quota fields, data region enforcement, retention policy, encryption policy, and tenant-level settings. | High |
| `src/agentic_rag/shared/db/models/documents.py` | Document, chunk, and embedding models. | Add chunk indexing status, OpenSearch index references, embedding job status, vector version strategy, parent-child chunk support, and partition/index review for very large tables. | High |
| `src/agentic_rag/shared/db/models/acl.py` | Document and chunk ACL models. | Add policy versioning, inherited ACLs, group expansion snapshots, deny rules, and efficient indexes for retrieval-time chunk filtering. | High |
| `src/agentic_rag/shared/db/models/ingestion_jobs.py` | Ingestion job model. | Add stage timestamps, worker lease fields, retry backoff fields, dead-letter reason, source connector metadata, and batch ingestion grouping. | High |
| `src/agentic_rag/shared/db/crud/documents.py` | Tenant-scoped document CRUD. | Add chunk CRUD, bulk chunk insert, bulk status updates, idempotent create-by-hash, pagination counts, lock-safe job updates, and retrieval-facing list queries. | High |
| `src/agentic_rag/shared/kafka/topics.py` | Kafka topic constants. | Add DLQ topics, retry topics, evaluation topics, tenant-aware topic naming policy, and topic retention documentation. | High |
| `src/agentic_rag/shared/kafka/events.py` | Kafka event schemas. | Add parser, metadata, chunking, embedding, indexing, retry, DLQ, and audit events with schema versioning and idempotency fields. | High |
| `src/agentic_rag/storage/object_store.py` | S3-compatible object storage client. | Add bucket readiness check, streaming upload integration, object metadata lookup, presigned URLs, multipart upload, server-side encryption options, and lifecycle policy notes. | High |
| `src/agentic_rag/shared/schemas/common.py` | Common API schema primitives. | Add pagination, error response, sort, filter, request ID, and batch operation response models. | Medium |
| `src/agentic_rag/shared/schemas/auth.py` | Auth and permission schemas. | Add tenant membership, workspace access, effective permissions, and token claim mapping schemas. | Medium |
| `src/agentic_rag/shared/schemas/documents.py` | Document API schemas. | Add upload request/response, object-store fields, document status transitions, bulk document operations, and ingestion linkage. | High |
| `src/agentic_rag/shared/schemas/chunks.py` | Chunk API and internal schemas. | Add chunk create/read/search schemas, chunk ACL summaries, citation fields, token window metadata, and embedding status fields. | High |
| `src/agentic_rag/shared/schemas/ingestion.py` | Ingestion schemas. | Add upload ingestion request, connector ingestion request, job progress response, retry response, and batch ingestion status response. | High |
| `src/agentic_rag/shared/schemas/retrieval.py` | Retrieval schemas. | Add metadata search, BM25 search, vector search, hybrid merge, reranker, citation, and ACL-filtered candidate schemas. | High |
| `src/agentic_rag/shared/schemas/query.py` | Query API schemas. | Add streaming response events, grounded answer response, citation response, cache metadata, and budget/timeout fields. | High |
| `src/agentic_rag/shared/schemas/agent.py` | Agent runtime schemas. | Add LangGraph state, step records, tool call records, checkpoint metadata, max step/tool limits, and loop detection fields. | High |
| `src/agentic_rag/shared/schemas/llm.py` | LLM gateway schemas. | Add model routing, token budget, provider timeout, retry policy, cost tracking, prompt policy, and response safety metadata. | High |
| `src/agentic_rag/shared/schemas/evaluation.py` | Evaluation schemas. | Add retrieval evaluation, answer faithfulness, citation accuracy, latency, cost, and regression test result schemas. | Medium |
| `tests/unit/api/test_documents.py` | Document API tests. | Add upload endpoint tests, authorization edge cases, ingestion job creation tests, and large file validation tests. | High |
| `tests/unit/api/test_health.py` | Health API tests. | Add readiness dependency status tests and degraded/unhealthy response tests. | Medium |
| `tests/unit/core/test_auth.py` | Auth tests. | Add OIDC JWKS cache tests, invalid issuer/audience tests, tenant claim mapping tests, and scope mapping tests. | High |
| `tests/unit/core/test_authorization.py` | Authorization tests. | Add chunk ACL filtering tests, workspace isolation tests, group access tests, deny-rule tests, and retrieval authorization tests. | High |
| `tests/unit/shared/db/test_document_crud.py` | Document CRUD tests. | Add bulk chunk insert tests, idempotency tests, status transition tests, soft delete restore tests, and tenant leak prevention tests. | High |
| `tests/unit/shared/db/test_models.py` | Model tests. | Add index/constraint coverage, relationship loading tests, JSON field tests, and model defaults for ingestion/chunk/ACL tables. | Medium |
| `tests/unit/shared/test_schemas.py` | Schema tests. | Add stricter validation tests for query, retrieval, agent, ingestion, and LLM gateway schemas. | Medium |
| `tests/unit/storage/test_object_store.py` | Object storage tests. | Add mocked error handling tests, metadata tests, streaming upload tests, and local MinIO integration tests later. | Medium |

## Recommended Next Implementation Order

| Step | Work |
|---|---|
| 1 | Add document upload API that writes raw files to object storage and creates an ingestion job. |
| 2 | Add ingestion job CRUD and status APIs. |
| 3 | Add parser and chunking worker contract with Kafka event schemas. |
| 4 | Add chunk CRUD with bulk insert and chunk-level ACL persistence. |
| 5 | Add OpenSearch indexing client and BM25 indexing worker. |
| 6 | Add selective embedding worker with pgvector writes. |
| 7 | Add retrieval service functions: metadata search, BM25 search, vector search, merge, ACL filter, rerank, context build. |
| 8 | Add query API and agent runtime skeleton with max steps, max tool calls, timeout, checkpointing, and loop protection. |
