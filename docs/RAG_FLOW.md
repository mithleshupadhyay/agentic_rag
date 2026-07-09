# Agentic RAG End-to-End Flow and Interview Guide

This document explains what happens in this `agentic_rag` project when a user
uploads a document and when a user chats with the system. It is written for
architecture and interview discussion, so it covers API flow, async processing,
chunk keys, BM25, vector indexing, hybrid retrieval, object storage, and the
current implementation limits.

## Current System Truth

Do not describe this repo as a MongoDB-based RAG system unless the
implementation is changed. The current implementation uses:

| Concern | Current implementation |
|---|---|
| API | FastAPI |
| Metadata DB | PostgreSQL through SQLAlchemy |
| Vector store | PostgreSQL with pgvector |
| BM25 search | OpenSearch |
| Object store | MinIO locally, S3-compatible API in production |
| Async processing | Kafka topics plus worker processes |
| Cache and rate/lock support | Redis |
| Agent runtime | LangGraph |
| LLM/embedding gateway | LiteLLM-facing gateway functions |

V1 ingestion currently supports UTF-8 text-like files such as `.txt`, `.md`,
`.json`, `.csv`, and similar MIME types. PDF/OCR/document-layout parsing is a
production extension, not the current parser behavior.

## Full System Flow

```mermaid
flowchart TB
    User["User / Client UI"] --> API["FastAPI API service"]
    API --> Auth["AuthN/AuthZ<br/>scope, tenant, workspace, ACL context"]

    subgraph Upload["Document upload path"]
        Auth --> UploadAPI["POST /documents/upload"]
        UploadAPI --> ValidateFile["Validate file name, size, metadata_json"]
        ValidateFile --> HashFile["Compute SHA-256 content_hash"]
        HashFile --> DocRow["Create documents row<br/>status=queued"]
        DocRow --> ObjectStore["MinIO / S3 raw object"]
        ObjectStore --> AttachKey["Attach object_key to document"]
        AttachKey --> JobRow["Create ingestion_jobs row<br/>stage=created"]
        JobRow --> ParseTopic["Kafka ingestion.parse<br/>document.parse_requested"]
    end

    subgraph Ingestion["Async ingestion workers"]
        ParseTopic --> IngestionWorker["Ingestion worker"]
        IngestionWorker --> ReadRaw["Read raw bytes from object store"]
        ReadRaw --> Decode["Decode supported text file"]
        Decode --> ChunkText["Split text into overlapping chunks"]
        ChunkText --> ChunkRows["Replace document_chunks rows<br/>copy ACL to chunk_acl"]
        ChunkRows --> ReadyDoc["documents.status=ready"]
        ChunkRows --> EmbedTopic["Kafka ingestion.embed"]
        ChunkRows --> IndexTopic["Kafka ingestion.index"]
    end

    subgraph Embedding["Embedding pipeline"]
        EmbedTopic --> EmbedWorker["Embedding worker"]
        EmbedWorker --> EmbedLLM["Embedding model through LLM gateway"]
        EmbedLLM --> Pgvector[("PostgreSQL + pgvector<br/>chunk_embeddings")]
    end

    subgraph BM25Index["BM25 indexing pipeline"]
        IndexTopic --> IndexWorker["Indexing worker"]
        IndexWorker --> OpenSearch[("OpenSearch<br/>chunks-write / chunks-read aliases")]
        IndexWorker --> MarkBM25["Mark chunk BM25 status<br/>indexed or failed"]
    end

    subgraph Query["Chat / query path"]
        Auth --> QueryAPI["POST /query or /query/stream"]
        QueryAPI --> QueryRun["Create query_run / agent_run"]
        QueryRun --> CacheCheck["Optional Redis cache lookup"]
        CacheCheck --> Retrieval["BM25, vector, or hybrid retrieval"]
        Retrieval --> ACLFilter["Tenant, workspace, ACL, denied-list filters"]
        ACLFilter --> Context["Context builder<br/>dedupe, clean, token budget"]
        Context --> LLM["Optional LLM answer synthesis"]
        LLM --> Verify["Grounding and citation verifier"]
        Verify --> Response["Answer + citations + context + confidence"]
    end

    Retrieval --> OpenSearch
    Retrieval --> Pgvector
    Retrieval --> DocDB[("PostgreSQL<br/>documents, chunks, ACLs, jobs, runs")]
    DocRow --> DocDB
    JobRow --> DocDB
    ChunkRows --> DocDB
    QueryRun --> DocDB
```

## Upload API Flow

Main endpoint: `POST /documents/upload`.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant API as FastAPI /documents/upload
    participant DB as PostgreSQL
    participant S3 as MinIO/S3
    participant K as Kafka

    U->>API: multipart file + workspace_id + metadata_json
    API->>API: sanitize file name
    API->>API: validate non-empty and max upload bytes
    API->>API: parse metadata_json as object
    API->>API: sha256(raw bytes) = document content_hash
    API->>DB: insert documents row with status=queued and document_acl
    API->>S3: put raw object bytes
    S3-->>API: bucket, object_key, etag, byte_size
    API->>DB: attach object_key to documents row
    API->>DB: insert ingestion_jobs row with status=queued
    API->>K: publish document.parse_requested to ingestion.parse
    API-->>U: document, ingestion_job_id, object_key, status
```

Important implementation details:

- The upload API reads the file bytes, computes `content_hash` with SHA-256,
  and stores the original file bytes in object storage.
- It creates a `documents` row before object upload, then attaches the
  `object_key` after upload succeeds.
- It creates one `ingestion_jobs` row for async processing.
- It publishes a Kafka event only if Kafka publishing is enabled.
- If object upload fails, the document is marked `failed`.
- If DB update after object upload fails, the API tries to delete the orphaned
  object.

## Object Store Keys

Current raw object key format:

```text
tenants/{tenant_id}/workspaces/{workspace_id_or_default}/documents/{document_id}/raw/{safe_file_name}
```

Example:

```text
tenants/tenant-a/workspaces/workspace-a/documents/11111111-1111-1111-1111-111111111111/raw/security-policy.md
```

```mermaid
flowchart LR
    Tenant["tenant_id"] --> Key["object_key"]
    Workspace["workspace_id or default"] --> Key
    Document["document_id UUID"] --> Key
    Raw["raw/"] --> Key
    File["safe_file_name"] --> Key
```

Current code does not store each chunk as an object-store object. Chunks are
stored as rows in `document_chunks`, indexed into OpenSearch, and optionally
embedded into `chunk_embeddings`.

If chunk-wise object storage is later added, use a deterministic key like:

```text
tenants/{tenant_id}/workspaces/{workspace_id_or_default}/documents/{document_id}/chunks/v1/{chunk_index}-{chunk_id}.txt
```

That keeps object paths tenant-scoped, workspace-scoped, document-scoped, and
stable across retries.

## Chunk Keys and IDs

When an interviewer asks "what is the key of each chunk?", answer with the
different keys used by different storage systems:

| Storage/system | Key |
|---|---|
| PostgreSQL chunk primary key | `document_chunks.id` UUID |
| Stable chunk order inside a document | unique `(document_id, chunk_index)` |
| Dedup/staleness key | `content_hash = sha256(chunk.content)` |
| OpenSearch BM25 document ID | `_id = chunk_id` |
| Embedding uniqueness | unique `(chunk_id, embedding_model, vector_version)` |
| Tenant/workspace filtering | `tenant_id`, `workspace_id` |
| ACL filtering | `chunk_acl` plus `acl_version`, allow lists, deny lists |

```mermaid
erDiagram
    DOCUMENTS ||--o{ DOCUMENT_CHUNKS : has
    DOCUMENT_CHUNKS ||--o| CHUNK_ACL : secures
    DOCUMENT_CHUNKS ||--o{ CHUNK_EMBEDDINGS : embeds
    DOCUMENTS ||--o{ INGESTION_JOBS : processes

    DOCUMENTS {
        uuid id PK
        string tenant_id
        string workspace_id
        string object_key
        string content_hash
        string status
        string owner_user_id
    }

    DOCUMENT_CHUNKS {
        uuid id PK
        uuid document_id FK
        int chunk_index
        text content
        string content_hash
        int token_count
        string bm25_index_status
    }

    CHUNK_EMBEDDINGS {
        uuid id PK
        uuid chunk_id FK
        vector embedding
        string embedding_model
        int vector_version
        string content_hash
    }

    INGESTION_JOBS {
        uuid id PK
        uuid document_id FK
        string object_key
        string status
        string current_stage
    }
```

## Ingestion Worker Flow

```mermaid
stateDiagram-v2
    [*] --> queued
    queued --> running_parse: claim job
    running_parse --> parsing_document: read object + decode text
    parsing_document --> chunking: update stage=chunk
    chunking --> ready: store chunks and chunk ACL
    ready --> publish_embedding: publish ingestion.embed
    ready --> publish_indexing: publish ingestion.index
    publish_embedding --> completed
    publish_indexing --> completed
    running_parse --> failed: exception
    parsing_document --> failed: unsupported file / decode error
    chunking --> failed: chunking or DB error
    failed --> queued: retry if retry_count < max_retries
    failed --> dlq: retries exhausted
    completed --> [*]
    dlq --> [*]
```

Detailed steps:

1. The worker claims a queued job using a lease. PostgreSQL can use
   `FOR UPDATE SKIP LOCKED` so multiple workers do not process the same job.
2. It reads `job.object_key` from MinIO/S3.
3. It decodes the file as UTF-8 text if the MIME type or extension is supported.
4. It chunks text using character windows with overlap. Defaults are
   `INGESTION_CHUNK_SIZE=2000` and `INGESTION_CHUNK_OVERLAP=200`.
5. It stores chunks in `document_chunks` and copies document ACL to `chunk_acl`.
6. It marks the document `ready`.
7. It publishes embedding and BM25 indexing events.
8. On failure, it stores error details, schedules retry with backoff, or sends
   to DLQ after retries are exhausted.

## Embedding and Vector DB Flow

```mermaid
sequenceDiagram
    autonumber
    participant K as Kafka ingestion.embed
    participant EW as Embedding worker
    participant DB as PostgreSQL + pgvector
    participant LLM as Embedding gateway

    K->>EW: document.embed_requested with chunk_ids
    EW->>DB: select chunks missing embedding for model/version
    DB-->>EW: chunk content
    EW->>LLM: embedding request with texts[]
    LLM-->>EW: vectors dimension=768
    EW->>DB: upsert chunk_embeddings
    DB-->>EW: written_count
```

Current pgvector storage:

```text
chunk_embeddings.embedding = Vector(768)
unique key = (chunk_id, embedding_model, vector_version)
```

Current code enables pgvector and stores vectors. It also creates B-tree indexes
for metadata filters such as tenant, model, document, and chunk. The current
migration does not define an explicit ANN vector index such as HNSW or IVFFlat.
For a production-scale interview answer, say:

```text
Yes, the vector DB should be indexed. For pgvector we add an ANN index on
chunk_embeddings.embedding, usually HNSW with vector_cosine_ops, and keep
B-tree filters on tenant_id, embedding_model, vector_version, workspace_id,
and document_id.
```

Example production index:

```sql
CREATE INDEX CONCURRENTLY ix_chunk_embeddings_embedding_hnsw
ON chunk_embeddings
USING hnsw (embedding vector_cosine_ops);
```

The query path generates an embedding for the user query, then searches:

```text
ORDER BY chunk_embeddings.embedding <=> query_embedding
```

That returns nearest chunks by cosine distance, after tenant, workspace,
document, metadata, and ACL filtering.

## BM25 Indexing Flow

```mermaid
sequenceDiagram
    autonumber
    participant K as Kafka ingestion.index
    participant IW as Indexing worker
    participant DB as PostgreSQL
    participant OS as OpenSearch

    K->>IW: document.index_requested with chunk_ids
    IW->>DB: list chunks where bm25_index_status=pending
    IW->>OS: ensure chunks-v1 index and aliases
    IW->>OS: bulk index chunks with _id=chunk_id
    OS-->>IW: bulk result
    IW->>DB: mark chunks indexed with index name and content hash
```

OpenSearch document shape for each chunk includes:

- `tenant_id`, `workspace_id`, `document_id`, `chunk_id`
- `content`, `document_title`, `file_name`
- `section_path`, `page_number`, offsets, token count
- document and chunk metadata
- ACL fields such as visibility, allowed users/groups/roles, denied users/groups
- classification and ACL version

OpenSearch uses `chunks-write` for writes and `chunks-read` for search. This
makes future blue/green reindexing easier because the application can write to
or read from aliases instead of hard-coding a physical index forever.

## What Is BM25?

BM25 is a lexical ranking algorithm used by search engines. It ranks a document
or chunk based on how well the query terms match the text.

In simple terms:

- If a query term appears in a chunk, the chunk becomes more relevant.
- Rare terms are weighted higher than common terms. This is the IDF part.
- Repeating a term helps, but the gain saturates. This prevents keyword spam.
- Very long chunks are normalized so they do not win only because they contain
  more words.

Typical BM25 scoring idea:

```text
score(query, chunk) = sum over query terms:
  IDF(term) * term_frequency_saturation * length_normalization
```

OpenSearch handles the inverted index and BM25 scoring internally. In this repo
the BM25 query uses a `multi_match` search over:

```text
content^3, document_title^2, file_name
```

So chunk content has the highest score impact, title has medium impact, and file
name helps as a smaller signal.

```mermaid
flowchart LR
    Query["security policy"] --> Analyzer["Analyzer<br/>tokenize, lowercase, normalize"]
    Analyzer --> Terms["security<br/>policy"]
    Terms --> InvertedIndex["Inverted index<br/>term -> chunk postings"]
    InvertedIndex --> BM25["BM25 scoring<br/>TF + IDF + length norm"]
    BM25 --> Filters["Tenant, workspace, ACL filters"]
    Filters --> Hits["Ranked authorized chunks"]
```

## Chat / Query Flow

There are two query styles:

| Endpoint | Behavior |
|---|---|
| `POST /query` | Non-streaming strategy-aware query path for BM25, vector, or hybrid retrieval |
| `POST /query/stream` | Streaming LangGraph agent runtime path |
| `POST /retrieval/bm25-search` | Direct BM25 retrieval API |
| `POST /retrieval/vector-search` | Direct vector retrieval API |
| `POST /retrieval/hybrid-search` | Direct hybrid retrieval API |
| `POST /retrieval/rerank` | Direct reranker API |

### Non-Streaming Query

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant API as POST /query
    participant Redis as Redis cache
    participant OS as OpenSearch BM25
    participant CB as Context builder
    participant LLM as LLM gateway
    participant V as Verifier
    participant DB as PostgreSQL

    U->>API: query, filters, limits
    API->>DB: create query_run
    API->>Redis: optional auth-scoped cache lookup
    alt cache hit
        Redis-->>API: cached response
        API->>DB: mark query_run completed
        API-->>U: response
    else cache miss
        API->>OS: BM25 search with tenant/workspace/ACL filters
        OS-->>API: candidate chunks
        API->>CB: build context within chunk/token budget
        alt LLM synthesis enabled and context exists
            API->>LLM: answer from authorized context only
            LLM-->>API: draft answer
            API->>V: verify answer support by citations
        else synthesis disabled
            API->>API: return retrieved context summary
        end
        API->>Redis: optional cache write
        API->>DB: mark query_run completed
        API-->>U: answer, citations, context, confidence
    end
```

### Streaming Agent Query

The streaming path uses a LangGraph-style workflow:

```mermaid
flowchart LR
    Start([Start]) --> Intent["classify_intent"]
    Intent --> Rewrite["rewrite_query"]
    Rewrite --> Plan["plan_filters"]
    Plan --> Strategy["select_retrieval_strategy"]
    Strategy --> BM25["bm25_search"]
    BM25 --> Context["build_context"]
    Context --> Generate["generate_answer"]
    Generate --> Verify["verify_grounding"]
    Verify --> End([End])
```

Current strategy selection sets BM25. The design already has schemas and APIs
for vector and hybrid retrieval, so the next step is to make the strategy node
choose between BM25, vector, and hybrid based on query intent, filters, and
confidence.

The agent runtime protects the system with guardrails:

- max steps
- max tool calls
- total timeout
- per-step timeout
- repeated tool-call detection
- no answer generation without authorized context
- cancellation checks from the database

## Retrieval Strategies

```mermaid
flowchart TB
    Query["User query"] --> Router["Retrieval strategy"]
    Router --> BM25["BM25<br/>exact lexical search"]
    Router --> Vector["Vector<br/>semantic similarity"]
    Router --> Hybrid["Hybrid<br/>merge BM25 + vector"]

    BM25 --> Merge["Candidate chunks"]
    Vector --> Merge
    Hybrid --> Merge
    Merge --> Rerank["Reranker"]
    Rerank --> Context["Context builder"]
    Context --> Answer["Grounded answer"]
```

BM25 is best for exact keywords, codes, names, IDs, and terms that must match
literally.

Vector search is best for semantic similarity, paraphrases, and cases where the
query uses different wording from the document.

Hybrid search runs both BM25 and vector search, merges candidates by rank, then
reranks. In this repo, the hybrid merge gives each retrieval source a
rank-based contribution like `0.5 / rank`, deduplicates by `chunk_id`, and then
passes candidates to the reranker.

## Context Building and Citations

The context builder does not blindly send all chunks to the LLM. It:

1. Keeps only the top candidates within `max_context_chunks`.
2. Deduplicates by chunk ID.
3. Deduplicates repeated normalized content.
4. Removes HTML tags and normalizes whitespace.
5. Truncates to `max_context_tokens`.
6. Creates citations with `document_id`, `chunk_id`, title, URI, page, section,
   quote, and score.

```mermaid
flowchart LR
    Candidates["Ranked candidates"] --> DedupeID["Dedupe chunk_id"]
    DedupeID --> DedupeText["Dedupe repeated content"]
    DedupeText --> Clean["Clean HTML/whitespace"]
    Clean --> Budget["Apply chunk and token budget"]
    Budget --> Citations["Attach citations"]
    Citations --> Context["Authorized context for LLM"]
```

## Security and ACL Flow

Every retrieval path is tenant-scoped and authorization-scoped.

```mermaid
flowchart TB
    UserContext["UserContext<br/>tenant, workspace, roles, groups, acl_version"] --> Filters["Retrieval filters"]
    Filters --> Tenant["tenant_id must match"]
    Filters --> Workspace["workspace_id must match if scoped"]
    Filters --> Deny["deny user/group checks"]
    Filters --> Allow["owner, public, tenant, allowed users, groups, roles"]
    Allow --> Results["Only authorized chunks"]
    Deny --> Blocked["Excluded chunks"]
```

The LLM only sees context after retrieval filtering. This is important because
authorization must happen before prompt construction, not after answer
generation.

## Query Cache Key

If Redis query caching is enabled, the cache key is authorization-scoped. The
hash includes:

- tenant ID
- workspace ID
- user ID
- roles and groups
- scopes
- ACL version
- query text
- filters
- retrieval limits
- context limits
- retrieval strategy
- LLM synthesis flag and model config

Format:

```text
{QUERY_CACHE_KEY_PREFIX}:bm25:{sha256(canonical_payload)}
```

Default prefix:

```text
agentic-rag:query
```

This avoids returning an answer cached for one user or ACL version to a
different user.

## Failure and Retry Flow

```mermaid
flowchart TB
    Work["Worker processes job"] --> Success{"Success?"}
    Success -->|Yes| Complete["Mark completed"]
    Success -->|No| Fail["Mark failed<br/>error_type + error_message"]
    Fail --> Retry{"retry_count < max_retries?"}
    Retry -->|Yes| Schedule["Set next_retry_at<br/>exponential backoff"]
    Schedule --> RetryTopic["Publish retry event"]
    Retry -->|No| DLQ["Publish DLQ event"]
```

This matters in interviews because ingestion, embedding, and indexing are
expensive and long-running. They should not block the upload API request.

## Status Lifecycle

```mermaid
flowchart LR
    D1["documents.status=queued"] --> D2["parsing"]
    D2 --> D3["indexing"]
    D3 --> D4["ready"]
    D1 --> DF["failed"]
    D2 --> DF
    D3 --> DF

    J1["ingestion_jobs.status=queued"] --> J2["running"]
    J2 --> J3["completed"]
    J2 --> JF["failed"]
    JF --> J2
```

Chunk BM25 status:

```text
pending -> indexed
pending -> failed
failed -> pending on retry/reindex
```

## Interview Short Answers

### What happens when a user uploads a document?

The upload API validates the file, computes a SHA-256 content hash, creates a
`documents` row with ACL metadata, uploads the raw bytes to MinIO/S3, attaches
the object key to the document row, creates an `ingestion_jobs` row, and
publishes a Kafka parse event. The API returns quickly with the document ID,
job ID, and object key. Parsing, chunking, embedding, and BM25 indexing happen
asynchronously in workers.

### What is the object store key?

Current raw file key:

```text
tenants/{tenant_id}/workspaces/{workspace_id_or_default}/documents/{document_id}/raw/{safe_file_name}
```

Chunks are not currently stored as object-store files. Their primary key is
`document_chunks.id`; OpenSearch uses that same chunk ID as `_id`; embeddings
are keyed by `(chunk_id, embedding_model, vector_version)`.

### What is the key of each chunk?

The main key is `document_chunks.id`, a UUID. The stable order key is
`(document_id, chunk_index)`. The content staleness/dedup key is `content_hash`.
The BM25 index document ID is `chunk_id`. The vector uniqueness key is
`(chunk_id, embedding_model, vector_version)`.

### Do we use indexing in vector DB?

The repo uses pgvector and stores each embedding as `Vector(768)` in
`chunk_embeddings`. The current migration has pgvector enabled and B-tree
metadata indexes, but it does not yet create an HNSW/IVFFlat ANN vector index.
For production, add an HNSW index on `embedding vector_cosine_ops` plus keep
tenant/model/document filters.

### What is BM25?

BM25 is lexical search ranking. It scores chunks by matching query terms,
boosting rare terms, applying diminishing returns for repeated terms, and
normalizing by chunk length. OpenSearch builds the inverted index and calculates
BM25 scores. In this project we search `content^3`, `document_title^2`, and
`file_name`, then apply tenant/workspace/ACL filters.

### What happens when a user chats?

For `POST /query`, the API creates a query run, optionally checks Redis cache,
runs the requested retrieval strategy with ACL filters, builds a token-limited
context, optionally calls the LLM to synthesize an answer, verifies grounding
against citations, records the run, and returns the answer plus citations and
context.

For `POST /query/stream`, the API runs the LangGraph agent flow:
`classify_intent -> rewrite_query -> plan_filters -> select_retrieval_strategy
-> bm25_search -> build_context -> generate_answer -> verify_grounding`, and
streams events/tokens back to the client.

### Why not vectorize everything only?

For large datasets, vectorizing everything is expensive and can be weaker for
exact terms, IDs, product names, legal clauses, and rare keywords. This design
uses metadata and BM25 first, vector search where semantic matching helps, and
hybrid search when both signals are useful.

### Why async ingestion?

Parsing, chunking, embedding, and indexing can take seconds or minutes and can
fail independently. The upload API should stay fast. Kafka and workers provide
backpressure, retries, DLQs, horizontal scaling, and operational isolation.

### Where is authorization enforced?

Authorization is enforced before context reaches the LLM. Retrieval filters by
tenant, workspace, ACL version, denied users/groups, ownership, visibility, and
allowed users/groups/roles. Only authorized chunks are passed to context
building and answer generation.

## Production Improvements To Mention Honestly

These are good interview follow-ups because they show you understand the gap
between V1 and production scale:

- Add PDF, DOCX, HTML, image, and OCR parsers.
- Store extracted text and optional chunk text in object storage for very large
  documents, while keeping searchable text in DB/OpenSearch.
- Add a pgvector HNSW or IVFFlat ANN index.
- Add dynamic retrieval routing in the agent strategy node: BM25 vs vector vs
  hybrid.
- Add blue/green OpenSearch reindex jobs using aliases.
- Add idempotency to every worker event path.
- Add background re-embedding when embedding model or vector version changes.
- Add ingestion progress endpoint and UI.
- Add per-tenant quotas for uploads, embeddings, LLM tokens, and indexing.
- Add stronger reranker model in production; current reranker is deterministic
  term coverage logic.
