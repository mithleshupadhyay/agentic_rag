from prometheus_client import Counter, Gauge, Histogram


QUERY_LIFECYCLE_TOTAL = Counter(
    "agentic_rag_query_lifecycle_total",
    "Total Agentic RAG query lifecycle events.",
    ("status", "retrieval_strategy", "synthesis_enabled"),
)
QUERY_LATENCY_SECONDS = Histogram(
    "agentic_rag_query_latency_seconds",
    "Agentic RAG query latency in seconds.",
    ("status", "retrieval_strategy", "synthesis_enabled"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
RETRIEVAL_LIFECYCLE_TOTAL = Counter(
    "agentic_rag_retrieval_lifecycle_total",
    "Total Agentic RAG retrieval lifecycle events.",
    ("status", "retrieval_strategy"),
)
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "agentic_rag_retrieval_latency_seconds",
    "Agentic RAG retrieval latency in seconds.",
    ("status", "retrieval_strategy"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
RETRIEVAL_RESULT_TOTAL = Counter(
    "agentic_rag_retrieval_result_total",
    "Total Agentic RAG retrieval result counts.",
    ("retrieval_strategy", "result"),
)
LLM_PROVIDER_CIRCUIT_STATE = Gauge(
    "agentic_rag_llm_provider_circuit_state",
    "LLM provider circuit breaker state by provider and model.",
    ("provider", "model", "state"),
)
LLM_PROVIDER_FAILURE_COUNT = Gauge(
    "agentic_rag_llm_provider_failure_count",
    "LLM provider consecutive circuit breaker failure count.",
    ("provider", "model"),
)
LLM_PROVIDER_RETRY_AFTER_SECONDS = Gauge(
    "agentic_rag_llm_provider_retry_after_seconds",
    "LLM provider circuit breaker retry-after seconds.",
    ("provider", "model"),
)
WORKER_JOB_LIFECYCLE_TOTAL = Counter(
    "agentic_rag_worker_job_lifecycle_total",
    "Total Agentic RAG worker job lifecycle events.",
    ("worker", "job_type", "status"),
)
WORKER_JOB_LATENCY_SECONDS = Histogram(
    "agentic_rag_worker_job_latency_seconds",
    "Agentic RAG worker job processing latency in seconds.",
    ("worker", "job_type", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
)
WORKER_QUEUE_LAG_SECONDS = Histogram(
    "agentic_rag_worker_queue_lag_seconds",
    "Agentic RAG worker queue lag in seconds before processing starts.",
    ("worker", "job_type", "source"),
    buckets=(0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 300.0, 900.0, 3600.0),
)
WORKER_ITEM_TOTAL = Counter(
    "agentic_rag_worker_item_total",
    "Total Agentic RAG worker item counts.",
    ("worker", "job_type", "item_type", "status"),
)
WORKER_RETRY_LIFECYCLE_TOTAL = Counter(
    "agentic_rag_worker_retry_lifecycle_total",
    "Total Agentic RAG worker retry and DLQ lifecycle events.",
    ("worker", "job_type", "status", "topic"),
)
