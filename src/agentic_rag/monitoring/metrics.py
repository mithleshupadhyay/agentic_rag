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
