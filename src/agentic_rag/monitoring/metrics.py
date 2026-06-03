from prometheus_client import Counter, Histogram


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
