import logging
import time
from uuid import uuid4

from fastapi import HTTPException

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.retrieval.context_builder import build_context
from agentic_rag.shared.schemas.query import QueryRequest, QueryResponse
from agentic_rag.shared.schemas.retrieval import ContextBuildRequest, RetrievalStrategy


logger = logging.getLogger(__name__)


def run_bm25_query(
    user_context: UserContext,
    request: QueryRequest,
) -> QueryResponse:
    logger.info(
        f"[Query] BM25 query started tenant={user_context.tenant_id} "
        f"user={user_context.id} retrieval_limit={request.retrieval_limit}"
    )
    started_at = time.perf_counter()

    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")

    filters = request.filters.model_copy(deep=True)
    if request.workspace_id:
        if filters.workspace_id and filters.workspace_id != request.workspace_id:
            logger.warning(
                f"[Query] Workspace mismatch workspace_id={request.workspace_id} "
                f"filters_workspace={filters.workspace_id}"
            )
            raise HTTPException(
                status_code=400,
                detail="workspace_id must match filters.workspace_id when both are provided.",
            )
        filters.workspace_id = request.workspace_id

    retrieval_response = search_bm25_chunks(
        user_context=user_context,
        query=query_text,
        filters=filters,
        limit=request.retrieval_limit,
    )

    context_response = build_context(
        ContextBuildRequest(
            query=query_text,
            chunks=retrieval_response.candidates,
            max_context_chunks=request.max_context_chunks,
            max_tokens=request.max_context_tokens,
        )
    )

    citations = [
        context_chunk.citation
        for context_chunk in context_response.context
    ]
    answer = (
        "LLM synthesis is not enabled yet. "
        f"Retrieved {len(context_response.context)} context chunks for this query."
    )
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    logger.info(
        f"[Query] BM25 query completed tenant={user_context.tenant_id} "
        f"user={user_context.id} candidates={len(retrieval_response.candidates)} "
        f"context_chunks={len(context_response.context)} latency_ms={latency_ms}"
    )
    return QueryResponse(
        agent_run_id=uuid4(),
        answer=answer,
        citations=citations,
        candidates=retrieval_response.candidates,
        context=context_response.context,
        context_token_count=context_response.token_count,
        confidence_score=0.0,
        retrieval_strategy=RetrievalStrategy.BM25,
        latency_ms=latency_ms,
        synthesis_enabled=False,
    )
