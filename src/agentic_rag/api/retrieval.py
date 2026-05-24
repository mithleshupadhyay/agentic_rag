import logging

from fastapi import APIRouter, Depends

from agentic_rag.core.dependencies import require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.shared.schemas.retrieval import BM25SearchRequest, RetrievalResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/bm25-search", response_model=RetrievalResponse)
def bm25_search_endpoint(
    payload: BM25SearchRequest,
    user_context: UserContext = Depends(require_scope("query:run")),
) -> RetrievalResponse:
    logger.info(
        f"[RetrievalAPI] BM25 search tenant={user_context.tenant_id} "
        f"user={user_context.id} limit={payload.limit}"
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query=payload.query,
        filters=payload.filters,
        limit=payload.limit,
    )

    logger.info(
        f"[RetrievalAPI] BM25 search completed tenant={user_context.tenant_id} "
        f"user={user_context.id} candidates={len(response.candidates)}"
    )
    return response
