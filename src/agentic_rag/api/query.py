import logging

from fastapi import APIRouter, Depends

from agentic_rag.core.dependencies import require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.query.bm25_query import run_bm25_query
from agentic_rag.shared.schemas.query import QueryRequest, QueryResponse


logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    payload: QueryRequest,
    user_context: UserContext = Depends(require_scope("query:run")),
) -> QueryResponse:
    logger.info(
        f"[QueryAPI] Query started tenant={user_context.tenant_id} "
        f"user={user_context.id}"
    )

    response = run_bm25_query(
        user_context=user_context,
        request=payload,
    )

    logger.info(
        f"[QueryAPI] Query completed tenant={user_context.tenant_id} "
        f"user={user_context.id} context_chunks={len(response.context)}"
    )
    return response
