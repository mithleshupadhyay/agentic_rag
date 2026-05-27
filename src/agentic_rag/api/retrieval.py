import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from agentic_rag.core.dependencies import require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.retrieval.hybrid_search import search_hybrid_chunks
from agentic_rag.retrieval.reranker import rerank_chunks
from agentic_rag.retrieval.vector_search import search_vector_chunks
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.retrieval import (
    BM25SearchRequest,
    HybridSearchRequest,
    RerankRequest,
    RerankResponse,
    RetrievalResponse,
    VectorSearchRequest,
)


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


@router.post("/hybrid-search", response_model=RetrievalResponse)
def hybrid_search_endpoint(
    payload: HybridSearchRequest,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_scope("query:run")),
) -> RetrievalResponse:
    logger.info(
        f"[RetrievalAPI] Hybrid search tenant={user_context.tenant_id} "
        f"user={user_context.id} limit={payload.limit}"
    )

    response = search_hybrid_chunks(
        db=db,
        user_context=user_context,
        query=payload.query,
        filters=payload.filters,
        limit=payload.limit,
        min_similarity=payload.min_similarity,
    )

    logger.info(
        f"[RetrievalAPI] Hybrid search completed tenant={user_context.tenant_id} "
        f"user={user_context.id} candidates={len(response.candidates)}"
    )
    return response


@router.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(
    payload: RerankRequest,
    user_context: UserContext = Depends(require_scope("query:run")),
) -> RerankResponse:
    logger.info(
        f"[RetrievalAPI] Rerank tenant={user_context.tenant_id} "
        f"user={user_context.id} candidates={len(payload.candidates)} "
        f"top_k={payload.top_k}"
    )

    response = rerank_chunks(
        query=payload.query,
        candidates=payload.candidates,
        top_k=payload.top_k,
    )

    logger.info(
        f"[RetrievalAPI] Rerank completed tenant={user_context.tenant_id} "
        f"user={user_context.id} chunks={len(response.chunks)}"
    )
    return response


@router.post("/vector-search", response_model=RetrievalResponse)
def vector_search_endpoint(
    payload: VectorSearchRequest,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_scope("query:run")),
) -> RetrievalResponse:
    logger.info(
        f"[RetrievalAPI] Vector search tenant={user_context.tenant_id} "
        f"user={user_context.id} limit={payload.limit}"
    )

    response = search_vector_chunks(
        db=db,
        user_context=user_context,
        query=payload.query,
        filters=payload.filters,
        limit=payload.limit,
        min_similarity=payload.min_similarity,
    )

    logger.info(
        f"[RetrievalAPI] Vector search completed tenant={user_context.tenant_id} "
        f"user={user_context.id} candidates={len(response.candidates)}"
    )
    return response
