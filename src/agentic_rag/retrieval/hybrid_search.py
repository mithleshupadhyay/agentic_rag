import logging
import time
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.retrieval.reranker import rerank_chunks
from agentic_rag.retrieval.vector_search import search_vector_chunks
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RetrievalFilters,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTool,
)


logger = logging.getLogger(__name__)


def search_hybrid_chunks(
    db: Session,
    user_context: UserContext,
    query: str,
    filters: Optional[RetrievalFilters] = None,
    limit: int = 20,
    min_similarity: float = 0.0,
) -> RetrievalResponse:
    logger.info(
        f"[Retrieval] Hybrid search started tenant={user_context.tenant_id} "
        f"user={user_context.id} limit={limit}"
    )
    started_at = time.perf_counter()
    query_text = query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Retrieval query is required.")

    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=400,
            detail="Retrieval limit must be between 1 and 200.",
        )

    if min_similarity < 0 or min_similarity > 1:
        raise HTTPException(
            status_code=400,
            detail="Minimum similarity must be between 0 and 1.",
        )

    filters = filters or RetrievalFilters()
    if filters.source_types or filters.tags or filters.metadata or filters.date_range:
        logger.warning(
            f"[Retrieval] Hybrid search rejected unsupported filters "
            f"tenant={user_context.tenant_id} user={user_context.id}"
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Hybrid search currently supports workspace_id and document_ids "
                "filters only."
            ),
        )

    bm25_response = search_bm25_chunks(
        user_context=user_context,
        query=query_text,
        filters=filters,
        limit=limit,
    )
    vector_response = search_vector_chunks(
        db=db,
        user_context=user_context,
        query=query_text,
        filters=filters,
        limit=limit,
        min_similarity=min_similarity,
    )

    merged_candidates: dict[str, CandidateChunk] = {}
    for index, candidate in enumerate(bm25_response.candidates):
        rank = index + 1
        rank_score = 0.5 / rank
        metadata = {
            **candidate.metadata,
            "retrieval_sources": [RetrievalTool.BM25_SEARCH.value],
            "bm25_score": candidate.score,
            "bm25_rank": rank,
            "bm25_rank_score": rank_score,
            "vector_score": None,
            "vector_rank": None,
            "vector_rank_score": 0.0,
            "hybrid_rank_score": rank_score,
        }
        citation = candidate.citation
        merged_candidates[str(candidate.chunk_id)] = CandidateChunk(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            content=candidate.content,
            score=rank_score,
            source=RetrievalStrategy.HYBRID.value,
            metadata=metadata,
            citation=Citation(
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                title=citation.title if citation else None,
                source_uri=citation.source_uri if citation else None,
                page_number=citation.page_number if citation else None,
                section_path=citation.section_path if citation else None,
                quote=citation.quote if citation else candidate.content,
                score=rank_score,
            ),
        )

    for index, candidate in enumerate(vector_response.candidates):
        rank = index + 1
        rank_score = 0.5 / rank
        chunk_key = str(candidate.chunk_id)
        existing_candidate = merged_candidates.get(chunk_key)
        if existing_candidate:
            sources = existing_candidate.metadata.get("retrieval_sources")
            retrieval_sources = sources if isinstance(sources, list) else []
            if RetrievalTool.VECTOR_SEARCH.value not in retrieval_sources:
                retrieval_sources.append(RetrievalTool.VECTOR_SEARCH.value)

            existing_candidate.score += rank_score
            existing_candidate.metadata["retrieval_sources"] = retrieval_sources
            existing_candidate.metadata["vector_score"] = candidate.score
            existing_candidate.metadata["vector_rank"] = rank
            existing_candidate.metadata["vector_rank_score"] = rank_score
            existing_candidate.metadata["hybrid_rank_score"] = existing_candidate.score

            if not existing_candidate.content and candidate.content:
                existing_candidate.content = candidate.content

            if existing_candidate.citation:
                existing_candidate.citation.score = existing_candidate.score
                if not existing_candidate.citation.quote and candidate.citation:
                    existing_candidate.citation.quote = candidate.citation.quote
            continue

        metadata = {
            **candidate.metadata,
            "retrieval_sources": [RetrievalTool.VECTOR_SEARCH.value],
            "bm25_score": None,
            "bm25_rank": None,
            "bm25_rank_score": 0.0,
            "vector_score": candidate.score,
            "vector_rank": rank,
            "vector_rank_score": rank_score,
            "hybrid_rank_score": rank_score,
        }
        citation = candidate.citation
        merged_candidates[chunk_key] = CandidateChunk(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            content=candidate.content,
            score=rank_score,
            source=RetrievalStrategy.HYBRID.value,
            metadata=metadata,
            citation=Citation(
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                title=citation.title if citation else None,
                source_uri=citation.source_uri if citation else None,
                page_number=citation.page_number if citation else None,
                section_path=citation.section_path if citation else None,
                quote=citation.quote if citation else candidate.content,
                score=rank_score,
            ),
        )

    candidates = list(merged_candidates.values())
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.metadata.get("bm25_rank") or limit + 1,
            candidate.metadata.get("vector_rank") or limit + 1,
            str(candidate.chunk_id),
        )
    )
    candidates = candidates[:limit]
    for candidate in candidates:
        candidate.metadata["pre_rerank_score"] = candidate.score
        candidate.metadata["pre_rerank_source"] = candidate.source

    rerank_response = rerank_chunks(
        query=query_text,
        candidates=candidates,
        top_k=limit,
    )

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        f"[Retrieval] Hybrid search completed tenant={user_context.tenant_id} "
        f"user={user_context.id} candidates={len(rerank_response.chunks)} "
        f"bm25_candidates={len(bm25_response.candidates)} "
        f"vector_candidates={len(vector_response.candidates)} latency_ms={latency_ms}"
    )
    return RetrievalResponse(
        strategy=RetrievalStrategy.HYBRID,
        candidates=rerank_response.chunks,
        latency_ms=latency_ms,
    )
