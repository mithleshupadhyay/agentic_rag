import logging
import re
import time
from collections.abc import Sequence

from fastapi import HTTPException

from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RerankResponse,
    RetrievalTool,
)


logger = logging.getLogger(__name__)


def rerank_chunks(
    query: str,
    candidates: Sequence[CandidateChunk],
    top_k: int = 12,
) -> RerankResponse:
    logger.info(
        f"[Retrieval] Rerank started candidates={len(candidates)} top_k={top_k}"
    )
    started_at = time.perf_counter()
    query_text = query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Rerank query is required.")

    if top_k < 1 or top_k > 100:
        raise HTTPException(
            status_code=400,
            detail="Rerank top_k must be between 1 and 100.",
        )

    query_terms = set(re.findall(r"[a-z0-9]+", query_text.lower()))
    scored_candidates: list[tuple[float, float, int, str, CandidateChunk]] = []
    for index, candidate in enumerate(candidates):
        searchable_parts = [candidate.content or ""]
        if candidate.citation:
            searchable_parts.append(candidate.citation.title or "")
            searchable_parts.append(candidate.citation.quote or "")
            searchable_parts.append(candidate.citation.section_path or "")

        for metadata_value in candidate.metadata.values():
            if isinstance(metadata_value, str):
                searchable_parts.append(metadata_value)
            elif isinstance(metadata_value, list):
                searchable_parts.extend(
                    item for item in metadata_value if isinstance(item, str)
                )

        candidate_terms = set(
            re.findall(r"[a-z0-9]+", " ".join(searchable_parts).lower())
        )
        matched_terms = query_terms.intersection(candidate_terms)
        query_coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0
        candidate_density = (
            len(matched_terms) / len(candidate_terms) if candidate_terms else 0.0
        )
        retrieval_sources = candidate.metadata.get("retrieval_sources")
        source_count = len(retrieval_sources) if isinstance(retrieval_sources, list) else 1
        source_signal = min(source_count, 2) / 2
        rerank_score = round(
            (query_coverage * 0.75)
            + (candidate_density * 0.15)
            + (source_signal * 0.10),
            6,
        )
        scored_candidates.append(
            (
                rerank_score,
                candidate.score,
                index,
                str(candidate.chunk_id),
                candidate,
            )
        )

    scored_candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
            item[3],
        )
    )

    reranked_chunks: list[CandidateChunk] = []
    for rank, (rerank_score, original_score, _index, _chunk_key, candidate) in enumerate(
        scored_candidates[:top_k],
        start=1,
    ):
        reranked_candidate = candidate.model_copy(deep=True)
        reranked_candidate.score = rerank_score
        reranked_candidate.source = RetrievalTool.RERANK.value
        reranked_candidate.metadata = {
            **candidate.metadata,
            "original_score": original_score,
            "original_source": (
                candidate.source.value
                if isinstance(candidate.source, RetrievalTool)
                else candidate.source
            ),
            "rerank_score": rerank_score,
            "rerank_rank": rank,
        }
        if reranked_candidate.citation:
            reranked_candidate.citation.score = rerank_score
        reranked_chunks.append(reranked_candidate)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        f"[Retrieval] Rerank completed candidates={len(candidates)} "
        f"returned={len(reranked_chunks)} latency_ms={latency_ms}"
    )
    return RerankResponse(chunks=reranked_chunks, latency_ms=latency_ms)
