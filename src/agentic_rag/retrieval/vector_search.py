import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.llm.gateway import generate_embeddings
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.embeddings import search_similar_chunks_by_embedding
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.llm import EmbeddingRequest, EmbeddingResponse
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RetrievalFilters,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTool,
)


logger = logging.getLogger(__name__)


def search_vector_chunks(
    db: Session,
    user_context: UserContext,
    query: str,
    filters: Optional[RetrievalFilters] = None,
    limit: int = 20,
    min_similarity: float = 0.0,
    embedding_client: Callable[[EmbeddingRequest], EmbeddingResponse] = generate_embeddings,
) -> RetrievalResponse:
    logger.info(
        f"[Retrieval] Vector search started tenant={user_context.tenant_id} "
        f"user={user_context.id} limit={limit} model={settings.embedding_model_name}"
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
    if user_context.workspace_id and filters.workspace_id:
        if user_context.workspace_id != filters.workspace_id:
            logger.warning(
                f"[Retrieval] Workspace filter denied "
                f"user_workspace={user_context.workspace_id} "
                f"requested_workspace={filters.workspace_id}"
            )
            return RetrievalResponse(
                strategy=RetrievalStrategy.VECTOR,
                candidates=[],
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )

    source_types = []
    for source_type in filters.source_types:
        clean_source_type = source_type.strip()
        if not clean_source_type:
            logger.warning(
                f"[Retrieval] Invalid vector source type filter "
                f"tenant={user_context.tenant_id} user={user_context.id}"
            )
            raise HTTPException(
                status_code=400,
                detail="Source type filters must be non-empty strings.",
            )
        source_types.append(clean_source_type)

    tags = []
    for tag in filters.tags:
        clean_tag = tag.strip()
        if not clean_tag:
            logger.warning(
                f"[Retrieval] Invalid vector tag filter "
                f"tenant={user_context.tenant_id} user={user_context.id}"
            )
            raise HTTPException(
                status_code=400,
                detail="Tag filters must be non-empty strings.",
            )
        tags.append(clean_tag)

    metadata_filters: dict[str, object] = {}
    if filters.metadata:
        for metadata_key, metadata_value in filters.metadata.items():
            clean_metadata_key = metadata_key.strip()
            if not clean_metadata_key or "." in clean_metadata_key:
                logger.warning(
                    f"[Retrieval] Invalid vector metadata filter key "
                    f"tenant={user_context.tenant_id} key={metadata_key}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Metadata filter keys must be non-empty top-level field names.",
                )

            if (
                isinstance(metadata_value, str)
                or isinstance(metadata_value, bool)
                or isinstance(metadata_value, int)
                or isinstance(metadata_value, float)
            ):
                metadata_filters[clean_metadata_key] = metadata_value
            else:
                logger.warning(
                    f"[Retrieval] Invalid vector metadata filter value "
                    f"tenant={user_context.tenant_id} key={clean_metadata_key} "
                    f"value_type={type(metadata_value).__name__}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Metadata filter values must be strings, numbers, or booleans.",
                )

    date_range_filters: dict[str, dict[str, datetime]] = {}
    if filters.date_range:
        allowed_date_fields = {"created_at", "updated_at"}
        allowed_date_operators = {"gt", "gte", "lt", "lte"}
        for date_field, date_limits in filters.date_range.items():
            if not isinstance(date_field, str):
                logger.warning(
                    f"[Retrieval] Invalid vector date range field "
                    f"tenant={user_context.tenant_id} field={date_field}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Date range fields must be created_at or updated_at.",
                )

            clean_date_field = date_field.strip()
            if clean_date_field not in allowed_date_fields:
                logger.warning(
                    f"[Retrieval] Unsupported vector date range field "
                    f"tenant={user_context.tenant_id} field={date_field}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Date range fields must be created_at or updated_at.",
                )

            if not isinstance(date_limits, dict) or not date_limits:
                logger.warning(
                    f"[Retrieval] Invalid vector date range value "
                    f"tenant={user_context.tenant_id} field={clean_date_field}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Date range values must be non-empty operator maps.",
                )

            date_range_filters[clean_date_field] = {}
            for date_operator, date_value in date_limits.items():
                if not isinstance(date_operator, str):
                    logger.warning(
                        f"[Retrieval] Invalid vector date range operator "
                        f"tenant={user_context.tenant_id} field={clean_date_field} "
                        f"operator={date_operator}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Date range operators must be gt, gte, lt, or lte.",
                    )

                clean_date_operator = date_operator.strip()
                if clean_date_operator not in allowed_date_operators:
                    logger.warning(
                        f"[Retrieval] Unsupported vector date range operator "
                        f"tenant={user_context.tenant_id} field={clean_date_field} "
                        f"operator={date_operator}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Date range operators must be gt, gte, lt, or lte.",
                    )

                if not isinstance(date_value, str) or not date_value.strip():
                    logger.warning(
                        f"[Retrieval] Invalid vector date range boundary "
                        f"tenant={user_context.tenant_id} field={clean_date_field} "
                        f"operator={clean_date_operator}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Date range values must be ISO 8601 strings.",
                    )

                clean_date_value = date_value.strip()
                normalized_date_value = clean_date_value
                if normalized_date_value.endswith("Z"):
                    normalized_date_value = normalized_date_value[:-1] + "+00:00"
                try:
                    parsed_date_value = datetime.fromisoformat(normalized_date_value)
                except ValueError:
                    logger.warning(
                        f"[Retrieval] Invalid vector date range ISO value "
                        f"tenant={user_context.tenant_id} field={clean_date_field} "
                        f"operator={clean_date_operator}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Date range values must be ISO 8601 strings.",
                    ) from None

                date_range_filters[clean_date_field][clean_date_operator] = (
                    parsed_date_value
                )

    embedding_response = embedding_client(
        EmbeddingRequest(
            auth=AuthContext(
                user_id=user_context.id,
                tenant_id=user_context.tenant_id,
                workspace_id=user_context.workspace_id,
                roles=user_context.roles or [],
                group_ids=user_context.group_ids or [],
                scopes=user_context.scopes or [],
                acl_version=user_context.acl_version,
            ),
            texts=[query_text],
            provider=settings.embedding_provider,
            model=settings.embedding_model_name,
            timeout_seconds=settings.embedding_timeout_seconds,
            metadata={"retrieval_tool": RetrievalTool.VECTOR_SEARCH.value},
        )
    )
    if len(embedding_response.embeddings) != 1:
        raise RuntimeError(
            "Embedding response count did not match vector search query count."
        )

    query_embedding = embedding_response.embeddings[0]
    if embedding_response.dimension != settings.embedding_dimension:
        logger.warning(
            f"[Retrieval] Vector search embedding dimension mismatch "
            f"tenant={user_context.tenant_id} model={embedding_response.model} "
            f"expected={settings.embedding_dimension} "
            f"actual={embedding_response.dimension}"
        )
        raise RuntimeError(
            "Embedding dimension does not match configured vector dimension "
            f"({embedding_response.dimension}!={settings.embedding_dimension})."
        )

    vector_search_limit = min(200, max(limit, limit * 3))
    search_results = search_similar_chunks_by_embedding(
        db=db,
        tenant_id=user_context.tenant_id,
        query_embedding=query_embedding,
        embedding_model=embedding_response.model,
        vector_version=settings.embedding_vector_version,
        embedding_dimension=settings.embedding_dimension,
        limit=vector_search_limit,
        min_similarity=min_similarity,
        workspace_id=user_context.workspace_id or filters.workspace_id,
        document_ids=filters.document_ids,
        source_types=source_types,
        metadata_filters=metadata_filters,
        tags=tags,
        date_range=date_range_filters,
        user_context=user_context,
    )

    deduped_candidates: dict[str, CandidateChunk] = {}
    dedupe_order: dict[str, int] = {}
    dedupe_distances: dict[str, float] = {}
    for result_index, result in enumerate(search_results):
        chunk = result.chunk
        document = chunk.document
        score = max(result.similarity, 0.0)
        candidate = CandidateChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            score=score,
            source=RetrievalTool.VECTOR_SEARCH,
            metadata={
                "workspace_id": chunk.workspace_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "classification_level": chunk.classification_level,
                "embedding_provider": embedding_response.provider,
                "embedding_model": embedding_response.model,
                "embedding_dimension": embedding_response.dimension,
                "vector_version": settings.embedding_vector_version,
                "distance": result.distance,
                "file_name": document.file_name if document else None,
                "source_type": document.source_type if document else None,
            },
            citation=Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                title=document.title if document else None,
                source_uri=document.source_uri if document else None,
                page_number=chunk.page_number,
                section_path=chunk.section_path,
                quote=chunk.content,
                score=score,
            ),
        )
        chunk_key = str(chunk.id)
        existing_candidate = deduped_candidates.get(chunk_key)
        if not existing_candidate:
            dedupe_order[chunk_key] = result_index
            dedupe_distances[chunk_key] = result.distance
            deduped_candidates[chunk_key] = candidate
            continue

        existing_distance_value = dedupe_distances.get(chunk_key, 1.0)
        candidate_distance_value = result.distance
        if (
            candidate.score > existing_candidate.score
            or (
                candidate.score == existing_candidate.score
                and candidate_distance_value < existing_distance_value
            )
        ):
            dedupe_distances[chunk_key] = result.distance
            deduped_candidates[chunk_key] = candidate

    candidates = list(deduped_candidates.values())
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            dedupe_distances.get(str(candidate.chunk_id), 1.0),
            dedupe_order[str(candidate.chunk_id)],
        )
    )
    candidates = candidates[:limit]

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        f"[Retrieval] Vector search completed tenant={user_context.tenant_id} "
        f"user={user_context.id} raw_candidates={len(search_results)} "
        f"deduplicated_candidates={len(deduped_candidates)} "
        f"candidates={len(candidates)} latency_ms={latency_ms}"
    )
    return RetrievalResponse(
        strategy=RetrievalStrategy.VECTOR,
        candidates=candidates,
        latency_ms=latency_ms,
    )
