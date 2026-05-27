import logging
import time
from collections.abc import Callable
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
    if filters.source_types or filters.tags or filters.metadata or filters.date_range:
        logger.warning(
            f"[Retrieval] Vector search rejected unsupported filters "
            f"tenant={user_context.tenant_id} user={user_context.id}"
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Vector search currently supports workspace_id and document_ids "
                "filters only."
            ),
        )

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

    search_results = search_similar_chunks_by_embedding(
        db=db,
        tenant_id=user_context.tenant_id,
        query_embedding=query_embedding,
        embedding_model=embedding_response.model,
        vector_version=settings.embedding_vector_version,
        embedding_dimension=settings.embedding_dimension,
        limit=limit,
        min_similarity=min_similarity,
        workspace_id=user_context.workspace_id or filters.workspace_id,
        document_ids=filters.document_ids,
        user_context=user_context,
    )

    candidates: list[CandidateChunk] = []
    for result in search_results:
        chunk = result.chunk
        document = chunk.document
        score = max(result.similarity, 0.0)
        candidates.append(
            CandidateChunk(
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
        )

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        f"[Retrieval] Vector search completed tenant={user_context.tenant_id} "
        f"user={user_context.id} candidates={len(candidates)} "
        f"latency_ms={latency_ms}"
    )
    return RetrievalResponse(
        strategy=RetrievalStrategy.VECTOR,
        candidates=candidates,
        latency_ms=latency_ms,
    )
