import logging
import time
from typing import Optional
from uuid import UUID

from fastapi import HTTPException

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.search.opensearch import OpenSearchClient
from agentic_rag.shared.schemas.auth import Visibility
from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    RetrievalFilters,
    RetrievalResponse,
    RetrievalStrategy,
    RetrievalTool,
)


logger = logging.getLogger(__name__)


def search_bm25_chunks(
    user_context: UserContext,
    query: str,
    filters: Optional[RetrievalFilters] = None,
    limit: int = 20,
    search_client: Optional[OpenSearchClient] = None,
) -> RetrievalResponse:
    logger.info(
        f"[Retrieval] BM25 search started tenant={user_context.tenant_id} "
        f"user={user_context.id} limit={limit}"
    )
    started_at = time.perf_counter()
    query_text = query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Retrieval query is required.")

    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Retrieval limit must be between 1 and 200.")

    filters = filters or RetrievalFilters()
    if user_context.workspace_id and filters.workspace_id:
        if user_context.workspace_id != filters.workspace_id:
            logger.warning(
                f"[Retrieval] Workspace filter denied user_workspace={user_context.workspace_id} "
                f"requested_workspace={filters.workspace_id}"
            )
            return RetrievalResponse(
                strategy=RetrievalStrategy.BM25,
                candidates=[],
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )

    user_roles = user_context.roles or []
    user_groups = user_context.group_ids or []
    workspace_id = user_context.workspace_id or filters.workspace_id

    filter_clauses: list[dict] = [
        {"term": {"tenant_id": user_context.tenant_id}},
        {"range": {"acl_version": {"lte": user_context.acl_version}}},
    ]
    if workspace_id:
        filter_clauses.append({"term": {"workspace_id": workspace_id}})
    if filters.document_ids:
        filter_clauses.append(
            {"terms": {"document_id": [str(document_id) for document_id in filters.document_ids]}}
        )
    if filters.source_types:
        filter_clauses.append({"terms": {"source_type": filters.source_types}})

    must_not_clauses: list[dict] = [
        {"term": {"denied_user_ids": user_context.id}},
    ]
    if user_groups:
        must_not_clauses.append({"terms": {"denied_group_ids": user_groups}})

    allowed_clauses: list[dict] = []
    if "admin" in user_roles:
        allowed_clauses.append({"match_all": {}})
    else:
        allowed_clauses.extend(
            [
                {"term": {"owner_user_id": user_context.id}},
                {"term": {"allowed_user_ids": user_context.id}},
                {"terms": {"visibility": [Visibility.PUBLIC.value, Visibility.TENANT.value]}},
            ]
        )
        if user_groups:
            allowed_clauses.append({"terms": {"allowed_group_ids": user_groups}})
        if user_roles:
            allowed_clauses.append({"terms": {"allowed_roles": user_roles}})

    filter_clauses.append(
        {
            "bool": {
                "should": allowed_clauses,
                "minimum_should_match": 1,
            }
        }
    )

    search_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["content^3", "document_title^2", "file_name"],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": filter_clauses,
                "must_not": must_not_clauses,
            }
        },
        "size": limit,
        "_source": [
            "tenant_id",
            "workspace_id",
            "document_id",
            "chunk_id",
            "chunk_index",
            "content",
            "token_count",
            "section_path",
            "page_number",
            "start_offset",
            "end_offset",
            "document_title",
            "file_name",
            "source_type",
            "source_uri",
            "classification_level",
        ],
        "highlight": {
            "fields": {
                "content": {
                    "fragment_size": 240,
                    "number_of_fragments": 1,
                }
            }
        },
    }

    owns_client = search_client is None
    search_client = search_client or OpenSearchClient()

    try:
        hits = search_client.search_chunks_bm25(search_body)
        candidates = []
        for hit in hits:
            source = hit.get("_source", {})
            highlight = hit.get("highlight", {})
            highlighted_content = highlight.get("content") or []
            quote = highlighted_content[0] if highlighted_content else source.get("content")
            score = float(hit.get("_score") or 0.0)
            document_id = UUID(source["document_id"])
            chunk_id = UUID(source["chunk_id"])

            candidates.append(
                CandidateChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=quote,
                    score=score,
                    source=RetrievalTool.BM25_SEARCH,
                    metadata={
                        "workspace_id": source.get("workspace_id"),
                        "chunk_index": source.get("chunk_index"),
                        "token_count": source.get("token_count"),
                        "start_offset": source.get("start_offset"),
                        "end_offset": source.get("end_offset"),
                        "file_name": source.get("file_name"),
                        "source_type": source.get("source_type"),
                        "classification_level": source.get("classification_level"),
                    },
                    citation=Citation(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        title=source.get("document_title"),
                        source_uri=source.get("source_uri"),
                        page_number=source.get("page_number"),
                        section_path=source.get("section_path"),
                        quote=quote,
                        score=score,
                    ),
                )
            )

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            f"[Retrieval] BM25 search completed tenant={user_context.tenant_id} "
            f"user={user_context.id} candidates={len(candidates)} latency_ms={latency_ms}"
        )
        return RetrievalResponse(
            strategy=RetrievalStrategy.BM25,
            candidates=candidates,
            latency_ms=latency_ms,
        )

    finally:
        if owns_client:
            search_client.close()
