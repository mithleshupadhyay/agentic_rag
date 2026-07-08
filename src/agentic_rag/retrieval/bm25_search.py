from datetime import datetime
import logging
import time
from typing import Optional
from uuid import UUID

from fastapi import HTTPException

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.monitoring.metrics import (
    RETRIEVAL_LATENCY_SECONDS,
    RETRIEVAL_LIFECYCLE_TOTAL,
    RETRIEVAL_RESULT_TOTAL,
)
from agentic_rag.search.opensearch import OpenSearchClient
from agentic_rag.shared.config import settings
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
        f"user={user_context.id} limit={limit} min_score={settings.bm25_min_score}"
    )
    started_at = time.perf_counter()
    retrieval_strategy = RetrievalStrategy.BM25.value
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
            latency_seconds = time.perf_counter() - started_at
            RETRIEVAL_LIFECYCLE_TOTAL.labels(
                status="workspace_denied",
                retrieval_strategy=retrieval_strategy,
            ).inc()
            RETRIEVAL_LATENCY_SECONDS.labels(
                status="workspace_denied",
                retrieval_strategy=retrieval_strategy,
            ).observe(latency_seconds)
            return RetrievalResponse(
                strategy=RetrievalStrategy.BM25,
                candidates=[],
                latency_ms=int(latency_seconds * 1000),
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
    if filters.metadata:
        for metadata_key, metadata_value in filters.metadata.items():
            clean_metadata_key = metadata_key.strip()
            if not clean_metadata_key or "." in clean_metadata_key:
                logger.warning(
                    f"[Retrieval] Invalid BM25 metadata filter key "
                    f"tenant={user_context.tenant_id} key={metadata_key}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Metadata filter keys must be non-empty top-level field names.",
                )

            if isinstance(metadata_value, bool):
                metadata_filter_value = str(metadata_value).lower()
            elif isinstance(metadata_value, (str, int, float)):
                metadata_filter_value = str(metadata_value)
            else:
                logger.warning(
                    f"[Retrieval] Invalid BM25 metadata filter value "
                    f"tenant={user_context.tenant_id} key={clean_metadata_key} "
                    f"value_type={type(metadata_value).__name__}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Metadata filter values must be strings, numbers, or booleans.",
                )

            document_metadata_field = (
                f"document_metadata.{clean_metadata_key}.keyword"
            )
            chunk_metadata_field = (
                f"chunk_metadata.{clean_metadata_key}.keyword"
            )
            filter_clauses.append(
                {
                    "bool": {
                        "should": [
                            {"term": {document_metadata_field: metadata_filter_value}},
                            {"term": {chunk_metadata_field: metadata_filter_value}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
    if filters.date_range:
        allowed_date_fields = {"created_at", "updated_at"}
        allowed_date_operators = {"gt", "gte", "lt", "lte"}
        for date_field, date_limits in filters.date_range.items():
            if not isinstance(date_field, str):
                logger.warning(
                    f"[Retrieval] Invalid BM25 date range field "
                    f"tenant={user_context.tenant_id} field={date_field}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Date range fields must be created_at or updated_at.",
                )

            clean_date_field = date_field.strip()
            if clean_date_field not in allowed_date_fields:
                logger.warning(
                    f"[Retrieval] Unsupported BM25 date range field "
                    f"tenant={user_context.tenant_id} field={date_field}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Date range fields must be created_at or updated_at.",
                )

            if not isinstance(date_limits, dict) or not date_limits:
                logger.warning(
                    f"[Retrieval] Invalid BM25 date range value "
                    f"tenant={user_context.tenant_id} field={clean_date_field}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Date range values must be non-empty operator maps.",
                )

            range_query = {}
            for date_operator, date_value in date_limits.items():
                if not isinstance(date_operator, str):
                    logger.warning(
                        f"[Retrieval] Invalid BM25 date range operator "
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
                        f"[Retrieval] Unsupported BM25 date range operator "
                        f"tenant={user_context.tenant_id} field={clean_date_field} "
                        f"operator={date_operator}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Date range operators must be gt, gte, lt, or lte.",
                    )

                if not isinstance(date_value, str) or not date_value.strip():
                    logger.warning(
                        f"[Retrieval] Invalid BM25 date range boundary "
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
                    datetime.fromisoformat(normalized_date_value)
                except ValueError:
                    logger.warning(
                        f"[Retrieval] Invalid BM25 date range ISO value "
                        f"tenant={user_context.tenant_id} field={clean_date_field} "
                        f"operator={clean_date_operator}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Date range values must be ISO 8601 strings.",
                    ) from None

                range_query[clean_date_operator] = clean_date_value

            filter_clauses.append({"range": {clean_date_field: range_query}})

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
        candidates: list[CandidateChunk] = []
        skipped_low_score_count = 0
        skipped_invalid_hit_count = 0
        deduplicated_hit_count = 0
        dedupe_positions: dict[str, int] = {}
        for hit in hits:
            if not isinstance(hit, dict):
                skipped_invalid_hit_count += 1
                logger.warning(
                    f"[Retrieval] Skipping invalid BM25 hit tenant={user_context.tenant_id} "
                    f"user={user_context.id} reason=hit_not_object"
                )
                continue

            source = hit.get("_source")
            if not isinstance(source, dict):
                skipped_invalid_hit_count += 1
                logger.warning(
                    f"[Retrieval] Skipping invalid BM25 hit tenant={user_context.tenant_id} "
                    f"user={user_context.id} reason=source_not_object"
                )
                continue

            highlight = hit.get("highlight") or {}
            if not isinstance(highlight, dict):
                highlight = {}
            highlighted_content = highlight.get("content") or []
            if not isinstance(highlighted_content, list):
                highlighted_content = []

            quote = None
            for highlighted_fragment in highlighted_content:
                if isinstance(highlighted_fragment, str) and highlighted_fragment.strip():
                    quote = highlighted_fragment
                    break
            if quote is None:
                quote = source.get("content")

            if not isinstance(quote, str) or not quote.strip():
                skipped_invalid_hit_count += 1
                logger.warning(
                    f"[Retrieval] Skipping invalid BM25 hit tenant={user_context.tenant_id} "
                    f"user={user_context.id} reason=missing_content"
                )
                continue

            try:
                score = float(hit.get("_score") or 0.0)
            except (TypeError, ValueError):
                skipped_invalid_hit_count += 1
                logger.warning(
                    f"[Retrieval] Skipping invalid BM25 hit tenant={user_context.tenant_id} "
                    f"user={user_context.id} reason=invalid_score"
                )
                continue

            if score < settings.bm25_min_score:
                skipped_low_score_count += 1
                continue

            source_document_id = source.get("document_id")
            source_chunk_id = source.get("chunk_id")
            source_section_path = source.get("section_path")
            try:
                document_id = UUID(str(source_document_id))
                chunk_id = UUID(str(source_chunk_id))
            except (TypeError, ValueError):
                skipped_invalid_hit_count += 1
                logger.warning(
                    f"[Retrieval] Skipping invalid BM25 hit tenant={user_context.tenant_id} "
                    f"user={user_context.id} document_id={source_document_id} "
                    f"chunk_id={source_chunk_id} reason=invalid_ids"
                )
                continue

            try:
                candidate = CandidateChunk(
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
                        section_path=source_section_path,
                        quote=quote,
                        score=score,
                    ),
                )
            except (TypeError, ValueError) as e:
                skipped_invalid_hit_count += 1
                logger.warning(
                    f"[Retrieval] Skipping invalid BM25 hit tenant={user_context.tenant_id} "
                    f"user={user_context.id} document_id={source_document_id} "
                    f"chunk_id={source_chunk_id} reason=candidate_validation_failed error={e}"
                )
                continue

            if isinstance(source_section_path, str) and source_section_path.strip():
                dedupe_key = f"{document_id}:section:{source_section_path.strip()}"
            else:
                dedupe_key = f"{document_id}:chunk:{chunk_id}"

            existing_candidate_index = dedupe_positions.get(dedupe_key)
            if existing_candidate_index is None:
                dedupe_positions[dedupe_key] = len(candidates)
                candidates.append(candidate)
            else:
                deduplicated_hit_count += 1
                if candidate.score > candidates[existing_candidate_index].score:
                    candidates[existing_candidate_index] = candidate

        latency_seconds = time.perf_counter() - started_at
        latency_ms = int(latency_seconds * 1000)
        RETRIEVAL_LIFECYCLE_TOTAL.labels(
            status="completed",
            retrieval_strategy=retrieval_strategy,
        ).inc()
        RETRIEVAL_LATENCY_SECONDS.labels(
            status="completed",
            retrieval_strategy=retrieval_strategy,
        ).observe(latency_seconds)
        if candidates:
            RETRIEVAL_RESULT_TOTAL.labels(
                retrieval_strategy=retrieval_strategy,
                result="returned_candidate",
            ).inc(len(candidates))
        if skipped_low_score_count:
            RETRIEVAL_RESULT_TOTAL.labels(
                retrieval_strategy=retrieval_strategy,
                result="skipped_low_score_hit",
            ).inc(skipped_low_score_count)
        if skipped_invalid_hit_count:
            RETRIEVAL_RESULT_TOTAL.labels(
                retrieval_strategy=retrieval_strategy,
                result="skipped_invalid_hit",
            ).inc(skipped_invalid_hit_count)
        if deduplicated_hit_count:
            RETRIEVAL_RESULT_TOTAL.labels(
                retrieval_strategy=retrieval_strategy,
                result="deduplicated_hit",
            ).inc(deduplicated_hit_count)

        logger.info(
            f"[Retrieval] BM25 search completed tenant={user_context.tenant_id} "
            f"user={user_context.id} candidates={len(candidates)} "
            f"skipped_low_score={skipped_low_score_count} "
            f"skipped_invalid_hits={skipped_invalid_hit_count} "
            f"deduplicated_hits={deduplicated_hit_count} "
            f"min_score={settings.bm25_min_score} latency_ms={latency_ms}"
        )
        return RetrievalResponse(
            strategy=RetrievalStrategy.BM25,
            candidates=candidates,
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_seconds = time.perf_counter() - started_at
        RETRIEVAL_LIFECYCLE_TOTAL.labels(
            status="failed",
            retrieval_strategy=retrieval_strategy,
        ).inc()
        RETRIEVAL_LATENCY_SECONDS.labels(
            status="failed",
            retrieval_strategy=retrieval_strategy,
        ).observe(latency_seconds)
        logger.exception(
            f"[Retrieval] BM25 search failed tenant={user_context.tenant_id} "
            f"user={user_context.id} latency_ms={int(latency_seconds * 1000)}: {e}"
        )
        raise

    finally:
        if owns_client:
            search_client.close()
