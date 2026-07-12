import hashlib
import json
import logging
import re
import time
from uuid import UUID, uuid4

from fastapi import HTTPException
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.llm.gateway import generate_chat_completion
from agentic_rag.query.answer_verifier import verify_answer_support
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.retrieval.context_builder import build_context
from agentic_rag.retrieval.hybrid_search import search_hybrid_chunks
from agentic_rag.retrieval.vector_search import search_vector_chunks
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.query_runs import (
    create_query_run,
    mark_query_run_completed,
    mark_query_run_failed,
)
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.llm import ChatCompletionRequest, LLMMessage
from agentic_rag.shared.schemas.query import (
    AnswerVerificationStatus,
    QueryCacheLookupStatus,
    QueryCacheWriteStatus,
    QueryRequest,
    QueryResponse,
)
from agentic_rag.shared.schemas.retrieval import (
    CandidateChunk,
    ContextBuildRequest,
    ContextChunk,
    RetrievalStrategy,
)


logger = logging.getLogger(__name__)


def build_extractive_answer(
    query_text: str,
    context_chunks: list[ContextChunk],
) -> str:
    query_terms = set(re.findall(r"[a-z0-9][a-z0-9_+.#-]{2,}", query_text.lower()))

    passage_candidates: list[tuple[int, int, int, str, str]] = []
    for context_index, context_chunk in enumerate(context_chunks, start=1):
        content = re.sub(r"\s+", " ", context_chunk.content).strip()
        if not content:
            continue

        raw_segments = re.split(r"(?<=[.!?])\s+|\s+•\s+|\n+", content)
        title = context_chunk.citation.title or "selected document"

        for segment_index, raw_segment in enumerate(raw_segments):
            segment = raw_segment.strip(" \t\r\n,;:-•")
            if len(segment) < 12:
                continue

            segment_is_complete = segment.endswith((".", "!", "?"))
            segment_terms = set(
                re.findall(r"[a-z0-9][a-z0-9_+.#-]*", segment.lower())
            )
            overlap_count = len(query_terms & segment_terms)

            if len(segment) > 380:
                segment_is_complete = False
                lower_segment = segment.lower()
                term_positions = [
                    lower_segment.find(term)
                    for term in query_terms
                    if lower_segment.find(term) >= 0
                ]
                first_match = min(term_positions) if term_positions else 0
                start = max(0, first_match - 90)
                end = min(len(segment), first_match + 290)

                if start > 0:
                    next_space = segment.find(" ", start)
                    if next_space >= 0 and next_space < end:
                        start = next_space + 1

                sentence_end = -1
                for marker in (". ", "! ", "? "):
                    marker_index = segment.find(marker, end)
                    if marker_index >= 0 and marker_index <= min(
                        len(segment),
                        end + 80,
                    ):
                        sentence_end = marker_index + 1
                        break
                if sentence_end >= 0:
                    end = sentence_end
                    segment_is_complete = True
                elif end < len(segment):
                    previous_space = segment.rfind(" ", start, end)
                    if previous_space > start:
                        end = previous_space

                segment = segment[start:end].strip(" \t\r\n,;:-•")

            if not segment.endswith((".", "!", "?")):
                segment = f"{segment.rstrip(',;:')}{'' if segment_is_complete else '...'}"

            passage_candidates.append(
                (
                    -context_index,
                    overlap_count,
                    -segment_index,
                    title,
                    f"{segment} [{context_index}]",
                )
            )

    if not passage_candidates:
        for context_index, context_chunk in enumerate(context_chunks, start=1):
            content = re.sub(r"\s+", " ", context_chunk.content).strip()
            if not content:
                continue

            title = context_chunk.citation.title or "selected document"
            if len(content) > 380:
                content = content[:380].rsplit(" ", 1)[0].strip()
            content = content.strip(" \t\r\n,;:-•")
            if not content:
                continue
            if not content.endswith((".", "!", "?")):
                content = f"{content.rstrip(',;:')}..."

            passage_candidates.append(
                (
                    -context_index,
                    0,
                    0,
                    title,
                    f"{content} [{context_index}]",
                )
            )
            if len(passage_candidates) >= 4:
                break

    if not passage_candidates:
        return (
            "I could not find relevant authorized context for this query. "
            "Try a more specific question or choose another indexed document."
        )

    passage_candidates.sort(reverse=True)
    selected_passages: list[tuple[str, str]] = []
    seen_passages: set[str] = set()
    for _, _, _, title, passage in passage_candidates:
        normalized_passage = passage.lower()
        if normalized_passage in seen_passages:
            continue
        selected_passages.append((title, passage))
        seen_passages.add(normalized_passage)
        if len(selected_passages) >= 4:
            break

    answer_lines = ["I found these relevant excerpts in the selected documents:"]
    for title, passage in selected_passages:
        answer_lines.append(f"- {passage} Source: {title}.")

    return "\n".join(answer_lines)


def run_bm25_query(
    user_context: UserContext,
    request: QueryRequest,
    db: Session | None = None,
    request_id: str | None = None,
    agent_run_id: UUID | None = None,
) -> QueryResponse:
    retrieval_strategy = request.retrieval_strategy
    logger.info(
        f"[Query] Query started tenant={user_context.tenant_id} "
        f"user={user_context.id} request_id={request_id} "
        f"retrieval_strategy={retrieval_strategy.value} "
        f"retrieval_limit={request.retrieval_limit}"
    )
    started_at = time.perf_counter()

    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")

    supported_strategies = {
        RetrievalStrategy.BM25,
        RetrievalStrategy.VECTOR,
        RetrievalStrategy.HYBRID,
    }
    if retrieval_strategy not in supported_strategies:
        raise HTTPException(
            status_code=400,
            detail="Query retrieval_strategy must be bm25, vector, or hybrid.",
        )

    if retrieval_strategy in {RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID}:
        if db is None:
            logger.warning(
                f"[Query] Query rejected because {retrieval_strategy.value} retrieval "
                f"requires a database session tenant={user_context.tenant_id} "
                f"user={user_context.id} request_id={request_id}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"{retrieval_strategy.value} query retrieval requires a database session.",
            )

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

    # Use a caller-provided id when streaming announced the run first.
    agent_run_id = agent_run_id or uuid4()
    query_run = None
    if db is not None:
        query_run = create_query_run(
            user_context=user_context,
            db=db,
            request=request,
            agent_run_id=agent_run_id,
            request_id=request_id,
        )

    try:
        redis_client = None
        query_cache_key = None
        cache_lookup_status = QueryCacheLookupStatus.DISABLED
        cache_write_status = QueryCacheWriteStatus.DISABLED
        if settings.query_cache_enabled:
            cache_lookup_status = QueryCacheLookupStatus.MISS
            cache_write_status = QueryCacheWriteStatus.NOT_ATTEMPTED
            try:
                # Build an authorization-scoped cache key.
                query_cache_payload = {
                    "tenant_id": user_context.tenant_id,
                    "workspace_id": filters.workspace_id or user_context.workspace_id,
                    "user_id": user_context.id,
                    "roles": sorted(user_context.roles or []),
                    "group_ids": sorted(user_context.group_ids or []),
                    "scopes": sorted(user_context.scopes or []),
                    "acl_version": user_context.acl_version,
                    "query": query_text,
                    "filters": filters.model_dump(mode="json"),
                    "retrieval_limit": request.retrieval_limit,
                    "max_context_chunks": request.max_context_chunks,
                    "max_context_tokens": request.max_context_tokens,
                    "retrieval_strategy": retrieval_strategy.value,
                    "llm_synthesis_enabled": settings.llm_synthesis_enabled,
                    "llm_provider": settings.llm_provider,
                    "llm_model": settings.default_llm_model,
                }
                query_cache_json = json.dumps(
                    query_cache_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                query_cache_hash = hashlib.sha256(
                    query_cache_json.encode("utf-8")
                ).hexdigest()
                query_cache_key = (
                    f"{settings.query_cache_key_prefix}:{retrieval_strategy.value}:"
                    f"{query_cache_hash}"
                )
                redis_client = Redis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                )
                cached_payload = redis_client.get(query_cache_key)
                if cached_payload:
                    # Return the cached answer under the current run id.
                    cached_text = (
                        cached_payload.decode("utf-8")
                        if isinstance(cached_payload, bytes)
                        else str(cached_payload)
                    )
                    cached_data = json.loads(cached_text)
                    cached_data["agent_run_id"] = str(agent_run_id)
                    cached_data["latency_ms"] = max(
                        0,
                        int((time.perf_counter() - started_at) * 1000),
                    )
                    cached_data["cache_lookup_status"] = (
                        QueryCacheLookupStatus.HIT.value
                    )
                    cached_data["cache_write_status"] = (
                        QueryCacheWriteStatus.NOT_ATTEMPTED.value
                    )
                    cached_data["cache_ttl_seconds"] = None
                    response = QueryResponse.model_validate(cached_data)
                    if db is not None and query_run is not None:
                        mark_query_run_completed(
                            db=db,
                            query_run=query_run,
                            response=response,
                        )
                    logger.info(
                        f"[Query] Query cache hit tenant={user_context.tenant_id} "
                        f"user={user_context.id} request_id={request_id} "
                        f"retrieval_strategy={retrieval_strategy.value} "
                        f"cache_key_hash={query_cache_hash}"
                    )
                    return response

                logger.info(
                    f"[Query] Query cache miss tenant={user_context.tenant_id} "
                    f"user={user_context.id} request_id={request_id} "
                    f"retrieval_strategy={retrieval_strategy.value} "
                    f"cache_key_hash={query_cache_hash}"
                )
            except (RedisError, TypeError, ValueError) as e:
                logger.warning(
                    f"[Query] Query cache skipped tenant={user_context.tenant_id} "
                    f"user={user_context.id} request_id={request_id} "
                    f"retrieval_strategy={retrieval_strategy.value} "
                    f"error_type={type(e).__name__}"
                )
                cache_lookup_status = QueryCacheLookupStatus.SKIPPED
                cache_write_status = QueryCacheWriteStatus.SKIPPED
                redis_client = None
                query_cache_key = None

        if retrieval_strategy == RetrievalStrategy.BM25:
            retrieval_response = search_bm25_chunks(
                user_context=user_context,
                query=query_text,
                filters=filters,
                limit=request.retrieval_limit,
            )
        elif retrieval_strategy == RetrievalStrategy.VECTOR:
            assert db is not None
            retrieval_response = search_vector_chunks(
                db=db,
                user_context=user_context,
                query=query_text,
                filters=filters,
                limit=request.retrieval_limit,
            )
        else:
            assert db is not None
            retrieval_response = search_hybrid_chunks(
                db=db,
                user_context=user_context,
                query=query_text,
                filters=filters,
                limit=request.retrieval_limit,
            )

        context_candidates = retrieval_response.candidates
        if len(filters.document_ids) > 1 and retrieval_response.candidates:
            query_terms = set(re.findall(r"[a-z0-9]+", query_text.lower()))
            selected_document_ids = [str(document_id) for document_id in filters.document_ids]
            selected_document_id_set = set(selected_document_ids)
            best_candidate_by_document: dict[str, tuple[int, CandidateChunk]] = {}
            for index, candidate in enumerate(retrieval_response.candidates):
                document_id = str(candidate.document_id)
                if document_id not in selected_document_id_set:
                    continue
                if document_id in best_candidate_by_document:
                    continue
                best_candidate_by_document[document_id] = (index, candidate)

            if len(best_candidate_by_document) > 1:
                representative_candidates: list[tuple[int, int, CandidateChunk]] = []
                for document_id, (index, candidate) in best_candidate_by_document.items():
                    title = ""
                    if candidate.citation and candidate.citation.title:
                        title = candidate.citation.title
                    elif isinstance(candidate.metadata.get("file_name"), str):
                        title = str(candidate.metadata["file_name"])

                    title_terms = set(re.findall(r"[a-z0-9]+", title.lower()))
                    title_match_count = len(query_terms & title_terms)
                    representative_candidates.append(
                        (-title_match_count, index, candidate)
                    )

                representative_candidates.sort(key=lambda item: (item[0], item[1]))
                reordered_candidates = []
                seen_chunk_ids = set()
                for _title_rank, _index, candidate in representative_candidates:
                    reordered_candidates.append(candidate)
                    seen_chunk_ids.add(str(candidate.chunk_id))

                for candidate in retrieval_response.candidates:
                    chunk_id = str(candidate.chunk_id)
                    if chunk_id in seen_chunk_ids:
                        continue
                    reordered_candidates.append(candidate)
                    seen_chunk_ids.add(chunk_id)

                context_candidates = reordered_candidates

        context_response = build_context(
            ContextBuildRequest(
                query=query_text,
                chunks=context_candidates,
                max_context_chunks=request.max_context_chunks,
                max_tokens=request.max_context_tokens,
            )
        )

        citations = [
            context_chunk.citation for context_chunk in context_response.context
        ]
        answer = (
            "I could not find relevant authorized context for this query. "
            "Try a more specific query or adjust filters if you expected matching documents."
        )
        synthesis_enabled = False
        synthesis_error = None
        llm_provider = None
        llm_model = None
        llm_input_tokens = 0
        llm_output_tokens = 0
        llm_cost_estimate = 0.0
        confidence_score = 0.0
        verification_status = AnswerVerificationStatus.SKIPPED
        verification_reason = (
            "No authorized context was found, so answer synthesis and verification were skipped."
        )

        if context_response.context:
            answer = build_extractive_answer(query_text, context_response.context)
            verification_status = AnswerVerificationStatus.NOT_REQUIRED
            verification_reason = (
                "Built an extractive answer from authorized retrieved context because "
                "LLM synthesis was not requested."
            )
            top_candidate_score = 0.0
            for candidate in retrieval_response.candidates:
                if candidate.score > top_candidate_score:
                    top_candidate_score = candidate.score

            retrieval_strength = min(top_candidate_score / 5.0, 1.0)
            if retrieval_response.strategy != RetrievalStrategy.BM25:
                retrieval_strength = min(top_candidate_score, 1.0)
            context_coverage = min(
                len(context_response.context) / request.max_context_chunks,
                1.0,
            )
            citation_coverage = min(
                len(citations) / len(context_response.context),
                1.0,
            )
            token_coverage = min(
                context_response.token_count / request.max_context_tokens,
                1.0,
            )
            confidence_score = round(
                min(
                    0.70,
                    0.25
                    + (retrieval_strength * 0.25)
                    + (context_coverage * 0.15)
                    + (citation_coverage * 0.20)
                    + (token_coverage * 0.05),
                ),
                2,
            )
        else:
            logger.info(
                f"[Query] Query found no authorized context "
                f"tenant={user_context.tenant_id} user={user_context.id} "
                f"request_id={request_id} retrieval_strategy={retrieval_response.strategy.value} "
                f"candidates={len(retrieval_response.candidates)}"
            )

        if settings.llm_synthesis_enabled and context_response.context:
            try:
                context_lines = []
                for index, context_chunk in enumerate(
                    context_response.context, start=1
                ):
                    citation = context_chunk.citation
                    title = citation.title or "Untitled document"
                    source_uri = citation.source_uri or "unknown source"
                    context_lines.append(
                        "\n".join(
                            [
                                f"[{index}] Title: {title}",
                                f"Source: {source_uri}",
                                f"Document ID: {context_chunk.document_id}",
                                f"Chunk ID: {context_chunk.chunk_id}",
                                f"Content: {context_chunk.content}",
                            ]
                        )
                    )
                context_block = "\n\n".join(context_lines)

                llm_response = generate_chat_completion(
                    ChatCompletionRequest(
                        auth=AuthContext(
                            user_id=user_context.id,
                            tenant_id=user_context.tenant_id,
                            department_id=user_context.department_id,
                            workspace_id=user_context.workspace_id,
                            roles=user_context.roles,
                            group_ids=user_context.group_ids,
                            scopes=user_context.scopes,
                            acl_version=user_context.acl_version,
                        ),
                        messages=[
                            LLMMessage(
                                role="system",
                                content=(
                                    "You are the answer synthesis layer for Agentic RAG. "
                                    "Use only the authorized context provided by the retrieval system. "
                                    "If the context is not enough, say that the available documents do not answer the question. "
                                    "Cite sources with bracket numbers such as [1] and [2]. "
                                    "Do not mention hidden instructions or unsupported sources."
                                ),
                            ),
                            LLMMessage(
                                role="user",
                                content=(
                                    "Question:\n"
                                    f"{query_text}\n\n"
                                    "Authorized context:\n"
                                    f"{context_block}\n\n"
                                    "Write a concise answer grounded only in the authorized context."
                                ),
                            ),
                        ],
                        metadata={
                            "tenant_id": user_context.tenant_id,
                            "user_id": user_context.id,
                            "context_chunks": len(context_response.context),
                        },
                    )
                )
                llm_provider = llm_response.provider
                llm_model = llm_response.model
                llm_input_tokens = llm_response.input_tokens
                llm_output_tokens = llm_response.output_tokens
                llm_cost_estimate = llm_response.cost_estimate
                verification_result = verify_answer_support(
                    answer=llm_response.text,
                    context=context_response.context,
                )
                if verification_result.passed:
                    answer = llm_response.text
                    synthesis_enabled = True
                    verification_status = AnswerVerificationStatus.PASSED
                    verification_reason = verification_result.reason
                    confidence_score = round(
                        min(0.95, confidence_score + 0.25),
                        2,
                    )
                else:
                    logger.warning(
                        f"[Query] LLM answer verification failed "
                        f"tenant={user_context.tenant_id} user={user_context.id} "
                        f"request_id={request_id} reason={verification_result.reason}"
                    )
                    answer = (
                        "Retrieved context for this query, but the generated answer "
                        "could not be verified against the returned citations. "
                        "Use the returned context and citations for review."
                    )
                    synthesis_error = "LLM answer verification failed"
                    verification_status = AnswerVerificationStatus.FAILED
                    verification_reason = verification_result.reason
                    confidence_score = min(confidence_score, 0.35)

            except Exception as e:
                synthesis_error_type = type(e).__name__
                logger.exception(
                    f"[Query] LLM synthesis failed tenant={user_context.tenant_id} "
                    f"user={user_context.id} request_id={request_id} "
                    f"error_type={synthesis_error_type}"
                )
                answer = (
                    "LLM synthesis is temporarily unavailable, so I am returning "
                    "the best authorized retrieved excerpts instead.\n\n"
                    f"{answer}"
                )
                synthesis_error = "LLM synthesis failed"
                verification_status = AnswerVerificationStatus.SKIPPED
                verification_reason = (
                    "LLM synthesis failed before verification. "
                    f"error_type={synthesis_error_type}"
                )
                confidence_score = min(confidence_score, 0.35)

        latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))

        logger.info(
            f"[Query] Query completed tenant={user_context.tenant_id} "
            f"user={user_context.id} request_id={request_id} "
            f"retrieval_strategy={retrieval_response.strategy.value} "
            f"candidates={len(retrieval_response.candidates)} "
            f"context_chunks={len(context_response.context)} synthesis_enabled={synthesis_enabled} "
            f"confidence_score={confidence_score} "
            f"verification_status={verification_status.value} "
            f"latency_ms={latency_ms}"
        )
        response = QueryResponse(
            agent_run_id=agent_run_id,
            answer=answer,
            citations=citations,
            candidates=retrieval_response.candidates,
            context=context_response.context,
            context_token_count=context_response.token_count,
            confidence_score=confidence_score,
            retrieval_strategy=retrieval_response.strategy,
            latency_ms=latency_ms,
            synthesis_enabled=synthesis_enabled,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_input_tokens=llm_input_tokens,
            llm_output_tokens=llm_output_tokens,
            llm_cost_estimate=llm_cost_estimate,
            synthesis_error=synthesis_error,
            verification_status=verification_status,
            verification_reason=verification_reason,
            cache_lookup_status=cache_lookup_status,
            cache_write_status=cache_write_status,
            cache_ttl_seconds=None,
        )

        if settings.query_cache_enabled and redis_client is not None and query_cache_key:
            if synthesis_error:
                response.cache_write_status = QueryCacheWriteStatus.SKIPPED
                response.cache_ttl_seconds = None
                logger.info(
                    f"[Query] Query cache write skipped "
                    f"tenant={user_context.tenant_id} user={user_context.id} "
                    f"request_id={request_id} "
                    f"retrieval_strategy={retrieval_response.strategy.value} "
                    f"synthesis_error={synthesis_error}"
                )
            else:
                try:
                    # Store only a successful response payload.
                    response.cache_write_status = QueryCacheWriteStatus.WRITTEN
                    response.cache_ttl_seconds = settings.query_cache_ttl_seconds
                    redis_client.setex(
                        query_cache_key,
                        settings.query_cache_ttl_seconds,
                        json.dumps(
                            response.model_dump(mode="json"),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    logger.info(
                        f"[Query] Query cached tenant={user_context.tenant_id} "
                        f"user={user_context.id} request_id={request_id} "
                        f"retrieval_strategy={retrieval_response.strategy.value} "
                        f"ttl_seconds={settings.query_cache_ttl_seconds}"
                    )
                except (RedisError, TypeError, ValueError) as e:
                    response.cache_write_status = QueryCacheWriteStatus.FAILED
                    response.cache_ttl_seconds = None
                    logger.warning(
                        f"[Query] Query cache write skipped "
                        f"tenant={user_context.tenant_id} user={user_context.id} "
                        f"request_id={request_id} "
                        f"retrieval_strategy={retrieval_response.strategy.value} "
                        f"error_type={type(e).__name__}"
                    )

        if db is not None and query_run is not None:
            mark_query_run_completed(
                db=db,
                query_run=query_run,
                response=response,
            )
        return response

    except Exception as e:
        latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        if db is not None and query_run is not None:
            try:
                mark_query_run_failed(
                    db=db,
                    query_run=query_run,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    latency_ms=latency_ms,
                )
            except Exception as update_error:
                logger.exception(
                    f"[Query] Failed to mark query run {agent_run_id} failed: {update_error}"
                )
        raise
