import json
import logging
import time
from collections.abc import Iterator
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agentic_rag.agent.graph import stream_agent_runtime_graph
from agentic_rag.core.dependencies import require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.monitoring.metrics import (
    QUERY_LATENCY_SECONDS,
    QUERY_LIFECYCLE_TOTAL,
)
from agentic_rag.query.bm25_query import run_bm25_query
from agentic_rag.shared.db.crud.query_runs import (
    cancel_query_run,
    get_query_run,
    list_query_runs,
)
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.common import PageResponse
from agentic_rag.shared.schemas.query import (
    AnswerVerificationStatus,
    QueryRequest,
    QueryResponse,
    QueryRunListItem,
    QueryRunRead,
    QueryRunSearchResponse,
    QueryRunStatus,
    QueryStreamEvent,
)
from agentic_rag.shared.schemas.agent import AgentStreamEventType
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.retrieval import RetrievalFilters, RetrievalStrategy


logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


def query_stream_event_messages(
    *,
    request_id: str | None,
    agent_run_id: UUID,
    user_context: UserContext,
    request_payload: QueryRequest,
    auth: AuthContext,
    db: Session,
    started_event_message: str,
    started_at: float,
) -> Iterator[str]:
    yield started_event_message

    answer_token_count = 0
    terminal_event_seen = False
    retrieval_strategy_label = RetrievalStrategy.BM25.value
    synthesis_enabled_label = "false"

    try:
        for agent_event in stream_agent_runtime_graph(
            agent_run_id=agent_run_id,
            auth=auth,
            query=request_payload.query,
            retrieval_filters=request_payload.filters,
            retrieval_limit=request_payload.retrieval_limit,
            max_context_chunks=request_payload.max_context_chunks,
            max_context_tokens=request_payload.max_context_tokens,
            db=db,
        ):
            if agent_event.event == AgentStreamEventType.AGENT_STARTED:
                continue

            if agent_event.event == AgentStreamEventType.AGENT_STEP_COMPLETED:
                event_data = {
                    "request_id": request_id,
                    "node_name": agent_event.node_name,
                    "status": agent_event.status.value if agent_event.status else None,
                    "step_number": agent_event.step_number,
                }
                event_data.update(agent_event.data)
                step_event = QueryStreamEvent(
                    event=agent_event.event.value,
                    agent_run_id=agent_run_id,
                    data=event_data,
                )
                step_payload = step_event.model_dump(mode="json")
                yield (
                    f"event: {step_event.event}\n"
                    f"data: {json.dumps(step_payload, separators=(',', ':'))}\n\n"
                )
                continue

            if agent_event.event == AgentStreamEventType.ANSWER_TOKEN:
                answer_token_count += 1
                synthesis_enabled_label = "true"
                event_data = {
                    "request_id": request_id,
                    "node_name": agent_event.node_name,
                    "status": agent_event.status.value if agent_event.status else None,
                    "step_number": agent_event.step_number,
                    "text_delta": agent_event.text_delta or "",
                }
                event_data.update(agent_event.data)
                token_event = QueryStreamEvent(
                    event=agent_event.event.value,
                    agent_run_id=agent_run_id,
                    data=event_data,
                )
                token_payload = token_event.model_dump(mode="json")
                yield (
                    f"event: {token_event.event}\n"
                    f"data: {json.dumps(token_payload, separators=(',', ':'))}\n\n"
                )
                continue

            if agent_event.event == AgentStreamEventType.AGENT_COMPLETED:
                terminal_event_seen = True
                latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
                if isinstance(agent_event.data.get("retrieval_strategy"), str):
                    retrieval_strategy_label = agent_event.data["retrieval_strategy"]
                response = QueryResponse.model_validate(
                    {
                        "agent_run_id": agent_run_id,
                        "answer": agent_event.data.get("answer") or "",
                        "citations": agent_event.data.get("citations", []),
                        "context": agent_event.data.get("context", []),
                        "context_token_count": agent_event.data.get(
                            "context_token_count",
                            0,
                        ),
                        "confidence_score": agent_event.data.get(
                            "confidence_score",
                            0.0,
                        ),
                        "retrieval_strategy": retrieval_strategy_label,
                        "latency_ms": latency_ms,
                        "synthesis_enabled": answer_token_count > 0,
                    }
                )
                completed_event = QueryStreamEvent(
                    event="query_completed",
                    agent_run_id=agent_run_id,
                    data={
                        "request_id": request_id,
                        "agent_status": (
                            agent_event.status.value
                            if agent_event.status
                            else None
                        ),
                        "stop_reason": agent_event.data.get("stop_reason"),
                        "response": response.model_dump(mode="json"),
                    },
                )
                completed_payload = completed_event.model_dump(mode="json")
                QUERY_LIFECYCLE_TOTAL.labels(
                    status="completed",
                    retrieval_strategy=retrieval_strategy_label,
                    synthesis_enabled=synthesis_enabled_label,
                ).inc()
                QUERY_LATENCY_SECONDS.labels(
                    status="completed",
                    retrieval_strategy=retrieval_strategy_label,
                    synthesis_enabled=synthesis_enabled_label,
                ).observe(max(response.latency_ms, 0) / 1000)
                logger.info(
                    f"[QueryAPI] Streaming query completed tenant={user_context.tenant_id} "
                    f"user={user_context.id} request_id={request_id} "
                    f"agent_run_id={agent_run_id} status={agent_event.status}"
                )
                yield (
                    f"event: {completed_event.event}\n"
                    f"data: {json.dumps(completed_payload, separators=(',', ':'))}\n\n"
                )
                continue

            if agent_event.event == AgentStreamEventType.AGENT_FAILED:
                terminal_event_seen = True
                latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
                failed_event = QueryStreamEvent(
                    event="query_failed",
                    agent_run_id=agent_run_id,
                    data={
                        "request_id": request_id,
                        "error_type": "AgentRuntimeFailed",
                        "error_message": (
                            agent_event.data.get("stop_reason")
                            or "Agent runtime stream failed."
                        ),
                        "agent_status": (
                            agent_event.status.value
                            if agent_event.status
                            else None
                        ),
                        "answer": agent_event.data.get("answer"),
                    },
                )
                failed_payload = failed_event.model_dump(mode="json")
                QUERY_LIFECYCLE_TOTAL.labels(
                    status="failed",
                    retrieval_strategy=retrieval_strategy_label,
                    synthesis_enabled=synthesis_enabled_label,
                ).inc()
                QUERY_LATENCY_SECONDS.labels(
                    status="failed",
                    retrieval_strategy=retrieval_strategy_label,
                    synthesis_enabled=synthesis_enabled_label,
                ).observe(max(latency_ms, 0) / 1000)
                logger.warning(
                    f"[QueryAPI] Streaming query failed tenant={user_context.tenant_id} "
                    f"user={user_context.id} request_id={request_id} "
                    f"agent_run_id={agent_run_id} status={agent_event.status}"
                )
                yield (
                    f"event: {failed_event.event}\n"
                    f"data: {json.dumps(failed_payload, separators=(',', ':'))}\n\n"
                )
                continue

        if not terminal_event_seen:
            latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
            failed_event = QueryStreamEvent(
                event="query_failed",
                agent_run_id=agent_run_id,
                data={
                    "request_id": request_id,
                    "error_type": "AgentRuntimeIncomplete",
                    "error_message": "Agent runtime stream ended without a terminal event.",
                },
            )
            failed_payload = failed_event.model_dump(mode="json")
            QUERY_LIFECYCLE_TOTAL.labels(
                status="failed",
                retrieval_strategy=retrieval_strategy_label,
                synthesis_enabled=synthesis_enabled_label,
            ).inc()
            QUERY_LATENCY_SECONDS.labels(
                status="failed",
                retrieval_strategy=retrieval_strategy_label,
                synthesis_enabled=synthesis_enabled_label,
            ).observe(max(latency_ms, 0) / 1000)
            logger.warning(
                f"[QueryAPI] Streaming query ended without terminal event "
                f"tenant={user_context.tenant_id} user={user_context.id} "
                f"request_id={request_id} agent_run_id={agent_run_id}"
            )
            yield (
                f"event: {failed_event.event}\n"
                f"data: {json.dumps(failed_payload, separators=(',', ':'))}\n\n"
            )

    except Exception as e:
        logger.exception(
            f"[QueryAPI] Streaming query failed tenant={user_context.tenant_id} "
            f"user={user_context.id} request_id={request_id} "
            f"agent_run_id={agent_run_id}: {e}"
        )
        QUERY_LIFECYCLE_TOTAL.labels(
            status="failed",
            retrieval_strategy=retrieval_strategy_label,
            synthesis_enabled=synthesis_enabled_label,
        ).inc()
        QUERY_LATENCY_SECONDS.labels(
            status="failed",
            retrieval_strategy=retrieval_strategy_label,
            synthesis_enabled=synthesis_enabled_label,
        ).observe(max(0.0, time.perf_counter() - started_at))
        failed_event = QueryStreamEvent(
            event="query_failed",
            agent_run_id=agent_run_id,
            data={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "request_id": request_id,
            },
        )
        failed_payload = failed_event.model_dump(mode="json")
        yield (
            f"event: {failed_event.event}\n"
            f"data: {json.dumps(failed_payload, separators=(',', ':'))}\n\n"
        )


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    request: Request,
    payload: QueryRequest,
    user_context: UserContext = Depends(require_scope("query:run")),
    db: Session = Depends(get_session),
) -> QueryResponse:
    request_id = getattr(request.state, "request_id", None)
    started_at = time.perf_counter()
    retrieval_strategy_label = RetrievalStrategy.BM25.value
    synthesis_enabled_label = "false"
    logger.info(
        f"[QueryAPI] Query started tenant={user_context.tenant_id} "
        f"user={user_context.id} request_id={request_id}"
    )
    QUERY_LIFECYCLE_TOTAL.labels(
        status="started",
        retrieval_strategy=retrieval_strategy_label,
        synthesis_enabled=synthesis_enabled_label,
    ).inc()

    try:
        response = run_bm25_query(
            user_context=user_context,
            request=payload,
            db=db,
            request_id=request_id,
        )

    except Exception:
        QUERY_LIFECYCLE_TOTAL.labels(
            status="failed",
            retrieval_strategy=retrieval_strategy_label,
            synthesis_enabled=synthesis_enabled_label,
        ).inc()
        QUERY_LATENCY_SECONDS.labels(
            status="failed",
            retrieval_strategy=retrieval_strategy_label,
            synthesis_enabled=synthesis_enabled_label,
        ).observe(max(0.0, time.perf_counter() - started_at))
        raise

    logger.info(
        f"[QueryAPI] Query completed tenant={user_context.tenant_id} "
        f"user={user_context.id} request_id={request_id} "
        f"context_chunks={len(response.context)}"
    )
    retrieval_strategy_label = response.retrieval_strategy.value
    synthesis_enabled_label = str(response.synthesis_enabled).lower()
    QUERY_LIFECYCLE_TOTAL.labels(
        status="completed",
        retrieval_strategy=retrieval_strategy_label,
        synthesis_enabled=synthesis_enabled_label,
    ).inc()
    QUERY_LATENCY_SECONDS.labels(
        status="completed",
        retrieval_strategy=retrieval_strategy_label,
        synthesis_enabled=synthesis_enabled_label,
    ).observe(max(response.latency_ms, 0) / 1000)
    return response


@router.post("/query/stream")
def query_stream_endpoint(
    request: Request,
    payload: QueryRequest,
    user_context: UserContext = Depends(require_scope("query:run")),
    db: Session = Depends(get_session),
) -> StreamingResponse:
    request_id = getattr(request.state, "request_id", None)
    agent_run_id = uuid4()
    started_at = time.perf_counter()
    retrieval_strategy_label = RetrievalStrategy.BM25.value
    synthesis_enabled_label = "false"
    logger.info(
        f"[QueryAPI] Streaming query started tenant={user_context.tenant_id} "
        f"user={user_context.id} request_id={request_id} agent_run_id={agent_run_id}"
    )
    QUERY_LIFECYCLE_TOTAL.labels(
        status="started",
        retrieval_strategy=retrieval_strategy_label,
        synthesis_enabled=synthesis_enabled_label,
    ).inc()

    stream_filters = payload.filters.model_copy(deep=True)
    if payload.workspace_id:
        if stream_filters.workspace_id and stream_filters.workspace_id != payload.workspace_id:
            logger.warning(
                f"[QueryAPI] Streaming query workspace mismatch "
                f"workspace_id={payload.workspace_id} "
                f"filters_workspace={stream_filters.workspace_id}"
            )
            raise HTTPException(
                status_code=400,
                detail="workspace_id must match filters.workspace_id when both are provided.",
            )
        stream_filters.workspace_id = payload.workspace_id

    if user_context.workspace_id:
        if stream_filters.workspace_id and stream_filters.workspace_id != user_context.workspace_id:
            logger.warning(
                f"[QueryAPI] Streaming query denied by workspace "
                f"user_workspace={user_context.workspace_id} "
                f"requested_workspace={stream_filters.workspace_id}"
            )
            raise HTTPException(status_code=403, detail="Workspace access denied.")
        stream_filters.workspace_id = user_context.workspace_id

    stream_request = payload.model_copy(
        update={
            "stream": True,
            "filters": stream_filters,
        }
    )
    auth = AuthContext(
        user_id=user_context.id,
        tenant_id=user_context.tenant_id,
        workspace_id=user_context.workspace_id or stream_filters.workspace_id,
        roles=user_context.roles or [],
        group_ids=user_context.group_ids or [],
        scopes=user_context.scopes or [],
        acl_version=user_context.acl_version,
        request_id=request_id,
    )

    # Prepare the started event.
    started_event = QueryStreamEvent(
        event="query_started",
        agent_run_id=agent_run_id,
        data={
            "request_id": request_id,
            "tenant_id": user_context.tenant_id,
            "user_id": user_context.id,
            "workspace_id": auth.workspace_id,
        },
    )
    started_payload = started_event.model_dump(mode="json")
    started_event_message = (
        f"event: {started_event.event}\n"
        f"data: {json.dumps(started_payload, separators=(',', ':'))}\n\n"
    )

    return StreamingResponse(
        query_stream_event_messages(
            request_id=request_id,
            agent_run_id=agent_run_id,
            user_context=user_context,
            request_payload=stream_request,
            auth=auth,
            db=db,
            started_event_message=started_event_message,
            started_at=started_at,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/query", response_model=QueryRunSearchResponse)
def list_query_run_endpoint(
    page: int = 1,
    size: int = 50,
    workspace_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    status: QueryRunStatus | None = None,
    user_context: UserContext = Depends(require_scope("query:run")),
    db: Session = Depends(get_session),
) -> QueryRunSearchResponse:
    if page < 1:
        raise HTTPException(status_code=422, detail="page must be greater than or equal to 1")
    if size < 1 or size > 500:
        raise HTTPException(status_code=422, detail="size must be between 1 and 500")

    logger.info(
        f"[QueryAPI] Listing query runs tenant={user_context.tenant_id} "
        f"user={user_context.id} page={page} size={size} "
        f"request_id={request_id} status={status.value if status else None}"
    )
    effective_workspace_id = workspace_id
    if user_context.workspace_id:
        if workspace_id and workspace_id != user_context.workspace_id:
            logger.warning(
                f"[QueryAPI] Query run list denied user_workspace={user_context.workspace_id} "
                f"requested_workspace={workspace_id}"
            )
            raise HTTPException(status_code=403, detail="Workspace access denied.")
        effective_workspace_id = user_context.workspace_id

    user_roles = user_context.roles or []
    effective_user_id = user_id
    if "admin" not in user_roles:
        if user_id and user_id != user_context.id:
            logger.warning(
                f"[QueryAPI] Query run list denied user={user_context.id} "
                f"requested_user={user_id}"
            )
            raise HTTPException(status_code=403, detail="Query run access denied.")
        effective_user_id = user_context.id

    query_runs, total = list_query_runs(
        db=db,
        tenant_id=user_context.tenant_id,
        skip=(page - 1) * size,
        limit=size,
        workspace_id=effective_workspace_id,
        user_id=effective_user_id,
        request_id=request_id,
        status=status,
    )
    logger.info(
        f"[QueryAPI] Listed {len(query_runs)} query runs tenant={user_context.tenant_id} "
        f"user={user_context.id}"
    )
    return QueryRunSearchResponse(
        items=[
            QueryRunListItem(
                agent_run_id=query_run.id,
                status=QueryRunStatus(query_run.status),
                workspace_id=query_run.workspace_id,
                user_id=query_run.user_id,
                request_id=query_run.request_id,
                conversation_id=query_run.conversation_id,
                query=query_run.query_text,
                retrieval_strategy=(
                    RetrievalStrategy(query_run.retrieval_strategy)
                    if query_run.retrieval_strategy
                    else None
                ),
                synthesis_enabled=query_run.synthesis_enabled,
                llm_provider=query_run.llm_provider,
                llm_model=query_run.llm_model,
                verification_status=AnswerVerificationStatus(
                    query_run.verification_status
                ),
                verification_reason=query_run.verification_reason,
                latency_ms=query_run.latency_ms,
                created_at=query_run.created_at,
                completed_at=query_run.completed_at,
            )
            for query_run in query_runs
        ],
        page=PageResponse(page=page, size=size, total=total),
    )


@router.get("/query/{agent_run_id}", response_model=QueryRunRead)
def get_query_run_endpoint(
    agent_run_id: UUID,
    user_context: UserContext = Depends(require_scope("query:run")),
    db: Session = Depends(get_session),
) -> QueryRunRead:
    logger.info(
        f"[QueryAPI] Fetching query run {agent_run_id} "
        f"tenant={user_context.tenant_id} user={user_context.id}"
    )
    query_run = get_query_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id=user_context.tenant_id,
    )
    if not query_run:
        raise HTTPException(status_code=404, detail="Query run not found.")

    if (
        user_context.workspace_id
        and query_run.workspace_id
        and user_context.workspace_id != query_run.workspace_id
    ):
        logger.warning(
            f"[QueryAPI] Query run {agent_run_id} denied by workspace "
            f"user_workspace={user_context.workspace_id} run_workspace={query_run.workspace_id}"
        )
        raise HTTPException(status_code=403, detail="Workspace access denied.")

    if "admin" not in (user_context.roles or []) and query_run.user_id != user_context.id:
        logger.warning(
            f"[QueryAPI] Query run {agent_run_id} denied for user={user_context.id} "
            f"owner={query_run.user_id}"
        )
        raise HTTPException(status_code=403, detail="Query run access denied.")

    logger.info(
        f"[QueryAPI] Fetched query run {agent_run_id} "
        f"tenant={user_context.tenant_id} user={user_context.id}"
    )

    response = None
    if query_run.response_payload:
        response = QueryResponse.model_validate(query_run.response_payload)

    return QueryRunRead(
        agent_run_id=query_run.id,
        status=QueryRunStatus(query_run.status),
        tenant_id=query_run.tenant_id,
        workspace_id=query_run.workspace_id,
        user_id=query_run.user_id,
        request_id=query_run.request_id,
        conversation_id=query_run.conversation_id,
        query=query_run.query_text,
        filters=RetrievalFilters.model_validate(query_run.filters),
        retrieval_limit=query_run.retrieval_limit,
        max_context_chunks=query_run.max_context_chunks,
        max_context_tokens=query_run.max_context_tokens,
        retrieval_strategy=(
            RetrievalStrategy(query_run.retrieval_strategy)
            if query_run.retrieval_strategy
            else None
        ),
        answer=query_run.answer,
        citations=query_run.citations.get("items", []),
        context_token_count=query_run.context_token_count,
        confidence_score=query_run.confidence_score,
        latency_ms=query_run.latency_ms,
        synthesis_enabled=query_run.synthesis_enabled,
        llm_provider=query_run.llm_provider,
        llm_model=query_run.llm_model,
        llm_input_tokens=query_run.llm_input_tokens,
        llm_output_tokens=query_run.llm_output_tokens,
        llm_cost_estimate=query_run.llm_cost_estimate,
        verification_status=AnswerVerificationStatus(query_run.verification_status),
        verification_reason=query_run.verification_reason,
        error_type=query_run.error_type,
        error_message=query_run.error_message,
        response_payload=query_run.response_payload,
        created_at=query_run.created_at,
        updated_at=query_run.updated_at,
        completed_at=query_run.completed_at,
        response=response,
    )


@router.post("/query/{agent_run_id}/cancel", response_model=QueryRunRead)
def cancel_query_run_endpoint(
    agent_run_id: UUID,
    user_context: UserContext = Depends(require_scope("query:run")),
    db: Session = Depends(get_session),
) -> QueryRunRead:
    logger.info(
        f"[QueryAPI] Cancelling query run {agent_run_id} "
        f"tenant={user_context.tenant_id} user={user_context.id}"
    )

    # Fetch the run in the current tenant before authorization checks.
    query_run = get_query_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id=user_context.tenant_id,
    )
    if not query_run:
        raise HTTPException(status_code=404, detail="Query run not found.")

    # Enforce workspace access first.
    if (
        user_context.workspace_id
        and query_run.workspace_id
        and user_context.workspace_id != query_run.workspace_id
    ):
        logger.warning(
            f"[QueryAPI] Query run {agent_run_id} cancellation denied by workspace "
            f"user_workspace={user_context.workspace_id} run_workspace={query_run.workspace_id}"
        )
        raise HTTPException(status_code=403, detail="Workspace access denied.")

    # Non-admin users can cancel only their own runs.
    if "admin" not in (user_context.roles or []) and query_run.user_id != user_context.id:
        logger.warning(
            f"[QueryAPI] Query run {agent_run_id} cancellation denied "
            f"for user={user_context.id} owner={query_run.user_id}"
        )
        raise HTTPException(status_code=403, detail="Query run access denied.")

    query_run = cancel_query_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id=user_context.tenant_id,
    )
    if not query_run:
        raise HTTPException(status_code=404, detail="Query run not found.")

    logger.info(
        f"[QueryAPI] Cancelled query run {agent_run_id} "
        f"tenant={user_context.tenant_id} user={user_context.id}"
    )
    QUERY_LIFECYCLE_TOTAL.labels(
        status="cancelled",
        retrieval_strategy=query_run.retrieval_strategy or RetrievalStrategy.BM25.value,
        synthesis_enabled=str(query_run.synthesis_enabled).lower(),
    ).inc()

    response = None
    if query_run.response_payload:
        response = QueryResponse.model_validate(query_run.response_payload)

    return QueryRunRead(
        agent_run_id=query_run.id,
        status=QueryRunStatus(query_run.status),
        tenant_id=query_run.tenant_id,
        workspace_id=query_run.workspace_id,
        user_id=query_run.user_id,
        request_id=query_run.request_id,
        conversation_id=query_run.conversation_id,
        query=query_run.query_text,
        filters=RetrievalFilters.model_validate(query_run.filters),
        retrieval_limit=query_run.retrieval_limit,
        max_context_chunks=query_run.max_context_chunks,
        max_context_tokens=query_run.max_context_tokens,
        retrieval_strategy=(
            RetrievalStrategy(query_run.retrieval_strategy)
            if query_run.retrieval_strategy
            else None
        ),
        answer=query_run.answer,
        citations=query_run.citations.get("items", []),
        context_token_count=query_run.context_token_count,
        confidence_score=query_run.confidence_score,
        latency_ms=query_run.latency_ms,
        synthesis_enabled=query_run.synthesis_enabled,
        llm_provider=query_run.llm_provider,
        llm_model=query_run.llm_model,
        llm_input_tokens=query_run.llm_input_tokens,
        llm_output_tokens=query_run.llm_output_tokens,
        llm_cost_estimate=query_run.llm_cost_estimate,
        verification_status=AnswerVerificationStatus(query_run.verification_status),
        verification_reason=query_run.verification_reason,
        error_type=query_run.error_type,
        error_message=query_run.error_message,
        response_payload=query_run.response_payload,
        created_at=query_run.created_at,
        updated_at=query_run.updated_at,
        completed_at=query_run.completed_at,
        response=response,
    )
