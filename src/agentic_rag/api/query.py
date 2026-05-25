import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from agentic_rag.core.dependencies import require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.query.bm25_query import run_bm25_query
from agentic_rag.shared.db.crud.query_runs import get_query_run, list_query_runs
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.common import PageResponse
from agentic_rag.shared.schemas.query import (
    QueryRequest,
    QueryResponse,
    QueryRunListItem,
    QueryRunRead,
    QueryRunSearchResponse,
    QueryRunStatus,
)
from agentic_rag.shared.schemas.retrieval import RetrievalFilters, RetrievalStrategy


logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    payload: QueryRequest,
    user_context: UserContext = Depends(require_scope("query:run")),
    db: Session = Depends(get_session),
) -> QueryResponse:
    logger.info(
        f"[QueryAPI] Query started tenant={user_context.tenant_id} "
        f"user={user_context.id}"
    )

    response = run_bm25_query(
        user_context=user_context,
        request=payload,
        db=db,
    )

    logger.info(
        f"[QueryAPI] Query completed tenant={user_context.tenant_id} "
        f"user={user_context.id} context_chunks={len(response.context)}"
    )
    return response


@router.get("/query", response_model=QueryRunSearchResponse)
def list_query_run_endpoint(
    page: int = 1,
    size: int = 50,
    workspace_id: str | None = None,
    user_id: str | None = None,
    user_context: UserContext = Depends(require_scope("query:run")),
    db: Session = Depends(get_session),
) -> QueryRunSearchResponse:
    if page < 1:
        raise HTTPException(status_code=422, detail="page must be greater than or equal to 1")
    if size < 1 or size > 500:
        raise HTTPException(status_code=422, detail="size must be between 1 and 500")

    logger.info(
        f"[QueryAPI] Listing query runs tenant={user_context.tenant_id} "
        f"user={user_context.id} page={page} size={size}"
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
        error_type=query_run.error_type,
        error_message=query_run.error_message,
        response_payload=query_run.response_payload,
        created_at=query_run.created_at,
        updated_at=query_run.updated_at,
        completed_at=query_run.completed_at,
        response=response,
    )
