import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.models import QueryRun
from agentic_rag.shared.schemas.query import QueryRequest, QueryResponse, QueryRunStatus


logger = logging.getLogger(__name__)


def create_query_run(
    user_context: UserContext,
    db: Session,
    request: QueryRequest,
    agent_run_id: UUID,
    request_id: Optional[str] = None,
) -> QueryRun:
    logger.info(
        f"[DB] Creating query run {agent_run_id} tenant={user_context.tenant_id} "
        f"user={user_context.id} request_id={request_id}"
    )

    try:
        db_obj = QueryRun(
            id=agent_run_id,
            tenant_id=user_context.tenant_id,
            workspace_id=request.workspace_id or user_context.workspace_id,
            user_id=user_context.id,
            request_id=request_id,
            conversation_id=request.conversation_id,
            query_text=request.query.strip(),
            filters=request.filters.model_dump(mode="json"),
            status=QueryRunStatus.RUNNING.value,
            citations={"items": []},
            candidates={"items": []},
            context={"items": []},
            response_payload={},
            retrieval_limit=request.retrieval_limit,
            max_context_chunks=request.max_context_chunks,
            max_context_tokens=request.max_context_tokens,
            context_token_count=0,
            synthesis_enabled=False,
            llm_input_tokens=0,
            llm_output_tokens=0,
            llm_cost_estimate=0.0,
        )

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(
            f"[DB] Created query run {db_obj.id} tenant={db_obj.tenant_id} "
            f"status={db_obj.status}"
        )
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to create query run {agent_run_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during query run creation.",
        )


def mark_query_run_completed(
    db: Session,
    query_run: QueryRun,
    response: QueryResponse,
) -> QueryRun:
    logger.info(f"[DB] Marking query run {query_run.id} completed")
    response_payload = response.model_dump(mode="json")

    query_run.status = QueryRunStatus.COMPLETED.value
    query_run.retrieval_strategy = response.retrieval_strategy.value
    query_run.answer = response.answer
    query_run.citations = {"items": response_payload.get("citations", [])}
    query_run.candidates = {"items": response_payload.get("candidates", [])}
    query_run.context = {"items": response_payload.get("context", [])}
    query_run.response_payload = response_payload
    query_run.context_token_count = response.context_token_count
    query_run.confidence_score = response.confidence_score
    query_run.latency_ms = response.latency_ms
    query_run.synthesis_enabled = response.synthesis_enabled
    query_run.llm_provider = response.llm_provider
    query_run.llm_model = response.llm_model
    query_run.llm_input_tokens = response.llm_input_tokens
    query_run.llm_output_tokens = response.llm_output_tokens
    query_run.llm_cost_estimate = response.llm_cost_estimate
    query_run.error_type = None
    query_run.error_message = response.synthesis_error
    query_run.completed_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(query_run)
        logger.info(f"[DB] Query run {query_run.id} completed")
        return query_run

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to complete query run {query_run.id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during query run completion.",
        )


def mark_query_run_failed(
    db: Session,
    query_run: QueryRun,
    error_type: str,
    error_message: str,
    latency_ms: Optional[int] = None,
) -> QueryRun:
    logger.warning(f"[DB] Marking query run {query_run.id} failed: {error_message}")

    query_run.status = QueryRunStatus.FAILED.value
    query_run.error_type = error_type[:128]
    query_run.error_message = error_message
    query_run.latency_ms = latency_ms
    query_run.completed_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(query_run)
        logger.info(f"[DB] Query run {query_run.id} failed")
        return query_run

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to mark query run {query_run.id} failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during query run failure update.",
        )


def get_query_run(
    db: Session,
    agent_run_id: UUID,
    tenant_id: str,
) -> Optional[QueryRun]:
    logger.info(f"[DB] Fetching query run {agent_run_id} tenant={tenant_id}")
    query_run = (
        db.query(QueryRun)
        .filter(
            QueryRun.id == agent_run_id,
            QueryRun.tenant_id == tenant_id,
        )
        .first()
    )
    if query_run:
        logger.info(f"[DB] Found query run {agent_run_id} tenant={tenant_id}")
    else:
        logger.warning(f"[DB] Query run {agent_run_id} not found tenant={tenant_id}")
    return query_run


def list_query_runs(
    db: Session,
    tenant_id: str,
    skip: int = 0,
    limit: int = 50,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> tuple[list[QueryRun], int]:
    logger.info(
        f"[DB] Listing query runs tenant={tenant_id} skip={skip} limit={limit} "
        f"workspace_id={workspace_id} user_id={user_id} request_id={request_id}"
    )
    query = db.query(QueryRun).filter(QueryRun.tenant_id == tenant_id)

    if workspace_id:
        query = query.filter(QueryRun.workspace_id == workspace_id)
    if user_id:
        query = query.filter(QueryRun.user_id == user_id)
    if request_id:
        query = query.filter(QueryRun.request_id == request_id)

    total = query.order_by(None).count()
    query_runs = (
        query.order_by(desc(QueryRun.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    logger.info(
        f"[DB] Listed {len(query_runs)} query runs of total={total} "
        f"tenant={tenant_id}"
    )
    return query_runs, total
