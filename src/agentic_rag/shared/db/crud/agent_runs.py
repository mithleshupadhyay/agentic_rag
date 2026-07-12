import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from agentic_rag.shared.db.models import (
    AgentCheckpoint as AgentCheckpointModel,
    AgentRun,
    AgentStep,
)
from agentic_rag.shared.schemas.agent import (
    AgentCheckpoint as AgentCheckpointSchema,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
    ToolCallRecord,
)
from agentic_rag.shared.schemas.auth import AuthContext


logger = logging.getLogger(__name__)


def create_agent_run(
    db: Session,
    agent_run_id: UUID,
    auth: AuthContext,
    query: str,
    limits: Optional[AgentLimits] = None,
) -> AgentRun:
    runtime_limits = limits or AgentLimits()

    # Use one UTC clock for persisted run timing.
    current_time = datetime.now(timezone.utc)
    timeout_at = current_time + timedelta(seconds=runtime_limits.total_timeout_seconds)

    logger.info(
        f"[DB] Creating agent run {agent_run_id} tenant={auth.tenant_id} "
        f"user={auth.user_id} workspace={auth.workspace_id}"
    )

    db_obj = AgentRun(
        id=agent_run_id,
        tenant_id=auth.tenant_id,
        department_id=auth.department_id,
        workspace_id=auth.workspace_id,
        user_id=auth.user_id,
        query_text=query.strip(),
        status=AgentRunStatus.RUNNING.value,
        confidence_score=0.0,
        total_steps=0,
        total_tool_calls=0,
        limits=runtime_limits.model_dump(mode="json"),
        timeout_at=timeout_at,
        started_at=current_time,
    )

    try:
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(
            f"[DB] Created agent run {db_obj.id} tenant={db_obj.tenant_id} "
            f"status={db_obj.status}"
        )
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to create agent run {agent_run_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during agent run creation.",
        )


def cancel_agent_run(
    db: Session,
    agent_run_id: UUID,
    tenant_id: str,
) -> Optional[AgentRun]:
    logger.info(f"[DB] Cancelling agent run {agent_run_id} tenant={tenant_id}")
    agent_run = (
        db.query(AgentRun)
        .filter(
            AgentRun.id == agent_run_id,
            AgentRun.tenant_id == tenant_id,
        )
        .first()
    )
    if not agent_run:
        logger.warning(
            f"[DB] Agent run {agent_run_id} not found for cancellation "
            f"tenant={tenant_id}"
        )
        return None

    # Only active runs can move to cancelled.
    if agent_run.status not in {
        AgentRunStatus.QUEUED.value,
        AgentRunStatus.RUNNING.value,
    }:
        logger.warning(
            f"[DB] Agent run {agent_run_id} cancellation rejected "
            f"tenant={tenant_id} status={agent_run.status}"
        )
        raise HTTPException(
            status_code=409,
            detail=f"Agent run cannot be cancelled from status {agent_run.status}.",
        )

    agent_run.status = AgentRunStatus.CANCELLED.value
    agent_run.completed_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(agent_run)
        logger.info(f"[DB] Agent run {agent_run.id} cancelled")
        return agent_run

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to cancel agent run {agent_run_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during agent run cancellation.",
        )


def is_agent_run_cancelled(
    db: Session,
    agent_run_id: UUID,
    tenant_id: str,
) -> bool:
    logger.info(
        f"[DB] Checking agent run cancellation {agent_run_id} tenant={tenant_id}"
    )

    # Refresh local ORM state before reading a cancellation written by another flow.
    db.expire_all()

    agent_run = (
        db.query(AgentRun)
        .filter(
            AgentRun.id == agent_run_id,
            AgentRun.tenant_id == tenant_id,
        )
        .first()
    )
    if not agent_run:
        logger.warning(
            f"[DB] Agent run {agent_run_id} not found for cancellation check "
            f"tenant={tenant_id}"
        )
        return False

    is_cancelled = agent_run.status == AgentRunStatus.CANCELLED.value
    logger.info(
        f"[DB] Agent run cancellation check run={agent_run_id} "
        f"tenant={tenant_id} cancelled={is_cancelled}"
    )
    return is_cancelled


def record_agent_run_step(
    db: Session,
    agent_run_id: UUID,
    tenant_id: str,
    node_name: AgentNodeName | str,
    step_number: int,
    status: str = "completed",
    finish_run: bool = False,
    tool_call: Optional[ToolCallRecord] = None,
    tool_name: Optional[str] = None,
    tool_input: Optional[dict[str, Any]] = None,
    tool_output_summary: Optional[str] = None,
    latency_ms: Optional[int] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Optional[AgentStep]:
    node_value = node_name.value if isinstance(node_name, AgentNodeName) else node_name

    logger.info(
        f"[DB] Recording agent step run={agent_run_id} tenant={tenant_id} "
        f"node={node_value} step_number={step_number}"
    )

    agent_run = (
        db.query(AgentRun)
        .filter(
            AgentRun.id == agent_run_id,
            AgentRun.tenant_id == tenant_id,
        )
        .first()
    )
    if not agent_run:
        logger.warning(
            f"[DB] Agent run {agent_run_id} not found for step record "
            f"tenant={tenant_id}"
        )
        return None

    persisted_tool_name = tool_name
    persisted_tool_input = tool_input or {}
    if tool_call is not None:
        persisted_tool_name = tool_call.tool_name
        persisted_tool_input = tool_call.arguments

    db_obj = AgentStep(
        tenant_id=tenant_id,
        agent_run_id=agent_run_id,
        node_name=node_value,
        step_number=step_number,
        tool_name=persisted_tool_name,
        tool_input=persisted_tool_input,
        tool_output_summary=tool_output_summary,
        latency_ms=latency_ms,
        status=status,
        error_type=error_type[:128] if error_type else None,
        error_message=error_message,
    )

    agent_run.total_steps = max(agent_run.total_steps, step_number)
    if tool_call is not None:
        agent_run.total_tool_calls += 1
    if status in {
        AgentRunStatus.HANDOFF_REQUIRED.value,
        AgentRunStatus.TIMED_OUT.value,
        AgentRunStatus.FAILED.value,
        AgentRunStatus.CANCELLED.value,
    } or (finish_run and status == AgentRunStatus.COMPLETED.value):
        agent_run.status = status
        agent_run.completed_at = datetime.now(timezone.utc)

    try:
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(
            f"[DB] Recorded agent step {db_obj.id} run={agent_run_id} "
            f"tenant={tenant_id} node={node_value}"
        )
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to record agent step run={agent_run_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during agent step creation.",
        )


def save_agent_checkpoint(
    db: Session,
    agent_run_id: UUID,
    tenant_id: str,
    checkpoint: AgentCheckpointSchema,
) -> Optional[AgentCheckpointModel]:
    logger.info(
        f"[DB] Saving agent checkpoint run={agent_run_id} tenant={tenant_id} "
        f"checkpoint_key={checkpoint.checkpoint_key}"
    )

    if checkpoint.agent_run_id != agent_run_id:
        logger.warning(
            f"[DB] Agent checkpoint rejected because run ids differ "
            f"agent_run_id={agent_run_id} checkpoint_run_id={checkpoint.agent_run_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="Agent checkpoint run id does not match agent run id.",
        )

    agent_run = (
        db.query(AgentRun)
        .filter(
            AgentRun.id == agent_run_id,
            AgentRun.tenant_id == tenant_id,
        )
        .first()
    )
    if not agent_run:
        logger.warning(
            f"[DB] Agent run {agent_run_id} not found for checkpoint tenant={tenant_id}"
        )
        return None

    db_obj = AgentCheckpointModel(
        tenant_id=tenant_id,
        agent_run_id=agent_run_id,
        checkpoint_key=checkpoint.checkpoint_key,
        state=checkpoint.state,
        created_at=checkpoint.created_at,
    )

    try:
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(
            f"[DB] Saved agent checkpoint {db_obj.id} run={agent_run_id} "
            f"tenant={tenant_id}"
        )
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(
            f"[DB] Failed to save agent checkpoint run={agent_run_id}: {e}"
        )
        raise HTTPException(
            status_code=400,
            detail="Database error during agent checkpoint creation.",
        )


def get_agent_run(
    db: Session,
    agent_run_id: UUID,
    tenant_id: str,
) -> Optional[AgentRun]:
    logger.info(f"[DB] Fetching agent run {agent_run_id} tenant={tenant_id}")
    agent_run = (
        db.query(AgentRun)
        .options(
            selectinload(AgentRun.steps),
            selectinload(AgentRun.checkpoints),
        )
        .filter(
            AgentRun.id == agent_run_id,
            AgentRun.tenant_id == tenant_id,
        )
        .first()
    )
    if agent_run:
        logger.info(f"[DB] Found agent run {agent_run_id} tenant={tenant_id}")
    else:
        logger.warning(f"[DB] Agent run {agent_run_id} not found tenant={tenant_id}")
    return agent_run
