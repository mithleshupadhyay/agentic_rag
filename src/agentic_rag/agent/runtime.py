import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from agentic_rag.shared.schemas.agent import (
    AgentCheckpoint,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
    AgentStateModel,
    ToolCallRecord,
)
from agentic_rag.shared.schemas.auth import AuthContext


logger = logging.getLogger(__name__)


SAFE_FALLBACK_ANSWER = (
    "I could not answer this confidently from the available authorized context."
)


@dataclass(frozen=True)
class AgentRuntimeDecision:
    should_stop: bool
    status: AgentRunStatus
    reason: str
    fallback_answer: str | None = None


@dataclass(frozen=True)
class AgentRuntimeStepResult:
    state: AgentStateModel
    checkpoint: AgentCheckpoint
    decision: AgentRuntimeDecision


def start_agent_state(
    *,
    agent_run_id: UUID,
    auth: AuthContext,
    query: str,
    limits: AgentLimits | None = None,
    now: datetime | None = None,
) -> AgentStateModel:
    runtime_limits = limits or AgentLimits()

    # Use one UTC clock for this run.
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    # Store the absolute run deadline.
    deadline_at = current_time + timedelta(seconds=runtime_limits.total_timeout_seconds)

    logger.info(
        f"[AgentRuntime] Agent state initialized "
        f"agent_run_id={agent_run_id} tenant_id={auth.tenant_id} "
        f"user_id={auth.user_id} deadline_at={deadline_at.isoformat()}"
    )
    return AgentStateModel(
        agent_run_id=agent_run_id,
        auth=auth,
        query=query,
        deadline_at=deadline_at,
    )


def record_agent_step(
    *,
    state: AgentStateModel,
    node_name: AgentNodeName | str,
    limits: AgentLimits | None = None,
    tool_call: ToolCallRecord | None = None,
    latency_ms: int | None = None,
    now: datetime | None = None,
) -> AgentRuntimeStepResult:
    runtime_limits = limits or AgentLimits()

    # Use one UTC clock for this step.
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    # Work on a copy of the state.
    node_value = node_name.value if isinstance(node_name, AgentNodeName) else node_name
    next_state = state.model_copy(deep=True)

    # Record the node visit.
    next_state.step_count += 1
    next_state.visited_nodes.append(node_value)

    # Record the tool call when this node used one.
    if tool_call is not None:
        next_state.tool_call_count += 1
        next_state.last_tool_calls.append(tool_call)

    # Use tool latency if no step latency was passed.
    step_latency_ms = latency_ms
    if step_latency_ms is None and tool_call is not None:
        step_latency_ms = tool_call.latency_ms

    # Evaluate guardrails before saving the checkpoint.
    decision = evaluate_agent_guardrails(
        state=next_state,
        limits=runtime_limits,
        now=current_time,
        step_latency_ms=step_latency_ms,
    )

    # Move to fallback when a guardrail stops the run.
    if decision.fallback_answer is not None:
        next_state.final_answer = SAFE_FALLBACK_ANSWER
        next_state.handoff_required = True
        logger.warning(
            f"[AgentRuntime] Safe fallback selected "
            f"agent_run_id={next_state.agent_run_id} reason={decision.reason}"
        )

    # Save checkpoint data after the node.
    checkpoint = AgentCheckpoint(
        agent_run_id=next_state.agent_run_id,
        checkpoint_key=f"step-{next_state.step_count:04d}-{node_value}",
        state=next_state.model_dump(mode="json"),
        created_at=current_time,
    )

    logger.info(
        f"[AgentRuntime] Agent step recorded "
        f"agent_run_id={next_state.agent_run_id} node_name={node_value} "
        f"step_count={next_state.step_count} "
        f"tool_call_count={next_state.tool_call_count} "
        f"status={decision.status.value}"
    )
    return AgentRuntimeStepResult(
        state=next_state,
        checkpoint=checkpoint,
        decision=decision,
    )


def evaluate_agent_guardrails(
    *,
    state: AgentStateModel,
    limits: AgentLimits | None = None,
    now: datetime | None = None,
    next_node_name: AgentNodeName | str | None = None,
    step_latency_ms: int | None = None,
) -> AgentRuntimeDecision:
    runtime_limits = limits or AgentLimits()

    # Compare guardrails with UTC times.
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    deadline_at = state.deadline_at
    if deadline_at.tzinfo is None:
        deadline_at = deadline_at.replace(tzinfo=timezone.utc)
    else:
        deadline_at = deadline_at.astimezone(timezone.utc)

    # Stop when the run deadline has passed.
    if current_time >= deadline_at:
        reason = "Agent run deadline exceeded."
        logger.warning(
            f"[AgentRuntime] Guardrail stopped run because deadline exceeded "
            f"agent_run_id={state.agent_run_id} deadline_at={state.deadline_at.isoformat()}"
        )
        return AgentRuntimeDecision(
            should_stop=True,
            status=AgentRunStatus.TIMED_OUT,
            reason=reason,
            fallback_answer=SAFE_FALLBACK_ANSWER,
        )

    # Stop when one node runs too long.
    if (
        step_latency_ms is not None
        and step_latency_ms > runtime_limits.step_timeout_seconds * 1000
    ):
        reason = "Agent step timeout exceeded."
        logger.warning(
            f"[AgentRuntime] Guardrail stopped run because step timeout exceeded "
            f"agent_run_id={state.agent_run_id} latency_ms={step_latency_ms} "
            f"step_timeout_seconds={runtime_limits.step_timeout_seconds}"
        )
        return AgentRuntimeDecision(
            should_stop=True,
            status=AgentRunStatus.TIMED_OUT,
            reason=reason,
            fallback_answer=SAFE_FALLBACK_ANSWER,
        )

    # Stop when the run used too many steps.
    if state.step_count > runtime_limits.max_steps:
        reason = "Agent step limit exceeded."
        logger.warning(
            f"[AgentRuntime] Guardrail stopped run because step limit exceeded "
            f"agent_run_id={state.agent_run_id} step_count={state.step_count} "
            f"max_steps={runtime_limits.max_steps}"
        )
        return AgentRuntimeDecision(
            should_stop=True,
            status=AgentRunStatus.HANDOFF_REQUIRED,
            reason=reason,
            fallback_answer=SAFE_FALLBACK_ANSWER,
        )

    # Stop when the run used too many tool calls.
    if state.tool_call_count > runtime_limits.max_tool_calls:
        reason = "Agent tool call limit exceeded."
        logger.warning(
            f"[AgentRuntime] Guardrail stopped run because tool call limit exceeded "
            f"agent_run_id={state.agent_run_id} "
            f"tool_call_count={state.tool_call_count} "
            f"max_tool_calls={runtime_limits.max_tool_calls}"
        )
        return AgentRuntimeDecision(
            should_stop=True,
            status=AgentRunStatus.HANDOFF_REQUIRED,
            reason=reason,
            fallback_answer=SAFE_FALLBACK_ANSWER,
        )

    # Count identical tool calls from the latest call backward.
    repeated_tool_count = 0
    if state.last_tool_calls:
        latest_tool_call = state.last_tool_calls[-1]
        latest_arguments = json.dumps(
            latest_tool_call.arguments,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        latest_signature = f"{latest_tool_call.tool_name}:{latest_arguments}"

        for tool_call in reversed(state.last_tool_calls):
            current_arguments = json.dumps(
                tool_call.arguments,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            current_signature = f"{tool_call.tool_name}:{current_arguments}"
            if current_signature != latest_signature:
                break
            repeated_tool_count += 1

    if repeated_tool_count > runtime_limits.max_same_tool_repeat:
        reason = "Agent repeated the same tool call too many times."
        logger.warning(
            f"[AgentRuntime] Guardrail stopped run because tool call repeated "
            f"agent_run_id={state.agent_run_id} "
            f"repeat_count={repeated_tool_count} "
            f"max_same_tool_repeat={runtime_limits.max_same_tool_repeat}"
        )
        return AgentRuntimeDecision(
            should_stop=True,
            status=AgentRunStatus.HANDOFF_REQUIRED,
            reason=reason,
            fallback_answer=SAFE_FALLBACK_ANSWER,
        )

    # Normalize the next node name before checking generation rules.
    next_node_value = None
    if next_node_name is not None:
        next_node_value = (
            next_node_name.value
            if isinstance(next_node_name, AgentNodeName)
            else next_node_name
        )

    # Block answer generation without authorized context.
    if next_node_value == AgentNodeName.GENERATE_ANSWER.value:
        if not state.authorized_chunks and not state.context:
            reason = "Answer generation requires authorized context."
            logger.warning(
                f"[AgentRuntime] Guardrail stopped generation without authorized context "
                f"agent_run_id={state.agent_run_id}"
            )
            return AgentRuntimeDecision(
                should_stop=True,
                status=AgentRunStatus.HANDOFF_REQUIRED,
                reason=reason,
                fallback_answer=SAFE_FALLBACK_ANSWER,
            )

    return AgentRuntimeDecision(
        should_stop=False,
        status=AgentRunStatus.RUNNING,
        reason="Agent runtime guardrails passed.",
    )
