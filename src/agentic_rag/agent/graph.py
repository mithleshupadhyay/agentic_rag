import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from agentic_rag.agent.runtime import record_agent_step, start_agent_state
from agentic_rag.shared.schemas.agent import (
    AgentCheckpoint,
    AgentGraphState,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
    AgentStateModel,
)
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.retrieval import RetrievalStrategy


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentGraphRunResult:
    state: AgentStateModel
    checkpoints: list[AgentCheckpoint]
    status: AgentRunStatus
    stop_reason: str | None = None


def classify_intent_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    logger.info(
        f"[AgentGraph] Classifying intent "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    agent_state.intent = "retrieve_and_answer"

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.CLASSIFY_INTENT,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
    }


def rewrite_query_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    logger.info(
        f"[AgentGraph] Rewriting query "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    agent_state.rewritten_query = agent_state.query.strip()

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.REWRITE_QUERY,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
    }


def plan_filters_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    logger.info(
        f"[AgentGraph] Planning filters "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    if "query" not in agent_state.filters:
        agent_state.filters["query"] = agent_state.rewritten_query or agent_state.query

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.PLAN_FILTERS,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
    }


def select_retrieval_strategy_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    logger.info(
        f"[AgentGraph] Selecting retrieval strategy "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    agent_state.retrieval_strategy = RetrievalStrategy.BM25

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.SELECT_RETRIEVAL_STRATEGY,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
    }


def build_agent_runtime_graph():
    logger.info("[AgentGraph] Building agent runtime graph")

    graph = StateGraph(AgentGraphState)
    graph.add_node(AgentNodeName.CLASSIFY_INTENT.value, classify_intent_node)
    graph.add_node(AgentNodeName.REWRITE_QUERY.value, rewrite_query_node)
    graph.add_node(AgentNodeName.PLAN_FILTERS.value, plan_filters_node)
    graph.add_node(AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value, select_retrieval_strategy_node)
    graph.add_edge(START, AgentNodeName.CLASSIFY_INTENT.value)
    graph.add_edge(AgentNodeName.CLASSIFY_INTENT.value, AgentNodeName.REWRITE_QUERY.value)
    graph.add_edge(AgentNodeName.REWRITE_QUERY.value, AgentNodeName.PLAN_FILTERS.value)
    graph.add_edge(AgentNodeName.PLAN_FILTERS.value, AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value)
    graph.add_edge(AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value, END)

    return graph.compile()


def run_agent_runtime_graph(
    *,
    agent_run_id: UUID,
    auth: AuthContext,
    query: str,
    limits: AgentLimits | None = None,
    now: datetime | None = None,
) -> AgentGraphRunResult:
    runtime_limits = limits or AgentLimits()

    # Use one UTC clock to initialize the graph run.
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    logger.info(
        f"[AgentGraph] Starting agent runtime graph "
        f"agent_run_id={agent_run_id} tenant_id={auth.tenant_id} user_id={auth.user_id}"
    )

    agent_state = start_agent_state(
        agent_run_id=agent_run_id,
        auth=auth,
        query=query,
        limits=runtime_limits,
        now=current_time,
    )
    graph_state = AgentGraphState(
        agent_state=agent_state,
        limits=runtime_limits,
        current_time=current_time,
    )

    compiled_graph = build_agent_runtime_graph()
    raw_result = compiled_graph.invoke(graph_state)
    completed_graph_state = AgentGraphState.model_validate(raw_result)

    logger.info(
        f"[AgentGraph] Agent runtime graph finished "
        f"agent_run_id={agent_run_id} "
        f"status={completed_graph_state.status.value} "
        f"step_count={completed_graph_state.agent_state.step_count} "
        f"checkpoint_count={len(completed_graph_state.checkpoints)}"
    )

    return AgentGraphRunResult(
        state=completed_graph_state.agent_state,
        checkpoints=completed_graph_state.checkpoints,
        status=completed_graph_state.status,
        stop_reason=completed_graph_state.stop_reason,
    )
