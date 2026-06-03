from datetime import datetime, timezone
from uuid import uuid4

from agentic_rag.agent.graph import build_agent_runtime_graph, run_agent_runtime_graph
from agentic_rag.agent.runtime import SAFE_FALLBACK_ANSWER, start_agent_state
from agentic_rag.shared.schemas.agent import (
    AgentGraphState,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
)
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.retrieval import RetrievalStrategy


def test_run_agent_runtime_graph_reaches_retrieval_boundary() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    auth = AuthContext(user_id="user-1", tenant_id="tenant-a")

    result = run_agent_runtime_graph(
        agent_run_id=agent_run_id,
        auth=auth,
        query="  Find the incident response policy  ",
        now=now,
    )

    assert result.status == AgentRunStatus.RUNNING
    assert result.stop_reason is None
    assert result.state.agent_run_id == agent_run_id
    assert result.state.auth == auth
    assert result.state.intent == "retrieve_and_answer"
    assert result.state.rewritten_query == "Find the incident response policy"
    assert result.state.filters == {"query": "Find the incident response policy"}
    assert result.state.retrieval_strategy == RetrievalStrategy.BM25
    assert result.state.step_count == 4
    assert result.state.tool_call_count == 0
    assert result.state.visited_nodes == [
        AgentNodeName.CLASSIFY_INTENT.value,
        AgentNodeName.REWRITE_QUERY.value,
        AgentNodeName.PLAN_FILTERS.value,
        AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value,
    ]


def test_run_agent_runtime_graph_returns_checkpoints_for_each_node() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = run_agent_runtime_graph(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
    )

    assert len(result.checkpoints) == 4
    assert result.checkpoints[0].checkpoint_key == "step-0001-classify_intent"
    assert result.checkpoints[1].checkpoint_key == "step-0002-rewrite_query"
    assert result.checkpoints[2].checkpoint_key == "step-0003-plan_filters"
    assert result.checkpoints[3].checkpoint_key == (
        "step-0004-select_retrieval_strategy"
    )
    assert result.checkpoints[-1].state["retrieval_strategy"] == RetrievalStrategy.BM25
    assert result.checkpoints[-1].state["step_count"] == 4


def test_run_agent_runtime_graph_stops_when_guardrail_fires() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    result = run_agent_runtime_graph(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=AgentLimits(max_steps=1),
        now=now,
    )

    assert result.status == AgentRunStatus.HANDOFF_REQUIRED
    assert result.stop_reason == "Agent step limit exceeded."
    assert result.state.final_answer == SAFE_FALLBACK_ANSWER
    assert result.state.handoff_required is True
    assert result.state.visited_nodes == [
        AgentNodeName.CLASSIFY_INTENT.value,
        AgentNodeName.REWRITE_QUERY.value,
    ]
    assert len(result.checkpoints) == 2
    assert result.checkpoints[-1].state["final_answer"] == SAFE_FALLBACK_ANSWER


def test_build_agent_runtime_graph_invokes_existing_state() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    auth = AuthContext(user_id="user-1", tenant_id="tenant-a")
    agent_state = start_agent_state(
        agent_run_id=agent_run_id,
        auth=auth,
        query="Find the incident response policy",
        now=now,
    )
    graph_state = AgentGraphState(agent_state=agent_state, current_time=now)

    compiled_graph = build_agent_runtime_graph()
    raw_result = compiled_graph.invoke(graph_state)
    completed_graph_state = AgentGraphState.model_validate(raw_result)

    assert completed_graph_state.agent_state.agent_run_id == agent_run_id
    assert completed_graph_state.status == AgentRunStatus.RUNNING
    assert completed_graph_state.agent_state.step_count == 4
    assert len(completed_graph_state.checkpoints) == 4
