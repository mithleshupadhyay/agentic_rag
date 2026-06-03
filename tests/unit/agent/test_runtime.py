from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agentic_rag.agent.runtime import (
    SAFE_FALLBACK_ANSWER,
    evaluate_agent_guardrails,
    record_agent_step,
    start_agent_state,
)
from agentic_rag.shared.schemas.agent import (
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
    ToolCallRecord,
)
from agentic_rag.shared.schemas.auth import AuthContext


def test_start_agent_state_sets_deadline_from_limits() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    auth = AuthContext(user_id="user-1", tenant_id="tenant-a")

    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=auth,
        query="Find the incident response policy",
        limits=AgentLimits(total_timeout_seconds=30),
        now=now,
    )

    assert state.auth == auth
    assert state.step_count == 0
    assert state.tool_call_count == 0
    assert state.deadline_at == now + timedelta(seconds=30)


def test_record_agent_step_updates_state_and_checkpoint() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
    )

    result = record_agent_step(
        state=state,
        node_name=AgentNodeName.CLASSIFY_INTENT,
        now=now + timedelta(seconds=1),
    )

    assert result.state.step_count == 1
    assert result.state.visited_nodes == [AgentNodeName.CLASSIFY_INTENT.value]
    assert result.decision.should_stop is False
    assert result.decision.status == AgentRunStatus.RUNNING
    assert result.checkpoint.checkpoint_key == "step-0001-classify_intent"
    assert result.checkpoint.state["step_count"] == 1
    assert result.checkpoint.state["visited_nodes"] == ["classify_intent"]


def test_step_limit_stops_run_and_sets_safe_fallback() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    limits = AgentLimits(max_steps=1)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=limits,
        now=now,
    )
    first_step = record_agent_step(
        state=state,
        node_name=AgentNodeName.CLASSIFY_INTENT,
        limits=limits,
        now=now + timedelta(seconds=1),
    )

    second_step = record_agent_step(
        state=first_step.state,
        node_name=AgentNodeName.REWRITE_QUERY,
        limits=limits,
        now=now + timedelta(seconds=2),
    )

    assert second_step.decision.should_stop is True
    assert second_step.decision.status == AgentRunStatus.HANDOFF_REQUIRED
    assert second_step.decision.fallback_answer == SAFE_FALLBACK_ANSWER
    assert second_step.state.final_answer == SAFE_FALLBACK_ANSWER
    assert second_step.state.handoff_required is True
    assert second_step.checkpoint.state["final_answer"] == SAFE_FALLBACK_ANSWER


def test_tool_call_limit_stops_run() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    limits = AgentLimits(max_tool_calls=1)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=limits,
        now=now,
    )
    tool_call = ToolCallRecord(
        tool_name="bm25_search",
        arguments={"query": "incident response"},
    )
    first_step = record_agent_step(
        state=state,
        node_name=AgentNodeName.BM25_SEARCH,
        limits=limits,
        tool_call=tool_call,
        now=now + timedelta(seconds=1),
    )

    second_step = record_agent_step(
        state=first_step.state,
        node_name=AgentNodeName.VECTOR_SEARCH,
        limits=limits,
        tool_call=ToolCallRecord(
            tool_name="vector_search",
            arguments={"query": "incident response"},
        ),
        now=now + timedelta(seconds=2),
    )

    assert second_step.decision.should_stop is True
    assert second_step.decision.reason == "Agent tool call limit exceeded."
    assert second_step.state.tool_call_count == 2


def test_deadline_guardrail_times_out_run() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=AgentLimits(total_timeout_seconds=10),
        now=now,
    )

    decision = evaluate_agent_guardrails(
        state=state,
        now=now + timedelta(seconds=10),
    )

    assert decision.should_stop is True
    assert decision.status == AgentRunStatus.TIMED_OUT
    assert decision.fallback_answer == SAFE_FALLBACK_ANSWER


def test_step_timeout_guardrail_times_out_run() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=AgentLimits(step_timeout_seconds=1),
        now=now,
    )

    result = record_agent_step(
        state=state,
        node_name=AgentNodeName.BM25_SEARCH,
        limits=AgentLimits(step_timeout_seconds=1),
        latency_ms=1001,
        now=now + timedelta(seconds=1),
    )

    assert result.decision.should_stop is True
    assert result.decision.status == AgentRunStatus.TIMED_OUT
    assert result.state.final_answer == SAFE_FALLBACK_ANSWER


def test_repeated_tool_call_guardrail_stops_on_third_same_call() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    limits = AgentLimits(max_same_tool_repeat=2)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=limits,
        now=now,
    )
    tool_call = ToolCallRecord(
        tool_name="metadata_search",
        arguments={"filters": {"department": "security"}},
    )

    first_step = record_agent_step(
        state=state,
        node_name=AgentNodeName.METADATA_SEARCH,
        limits=limits,
        tool_call=tool_call,
        now=now + timedelta(seconds=1),
    )
    second_step = record_agent_step(
        state=first_step.state,
        node_name=AgentNodeName.METADATA_SEARCH,
        limits=limits,
        tool_call=tool_call,
        now=now + timedelta(seconds=2),
    )
    third_step = record_agent_step(
        state=second_step.state,
        node_name=AgentNodeName.METADATA_SEARCH,
        limits=limits,
        tool_call=tool_call,
        now=now + timedelta(seconds=3),
    )

    assert len(third_step.state.last_tool_calls) == 3
    assert third_step.decision.should_stop is True
    assert third_step.decision.reason == (
        "Agent repeated the same tool call too many times."
    )
    assert third_step.state.handoff_required is True


def test_generation_without_authorized_context_is_blocked() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
    )

    decision = evaluate_agent_guardrails(
        state=state,
        next_node_name=AgentNodeName.GENERATE_ANSWER,
        now=now + timedelta(seconds=1),
    )

    assert decision.should_stop is True
    assert decision.status == AgentRunStatus.HANDOFF_REQUIRED
    assert decision.reason == "Answer generation requires authorized context."
    assert decision.fallback_answer == SAFE_FALLBACK_ANSWER


def test_record_agent_step_safe_fallback_does_not_mutate_original_state() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    limits = AgentLimits(max_steps=1)
    state = start_agent_state(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=limits,
        now=now,
    )
    first_step = record_agent_step(
        state=state,
        node_name=AgentNodeName.CLASSIFY_INTENT,
        limits=limits,
        now=now + timedelta(seconds=1),
    )

    fallback_step = record_agent_step(
        state=first_step.state,
        node_name=AgentNodeName.REWRITE_QUERY,
        limits=limits,
        now=now + timedelta(seconds=2),
    )

    assert state.final_answer is None
    assert state.handoff_required is False
    assert fallback_step.state.final_answer == SAFE_FALLBACK_ANSWER
    assert fallback_step.state.handoff_required is True
