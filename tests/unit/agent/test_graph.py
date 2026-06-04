from collections.abc import Iterator
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.agent import graph as agent_graph
from agentic_rag.agent.graph import (
    build_agent_runtime_graph,
    run_agent_runtime_graph,
    stream_agent_runtime_graph,
)
from agentic_rag.agent.runtime import SAFE_FALLBACK_ANSWER, start_agent_state
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.agent_runs import cancel_agent_run, get_agent_run
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.schemas.agent import (
    AgentGraphState,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
    AgentStreamEventType,
)
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.llm import (
    LLMResponse,
    LLMStreamEvent,
    LLMStreamEventType,
)
from agentic_rag.shared.schemas.retrieval import RetrievalFilters, RetrievalStrategy


class FakeSearchClient:
    def __init__(self, hits=None):
        self.document_id = uuid4()
        self.chunk_id = uuid4()
        self.search_body = None
        self.hits = hits
        if self.hits is None:
            self.hits = [
                {
                    "_score": 3.25,
                    "_source": {
                        "tenant_id": "tenant-a",
                        "workspace_id": "workspace-a",
                        "document_id": str(self.document_id),
                        "chunk_id": str(self.chunk_id),
                        "chunk_index": 1,
                        "content": "Full chunk content about incident response policy.",
                        "token_count": 7,
                        "section_path": "Security / Incident Response",
                        "page_number": 2,
                        "start_offset": 10,
                        "end_offset": 48,
                        "document_title": "Incident Response Policy",
                        "file_name": "incident-response.md",
                        "source_type": "upload",
                        "source_uri": "upload://incident-response.md",
                        "classification_level": "internal",
                    },
                    "highlight": {
                        "content": ["Highlighted incident response policy content."]
                    },
                },
            ]

    def search_chunks_bm25(self, search_body):
        self.search_body = search_body
        return self.hits

    def close(self):
        return None


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def test_run_agent_runtime_graph_generates_verified_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    search_client = FakeSearchClient()
    captured_llm_requests = []
    auth = AuthContext(
        user_id="user-1",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        roles=["analyst"],
        group_ids=["security"],
        acl_version=4,
    )

    def fake_generate_chat_completion(request):
        captured_llm_requests.append(request)
        return LLMResponse(
            text="The incident response policy content was found [1].",
            model="test-model",
            provider="test-provider",
            input_tokens=42,
            output_tokens=9,
            cost_estimate=0.002,
            latency_ms=15,
            metadata=request.metadata,
        )

    monkeypatch.setattr(
        agent_graph,
        "generate_chat_completion",
        fake_generate_chat_completion,
    )

    result = run_agent_runtime_graph(
        agent_run_id=agent_run_id,
        auth=auth,
        query="  Find the incident response policy  ",
        retrieval_filters=RetrievalFilters(workspace_id="workspace-a"),
        now=now,
        search_client=search_client,
    )
    search_body = search_client.search_body
    bool_query = search_body["query"]["bool"]
    filter_clauses = bool_query["filter"]
    must_not_clauses = bool_query["must_not"]
    acl_should = filter_clauses[-1]["bool"]["should"]

    assert result.status == AgentRunStatus.COMPLETED
    assert result.stop_reason is None
    assert result.state.agent_run_id == agent_run_id
    assert result.state.auth == auth
    assert result.state.intent == "retrieve_and_answer"
    assert result.state.rewritten_query == "Find the incident response policy"
    assert result.state.filters == {"query": "Find the incident response policy"}
    assert result.state.retrieval_strategy == RetrievalStrategy.BM25
    assert result.state.step_count == 8
    assert result.state.tool_call_count == 0
    assert len(result.state.retrieved_candidates) == 1
    assert len(result.state.authorized_chunks) == 1
    assert len(result.state.context) == 1
    assert len(result.state.citations) == 1
    assert (
        result.state.draft_answer
        == "The incident response policy content was found [1]."
    )
    assert (
        result.state.final_answer
        == "The incident response policy content was found [1]."
    )
    assert result.state.confidence_score == 1.0
    assert len(captured_llm_requests) == 1
    assert captured_llm_requests[0].metadata["tenant_id"] == "tenant-a"
    assert captured_llm_requests[0].metadata["user_id"] == "user-1"
    assert captured_llm_requests[0].metadata["context_chunks"] == 1
    assert "Highlighted incident response policy content." in (
        captured_llm_requests[0].messages[1].content
    )
    assert (
        result.state.context[0].content
        == "Highlighted incident response policy content."
    )
    assert {"term": {"tenant_id": "tenant-a"}} in filter_clauses
    assert {"term": {"workspace_id": "workspace-a"}} in filter_clauses
    assert {"range": {"acl_version": {"lte": 4}}} in filter_clauses
    assert {"term": {"denied_user_ids": "user-1"}} in must_not_clauses
    assert {"terms": {"denied_group_ids": ["security"]}} in must_not_clauses
    assert {"terms": {"allowed_group_ids": ["security"]}} in acl_should
    assert {"terms": {"allowed_roles": ["analyst"]}} in acl_should
    assert result.state.visited_nodes == [
        AgentNodeName.CLASSIFY_INTENT.value,
        AgentNodeName.REWRITE_QUERY.value,
        AgentNodeName.PLAN_FILTERS.value,
        AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value,
        AgentNodeName.BM25_SEARCH.value,
        AgentNodeName.BUILD_CONTEXT.value,
        AgentNodeName.GENERATE_ANSWER.value,
        AgentNodeName.VERIFY_GROUNDING.value,
    ]


def test_run_agent_runtime_graph_returns_checkpoints_for_each_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(
        agent_graph,
        "generate_chat_completion",
        lambda request: LLMResponse(
            text="The incident response policy content was found [1].",
            model="test-model",
            provider="test-provider",
            input_tokens=42,
            output_tokens=9,
            cost_estimate=0.002,
            latency_ms=15,
            metadata=request.metadata,
        ),
    )

    result = run_agent_runtime_graph(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
        search_client=FakeSearchClient(),
    )

    assert len(result.checkpoints) == 8
    assert result.checkpoints[0].checkpoint_key == "step-0001-classify_intent"
    assert result.checkpoints[1].checkpoint_key == "step-0002-rewrite_query"
    assert result.checkpoints[2].checkpoint_key == "step-0003-plan_filters"
    assert result.checkpoints[3].checkpoint_key == (
        "step-0004-select_retrieval_strategy"
    )
    assert result.checkpoints[4].checkpoint_key == "step-0005-bm25_search"
    assert result.checkpoints[5].checkpoint_key == "step-0006-build_context"
    assert result.checkpoints[6].checkpoint_key == "step-0007-generate_answer"
    assert result.checkpoints[7].checkpoint_key == "step-0008-verify_grounding"
    assert result.checkpoints[-1].state["retrieval_strategy"] == RetrievalStrategy.BM25
    assert result.checkpoints[-1].state["step_count"] == 8
    assert len(result.checkpoints[-1].state["context"]) == 1
    assert result.checkpoints[-1].state["final_answer"] == (
        "The incident response policy content was found [1]."
    )
    replay_metadata = result.checkpoints[-1].state["replay_metadata"]
    assert (
        replay_metadata[AgentNodeName.BM25_SEARCH.value]["tool_name"] == "bm25_search"
    )
    assert (
        replay_metadata[AgentNodeName.BM25_SEARCH.value]["output"]["candidate_count"]
        == 1
    )
    assert replay_metadata[AgentNodeName.BUILD_CONTEXT.value]["tool_name"] == (
        "context_builder"
    )
    assert (
        replay_metadata[AgentNodeName.BUILD_CONTEXT.value]["output"][
            "context_chunk_count"
        ]
        == 1
    )
    assert replay_metadata[AgentNodeName.GENERATE_ANSWER.value]["tool_name"] == (
        "llm_gateway"
    )
    assert (
        replay_metadata[AgentNodeName.GENERATE_ANSWER.value]["output"]["llm_model"]
        == "test-model"
    )
    assert replay_metadata[AgentNodeName.VERIFY_GROUNDING.value]["tool_name"] == (
        "answer_verifier"
    )
    assert (
        replay_metadata[AgentNodeName.VERIFY_GROUNDING.value]["output"]["passed"]
        is True
    )


def test_run_agent_runtime_graph_blocks_generation_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def fail_if_llm_is_called(request):
        raise AssertionError("LLM should not be called without authorized context.")

    monkeypatch.setattr(
        agent_graph,
        "generate_chat_completion",
        fail_if_llm_is_called,
    )

    result = run_agent_runtime_graph(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
        search_client=FakeSearchClient(hits=[]),
    )

    assert result.status == AgentRunStatus.HANDOFF_REQUIRED
    assert result.stop_reason == "Answer generation requires authorized context."
    assert result.state.context == []
    assert result.state.final_answer == SAFE_FALLBACK_ANSWER
    assert result.state.handoff_required is True
    assert result.state.visited_nodes == [
        AgentNodeName.CLASSIFY_INTENT.value,
        AgentNodeName.REWRITE_QUERY.value,
        AgentNodeName.PLAN_FILTERS.value,
        AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value,
        AgentNodeName.BM25_SEARCH.value,
        AgentNodeName.BUILD_CONTEXT.value,
        AgentNodeName.GENERATE_ANSWER.value,
    ]
    assert len(result.checkpoints) == 7
    assert result.checkpoints[-1].checkpoint_key == "step-0007-generate_answer"


def test_run_agent_runtime_graph_rejects_ungrounded_generated_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(
        agent_graph,
        "generate_chat_completion",
        lambda request: LLMResponse(
            text="The incident response policy content was found.",
            model="test-model",
            provider="test-provider",
            input_tokens=42,
            output_tokens=9,
            cost_estimate=0.002,
            latency_ms=15,
            metadata=request.metadata,
        ),
    )

    result = run_agent_runtime_graph(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
        search_client=FakeSearchClient(),
    )

    assert result.status == AgentRunStatus.HANDOFF_REQUIRED
    assert result.stop_reason == "Answer did not cite retrieved context."
    assert (
        result.state.draft_answer == "The incident response policy content was found."
    )
    assert result.state.final_answer == SAFE_FALLBACK_ANSWER
    assert result.state.handoff_required is True
    assert result.state.confidence_score == 0.0
    assert result.state.visited_nodes[-2:] == [
        AgentNodeName.GENERATE_ANSWER.value,
        AgentNodeName.VERIFY_GROUNDING.value,
    ]
    assert result.checkpoints[-1].checkpoint_key == "step-0008-verify_grounding"


def test_stream_agent_runtime_graph_yields_tokens_and_completed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    captured_llm_requests = []

    def fake_stream_chat_completion(request):
        captured_llm_requests.append(request)
        yield LLMStreamEvent(
            event=LLMStreamEventType.TOKEN,
            text_delta="The incident response ",
            model="test-model",
            provider="test-provider",
            metadata=request.metadata,
        )
        yield LLMStreamEvent(
            event=LLMStreamEventType.TOKEN,
            text_delta="policy content was found [1].",
            model="test-model",
            provider="test-provider",
            metadata=request.metadata,
        )
        yield LLMStreamEvent(
            event=LLMStreamEventType.COMPLETED,
            text="The incident response policy content was found [1].",
            model="test-model",
            provider="test-provider",
            input_tokens=42,
            output_tokens=9,
            cost_estimate=0.002,
            latency_ms=15,
            metadata=request.metadata,
        )

    monkeypatch.setattr(
        agent_graph,
        "stream_chat_completion",
        fake_stream_chat_completion,
    )

    events = list(
        stream_agent_runtime_graph(
            agent_run_id=agent_run_id,
            auth=AuthContext(
                user_id="user-1",
                tenant_id="tenant-a",
                workspace_id="workspace-a",
                roles=["analyst"],
                group_ids=["security"],
                acl_version=4,
            ),
            query="Find the incident response policy",
            retrieval_filters=RetrievalFilters(workspace_id="workspace-a"),
            now=now,
            search_client=FakeSearchClient(),
        )
    )

    event_types = [event.event for event in events]
    step_events = [
        event
        for event in events
        if event.event == AgentStreamEventType.AGENT_STEP_COMPLETED
    ]
    token_events = [
        event for event in events if event.event == AgentStreamEventType.ANSWER_TOKEN
    ]

    assert event_types[0] == AgentStreamEventType.AGENT_STARTED
    assert event_types[-1] == AgentStreamEventType.AGENT_COMPLETED
    assert [event.text_delta for event in token_events] == [
        "The incident response ",
        "policy content was found [1].",
    ]
    assert [event.node_name for event in step_events] == [
        AgentNodeName.CLASSIFY_INTENT.value,
        AgentNodeName.REWRITE_QUERY.value,
        AgentNodeName.PLAN_FILTERS.value,
        AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value,
        AgentNodeName.BM25_SEARCH.value,
        AgentNodeName.BUILD_CONTEXT.value,
        AgentNodeName.GENERATE_ANSWER.value,
        AgentNodeName.VERIFY_GROUNDING.value,
    ]
    assert step_events[-1].status == AgentRunStatus.COMPLETED
    assert events[-1].status == AgentRunStatus.COMPLETED
    assert events[-1].data["answer"] == (
        "The incident response policy content was found [1]."
    )
    assert events[-1].data["retrieval_strategy"] == "bm25"
    assert events[-1].data["context_token_count"] > 0
    assert len(events[-1].data["citations"]) == 1
    assert len(events[-1].data["context"]) == 1
    assert events[-1].data["checkpoint_count"] == 8
    assert len(captured_llm_requests) == 1
    assert captured_llm_requests[0].metadata["tenant_id"] == "tenant-a"
    assert captured_llm_requests[0].metadata["context_chunks"] == 1


def test_stream_agent_runtime_graph_persists_replay_metadata(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()

    def fake_stream_chat_completion(request):
        yield LLMStreamEvent(
            event=LLMStreamEventType.TOKEN,
            text_delta="The incident response ",
            model="test-model",
            provider="test-provider",
            metadata=request.metadata,
        )
        yield LLMStreamEvent(
            event=LLMStreamEventType.COMPLETED,
            text="The incident response policy content was found [1].",
            model="test-model",
            provider="test-provider",
            input_tokens=42,
            output_tokens=9,
            cost_estimate=0.002,
            latency_ms=15,
            metadata=request.metadata,
        )

    monkeypatch.setattr(
        agent_graph,
        "stream_chat_completion",
        fake_stream_chat_completion,
    )
    db.add(
        Tenant(
            tenant_id="tenant-a",
            name="Tenant A",
            slug="tenant-a",
            status="active",
            metadata_={},
        )
    )
    db.commit()

    events = list(
        stream_agent_runtime_graph(
            agent_run_id=agent_run_id,
            auth=AuthContext(
                user_id="user-1",
                tenant_id="tenant-a",
                workspace_id="workspace-a",
            ),
            query="Find the incident response policy",
            retrieval_filters=RetrievalFilters(workspace_id="workspace-a"),
            now=now,
            db=db,
            search_client=FakeSearchClient(),
        )
    )
    stored = get_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
    )

    assert events[-1].event == AgentStreamEventType.AGENT_COMPLETED
    assert stored is not None
    assert stored.status == AgentRunStatus.COMPLETED.value
    assert stored.total_steps == 8
    assert stored.total_tool_calls == 0
    stored_steps_by_node = {step.node_name: step for step in stored.steps}
    assert stored_steps_by_node[AgentNodeName.GENERATE_ANSWER.value].tool_name == (
        "llm_gateway"
    )
    assert (
        stored_steps_by_node[AgentNodeName.GENERATE_ANSWER.value].tool_input[
            "context_chunk_count"
        ]
        == 1
    )
    assert "Streamed answer generation produced" in (
        stored_steps_by_node[AgentNodeName.GENERATE_ANSWER.value].tool_output_summary
    )
    replay_metadata = stored.checkpoints[-1].state["replay_metadata"]
    assert (
        replay_metadata[AgentNodeName.GENERATE_ANSWER.value]["output"]["streamed"]
        is True
    )
    assert (
        replay_metadata[AgentNodeName.GENERATE_ANSWER.value]["output"][
            "token_event_count"
        ]
        == 1
    )


def test_stream_agent_runtime_graph_blocks_generation_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def fail_if_llm_stream_is_called(request):
        raise AssertionError("LLM stream should not be called without context.")

    monkeypatch.setattr(
        agent_graph,
        "stream_chat_completion",
        fail_if_llm_stream_is_called,
    )

    events = list(
        stream_agent_runtime_graph(
            agent_run_id=uuid4(),
            auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
            query="Find the incident response policy",
            now=now,
            search_client=FakeSearchClient(hits=[]),
        )
    )
    step_events = [
        event
        for event in events
        if event.event == AgentStreamEventType.AGENT_STEP_COMPLETED
    ]
    token_events = [
        event for event in events if event.event == AgentStreamEventType.ANSWER_TOKEN
    ]

    assert token_events == []
    assert step_events[-1].node_name == AgentNodeName.GENERATE_ANSWER.value
    assert step_events[-1].status == AgentRunStatus.HANDOFF_REQUIRED
    assert step_events[-1].data["stop_reason"] == (
        "Answer generation requires authorized context."
    )
    assert events[-1].event == AgentStreamEventType.AGENT_COMPLETED
    assert events[-1].status == AgentRunStatus.HANDOFF_REQUIRED
    assert events[-1].data["answer"] == SAFE_FALLBACK_ANSWER


def test_stream_agent_runtime_graph_emits_failed_event_when_llm_stream_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def fake_stream_chat_completion(request):
        yield LLMStreamEvent(
            event=LLMStreamEventType.TOKEN,
            text_delta="Partial answer ",
            model="test-model",
            provider="test-provider",
            metadata=request.metadata,
        )
        raise RuntimeError("stream failed")

    monkeypatch.setattr(
        agent_graph,
        "stream_chat_completion",
        fake_stream_chat_completion,
    )

    events = list(
        stream_agent_runtime_graph(
            agent_run_id=uuid4(),
            auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
            query="Find the incident response policy",
            now=now,
            search_client=FakeSearchClient(),
        )
    )
    step_events = [
        event
        for event in events
        if event.event == AgentStreamEventType.AGENT_STEP_COMPLETED
    ]
    token_events = [
        event for event in events if event.event == AgentStreamEventType.ANSWER_TOKEN
    ]

    assert [event.text_delta for event in token_events] == ["Partial answer "]
    assert step_events[-1].node_name == AgentNodeName.GENERATE_ANSWER.value
    assert step_events[-1].status == AgentRunStatus.HANDOFF_REQUIRED
    assert step_events[-1].data["error_type"] == "RuntimeError"
    assert events[-1].event == AgentStreamEventType.AGENT_FAILED
    assert events[-1].status == AgentRunStatus.HANDOFF_REQUIRED
    assert events[-1].data["answer"] == SAFE_FALLBACK_ANSWER


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


def test_build_agent_runtime_graph_invokes_existing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    auth = AuthContext(user_id="user-1", tenant_id="tenant-a")
    monkeypatch.setattr(
        agent_graph,
        "generate_chat_completion",
        lambda request: LLMResponse(
            text="The incident response policy content was found [1].",
            model="test-model",
            provider="test-provider",
            input_tokens=42,
            output_tokens=9,
            cost_estimate=0.002,
            latency_ms=15,
            metadata=request.metadata,
        ),
    )
    agent_state = start_agent_state(
        agent_run_id=agent_run_id,
        auth=auth,
        query="Find the incident response policy",
        now=now,
    )
    graph_state = AgentGraphState(
        agent_state=agent_state,
        current_time=now,
        search_client=FakeSearchClient(),
    )

    compiled_graph = build_agent_runtime_graph()
    raw_result = compiled_graph.invoke(graph_state)
    completed_graph_state = AgentGraphState.model_validate(raw_result)

    assert completed_graph_state.agent_state.agent_run_id == agent_run_id
    assert completed_graph_state.status == AgentRunStatus.COMPLETED
    assert completed_graph_state.agent_state.step_count == 8
    assert len(completed_graph_state.checkpoints) == 8


def test_run_agent_runtime_graph_persists_steps_and_checkpoints(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    monkeypatch.setattr(
        agent_graph,
        "generate_chat_completion",
        lambda request: LLMResponse(
            text="The incident response policy content was found [1].",
            model="test-model",
            provider="test-provider",
            input_tokens=42,
            output_tokens=9,
            cost_estimate=0.002,
            latency_ms=15,
            metadata=request.metadata,
        ),
    )
    db.add(
        Tenant(
            tenant_id="tenant-a",
            name="Tenant A",
            slug="tenant-a",
            status="active",
            metadata_={},
        )
    )
    db.commit()

    result = run_agent_runtime_graph(
        agent_run_id=agent_run_id,
        auth=AuthContext(
            user_id="user-1",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
        ),
        query="Find the incident response policy",
        retrieval_filters=RetrievalFilters(workspace_id="workspace-a"),
        retrieval_limit=5,
        max_context_chunks=3,
        max_context_tokens=500,
        now=now,
        db=db,
        search_client=FakeSearchClient(),
    )
    stored = get_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
    )

    assert result.status == AgentRunStatus.COMPLETED
    assert stored is not None
    assert stored.status == AgentRunStatus.COMPLETED.value
    assert stored.completed_at is not None
    assert stored.tenant_id == "tenant-a"
    assert stored.workspace_id == "workspace-a"
    assert stored.user_id == "user-1"
    assert stored.query_text == "Find the incident response policy"
    assert stored.total_steps == 8
    assert stored.total_tool_calls == 0
    assert len(result.state.context) == 1
    assert (
        result.state.final_answer
        == "The incident response policy content was found [1]."
    )
    assert [step.node_name for step in stored.steps] == result.state.visited_nodes
    assert [step.status for step in stored.steps] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    stored_steps_by_node = {step.node_name: step for step in stored.steps}
    assert stored_steps_by_node[AgentNodeName.BM25_SEARCH.value].tool_name == (
        "bm25_search"
    )
    assert (
        stored_steps_by_node[AgentNodeName.BM25_SEARCH.value].tool_input[
            "retrieval_limit"
        ]
        == 5
    )
    assert "authorized candidate chunks" in (
        stored_steps_by_node[AgentNodeName.BM25_SEARCH.value].tool_output_summary
    )
    assert stored_steps_by_node[AgentNodeName.BUILD_CONTEXT.value].tool_name == (
        "context_builder"
    )
    assert (
        stored_steps_by_node[AgentNodeName.BUILD_CONTEXT.value].tool_input[
            "max_context_chunks"
        ]
        == 3
    )
    assert "Context build selected" in (
        stored_steps_by_node[AgentNodeName.BUILD_CONTEXT.value].tool_output_summary
    )
    assert stored_steps_by_node[AgentNodeName.GENERATE_ANSWER.value].tool_name == (
        "llm_gateway"
    )
    assert (
        stored_steps_by_node[AgentNodeName.GENERATE_ANSWER.value].tool_input[
            "context_chunk_count"
        ]
        == 1
    )
    assert "Answer generation produced" in (
        stored_steps_by_node[AgentNodeName.GENERATE_ANSWER.value].tool_output_summary
    )
    assert stored_steps_by_node[AgentNodeName.VERIFY_GROUNDING.value].tool_name == (
        "answer_verifier"
    )
    assert (
        stored_steps_by_node[AgentNodeName.VERIFY_GROUNDING.value].tool_input[
            "citation_count"
        ]
        == 1
    )
    assert "Grounding verification passed" in (
        stored_steps_by_node[AgentNodeName.VERIFY_GROUNDING.value].tool_output_summary
    )
    assert [checkpoint.checkpoint_key for checkpoint in stored.checkpoints] == [
        "step-0001-classify_intent",
        "step-0002-rewrite_query",
        "step-0003-plan_filters",
        "step-0004-select_retrieval_strategy",
        "step-0005-bm25_search",
        "step-0006-build_context",
        "step-0007-generate_answer",
        "step-0008-verify_grounding",
    ]
    assert stored.checkpoints[-1].state["retrieval_strategy"] == RetrievalStrategy.BM25
    assert len(stored.checkpoints[-1].state["context"]) == 1
    assert stored.checkpoints[-1].state["final_answer"] == (
        "The incident response policy content was found [1]."
    )


def test_run_agent_runtime_graph_persists_guardrail_stop(db: Session) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    db.add(
        Tenant(
            tenant_id="tenant-a",
            name="Tenant A",
            slug="tenant-a",
            status="active",
            metadata_={},
        )
    )
    db.commit()

    result = run_agent_runtime_graph(
        agent_run_id=agent_run_id,
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        limits=AgentLimits(max_steps=1),
        now=now,
        db=db,
    )
    stored = get_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
    )

    assert result.status == AgentRunStatus.HANDOFF_REQUIRED
    assert stored is not None
    assert stored.status == AgentRunStatus.HANDOFF_REQUIRED.value
    assert stored.completed_at is not None
    assert stored.total_steps == 2
    assert len(stored.steps) == 2
    assert len(stored.checkpoints) == 2
    assert stored.steps[-1].status == AgentRunStatus.HANDOFF_REQUIRED.value
    assert stored.checkpoints[-1].state["final_answer"] == SAFE_FALLBACK_ANSWER


def test_run_agent_runtime_graph_stops_when_persisted_run_is_cancelled(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    db.add(
        Tenant(
            tenant_id="tenant-a",
            name="Tenant A",
            slug="tenant-a",
            status="active",
            metadata_={},
        )
    )
    db.commit()

    actual_cancelled_check = agent_graph.is_agent_run_cancelled
    cancellation_check_count = {"value": 0}

    def cancel_after_first_node_check(
        *,
        db: Session,
        agent_run_id: UUID,
        tenant_id: str,
    ) -> bool:
        cancellation_check_count["value"] += 1
        is_cancelled = actual_cancelled_check(
            db=db,
            agent_run_id=agent_run_id,
            tenant_id=tenant_id,
        )
        if cancellation_check_count["value"] == 1:
            cancel_agent_run(
                db=db,
                agent_run_id=agent_run_id,
                tenant_id=tenant_id,
            )
        return is_cancelled

    monkeypatch.setattr(
        agent_graph,
        "is_agent_run_cancelled",
        cancel_after_first_node_check,
    )

    result = run_agent_runtime_graph(
        agent_run_id=agent_run_id,
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
        db=db,
    )
    stored = get_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
    )

    assert result.status == AgentRunStatus.CANCELLED
    assert result.stop_reason == "Agent run was cancelled."
    assert result.state.intent == "retrieve_and_answer"
    assert result.state.rewritten_query is None
    assert result.state.visited_nodes == [
        AgentNodeName.CLASSIFY_INTENT.value,
        AgentNodeName.REWRITE_QUERY.value,
    ]
    assert len(result.checkpoints) == 2
    assert cancellation_check_count["value"] == 2
    assert stored is not None
    assert stored.status == AgentRunStatus.CANCELLED.value
    assert stored.completed_at is not None
    assert stored.total_steps == 2
    assert len(stored.steps) == 2
    assert len(stored.checkpoints) == 2
    assert [step.status for step in stored.steps] == [
        "completed",
        AgentRunStatus.CANCELLED.value,
    ]
    assert stored.checkpoints[-1].checkpoint_key == "step-0002-rewrite_query"
