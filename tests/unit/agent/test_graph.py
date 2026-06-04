from collections.abc import Iterator
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.agent import graph as agent_graph
from agentic_rag.agent.graph import build_agent_runtime_graph, run_agent_runtime_graph
from agentic_rag.agent.runtime import SAFE_FALLBACK_ANSWER, start_agent_state
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.agent_runs import cancel_agent_run, get_agent_run
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.schemas.agent import (
    AgentGraphState,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
)
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.retrieval import RetrievalFilters, RetrievalStrategy


class FakeSearchClient:
    def __init__(self):
        self.document_id = uuid4()
        self.chunk_id = uuid4()
        self.search_body = None

    def search_chunks_bm25(self, search_body):
        self.search_body = search_body
        return [
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
            }
        ]

    def close(self):
        return None


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def test_run_agent_runtime_graph_reaches_context_boundary() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    search_client = FakeSearchClient()
    auth = AuthContext(
        user_id="user-1",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        roles=["analyst"],
        group_ids=["security"],
        acl_version=4,
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

    assert result.status == AgentRunStatus.RUNNING
    assert result.stop_reason is None
    assert result.state.agent_run_id == agent_run_id
    assert result.state.auth == auth
    assert result.state.intent == "retrieve_and_answer"
    assert result.state.rewritten_query == "Find the incident response policy"
    assert result.state.filters == {"query": "Find the incident response policy"}
    assert result.state.retrieval_strategy == RetrievalStrategy.BM25
    assert result.state.step_count == 6
    assert result.state.tool_call_count == 0
    assert len(result.state.retrieved_candidates) == 1
    assert len(result.state.authorized_chunks) == 1
    assert len(result.state.context) == 1
    assert len(result.state.citations) == 1
    assert result.state.context[0].content == "Highlighted incident response policy content."
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
    ]


def test_run_agent_runtime_graph_returns_checkpoints_for_each_node() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = run_agent_runtime_graph(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
        now=now,
        search_client=FakeSearchClient(),
    )

    assert len(result.checkpoints) == 6
    assert result.checkpoints[0].checkpoint_key == "step-0001-classify_intent"
    assert result.checkpoints[1].checkpoint_key == "step-0002-rewrite_query"
    assert result.checkpoints[2].checkpoint_key == "step-0003-plan_filters"
    assert result.checkpoints[3].checkpoint_key == (
        "step-0004-select_retrieval_strategy"
    )
    assert result.checkpoints[4].checkpoint_key == "step-0005-bm25_search"
    assert result.checkpoints[5].checkpoint_key == "step-0006-build_context"
    assert result.checkpoints[-1].state["retrieval_strategy"] == RetrievalStrategy.BM25
    assert result.checkpoints[-1].state["step_count"] == 6
    assert len(result.checkpoints[-1].state["context"]) == 1


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
    graph_state = AgentGraphState(
        agent_state=agent_state,
        current_time=now,
        search_client=FakeSearchClient(),
    )

    compiled_graph = build_agent_runtime_graph()
    raw_result = compiled_graph.invoke(graph_state)
    completed_graph_state = AgentGraphState.model_validate(raw_result)

    assert completed_graph_state.agent_state.agent_run_id == agent_run_id
    assert completed_graph_state.status == AgentRunStatus.RUNNING
    assert completed_graph_state.agent_state.step_count == 6
    assert len(completed_graph_state.checkpoints) == 6


def test_run_agent_runtime_graph_persists_steps_and_checkpoints(db: Session) -> None:
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

    assert result.status == AgentRunStatus.RUNNING
    assert stored is not None
    assert stored.status == AgentRunStatus.RUNNING.value
    assert stored.tenant_id == "tenant-a"
    assert stored.workspace_id == "workspace-a"
    assert stored.user_id == "user-1"
    assert stored.query_text == "Find the incident response policy"
    assert stored.total_steps == 6
    assert stored.total_tool_calls == 0
    assert len(result.state.context) == 1
    assert [step.node_name for step in stored.steps] == result.state.visited_nodes
    assert [step.status for step in stored.steps] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert [checkpoint.checkpoint_key for checkpoint in stored.checkpoints] == [
        "step-0001-classify_intent",
        "step-0002-rewrite_query",
        "step-0003-plan_filters",
        "step-0004-select_retrieval_strategy",
        "step-0005-bm25_search",
        "step-0006-build_context",
    ]
    assert stored.checkpoints[-1].state["retrieval_strategy"] == RetrievalStrategy.BM25
    assert len(stored.checkpoints[-1].state["context"]) == 1


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
