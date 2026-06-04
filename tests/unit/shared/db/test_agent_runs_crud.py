from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.agent_runs import (
    cancel_agent_run,
    create_agent_run,
    get_agent_run,
    is_agent_run_cancelled,
    record_agent_run_step,
    save_agent_checkpoint,
)
from agentic_rag.shared.db.models import AgentCheckpoint as AgentCheckpointModel
from agentic_rag.shared.db.models import AgentStep, Tenant
from agentic_rag.shared.schemas.agent import (
    AgentCheckpoint,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
    ToolCallRecord,
)
from agentic_rag.shared.schemas.auth import AuthContext


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def test_create_agent_run_record_step_save_checkpoint_and_fetch(db: Session) -> None:
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
    created_before = datetime.now(timezone.utc)

    agent_run = create_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        auth=AuthContext(
            user_id="user-1",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
        ),
        query="  Find the incident response policy  ",
        limits=AgentLimits(total_timeout_seconds=30),
    )
    step = record_agent_run_step(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
        node_name=AgentNodeName.BM25_SEARCH,
        step_number=1,
        tool_call=ToolCallRecord(
            tool_name="bm25_search",
            arguments={"query": "incident response"},
            latency_ms=12,
        ),
        latency_ms=12,
    )
    checkpoint = save_agent_checkpoint(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
        checkpoint=AgentCheckpoint(
            agent_run_id=agent_run_id,
            checkpoint_key="step-0001-bm25_search",
            state={"step_count": 1, "tool_call_count": 1},
            created_at=now,
        ),
    )
    stored = get_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
    )

    assert agent_run.id == agent_run_id
    assert agent_run.tenant_id == "tenant-a"
    assert agent_run.workspace_id == "workspace-a"
    assert agent_run.user_id == "user-1"
    assert agent_run.query_text == "Find the incident response policy"
    assert agent_run.status == AgentRunStatus.RUNNING.value
    timeout_at = agent_run.timeout_at
    if timeout_at.tzinfo is None:
        timeout_at = timeout_at.replace(tzinfo=timezone.utc)
    assert created_before + timedelta(seconds=30) <= timeout_at
    assert timeout_at <= datetime.now(timezone.utc) + timedelta(seconds=30)
    assert step is not None
    assert step.node_name == AgentNodeName.BM25_SEARCH.value
    assert step.tool_name == "bm25_search"
    assert step.tool_input == {"query": "incident response"}
    assert step.latency_ms == 12
    assert checkpoint is not None
    assert checkpoint.checkpoint_key == "step-0001-bm25_search"
    assert checkpoint.state == {"step_count": 1, "tool_call_count": 1}
    assert stored is not None
    assert stored.total_steps == 1
    assert stored.total_tool_calls == 1
    assert len(stored.steps) == 1
    assert len(stored.checkpoints) == 1


def test_agent_run_crud_is_tenant_scoped(db: Session) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    agent_run_id = uuid4()
    db.add_all(
        [
            Tenant(
                tenant_id="tenant-a",
                name="Tenant A",
                slug="tenant-a",
                status="active",
                metadata_={},
            ),
            Tenant(
                tenant_id="tenant-b",
                name="Tenant B",
                slug="tenant-b",
                status="active",
                metadata_={},
            ),
        ]
    )
    db.commit()

    create_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
    )

    tenant_b_step = record_agent_run_step(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-b",
        node_name=AgentNodeName.CLASSIFY_INTENT,
        step_number=1,
    )
    tenant_b_checkpoint = save_agent_checkpoint(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-b",
        checkpoint=AgentCheckpoint(
            agent_run_id=agent_run_id,
            checkpoint_key="step-0001-classify_intent",
            state={"step_count": 1},
            created_at=now,
        ),
    )

    assert get_agent_run(db, agent_run_id, "tenant-a") is not None
    assert get_agent_run(db, agent_run_id, "tenant-b") is None
    assert tenant_b_step is None
    assert tenant_b_checkpoint is None
    assert db.query(AgentStep).count() == 0
    assert db.query(AgentCheckpointModel).count() == 0


def test_cancel_agent_run_marks_active_run_cancelled(db: Session) -> None:
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

    create_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
    )

    assert is_agent_run_cancelled(db, agent_run_id, "tenant-a") is False
    assert is_agent_run_cancelled(db, agent_run_id, "tenant-b") is False

    cancelled = cancel_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
    )

    assert cancelled is not None
    assert cancelled.status == AgentRunStatus.CANCELLED.value
    assert cancelled.completed_at is not None
    assert is_agent_run_cancelled(db, agent_run_id, "tenant-a") is True


def test_cancel_agent_run_rejects_terminal_status(db: Session) -> None:
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

    create_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
    )
    record_agent_run_step(
        db=db,
        agent_run_id=agent_run_id,
        tenant_id="tenant-a",
        node_name=AgentNodeName.HUMAN_HANDOFF,
        step_number=1,
        status=AgentRunStatus.HANDOFF_REQUIRED.value,
    )

    with pytest.raises(HTTPException) as exc_info:
        cancel_agent_run(
            db=db,
            agent_run_id=agent_run_id,
            tenant_id="tenant-a",
        )

    stored = get_agent_run(db, agent_run_id, "tenant-a")

    assert exc_info.value.status_code == 409
    assert stored is not None
    assert stored.status == AgentRunStatus.HANDOFF_REQUIRED.value


def test_save_agent_checkpoint_rejects_mismatched_run_id(db: Session) -> None:
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

    create_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
    )

    with pytest.raises(HTTPException) as exc_info:
        save_agent_checkpoint(
            db=db,
            agent_run_id=agent_run_id,
            tenant_id="tenant-a",
            checkpoint=AgentCheckpoint(
                agent_run_id=uuid4(),
                checkpoint_key="step-0001-classify_intent",
                state={"step_count": 1},
                created_at=now,
            ),
        )

    assert exc_info.value.status_code == 400
    assert db.query(AgentCheckpointModel).count() == 0


def test_create_agent_run_rolls_back_on_integrity_error(db: Session) -> None:
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

    create_agent_run(
        db=db,
        agent_run_id=agent_run_id,
        auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
        query="Find the incident response policy",
    )

    with pytest.raises(HTTPException) as exc_info:
        create_agent_run(
            db=db,
            agent_run_id=agent_run_id,
            auth=AuthContext(user_id="user-1", tenant_id="tenant-a"),
            query="Find the incident response policy again",
        )

    assert exc_info.value.status_code == 400
    stored = get_agent_run(db, agent_run_id, "tenant-a")
    assert stored is not None
    assert stored.query_text == "Find the incident response policy"
