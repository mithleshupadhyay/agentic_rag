from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agentic_rag.shared.kafka.events import (
    EmbedChunksPayload,
    EventEnvelope,
    EventType,
)
from agentic_rag.shared.schemas.agent import AgentLimits, AgentStateModel
from agentic_rag.shared.schemas.auth import AclPolicy, AuthContext, Visibility
from agentic_rag.shared.schemas.chunks import ChunkCreate, ChunkUpdate
from agentic_rag.shared.schemas.common import Citation, HealthResponse, PageRequest
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSourceType,
)
from agentic_rag.shared.schemas.query import QueryRequest, QueryResponse
from agentic_rag.shared.schemas.retrieval import RetrievalStrategy


def test_health_response_contract() -> None:
    response = HealthResponse(
        service="gateway-api",
        status="healthy",
        version="0.1.0",
    )

    assert response.model_dump()["service"] == "gateway-api"


def test_page_request_offset() -> None:
    assert PageRequest(page=3, size=25).offset == 50


def test_auth_context_defaults_are_isolated() -> None:
    first = AuthContext(user_id="u1", tenant_id="t1")
    second = AuthContext(user_id="u2", tenant_id="t1")

    first.roles.append("admin")

    assert first.roles == ["admin"]
    assert second.roles == []


def test_document_create_request_uses_acl_and_forbids_extra_fields() -> None:
    request = DocumentCreateRequest(
        source_type=DocumentSourceType.UPLOAD,
        title="PCI report",
        metadata={"customer": "acme"},
        acl=AclPolicy(
            visibility=Visibility.GROUP,
            allowed_group_ids=["security"],
        ),
    )

    assert request.acl.allowed_group_ids == ["security"]

    with pytest.raises(ValidationError):
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Bad payload",
            metadata={},
            acl=AclPolicy(),
            unexpected=True,
        )


def test_chunk_create_validates_token_count() -> None:
    with pytest.raises(ValidationError):
        ChunkCreate(
            document_id=uuid4(),
            chunk_index=0,
            content="hello",
            content_hash="abc",
            token_count=0,
        )


def test_chunk_update_is_partial_and_validates_acl_version() -> None:
    update = ChunkUpdate(metadata={"topic": "security"})

    assert update.metadata == {"topic": "security"}
    assert update.acl_version is None

    with pytest.raises(ValidationError):
        ChunkUpdate(acl_version=0)


def test_query_response_contract() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    response = QueryResponse(
        agent_run_id=uuid4(),
        answer="Use metadata and BM25 before vector search.",
        citations=[
            Citation(
                document_id=document_id,
                chunk_id=chunk_id,
                title="Architecture",
                score=0.91,
            )
        ],
        confidence_score=0.88,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        latency_ms=250,
    )

    assert response.citations[0].document_id == document_id
    assert response.retrieval_strategy == RetrievalStrategy.HYBRID


def test_query_request_defaults() -> None:
    request = QueryRequest(query="Find PCI documents")

    assert request.max_context_chunks == 12
    assert request.filters.document_ids == []


def test_agent_limits_validation() -> None:
    with pytest.raises(ValidationError):
        AgentLimits(max_steps=0)


def test_agent_state_model_contract() -> None:
    state = AgentStateModel(
        agent_run_id=uuid4(),
        auth=AuthContext(user_id="u1", tenant_id="t1"),
        query="What changed in policy documents?",
        deadline_at=datetime.now(timezone.utc),
    )

    assert state.step_count == 0
    assert state.retrieved_candidates == []


def test_kafka_event_envelope_and_payload() -> None:
    payload = EmbedChunksPayload(
        job_id=uuid4(),
        document_id=uuid4(),
        chunk_ids=[uuid4()],
        embedding_model="BAAI/bge-base-en-v1.5",
    )
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_EMBED_REQUESTED,
        tenant_id="tenant-a",
        correlation_id="req-1",
        payload=payload.model_dump(mode="json"),
    )

    assert envelope.event_version == 1
    assert envelope.payload["embedding_model"] == "BAAI/bge-base-en-v1.5"
