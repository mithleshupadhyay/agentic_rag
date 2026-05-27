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
from agentic_rag.shared.schemas.query import (
    QueryRequest,
    QueryResponse,
    QueryRunListItem,
    QueryRunRead,
    QueryRunStatus,
)
from agentic_rag.shared.schemas.retrieval import (
    BM25SearchRequest,
    HybridSearchRequest,
    RetrievalStrategy,
    VectorSearchRequest,
)


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
    assert response.candidates == []
    assert response.context == []
    assert response.context_token_count == 0
    assert response.synthesis_enabled is False
    assert response.llm_provider is None
    assert response.llm_model is None
    assert response.llm_input_tokens == 0
    assert response.llm_output_tokens == 0
    assert response.llm_cost_estimate == 0.0
    assert response.synthesis_error is None


def test_query_run_response_models_include_request_id() -> None:
    now = datetime.now(timezone.utc)
    agent_run_id = uuid4()
    read_model = QueryRunRead(
        agent_run_id=agent_run_id,
        status=QueryRunStatus.COMPLETED,
        tenant_id="tenant-a",
        user_id="user-1",
        request_id="request-id-1",
        query="Find policy",
        created_at=now,
        updated_at=now,
    )
    list_item = QueryRunListItem(
        agent_run_id=agent_run_id,
        status=QueryRunStatus.COMPLETED,
        user_id="user-1",
        request_id="request-id-1",
        query="Find policy",
        created_at=now,
    )

    assert read_model.request_id == "request-id-1"
    assert list_item.request_id == "request-id-1"


def test_query_request_defaults() -> None:
    request = QueryRequest(query="Find PCI documents")

    assert request.retrieval_limit == 20
    assert request.max_context_chunks == 12
    assert request.max_context_tokens == 6000
    assert request.filters.document_ids == []


def test_bm25_search_request_defaults() -> None:
    request = BM25SearchRequest(query="Find PCI documents")

    assert request.limit == 20
    assert request.filters.document_ids == []
    assert request.deadline_ms == 1500


def test_vector_search_request_defaults_and_similarity_validation() -> None:
    request = VectorSearchRequest(query="Find PCI documents")

    assert request.limit == 20
    assert request.min_similarity == 0.0
    assert request.filters.document_ids == []
    assert request.deadline_ms == 1500

    with pytest.raises(ValidationError):
        VectorSearchRequest(query="Find PCI documents", min_similarity=1.2)


def test_hybrid_search_request_defaults_and_similarity_validation() -> None:
    request = HybridSearchRequest(query="Find PCI documents")

    assert request.limit == 20
    assert request.min_similarity == 0.0
    assert request.filters.document_ids == []
    assert request.deadline_ms == 1500

    with pytest.raises(ValidationError):
        HybridSearchRequest(query="Find PCI documents", min_similarity=1.2)


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
        embedding_model="gemini/gemini-embedding-001",
    )
    envelope = EventEnvelope(
        event_type=EventType.DOCUMENT_EMBED_REQUESTED,
        tenant_id="tenant-a",
        correlation_id="req-1",
        payload=payload.model_dump(mode="json"),
    )

    assert envelope.event_version == 1
    assert envelope.payload["embedding_model"] == "gemini/gemini-embedding-001"
