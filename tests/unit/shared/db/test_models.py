from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from scripts import seed_local_db as db_seed
from agentic_rag.shared.config import Settings
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.models import (
    AgentCheckpoint,
    AgentRun,
    AgentStep,
    ChunkAcl,
    Document,
    DocumentAcl,
    DocumentChunk,
    IngestionJob,
    QueryRun,
    Tenant,
    User,
)


def test_database_url_uses_sync_driver_for_alembic() -> None:
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/app"
    )

    assert settings.sync_database_url == (
        "postgresql+psycopg://user:password@localhost:5432/app"
    )


def test_kafka_settings_use_kafka_environment_names() -> None:
    settings = Settings(
        KAFKA_PUBLISHING_ENABLED=True,
        KAFKA_CONSUMING_ENABLED=True,
        KAFKA_BOOTSTRAP_SERVERS="kafka-a:9092,kafka-b:9092",
        KAFKA_CLIENT_ID="worker-a",
        KAFKA_PRODUCER_FLUSH_TIMEOUT_SECONDS=2.5,
        KAFKA_INGESTION_CONSUMER_GROUP="ingestion-group-a",
        KAFKA_EMBEDDING_CONSUMER_GROUP="embedding-group-a",
        KAFKA_INDEXING_CONSUMER_GROUP="indexing-group-a",
    )

    assert settings.kafka_publishing_enabled is True
    assert settings.kafka_consuming_enabled is True
    assert settings.kafka_bootstrap_servers == "kafka-a:9092,kafka-b:9092"
    assert settings.kafka_client_id == "worker-a"
    assert settings.kafka_producer_flush_timeout_seconds == 2.5
    assert settings.kafka_ingestion_consumer_group == "ingestion-group-a"
    assert settings.kafka_embedding_consumer_group == "embedding-group-a"
    assert settings.kafka_indexing_consumer_group == "indexing-group-a"


def test_opensearch_settings_use_index_and_alias_names() -> None:
    settings = Settings(
        OPENSEARCH_CHUNK_INDEX="chunks-v2",
        OPENSEARCH_CHUNK_READ_ALIAS="chunks-read",
        OPENSEARCH_CHUNK_WRITE_ALIAS="chunks-write",
    )

    assert settings.opensearch_chunk_index == "chunks-v2"
    assert settings.opensearch_chunk_read_alias == "chunks-read"
    assert settings.opensearch_chunk_write_alias == "chunks-write"


def test_redis_and_llm_circuit_breaker_settings() -> None:
    settings = Settings(
        REDIS_URL="redis://redis:6379/1",
        REDIS_SOCKET_TIMEOUT_SECONDS=2.5,
        QUERY_CACHE_ENABLED=True,
        QUERY_CACHE_TTL_SECONDS=120,
        QUERY_CACHE_KEY_PREFIX="agentic-rag:test:query",
        LLM_CIRCUIT_BREAKER_STATE_BACKEND="redis",
        LLM_CIRCUIT_BREAKER_REDIS_KEY_PREFIX="agentic-rag:test:circuit",
    )

    assert settings.redis_url == "redis://redis:6379/1"
    assert settings.redis_socket_timeout_seconds == 2.5
    assert settings.query_cache_enabled is True
    assert settings.query_cache_ttl_seconds == 120
    assert settings.query_cache_key_prefix == "agentic-rag:test:query"
    assert settings.llm_circuit_breaker_state_backend == "redis"
    assert settings.llm_circuit_breaker_redis_key_prefix == "agentic-rag:test:circuit"


def test_metadata_contains_core_tables() -> None:
    expected_tables = {
        "tenants",
        "users",
        "roles",
        "groups",
        "user_roles",
        "user_groups",
        "documents",
        "document_chunks",
        "document_acl",
        "chunk_acl",
        "chunk_embeddings",
        "ingestion_jobs",
        "query_runs",
        "agent_runs",
        "agent_steps",
        "agent_checkpoints",
    }

    assert expected_tables.issubset(Base.metadata.tables)


def test_models_create_sqlite_schema_for_unit_tests() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    assert "documents" in inspect(engine).get_table_names()
    assert "document_chunks" in inspect(engine).get_table_names()
    assert "ingestion_jobs" in inspect(engine).get_table_names()
    assert "query_runs" in inspect(engine).get_table_names()
    assert "agent_runs" in inspect(engine).get_table_names()
    assert "agent_steps" in inspect(engine).get_table_names()
    assert "agent_checkpoints" in inspect(engine).get_table_names()


def test_seed_local_development_data_creates_tenant_and_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    test_settings = Settings(
        AUTH_PROVIDER="local",
        LOCAL_TENANT_ID="local-tenant",
        LOCAL_USER_ID="local-user",
        LOCAL_ACL_VERSION=3,
    )

    monkeypatch.setattr(db_seed, "settings", test_settings)
    monkeypatch.setattr(db_seed, "get_sync_session_factory", lambda: SessionLocal)

    db_seed.seed_local_development_data()
    db_seed.seed_local_development_data()

    with Session(engine) as session:
        tenants = session.scalars(select(Tenant)).all()
        users = session.scalars(select(User)).all()

        assert len(tenants) == 1
        assert tenants[0].tenant_id == "local-tenant"
        assert tenants[0].metadata_["source"] == "local-seed"
        assert len(users) == 1
        assert users[0].tenant_id == "local-tenant"
        assert users[0].external_subject == "local-user"
        assert users[0].acl_version == 3


def test_tenant_document_chunk_acl_flow_can_persist() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        tenant = Tenant(
            tenant_id="tenant-a",
            name="Tenant A",
            slug="tenant-a",
            status="active",
            metadata_={},
        )
        user = User(
            tenant_id="tenant-a",
            external_subject="user-1",
            email="user@example.com",
            status="active",
            acl_version=1,
            metadata_={},
        )
        document = Document(
            tenant_id="tenant-a",
            source_type="upload",
            title="Security policy",
            status="ready",
            acl_version=1,
            classification_level="internal",
            metadata_={"topic": "security"},
            is_deleted=False,
        )
        chunk = DocumentChunk(
            tenant_id="tenant-a",
            document=document,
            chunk_index=0,
            content="Only authorized users can read this policy.",
            content_hash="chunk-hash-1",
            token_count=8,
            metadata_={"section": "access"},
            acl_version=1,
            classification_level="internal",
            is_deleted=False,
        )
        document_acl = DocumentAcl(
            tenant_id="tenant-a",
            document=document,
            visibility="group",
            allowed_user_ids=[],
            allowed_group_ids=["security"],
            allowed_roles=[],
            denied_user_ids=[],
            denied_group_ids=[],
            acl_version=1,
        )
        chunk_acl = ChunkAcl(
            tenant_id="tenant-a",
            chunk=chunk,
            visibility="group",
            allowed_user_ids=[],
            allowed_group_ids=["security"],
            allowed_roles=[],
            denied_user_ids=[],
            denied_group_ids=[],
            acl_version=1,
        )
        ingestion_job = IngestionJob(
            tenant_id="tenant-a",
            document=document,
            source_type="upload",
            status="completed",
            current_stage="complete",
            retry_count=0,
            max_retries=3,
            metadata_={},
        )
        query_run = QueryRun(
            tenant_id="tenant-a",
            workspace_id=None,
            user_id="user-1",
            request_id="request-id-1",
            query_text="What does the policy say?",
            filters={},
            status="completed",
            retrieval_strategy="bm25",
            answer="Only authorized users can read this policy.",
            citations={"items": []},
            candidates={"items": []},
            context={"items": []},
            response_payload={},
            retrieval_limit=20,
            max_context_chunks=12,
            max_context_tokens=6000,
            context_token_count=8,
            synthesis_enabled=False,
            llm_input_tokens=0,
            llm_output_tokens=0,
            llm_cost_estimate=0.0,
        )
        agent_run = AgentRun(
            tenant_id="tenant-a",
            workspace_id=None,
            user_id="user-1",
            query_text="What does the policy say?",
            status="running",
            retrieval_strategy="bm25",
            confidence_score=0.0,
            total_steps=1,
            total_tool_calls=0,
            limits={"max_steps": 8},
            timeout_at=now,
            started_at=now,
        )
        agent_step = AgentStep(
            tenant_id="tenant-a",
            agent_run=agent_run,
            node_name="classify_intent",
            step_number=1,
            tool_input={},
            status="completed",
        )
        agent_checkpoint = AgentCheckpoint(
            tenant_id="tenant-a",
            agent_run=agent_run,
            checkpoint_key="step-0001-classify_intent",
            state={"step_count": 1},
        )

        session.add_all(
            [
                tenant,
                user,
                document,
                chunk,
                document_acl,
                chunk_acl,
                ingestion_job,
                query_run,
                agent_run,
                agent_step,
                agent_checkpoint,
            ]
        )
        session.commit()

        stored_document = session.scalars(select(Document)).one()

        assert stored_document.tenant_id == "tenant-a"
        assert stored_document.chunks[0].acl.allowed_group_ids == ["security"]
        assert stored_document.ingestion_jobs[0].status == "completed"
        stored_query_run = session.scalars(select(QueryRun)).one()
        assert stored_query_run.tenant_id == "tenant-a"
        assert stored_query_run.request_id == "request-id-1"
        assert stored_query_run.verification_status == "not_required"
        assert stored_query_run.verification_reason is None
        stored_agent_run = session.scalars(select(AgentRun)).one()
        assert stored_agent_run.tenant_id == "tenant-a"
        assert stored_agent_run.steps[0].node_name == "classify_intent"
        assert stored_agent_run.checkpoints[0].checkpoint_key == (
            "step-0001-classify_intent"
        )
