from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.models import (
    ChunkAcl,
    Document,
    DocumentAcl,
    DocumentChunk,
    IngestionJob,
    Tenant,
    User,
)
from agentic_rag.shared.config import Settings


def test_database_url_uses_sync_driver_for_alembic() -> None:
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/app"
    )

    assert settings.sync_database_url == (
        "postgresql+psycopg://user:password@localhost:5432/app"
    )


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
    }

    assert expected_tables.issubset(Base.metadata.tables)


def test_models_create_sqlite_schema_for_unit_tests() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    assert "documents" in inspect(engine).get_table_names()
    assert "document_chunks" in inspect(engine).get_table_names()
    assert "ingestion_jobs" in inspect(engine).get_table_names()


def test_tenant_document_chunk_acl_flow_can_persist() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
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

        session.add_all(
            [
                tenant,
                user,
                document,
                chunk,
                document_acl,
                chunk_acl,
                ingestion_job,
            ]
        )
        session.commit()

        stored_document = session.scalars(select(Document)).one()

        assert stored_document.tenant_id == "tenant-a"
        assert stored_document.chunks[0].acl.allowed_group_ids == ["security"]
        assert stored_document.ingestion_jobs[0].status == "completed"
