from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import (
    attach_document_object,
    create_document,
    create_ingestion_job_for_document,
    delete_document,
    get_document,
    list_documents,
    restore_document,
    search_documents,
    update_document_by_id,
)
from agentic_rag.shared.db.models import DocumentChunk, Tenant
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.common import PageRequest
from agentic_rag.shared.schemas.documents import (
    ClassificationLevel,
    DocumentCreateRequest,
    DocumentSearchRequest,
    DocumentSourceType,
    DocumentStatus,
    DocumentUpdateRequest,
    FileMetadata,
)


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def add_tenant(db: Session, tenant_id: str) -> None:
    db.add(
        Tenant(
            tenant_id=tenant_id,
            name=tenant_id.title(),
            slug=tenant_id,
            status="active",
            metadata_={},
        )
    )
    db.commit()


def test_create_document_persists_metadata_and_acl(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )
    request = DocumentCreateRequest(
        source_type=DocumentSourceType.UPLOAD,
        title="Security Policy",
        file=FileMetadata(
            file_name="security-policy.pdf",
            mime_type="application/pdf",
            byte_size=1024,
            content_hash="hash-1",
        ),
        metadata={"department": "security", "tags": ["policy"]},
        acl=AclPolicy(
            visibility=Visibility.GROUP,
            allowed_group_ids=["security"],
            acl_version=2,
        ),
    )

    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=request,
        object_key="raw/tenant-a/security-policy.pdf",
    )

    assert document.tenant_id == "tenant-a"
    assert document.workspace_id == "workspace-a"
    assert document.owner_user_id == "user-1"
    assert document.status == DocumentStatus.QUEUED
    assert document.file_name == "security-policy.pdf"
    assert document.object_key == "raw/tenant-a/security-policy.pdf"
    assert document.metadata_["department"] == "security"
    assert document.acl.visibility == Visibility.GROUP
    assert document.acl.allowed_group_ids == ["security"]
    assert document.acl.acl_version == 2


def test_attach_document_object_and_create_ingestion_job(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )
    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Uploaded policy",
            file=FileMetadata(
                file_name="policy.txt",
                mime_type="text/plain",
                byte_size=12,
                content_hash="hash-1",
            ),
            metadata={},
            acl=AclPolicy(),
        ),
    )

    document = attach_document_object(
        db=db,
        db_obj=document,
        object_key="tenants/tenant-a/workspaces/workspace-a/documents/raw/policy.txt",
    )
    ingestion_job = create_ingestion_job_for_document(
        user_context=user_context,
        db=db,
        document=document,
        idempotency_key="upload-policy-1",
        metadata={"content_hash": "hash-1"},
    )

    assert document.object_key == "tenants/tenant-a/workspaces/workspace-a/documents/raw/policy.txt"
    assert document.status == DocumentStatus.QUEUED
    assert ingestion_job.tenant_id == "tenant-a"
    assert ingestion_job.workspace_id == "workspace-a"
    assert ingestion_job.document_id == document.id
    assert ingestion_job.object_key == document.object_key
    assert ingestion_job.status == "queued"
    assert ingestion_job.current_stage == "created"
    assert ingestion_job.metadata_["content_hash"] == "hash-1"


def test_get_and_list_documents_are_tenant_scoped(db: Session) -> None:
    add_tenant(db, "tenant-a")
    add_tenant(db, "tenant-b")
    tenant_a_context = UserContext(
        id="user-a",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    tenant_b_context = UserContext(
        id="user-b",
        customer_id="tenant-b",
        tenant_id="tenant-b",
    )

    tenant_a_document = create_document(
        tenant_a_context,
        db,
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Tenant A document",
            metadata={},
            acl=AclPolicy(),
        ),
    )
    create_document(
        tenant_b_context,
        db,
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Tenant B document",
            metadata={},
            acl=AclPolicy(),
        ),
    )

    assert get_document(db, tenant_a_document.id, "tenant-a") is not None
    assert get_document(db, tenant_a_document.id, "tenant-b") is None

    tenant_a_documents = list_documents(db, "tenant-a")
    tenant_b_documents = list_documents(db, "tenant-b")

    assert [document.title for document in tenant_a_documents] == ["Tenant A document"]
    assert [document.title for document in tenant_b_documents] == ["Tenant B document"]


def test_update_document_changes_metadata_classification_and_acl(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    document = create_document(
        user_context,
        db,
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Old title",
            metadata={"department": "security"},
            acl=AclPolicy(),
        ),
    )

    updated = update_document_by_id(
        db,
        document.id,
        "tenant-a",
        DocumentUpdateRequest(
            title="New title",
            metadata={"department": "platform"},
            classification_level=ClassificationLevel.CONFIDENTIAL,
            acl=AclPolicy(
                visibility=Visibility.TENANT,
                allowed_roles=["admin"],
                acl_version=3,
            ),
        ),
    )

    assert updated.title == "New title"
    assert updated.metadata_ == {"department": "platform"}
    assert updated.classification_level == ClassificationLevel.CONFIDENTIAL
    assert updated.acl_version == 3
    assert updated.acl.visibility == Visibility.TENANT
    assert updated.acl.allowed_roles == ["admin"]


def test_update_document_by_id_rejects_cross_tenant_access(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    document = create_document(
        user_context,
        db,
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Private document",
            metadata={},
            acl=AclPolicy(),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        update_document_by_id(
            db,
            document.id,
            "tenant-b",
            DocumentUpdateRequest(title="Bad update"),
        )

    assert exc_info.value.status_code == 404


def test_delete_and_restore_document_soft_deletes_chunks(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    document = create_document(
        user_context,
        db,
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Chunked document",
            metadata={},
            acl=AclPolicy(),
        ),
    )
    chunk = DocumentChunk(
        tenant_id="tenant-a",
        document_id=document.id,
        chunk_index=0,
        content="Only authorized users should read this.",
        content_hash="chunk-hash-1",
        token_count=6,
        metadata_={},
        acl_version=1,
        classification_level="internal",
        is_deleted=False,
    )
    db.add(chunk)
    db.commit()

    delete_document(db, document.id, "tenant-a")

    assert get_document(db, document.id, "tenant-a") is None
    assert db.get(DocumentChunk, chunk.id).is_deleted is True

    restored = restore_document(db, document.id, "tenant-a")

    assert restored is not None
    assert restored.is_deleted is False
    assert restored.status == DocumentStatus.QUEUED
    assert db.get(DocumentChunk, chunk.id).is_deleted is False


def test_search_documents_applies_filters_and_pagination(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    first = create_document(
        user_context,
        db,
        DocumentCreateRequest(
            workspace_id="workspace-a",
            source_type=DocumentSourceType.UPLOAD,
            title="Security policy",
            metadata={"department": "security"},
            acl=AclPolicy(),
        ),
    )
    second = create_document(
        user_context,
        db,
        DocumentCreateRequest(
            workspace_id="workspace-a",
            source_type=DocumentSourceType.URL,
            title="Engineering runbook",
            metadata={"department": "engineering"},
            acl=AclPolicy(),
        ),
    )
    first.status = "ready"
    second.status = "ready"
    db.commit()

    items, total = search_documents(
        db,
        "tenant-a",
        DocumentSearchRequest(
            page=PageRequest(page=1, size=10),
            workspace_id="workspace-a",
            source_type=DocumentSourceType.UPLOAD,
            status=DocumentStatus.READY,
            owner_user_id="user-1",
            metadata_filters={"department": "security"},
        ),
    )

    assert total == 1
    assert [document.title for document in items] == ["Security policy"]


def test_search_documents_filters_by_metadata_tags(db: Session) -> None:
    add_tenant(db, "tenant-a")
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
    )
    create_document(
        user_context,
        db,
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Tagged policy",
            metadata={"tags": ["policy", "security"]},
            acl=AclPolicy(),
        ),
    )
    create_document(
        user_context,
        db,
        DocumentCreateRequest(
            source_type=DocumentSourceType.UPLOAD,
            title="Untagged runbook",
            metadata={"tags": ["runbook"]},
            acl=AclPolicy(),
        ),
    )

    items, total = search_documents(
        db,
        "tenant-a",
        DocumentSearchRequest(tags=["security"]),
    )

    assert total == 1
    assert [document.title for document in items] == ["Tagged policy"]
