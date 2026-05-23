from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.main import app
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import DocumentSourceType


@pytest.fixture()
def client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
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
        session.commit()

    current_user = UserContext(
        id="owner-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        roles=[],
        group_ids=[],
        scopes=[
            "documents:read",
            "documents:write",
            "documents:delete",
        ],
        acl_version=5,
    )

    def override_get_session() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    async def override_get_current_user() -> UserContext:
        return current_user

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user
    test_client = TestClient(app)
    test_client.current_user = current_user  # type: ignore[attr-defined]

    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)


def set_current_user(
    client: TestClient,
    user_id: str,
    tenant_id: str = "tenant-a",
    roles: list[str] | None = None,
    group_ids: list[str] | None = None,
    scopes: list[str] | None = None,
    acl_version: int = 5,
) -> None:
    client.current_user.id = user_id  # type: ignore[attr-defined]
    client.current_user.customer_id = tenant_id  # type: ignore[attr-defined]
    client.current_user.tenant_id = tenant_id  # type: ignore[attr-defined]
    client.current_user.roles = roles or []  # type: ignore[attr-defined]
    client.current_user.group_ids = group_ids or []  # type: ignore[attr-defined]
    client.current_user.scopes = scopes or [  # type: ignore[attr-defined]
        "documents:read",
        "documents:write",
        "documents:delete",
    ]
    client.current_user.acl_version = acl_version  # type: ignore[attr-defined]


def create_document_payload(
    title: str = "Security Policy",
    visibility: Visibility = Visibility.PRIVATE,
    allowed_group_ids: list[str] | None = None,
    allowed_roles: list[str] | None = None,
) -> dict:
    return {
        "source_type": DocumentSourceType.UPLOAD,
        "title": title,
        "metadata": {
            "department": "security",
            "tags": ["security", "policy"],
        },
        "acl": AclPolicy(
            visibility=visibility,
            allowed_group_ids=allowed_group_ids or [],
            allowed_roles=allowed_roles or [],
            acl_version=5,
        ).model_dump(mode="json"),
    }


class FakeObjectStore:
    def __init__(self):
        self.uploads = []
        self.deleted = []

    def build_object_key(
        self,
        tenant_id: str,
        workspace_id: str | None,
        document_id,
        file_name: str,
    ) -> str:
        workspace = workspace_id or "default"
        return (
            f"tenants/{tenant_id}/workspaces/{workspace}/"
            f"documents/{document_id}/raw/{file_name}"
        )

    def put_bytes(
        self,
        object_key: str,
        data: bytes,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict:
        self.uploads.append(
            {
                "object_key": object_key,
                "data": data,
                "content_type": content_type,
                "metadata": metadata,
            }
        )
        return {
            "bucket": "agentic-rag-test",
            "object_key": object_key,
            "byte_size": len(data),
            "content_hash": "cc1e6f388e7836a3dcd07341509c3afa63759a3ca7e3fbbc4233f6ec64d85dec",
            "etag": '"etag-1"',
        }

    def delete_object(self, object_key: str) -> None:
        self.deleted.append(object_key)


def test_create_document_endpoint_returns_document(client: TestClient) -> None:
    response = client.post(
        "/documents",
        json=create_document_payload(),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["tenant_id"] == "tenant-a"
    assert body["owner_user_id"] == "owner-1"
    assert body["title"] == "Security Policy"
    assert body["metadata"]["department"] == "security"


def test_upload_document_endpoint_stores_object_and_creates_ingestion_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = FakeObjectStore()
    monkeypatch.setattr(
        "agentic_rag.api.documents.ObjectStoreClient",
        lambda: fake_store,
    )

    response = client.post(
        "/documents/upload",
        files={"file": ("security.txt", b"security policy", "text/plain")},
        data={
            "workspace_id": "workspace-a",
            "title": "Uploaded Security Policy",
            "metadata_json": '{"department": "security"}',
            "idempotency_key": "upload-security-1",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["bucket"] == "agentic-rag-test"
    assert body["object_key"] == fake_store.uploads[0]["object_key"]
    assert body["byte_size"] == len(b"security policy")
    assert body["ingestion_status"] == "queued"
    assert body["ingestion_stage"] == "created"
    assert body["document"]["tenant_id"] == "tenant-a"
    assert body["document"]["workspace_id"] == "workspace-a"
    assert body["document"]["title"] == "Uploaded Security Policy"
    assert body["document"]["file_name"] == "security.txt"
    assert body["document"]["object_key"] == body["object_key"]
    assert body["document"]["metadata"]["department"] == "security"
    assert fake_store.uploads[0]["content_type"] == "text/plain"
    assert fake_store.uploads[0]["metadata"]["tenant_id"] == "tenant-a"
    assert fake_store.uploads[0]["metadata"]["document_id"] == body["document"]["id"]


def test_upload_document_endpoint_rejects_invalid_metadata(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("security.txt", b"security policy", "text/plain")},
        data={"metadata_json": "[]"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "metadata_json must be a valid JSON object."


def test_upload_document_endpoint_deletes_object_when_job_creation_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = FakeObjectStore()
    monkeypatch.setattr(
        "agentic_rag.api.documents.ObjectStoreClient",
        lambda: fake_store,
    )

    def fail_create_ingestion_job_for_document(*args, **kwargs):
        raise HTTPException(status_code=400, detail="Failed to create ingestion job.")

    monkeypatch.setattr(
        "agentic_rag.api.documents.create_ingestion_job_for_document",
        fail_create_ingestion_job_for_document,
    )

    response = client.post(
        "/documents/upload",
        files={"file": ("security.txt", b"security policy", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Failed to create ingestion job."
    assert fake_store.deleted == [fake_store.uploads[0]["object_key"]]


def test_create_document_endpoint_requires_write_scope(client: TestClient) -> None:
    set_current_user(
        client,
        user_id="owner-1",
        scopes=["documents:read"],
    )

    response = client.post(
        "/documents",
        json=create_document_payload(),
    )

    assert response.status_code == 403


def test_get_document_endpoint_enforces_document_acl(client: TestClient) -> None:
    create_response = client.post(
        "/documents",
        json=create_document_payload(),
    )
    document_id = create_response.json()["id"]

    set_current_user(client, user_id="user-2")
    denied_response = client.get(f"/documents/{document_id}")

    set_current_user(client, user_id="owner-1")
    allowed_response = client.get(f"/documents/{document_id}")

    assert denied_response.status_code == 403
    assert allowed_response.status_code == 200
    assert allowed_response.json()["id"] == document_id


def test_list_documents_filters_unreadable_documents(client: TestClient) -> None:
    client.post(
        "/documents",
        json=create_document_payload(
            title="Private owner document",
            visibility=Visibility.PRIVATE,
        ),
    )
    client.post(
        "/documents",
        json=create_document_payload(
            title="Tenant visible document",
            visibility=Visibility.TENANT,
        ),
    )

    set_current_user(client, user_id="user-2")
    response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["page"]["total"] == 1
    assert [item["title"] for item in body["items"]] == ["Tenant visible document"]


def test_search_documents_endpoint_applies_metadata_and_acl(client: TestClient) -> None:
    client.post(
        "/documents",
        json=create_document_payload(
            title="Security group document",
            visibility=Visibility.GROUP,
            allowed_group_ids=["security"],
        ),
    )
    client.post(
        "/documents",
        json=create_document_payload(
            title="Manager document",
            visibility=Visibility.GROUP,
            allowed_roles=["manager"],
        ),
    )

    set_current_user(client, user_id="user-2", group_ids=["security"])
    response = client.post(
        "/documents/search",
        json={
            "page": {"page": 1, "size": 10},
            "metadata_filters": {"department": "security"},
            "tags": ["policy"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["page"]["total"] == 1
    assert [item["title"] for item in body["items"]] == ["Security group document"]


def test_update_document_endpoint_requires_write_permission(client: TestClient) -> None:
    create_response = client.post(
        "/documents",
        json=create_document_payload(),
    )
    document_id = create_response.json()["id"]

    set_current_user(client, user_id="user-2")
    denied_response = client.patch(
        f"/documents/{document_id}",
        json={"title": "Bad update"},
    )

    set_current_user(client, user_id="owner-1")
    allowed_response = client.patch(
        f"/documents/{document_id}",
        json={"title": "Updated policy"},
    )

    assert denied_response.status_code == 403
    assert allowed_response.status_code == 200
    assert allowed_response.json()["title"] == "Updated policy"


def test_delete_and_restore_document_endpoints(client: TestClient) -> None:
    create_response = client.post(
        "/documents",
        json=create_document_payload(),
    )
    document_id = create_response.json()["id"]

    delete_response = client.delete(f"/documents/{document_id}")
    read_deleted_response = client.get(f"/documents/{document_id}")
    restore_response = client.post(f"/documents/{document_id}/restore")
    read_restored_response = client.get(f"/documents/{document_id}")

    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "id": document_id,
        "status": "deleted",
    }
    assert read_deleted_response.status_code == 404
    assert restore_response.status_code == 200
    assert restore_response.json()["is_deleted"] is False
    assert read_restored_response.status_code == 200


def test_cross_tenant_document_is_not_found(client: TestClient) -> None:
    create_response = client.post(
        "/documents",
        json=create_document_payload(),
    )
    document_id = create_response.json()["id"]

    set_current_user(client, user_id="owner-2", tenant_id="tenant-b")
    response = client.get(f"/documents/{document_id}")

    assert response.status_code == 404
