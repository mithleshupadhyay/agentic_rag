import json
from collections.abc import Iterator

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.search.opensearch import OpenSearchClient, build_chunk_search_document
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def add_chunk(db: Session):
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
            workspace_id="workspace-a",
            source_type=DocumentSourceType.UPLOAD,
            source_uri="upload://security.md",
            title="Security Guide",
            metadata={"department": "security"},
            acl=AclPolicy(
                visibility=Visibility.GROUP,
                allowed_group_ids=["security"],
                allowed_roles=["analyst"],
                acl_version=4,
            ),
        ),
    )
    document.file_name = "security.md"
    document.status = "ready"
    db.commit()
    db.refresh(document)
    chunks = replace_document_chunks(
        db=db,
        document=document,
        chunks=[
            {
                "chunk_index": 0,
                "content": "Security policy content for BM25 indexing.",
                "content_hash": "hash-1",
                "token_count": 6,
                "start_offset": 0,
                "end_offset": 42,
                "metadata": {"section": "overview"},
            }
        ],
    )
    return chunks[0]


def test_build_chunk_search_document_includes_tenant_acl_and_content(db: Session) -> None:
    chunk = add_chunk(db)

    payload = build_chunk_search_document(chunk)

    assert payload["tenant_id"] == "tenant-a"
    assert payload["workspace_id"] == "workspace-a"
    assert payload["chunk_id"] == str(chunk.id)
    assert payload["content"] == "Security policy content for BM25 indexing."
    assert payload["document_title"] == "Security Guide"
    assert payload["file_name"] == "security.md"
    assert payload["visibility"] == Visibility.GROUP
    assert payload["allowed_group_ids"] == ["security"]
    assert payload["allowed_roles"] == ["analyst"]
    assert payload["acl_version"] == 4


def test_opensearch_client_creates_index_and_bulk_indexes_chunk(db: Session) -> None:
    chunk = add_chunk(db)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(404, request=request)
        if request.method == "PUT":
            return httpx.Response(200, json={"acknowledged": True}, request=request)
        if request.method == "POST" and request.url.path == "/_bulk":
            body_lines = request.content.decode("utf-8").strip().splitlines()
            assert json.loads(body_lines[0])["index"]["_index"] == "chunks-test"
            assert json.loads(body_lines[1])["tenant_id"] == "tenant-a"
            return httpx.Response(200, json={"errors": False, "items": []}, request=request)
        return httpx.Response(500, request=request)

    client = OpenSearchClient(
        base_url="http://opensearch:9200",
        username="",
        password="",
        index_name="chunks-test",
        transport=httpx.MockTransport(handler),
    )

    client.ensure_chunk_index()
    indexed_count = client.bulk_index_chunks([chunk])
    client.close()

    assert indexed_count == 1
    assert [request.method for request in requests] == ["GET", "PUT", "POST"]
