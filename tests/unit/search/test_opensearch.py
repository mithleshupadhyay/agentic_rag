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
    assert payload["owner_user_id"] == "user-1"
    assert payload["file_name"] == "security.md"
    assert payload["document_metadata"] == {"department": "security"}
    assert payload["chunk_metadata"] == {"section": "overview"}
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
            payload = json.loads(request.content.decode("utf-8"))
            properties = payload["mappings"]["properties"]
            assert properties["document_metadata"] == {"type": "object", "dynamic": True}
            assert properties["chunk_metadata"] == {"type": "object", "dynamic": True}
            assert payload["aliases"]["chunks-read-test"] == {}
            assert payload["aliases"]["chunks-write-test"] == {"is_write_index": True}
            return httpx.Response(200, json={"acknowledged": True}, request=request)
        if request.method == "POST" and request.url.path == "/_bulk":
            body_lines = request.content.decode("utf-8").strip().splitlines()
            assert json.loads(body_lines[0])["index"]["_index"] == "chunks-write-test"
            assert json.loads(body_lines[1])["tenant_id"] == "tenant-a"
            return httpx.Response(200, json={"errors": False, "items": []}, request=request)
        return httpx.Response(500, request=request)

    client = OpenSearchClient(
        base_url="http://opensearch:9200",
        username="",
        password="",
        index_name="chunks-test",
        chunk_read_alias="chunks-read-test",
        chunk_write_alias="chunks-write-test",
        transport=httpx.MockTransport(handler),
    )

    client.ensure_chunk_index()
    indexed_count = client.bulk_index_chunks([chunk])
    client.close()

    assert indexed_count == 1
    assert [request.method for request in requests] == ["GET", "PUT", "POST"]


def test_opensearch_client_ensures_aliases_for_existing_chunk_index() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/chunks-test":
            return httpx.Response(200, json={}, request=request)
        if request.method == "POST" and request.url.path == "/_aliases":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["actions"] == [
                {
                    "add": {
                        "index": "chunks-test",
                        "alias": "chunks-read-test",
                    }
                },
                {
                    "add": {
                        "index": "chunks-test",
                        "alias": "chunks-write-test",
                        "is_write_index": True,
                    }
                },
            ]
            return httpx.Response(200, json={"acknowledged": True}, request=request)
        return httpx.Response(500, request=request)

    client = OpenSearchClient(
        base_url="http://opensearch:9200",
        username="",
        password="",
        index_name="chunks-test",
        chunk_read_alias="chunks-read-test",
        chunk_write_alias="chunks-write-test",
        transport=httpx.MockTransport(handler),
    )

    client.ensure_chunk_index()
    client.close()

    assert [request.method for request in requests] == ["GET", "POST"]


def test_opensearch_client_reports_bulk_index_item_failures(db: Session) -> None:
    chunk = add_chunk(db)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/_bulk":
            return httpx.Response(
                200,
                json={
                    "errors": True,
                    "items": [
                        {
                            "index": {
                                "_index": "chunks-test",
                                "_id": str(chunk.id),
                                "status": 400,
                                "error": {
                                    "type": "mapper_parsing_exception",
                                    "reason": "raw private chunk content should not leak",
                                },
                            }
                        }
                    ],
                },
                request=request,
            )
        return httpx.Response(500, request=request)

    client = OpenSearchClient(
        base_url="http://opensearch:9200",
        username="",
        password="",
        index_name="chunks-test",
        chunk_read_alias="chunks-read-test",
        chunk_write_alias="chunks-write-test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.bulk_index_chunks([chunk])

    client.close()

    error_message = str(exc_info.value)
    assert "failed_count=1" in error_message
    assert "statuses=400:1" in error_message
    assert "error_types=mapper_parsing_exception:1" in error_message
    assert "raw private chunk content" not in error_message


def test_opensearch_client_searches_bm25_chunks() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/chunks-read-test/_search":
            body = json.loads(request.content.decode("utf-8"))
            assert body["query"]["bool"]["filter"][0]["term"]["tenant_id"] == "tenant-a"
            return httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [
                            {
                                "_score": 2.5,
                                "_source": {
                                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                                },
                            }
                        ]
                    }
                },
                request=request,
            )
        return httpx.Response(500, request=request)

    client = OpenSearchClient(
        base_url="http://opensearch:9200",
        username="",
        password="",
        index_name="chunks-test",
        chunk_read_alias="chunks-read-test",
        chunk_write_alias="chunks-write-test",
        transport=httpx.MockTransport(handler),
    )

    hits = client.search_chunks_bm25(
        {
            "query": {
                "bool": {
                    "filter": [{"term": {"tenant_id": "tenant-a"}}],
                }
            }
        }
    )
    client.close()

    assert hits[0]["_score"] == 2.5
    assert [request.method for request in requests] == ["POST"]


def test_opensearch_client_retries_bm25_search_retryable_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_score": 3.0,
                            "_source": {
                                "chunk_id": "00000000-0000-0000-0000-000000000001",
                            },
                        }
                    ]
                }
            },
            request=request,
        )

    client = OpenSearchClient(
        base_url="http://opensearch:9200",
        username="",
        password="",
        index_name="chunks-test",
        chunk_read_alias="chunks-read-test",
        chunk_write_alias="chunks-write-test",
        search_retry_attempts=2,
        search_retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )

    hits = client.search_chunks_bm25({"query": {"match_all": {}}})
    client.close()

    assert hits[0]["_score"] == 3.0
    assert len(requests) == 2
    assert [request.url.path for request in requests] == [
        "/chunks-read-test/_search",
        "/chunks-read-test/_search",
    ]


def test_opensearch_client_raises_after_bm25_search_retries_exhausted() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, request=request)

    client = OpenSearchClient(
        base_url="http://opensearch:9200",
        username="",
        password="",
        index_name="chunks-test",
        chunk_read_alias="chunks-read-test",
        chunk_write_alias="chunks-write-test",
        search_retry_attempts=2,
        search_retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.search_chunks_bm25({"query": {"match_all": {}}})

    client.close()

    assert "HTTP status error" in str(exc_info.value)
    assert len(requests) == 2
