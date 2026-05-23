from uuid import uuid4

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.schemas.retrieval import (
    RetrievalFilters,
    RetrievalStrategy,
    RetrievalTool,
)
from agentic_rag.retrieval.bm25_search import search_bm25_chunks


class FakeSearchClient:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.search_body = None
        self.closed = False

    def search_chunks_bm25(self, search_body):
        self.search_body = search_body
        return self.hits

    def close(self):
        self.closed = True


def test_search_bm25_chunks_builds_tenant_acl_filters_and_candidates() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    search_client = FakeSearchClient(
        hits=[
            {
                "_score": 3.25,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(chunk_id),
                    "chunk_index": 1,
                    "content": "Full chunk content about security policy.",
                    "token_count": 7,
                    "section_path": "Security / Policy",
                    "page_number": 2,
                    "start_offset": 10,
                    "end_offset": 48,
                    "document_title": "Security Policy",
                    "file_name": "security.md",
                    "source_type": "upload",
                    "source_uri": "upload://security.md",
                    "classification_level": "internal",
                },
                "highlight": {
                    "content": ["Highlighted security policy content."]
                },
            }
        ]
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        roles=["analyst"],
        group_ids=["security"],
        acl_version=4,
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        filters=RetrievalFilters(
            workspace_id="workspace-a",
            document_ids=[document_id],
            source_types=["upload"],
        ),
        limit=5,
        search_client=search_client,
    )

    search_body = search_client.search_body
    bool_query = search_body["query"]["bool"]
    filter_clauses = bool_query["filter"]
    must_not_clauses = bool_query["must_not"]
    acl_should = filter_clauses[-1]["bool"]["should"]

    assert response.strategy == RetrievalStrategy.BM25
    assert len(response.candidates) == 1
    assert response.candidates[0].source == RetrievalTool.BM25_SEARCH
    assert response.candidates[0].chunk_id == chunk_id
    assert response.candidates[0].document_id == document_id
    assert response.candidates[0].content == "Highlighted security policy content."
    assert response.candidates[0].citation.title == "Security Policy"
    assert response.candidates[0].citation.page_number == 2
    assert {"term": {"tenant_id": "tenant-a"}} in filter_clauses
    assert {"term": {"workspace_id": "workspace-a"}} in filter_clauses
    assert {"terms": {"document_id": [str(document_id)]}} in filter_clauses
    assert {"terms": {"source_type": ["upload"]}} in filter_clauses
    assert {"range": {"acl_version": {"lte": 4}}} in filter_clauses
    assert {"term": {"denied_user_ids": "user-1"}} in must_not_clauses
    assert {"terms": {"denied_group_ids": ["security"]}} in must_not_clauses
    assert {"term": {"owner_user_id": "user-1"}} in acl_should
    assert {"term": {"allowed_user_ids": "user-1"}} in acl_should
    assert {"terms": {"allowed_group_ids": ["security"]}} in acl_should
    assert {"terms": {"allowed_roles": ["analyst"]}} in acl_should
    assert {"terms": {"visibility": ["public", "tenant"]}} in acl_should
    assert search_body["size"] == 5


def test_search_bm25_chunks_uses_admin_acl_clause() -> None:
    search_client = FakeSearchClient()
    user_context = UserContext(
        id="admin-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        roles=["admin"],
        group_ids=[],
        acl_version=2,
    )

    search_bm25_chunks(
        user_context=user_context,
        query="architecture",
        search_client=search_client,
    )

    acl_should = search_client.search_body["query"]["bool"]["filter"][-1]["bool"]["should"]
    assert acl_should == [{"match_all": {}}]


def test_search_bm25_chunks_returns_empty_for_workspace_mismatch() -> None:
    search_client = FakeSearchClient()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        filters=RetrievalFilters(workspace_id="workspace-b"),
        search_client=search_client,
    )

    assert response.candidates == []
    assert search_client.search_body is None
