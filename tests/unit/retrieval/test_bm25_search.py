import logging
from uuid import uuid4

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.shared.schemas.retrieval import (
    RetrievalFilters,
    RetrievalStrategy,
    RetrievalTool,
)


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


def test_search_bm25_chunks_adds_exact_metadata_filters() -> None:
    search_client = FakeSearchClient()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        filters=RetrievalFilters(
            metadata={
                "department": "security",
                "published": True,
                "priority": 3,
            }
        ),
        search_client=search_client,
    )

    filter_clauses = search_client.search_body["query"]["bool"]["filter"]

    assert {
        "bool": {
            "should": [
                {"term": {"document_metadata.department.keyword": "security"}},
                {"term": {"chunk_metadata.department.keyword": "security"}},
            ],
            "minimum_should_match": 1,
        }
    } in filter_clauses
    assert {
        "bool": {
            "should": [
                {"term": {"document_metadata.published": True}},
                {"term": {"chunk_metadata.published": True}},
            ],
            "minimum_should_match": 1,
        }
    } in filter_clauses
    assert {
        "bool": {
            "should": [
                {"term": {"document_metadata.priority": 3}},
                {"term": {"chunk_metadata.priority": 3}},
            ],
            "minimum_should_match": 1,
        }
    } in filter_clauses


def test_search_bm25_chunks_rejects_nested_metadata_filter_key() -> None:
    search_client = FakeSearchClient()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        search_bm25_chunks(
            user_context=user_context,
            query="security policy",
            filters=RetrievalFilters(metadata={"security.department": "security"}),
            search_client=search_client,
        )

    assert exc_info.value.status_code == 400
    assert search_client.search_body is None


def test_search_bm25_chunks_adds_date_range_filters() -> None:
    search_client = FakeSearchClient()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        filters=RetrievalFilters(
            date_range={
                "created_at": {
                    "gte": "2026-06-01T00:00:00Z",
                    "lte": "2026-06-05T23:59:59Z",
                },
                "updated_at": {
                    "gt": "2026-06-02T00:00:00+00:00",
                    "lt": "2026-06-06T00:00:00+00:00",
                },
            }
        ),
        search_client=search_client,
    )

    filter_clauses = search_client.search_body["query"]["bool"]["filter"]

    assert {
        "range": {
            "created_at": {
                "gte": "2026-06-01T00:00:00Z",
                "lte": "2026-06-05T23:59:59Z",
            }
        }
    } in filter_clauses
    assert {
        "range": {
            "updated_at": {
                "gt": "2026-06-02T00:00:00+00:00",
                "lt": "2026-06-06T00:00:00+00:00",
            }
        }
    } in filter_clauses


def test_search_bm25_chunks_rejects_unknown_date_range_field() -> None:
    search_client = FakeSearchClient()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        search_bm25_chunks(
            user_context=user_context,
            query="security policy",
            filters=RetrievalFilters(
                date_range={
                    "document_date": {
                        "gte": "2026-06-01T00:00:00Z",
                    }
                }
            ),
            search_client=search_client,
        )

    assert exc_info.value.status_code == 400
    assert search_client.search_body is None


def test_search_bm25_chunks_rejects_invalid_date_range_value() -> None:
    search_client = FakeSearchClient()
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        search_bm25_chunks(
            user_context=user_context,
            query="security policy",
            filters=RetrievalFilters(
                date_range={
                    "created_at": {
                        "gte": "not-a-date",
                    }
                }
            ),
            search_client=search_client,
        )

    assert exc_info.value.status_code == 400
    assert search_client.search_body is None


def test_search_bm25_chunks_skips_invalid_hits_and_keeps_valid_hit(caplog) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    search_client = FakeSearchClient(
        hits=[
            [],
            {
                "_score": 2.5,
                "_source": None,
            },
            {
                "_score": "bad-score",
                "_source": {
                    "document_id": str(uuid4()),
                    "chunk_id": str(uuid4()),
                    "content": "Invalid score content.",
                },
            },
            {
                "_score": 2.5,
                "_source": {
                    "document_id": "not-a-uuid",
                    "chunk_id": str(uuid4()),
                    "content": "Invalid id content.",
                },
            },
            {
                "_score": 2.5,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(chunk_id),
                    "chunk_index": 1,
                    "content": "Valid fallback content.",
                    "token_count": 4,
                    "document_title": "Valid Result",
                    "source_uri": "upload://valid.md",
                },
            },
        ]
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    caplog.set_level(logging.WARNING, logger="agentic_rag.retrieval.bm25_search")

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        search_client=search_client,
    )

    assert len(response.candidates) == 1
    assert response.candidates[0].document_id == document_id
    assert response.candidates[0].chunk_id == chunk_id
    assert response.candidates[0].content == "Valid fallback content."
    assert "Skipping invalid BM25 hit" in caplog.text


def test_search_bm25_chunks_handles_invalid_highlight_fragments() -> None:
    highlighted_document_id = uuid4()
    highlighted_chunk_id = uuid4()
    fallback_document_id = uuid4()
    fallback_chunk_id = uuid4()
    search_client = FakeSearchClient(
        hits=[
            {
                "_score": 3.0,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(highlighted_document_id),
                    "chunk_id": str(highlighted_chunk_id),
                    "chunk_index": 1,
                    "content": "Raw content should not be used when a highlight is valid.",
                    "token_count": 8,
                    "document_title": "Highlighted Result",
                    "source_uri": "upload://highlighted.md",
                },
                "highlight": {
                    "content": [None, "", "   ", "First valid highlighted fragment."]
                },
            },
            {
                "_score": 2.0,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(fallback_document_id),
                    "chunk_id": str(fallback_chunk_id),
                    "chunk_index": 2,
                    "content": "Fallback raw content.",
                    "token_count": 3,
                    "document_title": "Fallback Result",
                    "source_uri": "upload://fallback.md",
                },
                "highlight": {
                    "content": [None, "", "   "]
                },
            },
        ]
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        search_client=search_client,
    )

    assert len(response.candidates) == 2
    assert response.candidates[0].chunk_id == highlighted_chunk_id
    assert response.candidates[0].content == "First valid highlighted fragment."
    assert response.candidates[0].citation.quote == "First valid highlighted fragment."
    assert response.candidates[1].chunk_id == fallback_chunk_id
    assert response.candidates[1].content == "Fallback raw content."
    assert response.candidates[1].citation.quote == "Fallback raw content."


def test_search_bm25_chunks_deduplicates_document_section_results() -> None:
    document_id = uuid4()
    lower_score_chunk_id = uuid4()
    higher_score_chunk_id = uuid4()
    first_no_section_chunk_id = uuid4()
    second_no_section_chunk_id = uuid4()
    search_client = FakeSearchClient(
        hits=[
            {
                "_score": 1.0,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(lower_score_chunk_id),
                    "chunk_index": 1,
                    "content": "Lower score duplicate section.",
                    "token_count": 4,
                    "section_path": "Security / Policy",
                    "document_title": "Security Policy",
                    "source_uri": "upload://security.md",
                },
            },
            {
                "_score": 4.0,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(higher_score_chunk_id),
                    "chunk_index": 2,
                    "content": "Higher score duplicate section.",
                    "token_count": 5,
                    "section_path": "Security / Policy",
                    "document_title": "Security Policy",
                    "source_uri": "upload://security.md",
                },
            },
            {
                "_score": 2.0,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(first_no_section_chunk_id),
                    "chunk_index": 3,
                    "content": "First chunk without section.",
                    "token_count": 4,
                    "document_title": "Security Policy",
                    "source_uri": "upload://security.md",
                },
            },
            {
                "_score": 1.5,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(second_no_section_chunk_id),
                    "chunk_index": 4,
                    "content": "Second chunk without section.",
                    "token_count": 4,
                    "document_title": "Security Policy",
                    "source_uri": "upload://security.md",
                },
            },
        ]
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        search_client=search_client,
    )

    chunk_ids = [candidate.chunk_id for candidate in response.candidates]

    assert len(response.candidates) == 3
    assert response.candidates[0].chunk_id == higher_score_chunk_id
    assert response.candidates[0].score == 4.0
    assert response.candidates[0].content == "Higher score duplicate section."
    assert lower_score_chunk_id not in chunk_ids
    assert first_no_section_chunk_id in chunk_ids
    assert second_no_section_chunk_id in chunk_ids


def test_search_bm25_chunks_records_retrieval_metrics(monkeypatch) -> None:
    document_id = uuid4()
    first_chunk_id = uuid4()
    duplicate_chunk_id = uuid4()
    low_score_document_id = uuid4()
    low_score_chunk_id = uuid4()
    lifecycle_labels = {
        "status": "completed",
        "retrieval_strategy": "bm25",
    }
    returned_candidate_labels = {
        "retrieval_strategy": "bm25",
        "result": "returned_candidate",
    }
    skipped_invalid_hit_labels = {
        "retrieval_strategy": "bm25",
        "result": "skipped_invalid_hit",
    }
    skipped_low_score_hit_labels = {
        "retrieval_strategy": "bm25",
        "result": "skipped_low_score_hit",
    }
    deduplicated_hit_labels = {
        "retrieval_strategy": "bm25",
        "result": "deduplicated_hit",
    }
    lifecycle_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_lifecycle_total",
            lifecycle_labels,
        )
        or 0
    )
    latency_count_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_latency_seconds_count",
            lifecycle_labels,
        )
        or 0
    )
    returned_candidate_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            returned_candidate_labels,
        )
        or 0
    )
    skipped_invalid_hit_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            skipped_invalid_hit_labels,
        )
        or 0
    )
    skipped_low_score_hit_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            skipped_low_score_hit_labels,
        )
        or 0
    )
    deduplicated_hit_before = (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            deduplicated_hit_labels,
        )
        or 0
    )
    search_client = FakeSearchClient(
        hits=[
            [],
            {
                "_score": 0.5,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(low_score_document_id),
                    "chunk_id": str(low_score_chunk_id),
                    "chunk_index": 1,
                    "content": "Low score match.",
                    "token_count": 3,
                    "document_title": "Low Score",
                    "source_uri": "upload://low-score.md",
                },
            },
            {
                "_score": 2.0,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(first_chunk_id),
                    "chunk_index": 2,
                    "content": "Best section match.",
                    "token_count": 4,
                    "section_path": "Security / Policy",
                    "document_title": "Security Policy",
                    "source_uri": "upload://security.md",
                },
            },
            {
                "_score": 1.5,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(duplicate_chunk_id),
                    "chunk_index": 3,
                    "content": "Duplicate section match.",
                    "token_count": 4,
                    "section_path": "Security / Policy",
                    "document_title": "Security Policy",
                    "source_uri": "upload://security.md",
                },
            },
        ]
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    monkeypatch.setattr(
        "agentic_rag.retrieval.bm25_search.settings.bm25_min_score",
        1.0,
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        search_client=search_client,
    )

    assert len(response.candidates) == 1
    assert response.candidates[0].chunk_id == first_chunk_id
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_lifecycle_total",
            lifecycle_labels,
        )
        == lifecycle_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_latency_seconds_count",
            lifecycle_labels,
        )
        == latency_count_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            returned_candidate_labels,
        )
        == returned_candidate_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            skipped_invalid_hit_labels,
        )
        == skipped_invalid_hit_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            skipped_low_score_hit_labels,
        )
        == skipped_low_score_hit_before + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "agentic_rag_retrieval_result_total",
            deduplicated_hit_labels,
        )
        == deduplicated_hit_before + 1
    )


def test_search_bm25_chunks_filters_low_score_candidates(monkeypatch) -> None:
    high_score_document_id = uuid4()
    high_score_chunk_id = uuid4()
    low_score_document_id = uuid4()
    low_score_chunk_id = uuid4()
    search_client = FakeSearchClient(
        hits=[
            {
                "_score": 2.75,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(high_score_document_id),
                    "chunk_id": str(high_score_chunk_id),
                    "chunk_index": 1,
                    "content": "Strong security policy match.",
                    "token_count": 4,
                    "document_title": "Security Policy",
                    "source_uri": "upload://security.md",
                },
            },
            {
                "_score": 0.75,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(low_score_document_id),
                    "chunk_id": str(low_score_chunk_id),
                    "chunk_index": 2,
                    "content": "Weak policy match.",
                    "token_count": 3,
                    "document_title": "Weak Match",
                    "source_uri": "upload://weak.md",
                },
            },
        ]
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    monkeypatch.setattr(
        "agentic_rag.retrieval.bm25_search.settings.bm25_min_score",
        2.0,
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        search_client=search_client,
    )

    assert len(response.candidates) == 1
    assert response.candidates[0].chunk_id == high_score_chunk_id
    assert response.candidates[0].score == 2.75
    assert response.candidates[0].citation.title == "Security Policy"


def test_search_bm25_chunks_keeps_zero_score_candidates_by_default(monkeypatch) -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    search_client = FakeSearchClient(
        hits=[
            {
                "_score": 0.0,
                "_source": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "document_id": str(document_id),
                    "chunk_id": str(chunk_id),
                    "chunk_index": 1,
                    "content": "Default threshold keeps this result.",
                    "token_count": 5,
                    "document_title": "Default Threshold",
                    "source_uri": "upload://default.md",
                },
            }
        ]
    )
    user_context = UserContext(
        id="user-1",
        customer_id="tenant-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        acl_version=4,
    )

    monkeypatch.setattr(
        "agentic_rag.retrieval.bm25_search.settings.bm25_min_score",
        0.0,
    )

    response = search_bm25_chunks(
        user_context=user_context,
        query="security policy",
        search_client=search_client,
    )

    assert len(response.candidates) == 1
    assert response.candidates[0].chunk_id == chunk_id
    assert response.candidates[0].content == "Default threshold keeps this result."
    assert response.candidates[0].citation.quote == "Default threshold keeps this result."
    assert response.candidates[0].score == 0.0


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
