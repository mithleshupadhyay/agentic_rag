import json
import logging
from typing import Any, Optional

import httpx

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.models import DocumentChunk


logger = logging.getLogger(__name__)


def build_chunk_search_document(chunk: DocumentChunk) -> dict[str, Any]:
    document = chunk.document
    chunk_acl = chunk.acl

    return {
        "tenant_id": chunk.tenant_id,
        "workspace_id": chunk.workspace_id,
        "document_id": str(chunk.document_id),
        "chunk_id": str(chunk.id),
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "token_count": chunk.token_count,
        "section_path": chunk.section_path,
        "page_number": chunk.page_number,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "document_title": document.title if document else None,
        "owner_user_id": document.owner_user_id if document else None,
        "file_name": document.file_name if document else None,
        "source_type": document.source_type if document else None,
        "source_uri": document.source_uri if document else None,
        "document_metadata": document.metadata_ if document else {},
        "chunk_metadata": chunk.metadata_,
        "visibility": chunk_acl.visibility if chunk_acl else "private",
        "allowed_user_ids": chunk_acl.allowed_user_ids if chunk_acl else [],
        "allowed_group_ids": chunk_acl.allowed_group_ids if chunk_acl else [],
        "allowed_roles": chunk_acl.allowed_roles if chunk_acl else [],
        "denied_user_ids": chunk_acl.denied_user_ids if chunk_acl else [],
        "denied_group_ids": chunk_acl.denied_group_ids if chunk_acl else [],
        "acl_version": chunk_acl.acl_version if chunk_acl else chunk.acl_version,
        "classification_level": chunk.classification_level,
        "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
        "updated_at": chunk.updated_at.isoformat() if chunk.updated_at else None,
    }


class OpenSearchClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = (base_url or settings.opensearch_url).rstrip("/")
        self.username = username if username is not None else settings.opensearch_username
        self.password = password if password is not None else settings.opensearch_password
        self.index_name = index_name or settings.opensearch_chunk_index
        self.timeout_seconds = timeout_seconds or settings.opensearch_request_timeout_seconds
        auth = (self.username, self.password) if self.username and self.password else None
        self.client = httpx.Client(
            base_url=self.base_url,
            auth=auth,
            timeout=self.timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def ensure_chunk_index(self, index_name: Optional[str] = None) -> None:
        index_name = index_name or self.index_name
        logger.info(f"[OpenSearch] Ensuring chunk index {index_name}")
        response = self.client.get(f"/{index_name}")
        if response.status_code == 200:
            logger.info(f"[OpenSearch] Chunk index exists {index_name}")
            return
        if response.status_code != 404:
            response.raise_for_status()

        payload = {
            "settings": {
                "index": {
                    "number_of_shards": settings.opensearch_index_shards,
                    "number_of_replicas": settings.opensearch_index_replicas,
                }
            },
            "mappings": {
                "dynamic": "false",
                "properties": {
                    "tenant_id": {"type": "keyword"},
                    "workspace_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content": {"type": "text"},
                    "content_hash": {"type": "keyword"},
                    "token_count": {"type": "integer"},
                    "section_path": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "start_offset": {"type": "integer"},
                    "end_offset": {"type": "integer"},
                    "document_title": {"type": "text"},
                    "owner_user_id": {"type": "keyword"},
                    "file_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "source_type": {"type": "keyword"},
                    "source_uri": {"type": "keyword"},
                    "document_metadata": {"type": "object", "enabled": False},
                    "chunk_metadata": {"type": "object", "enabled": False},
                    "visibility": {"type": "keyword"},
                    "allowed_user_ids": {"type": "keyword"},
                    "allowed_group_ids": {"type": "keyword"},
                    "allowed_roles": {"type": "keyword"},
                    "denied_user_ids": {"type": "keyword"},
                    "denied_group_ids": {"type": "keyword"},
                    "acl_version": {"type": "integer"},
                    "classification_level": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                },
            },
        }
        create_response = self.client.put(f"/{index_name}", json=payload)
        create_response.raise_for_status()
        logger.info(f"[OpenSearch] Created chunk index {index_name}")

    def bulk_index_chunks(
        self,
        chunks: list[DocumentChunk],
        index_name: Optional[str] = None,
    ) -> int:
        index_name = index_name or self.index_name
        if not chunks:
            logger.info("[OpenSearch] No chunks to index")
            return 0

        lines = []
        for chunk in chunks:
            lines.append(json.dumps({"index": {"_index": index_name, "_id": str(chunk.id)}}))
            lines.append(json.dumps(build_chunk_search_document(chunk), default=str))

        payload = "\n".join(lines) + "\n"
        logger.info(f"[OpenSearch] Bulk indexing {len(chunks)} chunks index={index_name}")
        response = self.client.post(
            "/_bulk",
            content=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("errors"):
            logger.error(f"[OpenSearch] Bulk index returned errors: {response_data}")
            raise RuntimeError("OpenSearch bulk indexing failed for one or more chunks")

        logger.info(f"[OpenSearch] Bulk indexed {len(chunks)} chunks index={index_name}")
        return len(chunks)

    def search_chunks_bm25(
        self,
        search_body: dict[str, Any],
        index_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        index_name = index_name or self.index_name
        logger.info(f"[OpenSearch] Searching BM25 chunks index={index_name}")
        response = self.client.post(f"/{index_name}/_search", json=search_body)
        response.raise_for_status()
        response_data = response.json()
        hits = response_data.get("hits", {}).get("hits", [])
        logger.info(f"[OpenSearch] BM25 search returned {len(hits)} chunks")
        return hits
