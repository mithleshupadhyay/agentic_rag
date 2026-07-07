import hashlib
import logging
import os
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.search.opensearch import OpenSearchClient
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import DocumentChunk, Tenant
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSourceType,
    FileMetadata,
)
from agentic_rag.shared.schemas.retrieval import RetrievalFilters, RetrievalStrategy


logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _ensure_tenant(db: Session, tenant_id: str, smoke_id: str) -> None:
    if db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first():
        return

    db.add(
        Tenant(
            tenant_id=tenant_id,
            name=f"BM25 Smoke {tenant_id}",
            slug=tenant_id,
            status="active",
            metadata_={"smoke": True, "smoke_id": smoke_id},
        )
    )
    db.commit()


def _create_document_chunk(
    db: Session,
    user_context: UserContext,
    workspace_id: str,
    title: str,
    content: str,
    acl: AclPolicy,
    smoke_id: str,
    chunk_index: int = 0,
) -> DocumentChunk:
    content_hash = _content_hash(content)
    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=DocumentCreateRequest(
            workspace_id=workspace_id,
            source_type=DocumentSourceType.UPLOAD,
            source_uri=f"upload://bm25-smoke/{smoke_id}/{title}.txt",
            title=title,
            file=FileMetadata(
                file_name=f"{title.lower().replace(' ', '-')}.txt",
                mime_type="text/plain",
                byte_size=len(content.encode("utf-8")),
                content_hash=content_hash,
            ),
            metadata={
                "smoke": True,
                "smoke_id": smoke_id,
                "department": "security",
            },
            acl=acl,
        ),
    )
    chunks = replace_document_chunks(
        db=db,
        document=document,
        chunks=[
            {
                "chunk_index": chunk_index,
                "content": content,
                "content_hash": content_hash,
                "token_count": max(1, len(content.split())),
                "start_offset": 0,
                "end_offset": len(content),
                "section_path": f"BM25 Smoke / {chunk_index}",
                "metadata": {
                    "smoke": True,
                    "smoke_id": smoke_id,
                    "section": "security",
                },
            }
        ],
    )
    return chunks[0]


def _index_smoke_chunks(chunks: list[DocumentChunk]) -> None:
    search_client = OpenSearchClient()
    try:
        search_client.ensure_chunk_index()
        indexed_count = search_client.bulk_index_chunks(chunks)
        refresh_response = search_client.client.post(f"/{search_client.index_name}/_refresh")
        refresh_response.raise_for_status()
        logger.info(
            f"[BM25Smoke] Indexed and refreshed OpenSearch chunks "
            f"count={indexed_count} index={search_client.index_name}"
        )
    finally:
        search_client.close()


def main() -> int:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )

    tenant_id = settings.local_tenant_id
    workspace_id = settings.local_workspace_id or "local-workspace"
    user_id = settings.local_user_id
    group_id = "bm25-smoke-group"
    smoke_id = str(uuid4())
    other_tenant_id = f"bm25-smoke-{smoke_id[:8]}"
    other_workspace_id = f"bm25-smoke-other-workspace-{smoke_id[:8]}"
    SessionLocal = get_sync_session_factory()

    logger.info(f"[BM25Smoke] Preparing OpenSearch BM25 smoke tenant={tenant_id}")
    with SessionLocal() as db:
        _ensure_tenant(db, tenant_id, smoke_id)
        _ensure_tenant(db, other_tenant_id, smoke_id)

        user_context = UserContext(
            id=user_id,
            customer_id=tenant_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            roles=["reader"],
            group_ids=[group_id],
            scopes=["query:run"],
            acl_version=settings.local_acl_version,
        )
        allowed_acl = AclPolicy(
            visibility=Visibility.GROUP,
            allowed_group_ids=[group_id],
            acl_version=settings.local_acl_version,
        )
        primary_allowed_chunk = _create_document_chunk(
            db=db,
            user_context=user_context,
            workspace_id=workspace_id,
            title="BM25 smoke primary allowed document",
            content=(
                "Quarterly access review policy confirms quarterly access review policy "
                "owners must verify entitlements before renewal."
            ),
            acl=allowed_acl,
            smoke_id=smoke_id,
            chunk_index=0,
        )
        secondary_allowed_chunk = _create_document_chunk(
            db=db,
            user_context=user_context,
            workspace_id=workspace_id,
            title="BM25 smoke secondary allowed document",
            content=(
                "Quarterly access review evidence must include reviewer notes and "
                "approval dates."
            ),
            acl=allowed_acl,
            smoke_id=smoke_id,
            chunk_index=0,
        )

        private_owner_context = UserContext(
            id=f"other-user-{smoke_id[:8]}",
            customer_id=tenant_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            roles=["reader"],
            group_ids=[],
            scopes=["query:run"],
            acl_version=settings.local_acl_version,
        )
        private_chunk = _create_document_chunk(
            db=db,
            user_context=private_owner_context,
            workspace_id=workspace_id,
            title="BM25 smoke private document",
            content=(
                "Quarterly access review policy private material should not be "
                "returned."
            ),
            acl=AclPolicy(
                visibility=Visibility.PRIVATE,
                allowed_user_ids=[private_owner_context.id],
                acl_version=settings.local_acl_version,
            ),
            smoke_id=smoke_id,
            chunk_index=0,
        )

        other_workspace_chunk = _create_document_chunk(
            db=db,
            user_context=user_context,
            workspace_id=other_workspace_id,
            title="BM25 smoke other workspace document",
            content=(
                "Quarterly access review policy other workspace material should not "
                "be returned."
            ),
            acl=allowed_acl,
            smoke_id=smoke_id,
            chunk_index=0,
        )

        other_tenant_context = UserContext(
            id=user_id,
            customer_id=other_tenant_id,
            tenant_id=other_tenant_id,
            workspace_id=workspace_id,
            roles=["reader"],
            group_ids=[group_id],
            scopes=["query:run"],
            acl_version=settings.local_acl_version,
        )
        other_tenant_chunk = _create_document_chunk(
            db=db,
            user_context=other_tenant_context,
            workspace_id=workspace_id,
            title="BM25 smoke other tenant document",
            content=(
                "Quarterly access review policy other tenant material should not be "
                "returned."
            ),
            acl=allowed_acl,
            smoke_id=smoke_id,
            chunk_index=0,
        )

        all_chunks = [
            primary_allowed_chunk,
            secondary_allowed_chunk,
            private_chunk,
            other_workspace_chunk,
            other_tenant_chunk,
        ]
        _index_smoke_chunks(all_chunks)

        response = search_bm25_chunks(
            user_context=user_context,
            query="quarterly access review policy",
            filters=RetrievalFilters(
                workspace_id=workspace_id,
                document_ids=[chunk.document_id for chunk in all_chunks],
            ),
            limit=5,
        )

        expected_chunk_ids = [primary_allowed_chunk.id, secondary_allowed_chunk.id]
        actual_chunk_ids = [candidate.chunk_id for candidate in response.candidates]
        blocked_chunk_ids: set[UUID] = {
            private_chunk.id,
            other_workspace_chunk.id,
            other_tenant_chunk.id,
        }
        if response.strategy != RetrievalStrategy.BM25:
            logger.error(f"[BM25Smoke] Unexpected strategy={response.strategy}")
            return 1
        if actual_chunk_ids != expected_chunk_ids:
            logger.error(
                f"[BM25Smoke] Unexpected BM25 result order "
                f"expected={expected_chunk_ids} actual={actual_chunk_ids}"
            )
            return 1
        if blocked_chunk_ids.intersection(actual_chunk_ids):
            logger.error("[BM25Smoke] Unauthorized BM25 chunks were returned")
            return 1
        if response.candidates[0].score < response.candidates[1].score:
            logger.error("[BM25Smoke] BM25 results were not score ordered")
            return 1
        if response.candidates[0].source != "bm25_search":
            logger.error(
                f"[BM25Smoke] Unexpected retrieval source="
                f"{response.candidates[0].source}"
            )
            return 1

    logger.info(
        f"[BM25Smoke] OpenSearch BM25 retrieval smoke ok tenant={tenant_id} "
        f"workspace={workspace_id} smoke_id={smoke_id}"
    )
    print("bm25 retrieval smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
