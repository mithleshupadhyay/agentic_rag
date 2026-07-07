import hashlib
import logging
import os
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.retrieval.vector_search import search_vector_chunks
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.documents import create_document
from agentic_rag.shared.db.crud.embeddings import create_chunk_embedding
from agentic_rag.shared.db.crud.ingestion import replace_document_chunks
from agentic_rag.shared.db.models import DocumentChunk, Tenant
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.shared.schemas.auth import AclPolicy, Visibility
from agentic_rag.shared.schemas.chunks import ChunkEmbeddingCreate
from agentic_rag.shared.schemas.documents import DocumentCreateRequest, DocumentSourceType
from agentic_rag.shared.schemas.llm import EmbeddingRequest, EmbeddingResponse
from agentic_rag.shared.schemas.retrieval import RetrievalFilters, RetrievalStrategy


logger = logging.getLogger(__name__)


def _vector_at(index: int, value: float = 1.0) -> list[float]:
    vector = [0.0] * settings.embedding_dimension
    vector[index] = value
    return vector


def _ensure_tenant(db: Session, tenant_id: str, smoke_id: str) -> None:
    if db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first():
        return

    db.add(
        Tenant(
            tenant_id=tenant_id,
            name=f"Vector Smoke {tenant_id}",
            slug=tenant_id,
            status="active",
            metadata_={"smoke": True, "smoke_id": smoke_id},
        )
    )
    db.commit()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _create_chunk_embeddings(
    db: Session,
    tenant_id: str,
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
) -> None:
    for chunk, vector in zip(chunks, vectors, strict=True):
        create_chunk_embedding(
            db=db,
            tenant_id=tenant_id,
            obj_in=ChunkEmbeddingCreate(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                embedding=vector,
                embedding_model=settings.embedding_model_name,
                embedding_dimension=settings.embedding_dimension,
                content_hash=chunk.content_hash,
                vector_version=settings.embedding_vector_version,
                metadata={"created_by": "vector_retrieval_smoke"},
            ),
        )


def _create_document_chunks(
    db: Session,
    user_context: UserContext,
    workspace_id: str,
    title: str,
    contents: list[str],
    acl: AclPolicy,
    smoke_id: str,
) -> list[DocumentChunk]:
    document = create_document(
        user_context=user_context,
        db=db,
        obj_in=DocumentCreateRequest(
            workspace_id=workspace_id,
            source_type=DocumentSourceType.UPLOAD,
            source_uri=f"upload://vector-smoke/{smoke_id}/{title}.txt",
            title=title,
            metadata={
                "smoke": True,
                "smoke_id": smoke_id,
                "tags": ["vector-smoke"],
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
                "chunk_index": index,
                "content": content,
                "content_hash": _content_hash(content),
                "token_count": max(1, len(content.split())),
                "start_offset": 0,
                "end_offset": len(content),
                "section_path": f"Smoke / {index}",
                "metadata": {
                    "smoke": True,
                    "smoke_id": smoke_id,
                    "tags": ["vector-smoke"],
                    "section": "security",
                },
            }
            for index, content in enumerate(contents)
        ],
    )
    return chunks


def main() -> int:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )

    if settings.embedding_dimension != 768:
        logger.error(
            f"[VectorSmoke] EMBEDDING_DIMENSION must be 768 for the current "
            f"pgvector schema, got {settings.embedding_dimension}"
        )
        return 1

    tenant_id = settings.local_tenant_id
    workspace_id = settings.local_workspace_id or "local-workspace"
    user_id = settings.local_user_id
    group_id = "vector-smoke-group"
    smoke_id = str(uuid4())
    other_tenant_id = f"vector-smoke-{smoke_id[:8]}"
    other_workspace_id = f"vector-smoke-other-workspace-{smoke_id[:8]}"
    SessionLocal = get_sync_session_factory()

    logger.info(f"[VectorSmoke] Preparing pgvector retrieval smoke tenant={tenant_id}")
    with SessionLocal() as db:
        dialect_name = db.get_bind().dialect.name
        if dialect_name != "postgresql":
            logger.error(
                f"[VectorSmoke] This smoke check must run against PostgreSQL, "
                f"got dialect={dialect_name}"
            )
            return 1

        extension_name = db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        if extension_name != "vector":
            logger.error("[VectorSmoke] PostgreSQL vector extension is not installed")
            return 1

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
        allowed_chunks = _create_document_chunks(
            db=db,
            user_context=user_context,
            workspace_id=workspace_id,
            title="Vector smoke allowed document",
            contents=[
                "Password rotation must happen every ninety days.",
                "Access reviews must be completed every quarter.",
            ],
            acl=allowed_acl,
            smoke_id=smoke_id,
        )
        _create_chunk_embeddings(
            db=db,
            tenant_id=tenant_id,
            chunks=allowed_chunks,
            vectors=[
                _vector_at(0),
                [0.8, 0.6] + [0.0] * (settings.embedding_dimension - 2),
            ],
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
        private_chunks = _create_document_chunks(
            db=db,
            user_context=private_owner_context,
            workspace_id=workspace_id,
            title="Vector smoke private document",
            contents=["This private chunk should never reach the vector result."],
            acl=AclPolicy(
                visibility=Visibility.PRIVATE,
                allowed_user_ids=[private_owner_context.id],
                acl_version=settings.local_acl_version,
            ),
            smoke_id=smoke_id,
        )
        _create_chunk_embeddings(
            db=db,
            tenant_id=tenant_id,
            chunks=private_chunks,
            vectors=[_vector_at(0)],
        )

        other_workspace_chunks = _create_document_chunks(
            db=db,
            user_context=user_context,
            workspace_id=other_workspace_id,
            title="Vector smoke other workspace document",
            contents=["This other workspace chunk should be filtered out."],
            acl=allowed_acl,
            smoke_id=smoke_id,
        )
        _create_chunk_embeddings(
            db=db,
            tenant_id=tenant_id,
            chunks=other_workspace_chunks,
            vectors=[_vector_at(0)],
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
        other_tenant_chunks = _create_document_chunks(
            db=db,
            user_context=other_tenant_context,
            workspace_id=workspace_id,
            title="Vector smoke other tenant document",
            contents=["This other tenant chunk should be filtered out."],
            acl=allowed_acl,
            smoke_id=smoke_id,
        )
        _create_chunk_embeddings(
            db=db,
            tenant_id=other_tenant_id,
            chunks=other_tenant_chunks,
            vectors=[_vector_at(0)],
        )

        def fake_embedding_client(request: EmbeddingRequest) -> EmbeddingResponse:
            if request.texts != ["password rotation policy"]:
                raise AssertionError("Unexpected vector smoke embedding text.")

            return EmbeddingResponse(
                embeddings=[_vector_at(0)],
                model=settings.embedding_model_name,
                provider=settings.embedding_provider,
                dimension=settings.embedding_dimension,
                latency_ms=1,
            )

        response = search_vector_chunks(
            db=db,
            user_context=user_context,
            query="password rotation policy",
            filters=RetrievalFilters(
                workspace_id=workspace_id,
                metadata={"smoke_id": smoke_id},
                tags=["vector-smoke"],
            ),
            limit=5,
            min_similarity=0.0,
            embedding_client=fake_embedding_client,
        )

        expected_chunk_ids = [allowed_chunks[0].id, allowed_chunks[1].id]
        actual_chunk_ids = [candidate.chunk_id for candidate in response.candidates]
        blocked_chunk_ids = {
            private_chunks[0].id,
            other_workspace_chunks[0].id,
            other_tenant_chunks[0].id,
        }
        if response.strategy != RetrievalStrategy.VECTOR:
            logger.error(f"[VectorSmoke] Unexpected strategy={response.strategy}")
            return 1
        if actual_chunk_ids != expected_chunk_ids:
            logger.error(
                f"[VectorSmoke] Unexpected vector result order "
                f"expected={expected_chunk_ids} actual={actual_chunk_ids}"
            )
            return 1
        if blocked_chunk_ids.intersection(actual_chunk_ids):
            logger.error("[VectorSmoke] Unauthorized vector chunks were returned")
            return 1
        if response.candidates[0].score < response.candidates[1].score:
            logger.error("[VectorSmoke] Vector results were not similarity ordered")
            return 1
        if response.candidates[0].source != "vector_search":
            logger.error(
                f"[VectorSmoke] Unexpected retrieval source="
                f"{response.candidates[0].source}"
            )
            return 1

    logger.info(
        f"[VectorSmoke] pgvector retrieval smoke ok tenant={tenant_id} "
        f"workspace={workspace_id} smoke_id={smoke_id}"
    )
    print("vector retrieval smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
