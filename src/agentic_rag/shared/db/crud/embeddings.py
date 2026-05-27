import logging
from dataclasses import dataclass
from math import sqrt
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import not_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.models import ChunkAcl, ChunkEmbedding, Document, DocumentChunk
from agentic_rag.shared.schemas.auth import Visibility
from agentic_rag.shared.schemas.chunks import ChunkEmbeddingCreate


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkVectorSearchResult:
    chunk: DocumentChunk
    similarity: float
    distance: float


def embedding_exists(
    db: Session,
    tenant_id: str,
    chunk_id: UUID,
    embedding_model: str,
    vector_version: int = 1,
    content_hash: Optional[str] = None,
) -> bool:
    logger.info(
        f"[DB] Checking chunk embedding tenant={tenant_id} chunk={chunk_id} "
        f"model={embedding_model} vector_version={vector_version}"
    )
    query = db.query(ChunkEmbedding).filter(
        ChunkEmbedding.tenant_id == tenant_id,
        ChunkEmbedding.chunk_id == chunk_id,
        ChunkEmbedding.embedding_model == embedding_model,
        ChunkEmbedding.vector_version == vector_version,
    )
    if content_hash is not None:
        query = query.filter(ChunkEmbedding.content_hash == content_hash)

    exists = query.first() is not None
    logger.info(
        f"[DB] Chunk embedding exists={exists} tenant={tenant_id} "
        f"chunk={chunk_id} model={embedding_model}"
    )
    return exists


def create_chunk_embedding(
    db: Session,
    tenant_id: str,
    obj_in: ChunkEmbeddingCreate,
) -> ChunkEmbedding:
    logger.info(
        f"[DB] Creating chunk embedding tenant={tenant_id} "
        f"chunk={obj_in.chunk_id} model={obj_in.embedding_model} "
        f"vector_version={obj_in.vector_version}"
    )
    if obj_in.embedding_dimension != len(obj_in.embedding):
        logger.warning(
            f"[DB] Embedding dimension mismatch chunk={obj_in.chunk_id} "
            f"declared={obj_in.embedding_dimension} actual={len(obj_in.embedding)}"
        )
        raise HTTPException(
            status_code=400,
            detail="Embedding dimension must match embedding vector length.",
        )

    chunk = (
        db.query(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.id == obj_in.chunk_id,
            DocumentChunk.document_id == obj_in.document_id,
            DocumentChunk.is_deleted.is_(False),
            Document.is_deleted.is_(False),
        )
        .first()
    )
    if not chunk:
        logger.warning(
            f"[DB] Chunk not found for embedding tenant={tenant_id} "
            f"chunk={obj_in.chunk_id}"
        )
        raise HTTPException(status_code=404, detail="Chunk not found.")

    existing_embedding = (
        db.query(ChunkEmbedding)
        .filter(
            ChunkEmbedding.tenant_id == tenant_id,
            ChunkEmbedding.chunk_id == obj_in.chunk_id,
            ChunkEmbedding.embedding_model == obj_in.embedding_model,
            ChunkEmbedding.vector_version == obj_in.vector_version,
        )
        .first()
    )
    if existing_embedding:
        if existing_embedding.content_hash == obj_in.content_hash:
            logger.info(
                f"[DB] Chunk embedding already exists tenant={tenant_id} "
                f"chunk={obj_in.chunk_id} model={obj_in.embedding_model}"
            )
            return existing_embedding

        logger.info(
            f"[DB] Updating stale chunk embedding tenant={tenant_id} "
            f"chunk={obj_in.chunk_id} model={obj_in.embedding_model}"
        )
        existing_embedding.workspace_id = chunk.workspace_id
        existing_embedding.document_id = chunk.document_id
        existing_embedding.embedding = obj_in.embedding
        existing_embedding.embedding_dimension = obj_in.embedding_dimension
        existing_embedding.content_hash = obj_in.content_hash
        existing_embedding.metadata_ = obj_in.metadata

        try:
            db.commit()
            db.refresh(existing_embedding)
            logger.info(
                f"[DB] Updated chunk embedding {existing_embedding.id} "
                f"tenant={tenant_id}"
            )
            return existing_embedding

        except IntegrityError as e:
            db.rollback()
            logger.exception(
                f"[DB] Failed to update chunk embedding chunk={obj_in.chunk_id}: {e}"
            )
            raise HTTPException(
                status_code=400,
                detail="Database error during chunk embedding update.",
            )

    db_obj = ChunkEmbedding(
        tenant_id=tenant_id,
        workspace_id=chunk.workspace_id,
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        embedding=obj_in.embedding,
        embedding_model=obj_in.embedding_model,
        embedding_dimension=obj_in.embedding_dimension,
        content_hash=obj_in.content_hash,
        vector_version=obj_in.vector_version,
        metadata_=obj_in.metadata,
    )

    try:
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(f"[DB] Created chunk embedding {db_obj.id} tenant={tenant_id}")
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to create chunk embedding: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during chunk embedding creation.",
        )


def bulk_create_chunk_embeddings(
    db: Session,
    tenant_id: str,
    embeddings: list[ChunkEmbeddingCreate],
) -> int:
    logger.info(
        f"[DB] Bulk creating chunk embeddings tenant={tenant_id} "
        f"count={len(embeddings)}"
    )
    if not embeddings:
        return 0

    chunk_ids = [embedding.chunk_id for embedding in embeddings]
    chunks = (
        db.query(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.id.in_(chunk_ids),
            DocumentChunk.is_deleted.is_(False),
            Document.is_deleted.is_(False),
        )
        .all()
    )
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    existing_embeddings = (
        db.query(ChunkEmbedding)
        .filter(
            ChunkEmbedding.tenant_id == tenant_id,
            ChunkEmbedding.chunk_id.in_(chunk_ids),
        )
        .all()
    )
    existing_by_key = {
        (
            existing.chunk_id,
            existing.embedding_model,
            existing.vector_version,
        ): existing
        for existing in existing_embeddings
    }

    written_count = 0
    try:
        for embedding_payload in embeddings:
            if embedding_payload.embedding_dimension != len(embedding_payload.embedding):
                logger.warning(
                    f"[DB] Embedding dimension mismatch chunk={embedding_payload.chunk_id} "
                    f"declared={embedding_payload.embedding_dimension} "
                    f"actual={len(embedding_payload.embedding)}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Embedding dimension must match embedding vector length.",
                )

            chunk = chunks_by_id.get(embedding_payload.chunk_id)
            if not chunk or chunk.document_id != embedding_payload.document_id:
                logger.warning(
                    f"[DB] Chunk not found during bulk embedding tenant={tenant_id} "
                    f"chunk={embedding_payload.chunk_id}"
                )
                raise HTTPException(status_code=404, detail="Chunk not found.")

            existing_embedding = existing_by_key.get(
                (
                    embedding_payload.chunk_id,
                    embedding_payload.embedding_model,
                    embedding_payload.vector_version,
                )
            )
            if existing_embedding:
                if existing_embedding.content_hash == embedding_payload.content_hash:
                    logger.info(
                        f"[DB] Skipping existing chunk embedding tenant={tenant_id} "
                        f"chunk={embedding_payload.chunk_id} "
                        f"model={embedding_payload.embedding_model}"
                    )
                    continue

                existing_embedding.workspace_id = chunk.workspace_id
                existing_embedding.document_id = chunk.document_id
                existing_embedding.embedding = embedding_payload.embedding
                existing_embedding.embedding_dimension = (
                    embedding_payload.embedding_dimension
                )
                existing_embedding.content_hash = embedding_payload.content_hash
                existing_embedding.metadata_ = embedding_payload.metadata
                written_count += 1
                continue

            db_obj = ChunkEmbedding(
                tenant_id=tenant_id,
                workspace_id=chunk.workspace_id,
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                embedding=embedding_payload.embedding,
                embedding_model=embedding_payload.embedding_model,
                embedding_dimension=embedding_payload.embedding_dimension,
                content_hash=embedding_payload.content_hash,
                vector_version=embedding_payload.vector_version,
                metadata_=embedding_payload.metadata,
            )
            db.add(db_obj)
            existing_by_key[
                (
                    embedding_payload.chunk_id,
                    embedding_payload.embedding_model,
                    embedding_payload.vector_version,
                )
            ] = db_obj
            written_count += 1

        db.commit()
        logger.info(
            f"[DB] Bulk chunk embeddings written tenant={tenant_id} "
            f"written_count={written_count}"
        )
        return written_count

    except HTTPException:
        db.rollback()
        raise

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to bulk create chunk embeddings: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during bulk chunk embedding creation.",
        )


def get_chunks_missing_embedding(
    db: Session,
    tenant_id: str,
    embedding_model: str,
    vector_version: int = 1,
    limit: int = 100,
    document_id: Optional[UUID] = None,
) -> list[DocumentChunk]:
    logger.info(
        f"[DB] Listing chunks missing embedding tenant={tenant_id} "
        f"model={embedding_model} vector_version={vector_version} "
        f"limit={limit} document_id={document_id}"
    )
    query = (
        db.query(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .outerjoin(
            ChunkEmbedding,
            (ChunkEmbedding.tenant_id == DocumentChunk.tenant_id)
            & (ChunkEmbedding.chunk_id == DocumentChunk.id)
            & (ChunkEmbedding.embedding_model == embedding_model)
            & (ChunkEmbedding.vector_version == vector_version)
            & (ChunkEmbedding.content_hash == DocumentChunk.content_hash),
        )
        .options(selectinload(DocumentChunk.document), selectinload(DocumentChunk.acl))
        .filter(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.is_deleted.is_(False),
            DocumentChunk.content != "",
            Document.is_deleted.is_(False),
            Document.status == "ready",
            ChunkEmbedding.id.is_(None),
        )
        .order_by(DocumentChunk.updated_at.asc(), DocumentChunk.created_at.asc())
        .limit(limit)
    )
    if document_id:
        query = query.filter(DocumentChunk.document_id == document_id)

    bind = db.get_bind()
    dialect_name = bind.dialect.name if bind else ""
    if dialect_name == "postgresql":
        query = query.with_for_update(skip_locked=True, of=DocumentChunk)

    chunks = query.all()
    logger.info(
        f"[DB] Found {len(chunks)} chunks missing embedding tenant={tenant_id}"
    )
    return chunks


def search_similar_chunks_by_embedding(
    db: Session,
    tenant_id: str,
    query_embedding: list[float],
    embedding_model: str,
    vector_version: int = 1,
    embedding_dimension: int = 768,
    limit: int = 20,
    min_similarity: Optional[float] = None,
    workspace_id: Optional[str] = None,
    document_ids: Optional[list[UUID]] = None,
    user_context: Optional[UserContext] = None,
) -> list[ChunkVectorSearchResult]:
    logger.info(
        f"[DB] Searching similar chunks tenant={tenant_id} model={embedding_model} "
        f"vector_version={vector_version} limit={limit} workspace_id={workspace_id} "
        f"document_count={len(document_ids or [])}"
    )
    if not query_embedding:
        raise HTTPException(status_code=400, detail="Query embedding is required.")
    if len(query_embedding) != embedding_dimension:
        raise HTTPException(
            status_code=400,
            detail="Query embedding dimension must match configured embedding dimension.",
        )
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=400,
            detail="Vector search limit must be between 1 and 200.",
        )
    if min_similarity is not None and not 0 <= min_similarity <= 1:
        raise HTTPException(
            status_code=400,
            detail="Minimum similarity must be between 0 and 1.",
        )
    if user_context and user_context.tenant_id != tenant_id:
        logger.warning(
            f"[DB] Vector search denied by tenant mismatch tenant={tenant_id} "
            f"user_tenant={user_context.tenant_id}"
        )
        return []

    bind = db.get_bind()
    dialect_name = bind.dialect.name if bind else ""

    if dialect_name == "sqlite":
        query = (
            db.query(ChunkEmbedding, DocumentChunk)
            .join(DocumentChunk, DocumentChunk.id == ChunkEmbedding.chunk_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .options(
                selectinload(ChunkEmbedding.chunk).selectinload(DocumentChunk.acl),
                selectinload(ChunkEmbedding.chunk).selectinload(DocumentChunk.document),
            )
            .filter(
                ChunkEmbedding.tenant_id == tenant_id,
                ChunkEmbedding.embedding_model == embedding_model,
                ChunkEmbedding.vector_version == vector_version,
                ChunkEmbedding.embedding_dimension == embedding_dimension,
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.is_deleted.is_(False),
                Document.is_deleted.is_(False),
                Document.status == "ready",
            )
        )
        if workspace_id:
            query = query.filter(DocumentChunk.workspace_id == workspace_id)
        if document_ids:
            query = query.filter(DocumentChunk.document_id.in_(document_ids))

        query_embedding_norm = sqrt(sum(value * value for value in query_embedding))
        user_group_set = set(user_context.group_ids or []) if user_context else set()
        user_role_set = set(user_context.roles or []) if user_context else set()
        results: list[ChunkVectorSearchResult] = []
        for embedding, chunk in query.all():
            # Step: apply chunk ACL checks for the requesting user.
            if user_context:
                if chunk.tenant_id != user_context.tenant_id:
                    continue

                acl = chunk.acl
                acl_version = acl.acl_version if acl else chunk.acl_version
                if user_context.acl_version < acl_version:
                    continue

                if acl:
                    if user_context.id in acl.denied_user_ids:
                        continue
                    if user_group_set.intersection(acl.denied_group_ids):
                        continue

                allowed = False
                if "admin" in user_role_set:
                    allowed = True
                elif chunk.document and chunk.document.owner_user_id == user_context.id:
                    allowed = True
                elif acl:
                    if str(acl.visibility) in (
                        Visibility.PUBLIC.value,
                        Visibility.TENANT.value,
                    ):
                        allowed = True
                    elif user_context.id in acl.allowed_user_ids:
                        allowed = True
                    elif user_group_set.intersection(acl.allowed_group_ids):
                        allowed = True
                    elif user_role_set.intersection(acl.allowed_roles):
                        allowed = True

                if not allowed:
                    continue

            # Step: compute cosine distance in SQLite for unit-test fallback.
            stored_embedding = list(embedding.embedding)
            if len(stored_embedding) != len(query_embedding):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Query embedding dimension must match stored embedding "
                        "dimension."
                    ),
                )

            stored_embedding_norm = sqrt(
                sum(value * value for value in stored_embedding)
            )
            if query_embedding_norm == 0 or stored_embedding_norm == 0:
                distance = 1.0
            else:
                similarity_score = sum(
                    query_value * stored_value
                    for query_value, stored_value in zip(
                        query_embedding,
                        stored_embedding,
                    )
                ) / (query_embedding_norm * stored_embedding_norm)
                similarity_score = max(min(similarity_score, 1.0), -1.0)
                distance = 1.0 - similarity_score

            similarity = 1.0 - distance
            if min_similarity is not None and similarity < min_similarity:
                continue
            results.append(
                ChunkVectorSearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    distance=distance,
                )
            )

        results.sort(key=lambda result: (result.distance, result.chunk.created_at))
        logger.info(
            f"[DB] Vector search returned {len(results[:limit])} chunks "
            f"tenant={tenant_id}"
        )
        return results[:limit]

    distance_expr = ChunkEmbedding.embedding.cosine_distance(query_embedding)  # type: ignore[attr-defined]
    query = (
        db.query(DocumentChunk, distance_expr.label("distance"))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .outerjoin(ChunkAcl, ChunkAcl.chunk_id == DocumentChunk.id)
        .options(selectinload(DocumentChunk.document), selectinload(DocumentChunk.acl))
        .filter(
            ChunkEmbedding.tenant_id == tenant_id,
            ChunkEmbedding.embedding_model == embedding_model,
            ChunkEmbedding.vector_version == vector_version,
            ChunkEmbedding.embedding_dimension == embedding_dimension,
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.is_deleted.is_(False),
            Document.is_deleted.is_(False),
            Document.status == "ready",
        )
    )
    if workspace_id:
        query = query.filter(DocumentChunk.workspace_id == workspace_id)
    if document_ids:
        query = query.filter(DocumentChunk.document_id.in_(document_ids))
    if min_similarity is not None:
        query = query.filter(distance_expr <= 1.0 - min_similarity)

    if user_context:
        user_groups = user_context.group_ids or []
        user_roles = user_context.roles or []
        query = query.filter(
            DocumentChunk.acl_version <= user_context.acl_version,
            or_(ChunkAcl.id.is_(None), ChunkAcl.acl_version <= user_context.acl_version),
        )

        denied_clauses = [ChunkAcl.denied_user_ids.contains([user_context.id])]
        for group_id in user_groups:
            denied_clauses.append(ChunkAcl.denied_group_ids.contains([group_id]))
        query = query.filter(
            or_(ChunkAcl.id.is_(None), not_(or_(*denied_clauses)))
        )

        if "admin" not in user_roles:
            allowed_clauses = [
                Document.owner_user_id == user_context.id,
                ChunkAcl.visibility.in_(
                    [Visibility.PUBLIC.value, Visibility.TENANT.value]
                ),
                ChunkAcl.allowed_user_ids.contains([user_context.id]),
            ]
            for group_id in user_groups:
                allowed_clauses.append(ChunkAcl.allowed_group_ids.contains([group_id]))
            for role in user_roles:
                allowed_clauses.append(ChunkAcl.allowed_roles.contains([role]))
            query = query.filter(or_(*allowed_clauses))

    rows = (
        query.order_by(distance_expr.asc(), DocumentChunk.created_at.asc())
        .limit(limit)
        .all()
    )
    results = [
        ChunkVectorSearchResult(
            chunk=chunk,
            similarity=1.0 - float(distance),
            distance=float(distance),
        )
        for chunk, distance in rows
    ]
    logger.info(f"[DB] Vector search returned {len(results)} chunks tenant={tenant_id}")
    return results
