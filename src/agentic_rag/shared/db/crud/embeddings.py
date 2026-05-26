import logging
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from agentic_rag.shared.db.models import ChunkEmbedding, Document, DocumentChunk
from agentic_rag.shared.schemas.chunks import ChunkEmbeddingCreate


logger = logging.getLogger(__name__)


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
        query = query.with_for_update(skip_locked=True)

    chunks = query.all()
    logger.info(
        f"[DB] Found {len(chunks)} chunks missing embedding tenant={tenant_id}"
    )
    return chunks
