import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.models import Document, DocumentChunk


logger = logging.getLogger(__name__)


def list_chunks_pending_bm25_index(
    db: Session,
    limit: Optional[int] = None,
    index_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> list[DocumentChunk]:
    limit = limit or settings.bm25_index_batch_size
    index_name = index_name or settings.opensearch_chunk_index
    logger.info(
        f"[DB] Listing chunks pending BM25 index limit={limit} "
        f"index={index_name} tenant={tenant_id}"
    )

    query = (
        db.query(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .options(
            selectinload(DocumentChunk.document),
            selectinload(DocumentChunk.acl),
        )
        .filter(
            DocumentChunk.is_deleted.is_(False),
            DocumentChunk.content != "",
            Document.is_deleted.is_(False),
            Document.status == "ready",
            or_(
                DocumentChunk.bm25_index_status != "indexed",
                DocumentChunk.bm25_index_name.is_(None),
                DocumentChunk.bm25_index_name != index_name,
                DocumentChunk.bm25_index_content_hash.is_(None),
                DocumentChunk.bm25_index_content_hash != DocumentChunk.content_hash,
            ),
        )
        .order_by(DocumentChunk.updated_at.asc(), DocumentChunk.created_at.asc())
        .limit(limit)
    )

    if tenant_id:
        query = query.filter(DocumentChunk.tenant_id == tenant_id)

    bind = db.get_bind()
    dialect_name = bind.dialect.name if bind else ""
    if dialect_name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    chunks = query.all()
    logger.info(f"[DB] Found {len(chunks)} chunks pending BM25 index")
    return chunks


def mark_chunk_bm25_indexed(
    db: Session,
    chunk: DocumentChunk,
    index_name: str,
) -> DocumentChunk:
    logger.info(
        f"[DB] Marking chunk {chunk.id} BM25 indexed "
        f"tenant={chunk.tenant_id} index={index_name}"
    )
    chunk.bm25_index_status = "indexed"
    chunk.bm25_index_name = index_name
    chunk.bm25_index_content_hash = chunk.content_hash
    chunk.bm25_indexed_at = datetime.now(timezone.utc)
    chunk.bm25_index_error = None

    try:
        db.commit()
        db.refresh(chunk)
        logger.info(f"[DB] Marked chunk {chunk.id} BM25 indexed")
        return chunk

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to mark chunk {chunk.id} indexed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during chunk BM25 index status update.",
        )


def mark_chunk_bm25_failed(
    db: Session,
    chunk: DocumentChunk,
    error_message: str,
) -> DocumentChunk:
    logger.warning(
        f"[DB] Marking chunk {chunk.id} BM25 index failed "
        f"tenant={chunk.tenant_id}: {error_message}"
    )
    chunk.bm25_index_status = "failed"
    chunk.bm25_index_error = error_message

    try:
        db.commit()
        db.refresh(chunk)
        logger.info(f"[DB] Marked chunk {chunk.id} BM25 failed")
        return chunk

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to mark chunk {chunk.id} BM25 failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during chunk BM25 index failure update.",
        )
