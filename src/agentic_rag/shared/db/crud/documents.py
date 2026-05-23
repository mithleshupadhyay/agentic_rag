import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import asc, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session, noload, selectinload

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.models import Document, DocumentAcl, DocumentChunk, IngestionJob
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSearchRequest,
    DocumentUpdateRequest,
)


logger = logging.getLogger(__name__)


# --- CREATE ---
def create_document(
    user_context: UserContext,
    db: Session,
    obj_in: DocumentCreateRequest,
    object_key: Optional[str] = None,
) -> Document:
    try:
        logger.info(
            f"[DB] Creating document tenant={user_context.tenant_id} "
            f"user={user_context.id} source_type={obj_in.source_type}"
        )
        file_metadata = obj_in.file
        document_acl = obj_in.acl

        db_obj = Document(
            tenant_id=user_context.tenant_id,
            workspace_id=obj_in.workspace_id or user_context.workspace_id,
            source_type=obj_in.source_type,
            source_uri=obj_in.source_uri,
            object_key=object_key,
            title=obj_in.title,
            file_name=file_metadata.file_name if file_metadata else None,
            mime_type=file_metadata.mime_type if file_metadata else None,
            byte_size=file_metadata.byte_size if file_metadata else None,
            content_hash=file_metadata.content_hash if file_metadata else None,
            status="queued",
            owner_user_id=user_context.id,
            acl_version=document_acl.acl_version,
            classification_level="internal",
            metadata_=obj_in.metadata,
            created_by=user_context.id,
        )
        db_obj.acl = DocumentAcl(
            tenant_id=user_context.tenant_id,
            document=db_obj,
            visibility=document_acl.visibility,
            allowed_user_ids=document_acl.allowed_user_ids,
            allowed_group_ids=document_acl.allowed_group_ids,
            allowed_roles=document_acl.allowed_roles,
            denied_user_ids=document_acl.denied_user_ids,
            denied_group_ids=document_acl.denied_group_ids,
            acl_version=document_acl.acl_version,
        )

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        _ = db_obj.acl
        logger.info(
            f"[DB] Created document {db_obj.id} tenant={db_obj.tenant_id} "
            f"status={db_obj.status}"
        )
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to create document: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during document creation.",
        )


def attach_document_object(
    db: Session,
    db_obj: Document,
    object_key: str,
) -> Document:
    logger.info(f"[DB] Attaching object key to document {db_obj.id}")
    db_obj.object_key = object_key
    db_obj.status = "queued"

    try:
        db.commit()
        db.refresh(db_obj)
        _ = db_obj.acl
        logger.info(f"[DB] Attached object key to document {db_obj.id}")
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to attach object key to document {db_obj.id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during document object update.",
        )


def mark_document_failed(
    db: Session,
    db_obj: Document,
) -> Document:
    logger.warning(f"[DB] Marking document {db_obj.id} as failed")
    db_obj.status = "failed"

    try:
        db.commit()
        db.refresh(db_obj)
        _ = db_obj.acl
        logger.info(f"[DB] Marked document {db_obj.id} as failed")
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to mark document {db_obj.id} as failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during document status update.",
        )


def create_ingestion_job_for_document(
    user_context: UserContext,
    db: Session,
    document: Document,
    idempotency_key: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> IngestionJob:
    try:
        logger.info(
            f"[DB] Creating ingestion job document={document.id} "
            f"tenant={user_context.tenant_id}"
        )
        db_obj = IngestionJob(
            tenant_id=user_context.tenant_id,
            workspace_id=document.workspace_id,
            document_id=document.id,
            source_type=document.source_type,
            source_uri=document.source_uri,
            object_key=document.object_key,
            status="queued",
            current_stage="created",
            retry_count=0,
            max_retries=3,
            idempotency_key=idempotency_key,
            metadata_=metadata or {},
            created_by=user_context.id,
        )

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(
            f"[DB] Created ingestion job {db_obj.id} document={document.id} "
            f"tenant={db_obj.tenant_id}"
        )
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to create ingestion job for document {document.id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during ingestion job creation.",
        )


# --- GET ONE ---
def get_document(
    db: Session,
    id: UUID,
    tenant_id: str,
    *,
    include_deleted: bool = False,
) -> Optional[Document]:
    logger.info(
        f"[DB] Fetching document {id} tenant={tenant_id} "
        f"include_deleted={include_deleted}"
    )
    query = (
        db.query(Document)
        .options(
            selectinload(Document.acl),
            noload(Document.chunks),
            noload(Document.ingestion_jobs),
        )
        .filter(
            Document.id == id,
            Document.tenant_id == tenant_id,
        )
    )
    if not include_deleted:
        query = query.filter(Document.is_deleted.is_(False))
    document = query.first()
    if document:
        logger.info(f"[DB] Found document {id} tenant={tenant_id}")
    else:
        logger.warning(f"[DB] Document {id} not found tenant={tenant_id}")
    return document


# --- LIST MANY ---
def list_documents(
    db: Session,
    tenant_id: str,
    skip: int = 0,
    limit: int = 50,
    sort: Optional[List[str]] = None,
    sort_dir: Optional[List[str]] = None,
) -> List[Document]:
    sort = sort or ["created_at"]
    sort_dir = sort_dir or ["desc"]
    logger.info(
        f"[DB] Listing documents tenant={tenant_id} skip={skip} limit={limit} "
        f"sort={sort} sort_dir={sort_dir}"
    )

    query = (
        db.query(Document)
        .options(
            noload(Document.chunks),
            noload(Document.ingestion_jobs),
        )
        .filter(
            Document.tenant_id == tenant_id,
            Document.is_deleted.is_(False),
        )
    )
    query = apply_sorting(query, Document, sort, sort_dir)
    documents = query.offset(skip).limit(limit).all()
    logger.info(f"[DB] Listed {len(documents)} documents tenant={tenant_id}")
    return documents


# --- UPDATE ---
def update_document(
    db: Session,
    db_obj: Document,
    obj_in: DocumentUpdateRequest,
) -> Document:
    logger.info(f"[DB] Updating document {db_obj.id} tenant={db_obj.tenant_id}")
    if obj_in.title is not None:
        db_obj.title = obj_in.title
    if obj_in.metadata is not None:
        db_obj.metadata_ = obj_in.metadata
    if obj_in.classification_level is not None:
        db_obj.classification_level = obj_in.classification_level
    if obj_in.acl is not None:
        db_obj.acl_version = obj_in.acl.acl_version
        if db_obj.acl:
            db_obj.acl.visibility = obj_in.acl.visibility
            db_obj.acl.allowed_user_ids = obj_in.acl.allowed_user_ids
            db_obj.acl.allowed_group_ids = obj_in.acl.allowed_group_ids
            db_obj.acl.allowed_roles = obj_in.acl.allowed_roles
            db_obj.acl.denied_user_ids = obj_in.acl.denied_user_ids
            db_obj.acl.denied_group_ids = obj_in.acl.denied_group_ids
            db_obj.acl.acl_version = obj_in.acl.acl_version
        else:
            db_obj.acl = DocumentAcl(
                tenant_id=db_obj.tenant_id,
                document=db_obj,
                visibility=obj_in.acl.visibility,
                allowed_user_ids=obj_in.acl.allowed_user_ids,
                allowed_group_ids=obj_in.acl.allowed_group_ids,
                allowed_roles=obj_in.acl.allowed_roles,
                denied_user_ids=obj_in.acl.denied_user_ids,
                denied_group_ids=obj_in.acl.denied_group_ids,
                acl_version=obj_in.acl.acl_version,
            )

    try:
        db.commit()
        db.refresh(db_obj)
        _ = db_obj.acl
        logger.info(f"[DB] Updated document {db_obj.id} tenant={db_obj.tenant_id}")
        return db_obj

    except IntegrityError as e:
        db.rollback()
        logger.exception(f"[DB] Failed to update document {db_obj.id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Database error during document update.",
        )


def update_document_by_id(
    db: Session,
    document_id: UUID,
    tenant_id: str,
    obj_in: DocumentUpdateRequest,
) -> Document:
    logger.info(
        f"[DB] Updating document by id {document_id} tenant={tenant_id}"
    )
    db_obj = get_document(db, document_id, tenant_id)

    if not db_obj:
        logger.warning(
            f"[DB] Document {document_id} not found for update tenant={tenant_id}"
        )
        raise HTTPException(status_code=404, detail="Document not found.")

    return update_document(db, db_obj, obj_in)


# --- DELETE ---
def delete_document(
    db: Session,
    id: UUID,
    tenant_id: str,
    *,
    hard_delete: bool = False,
) -> None:
    logger.info(
        f"[DB] Deleting document {id} tenant={tenant_id} hard_delete={hard_delete}"
    )
    obj = (
        db.query(Document)
        .filter(
            Document.id == id,
            Document.tenant_id == tenant_id,
        )
        .first()
    )

    if obj:
        if hard_delete:
            db.delete(obj)
            logger.info(f"[DB] Hard deleted document {id} tenant={tenant_id}")
        else:
            deleted_at = datetime.now(timezone.utc)
            obj.status = "deleted"
            obj.is_deleted = True
            obj.deleted_at = deleted_at
            (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.document_id == id,
                    DocumentChunk.is_deleted.is_(False),
                )
                .update(
                    {
                        "is_deleted": True,
                        "deleted_at": deleted_at,
                    },
                    synchronize_session=False,
                )
            )
            logger.info(f"[DB] Soft deleted document {id} tenant={tenant_id}")
        db.commit()
    else:
        logger.warning(f"[DB] Document {id} not found for delete tenant={tenant_id}")


# --- RESTORE DELETED ---
def restore_document(
    db: Session,
    id: UUID,
    tenant_id: str,
) -> Optional[Document]:
    logger.info(f"[DB] Restoring document {id} tenant={tenant_id}")
    obj = (
        db.query(Document)
        .options(selectinload(Document.acl))
        .filter(
            Document.id == id,
            Document.tenant_id == tenant_id,
            Document.is_deleted.is_(True),
        )
        .first()
    )

    if obj:
        obj.status = "queued"
        obj.is_deleted = False
        obj.deleted_at = None
        (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.document_id == id,
                DocumentChunk.is_deleted.is_(True),
            )
            .update(
                {
                    "is_deleted": False,
                    "deleted_at": None,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        db.refresh(obj)
        _ = obj.acl
        logger.info(f"[DB] Restored document {id} tenant={tenant_id}")
        return obj

    logger.warning(f"[DB] Document {id} not found for restore tenant={tenant_id}")
    return None


# --- SEARCH ---
def apply_sorting(
    query: Query,
    model,
    sort_fields: List[str],
    directions: List[str],
) -> Query:
    for field, dir_ in zip(sort_fields, directions):
        column = getattr(model, field, None)
        if column is not None:
            query = query.order_by(asc(column) if dir_ == "asc" else desc(column))
    return query


def search_documents(
    db: Session,
    tenant_id: str,
    req: DocumentSearchRequest,
    sort: Optional[List[str]] = None,
    sort_dir: Optional[List[str]] = None,
) -> Tuple[List[Document], int]:
    sort = sort or ["created_at"]
    sort_dir = sort_dir or ["desc"]
    logger.info(
        f"[DB] Searching documents tenant={tenant_id} page={req.page.page} "
        f"size={req.page.size} sort={sort} sort_dir={sort_dir}"
    )

    query = (
        db.query(Document)
        .options(
            noload(Document.chunks),
            noload(Document.ingestion_jobs),
        )
        .filter(
            Document.tenant_id == tenant_id,
            Document.is_deleted.is_(False),
        )
    )

    if req.workspace_id:
        query = query.filter(Document.workspace_id == req.workspace_id)
    if req.source_type:
        query = query.filter(Document.source_type == req.source_type)
    if req.status:
        query = query.filter(Document.status == req.status)
    if req.owner_user_id:
        query = query.filter(Document.owner_user_id == req.owner_user_id)
    for key, value in req.metadata_filters.items():
        if isinstance(value, str):
            query = query.filter(Document.metadata_[key].as_string() == value)
        else:
            query = query.filter(Document.metadata_[key] == value)
    for tag in req.tags:
        bind = db.get_bind()
        dialect_name = bind.dialect.name if bind else ""
        if dialect_name == "sqlite":
            tag_value = tag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query = query.filter(
                Document.metadata_["tags"].as_string().like(
                    f'%"{tag_value}"%',
                    escape="\\",
                )
            )
        else:
            query = query.filter(Document.metadata_["tags"].contains([tag]))

    total = query.order_by(None).count()
    query = apply_sorting(query, Document, sort, sort_dir)
    documents = query.offset(req.page.offset).limit(req.page.size).all()
    logger.info(
        f"[DB] Search returned {len(documents)} documents of total={total} "
        f"tenant={tenant_id}"
    )
    return documents, total
