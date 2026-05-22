from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import asc, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session, noload, selectinload

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.models import Document, DocumentAcl, DocumentChunk
from agentic_rag.shared.schemas.documents import (
    DocumentCreateRequest,
    DocumentSearchRequest,
    DocumentUpdateRequest,
)


# --- CREATE ---
def create_document(
    user_context: UserContext,
    db: Session,
    obj_in: DocumentCreateRequest,
    object_key: Optional[str] = None,
) -> Document:
    try:
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
        return db_obj

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Database error during document creation.",
        )


# --- GET ONE ---
def get_document(db: Session, id: UUID, tenant_id: str) -> Optional[Document]:
    return (
        db.query(Document)
        .options(
            selectinload(Document.acl),
            noload(Document.chunks),
            noload(Document.ingestion_jobs),
        )
        .filter(
            Document.id == id,
            Document.tenant_id == tenant_id,
            Document.is_deleted.is_(False),
        )
        .first()
    )


# --- LIST MANY ---
def list_documents(
    db: Session,
    tenant_id: str,
    skip: int = 0,
    limit: int = 50,
    sort: Optional[list[str]] = None,
    sort_dir: Optional[list[str]] = None,
) -> list[Document]:
    sort = sort or ["created_at"]
    sort_dir = sort_dir or ["desc"]

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
    return query.offset(skip).limit(limit).all()


# --- UPDATE ---
def update_document(
    db: Session,
    db_obj: Document,
    obj_in: DocumentUpdateRequest,
) -> Document:
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
        return db_obj

    except IntegrityError:
        db.rollback()
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
    db_obj = get_document(db, document_id, tenant_id)

    if not db_obj:
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
        db.commit()


# --- RESTORE DELETED ---
def restore_document(
    db: Session,
    id: UUID,
    tenant_id: str,
) -> Optional[Document]:
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
        return obj

    return None


# --- SEARCH ---
def apply_sorting(
    query: Query,
    model,
    sort_fields: list[str],
    directions: list[str],
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
    sort: Optional[list[str]] = None,
    sort_dir: Optional[list[str]] = None,
) -> tuple[list[Document], int]:
    sort = sort or ["created_at"]
    sort_dir = sort_dir or ["desc"]

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
    return query.offset(req.page.offset).limit(req.page.size).all(), total
