import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from agentic_rag.core.authorization import (
    can_read_document,
    require_document_permission,
)
from agentic_rag.core.dependencies import require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.db.crud.documents import (
    create_document,
    delete_document,
    get_document,
    restore_document,
    search_documents,
    update_document,
)
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.auth import PermissionAction
from agentic_rag.shared.schemas.common import PageRequest, PageResponse
from agentic_rag.shared.schemas.documents import (
    DocumentActionResponse,
    DocumentCreateRequest,
    DocumentListItem,
    DocumentRead,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
def create_document_endpoint(
    payload: DocumentCreateRequest,
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:write")),
) -> DocumentRead:
    logger.info(
        f"[DocumentAPI] Create document tenant={user_ctx.tenant_id}, "
        f"user={user_ctx.id}, source_type={payload.source_type}"
    )

    document = create_document(
        user_context=user_ctx,
        db=db,
        obj_in=payload,
    )

    logger.info(f"[DocumentAPI] Created document {document.id}")
    return DocumentRead.model_validate(document)


@router.get("", response_model=DocumentSearchResponse)
def list_documents_endpoint(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:read")),
) -> DocumentSearchResponse:
    logger.info(
        f"[DocumentAPI] List documents tenant={user_ctx.tenant_id}, "
        f"user={user_ctx.id}, page={page}, size={size}"
    )

    req = DocumentSearchRequest(page=PageRequest(page=page, size=size))
    documents, total = search_documents(
        db=db,
        tenant_id=user_ctx.tenant_id,
        req=req,
        sort=user_ctx.default_sort,
        sort_dir=user_ctx.default_sort_dir,
    )

    # Filter the tenant-scoped DB result by user/role/group ACL.
    readable_documents = []
    for document in documents:
        if can_read_document(user_ctx, document):
            readable_documents.append(document)

    logger.info(
        f"[DocumentAPI] Listed {len(readable_documents)} readable documents "
        f"from total={total}"
    )
    return DocumentSearchResponse(
        items=[
            DocumentListItem.model_validate(document)
            for document in readable_documents
        ],
        page=PageResponse(
            page=page,
            size=size,
            total=len(readable_documents),
        ),
    )


@router.post("/search", response_model=DocumentSearchResponse)
def search_documents_endpoint(
    payload: DocumentSearchRequest,
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:read")),
) -> DocumentSearchResponse:
    logger.info(
        f"[DocumentAPI] Search documents tenant={user_ctx.tenant_id}, "
        f"user={user_ctx.id}, page={payload.page.page}, size={payload.page.size}"
    )

    documents, total = search_documents(
        db=db,
        tenant_id=user_ctx.tenant_id,
        req=payload,
        sort=user_ctx.default_sort,
        sort_dir=user_ctx.default_sort_dir,
    )

    # Search is still tenant-scoped first, then filtered by ACL.
    readable_documents = []
    for document in documents:
        if can_read_document(user_ctx, document):
            readable_documents.append(document)

    logger.info(
        f"[DocumentAPI] Search returned {len(readable_documents)} readable documents "
        f"from total={total}"
    )
    return DocumentSearchResponse(
        items=[
            DocumentListItem.model_validate(document)
            for document in readable_documents
        ],
        page=PageResponse(
            page=payload.page.page,
            size=payload.page.size,
            total=len(readable_documents),
        ),
    )


@router.get("/{document_id}", response_model=DocumentRead)
def get_document_endpoint(
    document_id: UUID,
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:read")),
) -> DocumentRead:
    logger.info(
        f"[DocumentAPI] Get document {document_id} "
        f"tenant={user_ctx.tenant_id}, user={user_ctx.id}"
    )

    document = get_document(
        db=db,
        id=document_id,
        tenant_id=user_ctx.tenant_id,
    )
    if not document:
        logger.warning(f"[DocumentAPI] Document {document_id} not found")
        raise HTTPException(status_code=404, detail="Document not found.")

    # User-level permission is checked after tenant-scoped fetch.
    require_document_permission(
        user_context=user_ctx,
        document=document,
        action=PermissionAction.READ,
    )

    logger.info(f"[DocumentAPI] Get document allowed {document_id}")
    return DocumentRead.model_validate(document)


@router.patch("/{document_id}", response_model=DocumentRead)
def update_document_endpoint(
    document_id: UUID,
    payload: DocumentUpdateRequest,
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:write")),
) -> DocumentRead:
    logger.info(
        f"[DocumentAPI] Update document {document_id} "
        f"tenant={user_ctx.tenant_id}, user={user_ctx.id}"
    )

    document = get_document(
        db=db,
        id=document_id,
        tenant_id=user_ctx.tenant_id,
    )
    if not document:
        logger.warning(f"[DocumentAPI] Document {document_id} not found for update")
        raise HTTPException(status_code=404, detail="Document not found.")

    require_document_permission(
        user_context=user_ctx,
        document=document,
        action=PermissionAction.WRITE,
    )

    updated_document = update_document(db=db, db_obj=document, obj_in=payload)
    logger.info(f"[DocumentAPI] Updated document {document_id}")
    return DocumentRead.model_validate(updated_document)


@router.delete("/{document_id}", response_model=DocumentActionResponse)
def delete_document_endpoint(
    document_id: UUID,
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:delete")),
) -> DocumentActionResponse:
    logger.info(
        f"[DocumentAPI] Delete document {document_id} "
        f"tenant={user_ctx.tenant_id}, user={user_ctx.id}"
    )

    document = get_document(
        db=db,
        id=document_id,
        tenant_id=user_ctx.tenant_id,
    )
    if not document:
        logger.warning(f"[DocumentAPI] Document {document_id} not found for delete")
        raise HTTPException(status_code=404, detail="Document not found.")

    require_document_permission(
        user_context=user_ctx,
        document=document,
        action=PermissionAction.DELETE,
    )

    delete_document(db=db, id=document_id, tenant_id=user_ctx.tenant_id)
    logger.info(f"[DocumentAPI] Deleted document {document_id}")
    return DocumentActionResponse(id=document_id, status="deleted")


@router.post("/{document_id}/restore", response_model=DocumentRead)
def restore_document_endpoint(
    document_id: UUID,
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:write")),
) -> DocumentRead:
    logger.info(
        f"[DocumentAPI] Restore document {document_id} "
        f"tenant={user_ctx.tenant_id}, user={user_ctx.id}"
    )

    document = get_document(
        db=db,
        id=document_id,
        tenant_id=user_ctx.tenant_id,
        include_deleted=True,
    )
    if not document:
        logger.warning(f"[DocumentAPI] Document {document_id} not found for restore")
        raise HTTPException(status_code=404, detail="Document not found.")

    if not document.is_deleted:
        logger.info(f"[DocumentAPI] Restore skipped for active document {document_id}")
        return DocumentRead.model_validate(document)

    require_document_permission(
        user_context=user_ctx,
        document=document,
        action=PermissionAction.WRITE,
    )

    restored_document = restore_document(
        db=db,
        id=document_id,
        tenant_id=user_ctx.tenant_id,
    )
    if not restored_document:
        logger.warning(f"[DocumentAPI] Restore failed for document {document_id}")
        raise HTTPException(status_code=404, detail="Document not found.")

    logger.info(f"[DocumentAPI] Restored document {document_id}")
    return DocumentRead.model_validate(restored_document)
