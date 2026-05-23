import hashlib
import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from agentic_rag.core.authorization import (
    can_read_document,
    require_document_permission,
)
from agentic_rag.core.dependencies import require_scope
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.documents import (
    attach_document_object,
    create_document,
    create_ingestion_job_for_document,
    delete_document,
    get_document,
    mark_document_failed,
    restore_document,
    search_documents,
    update_document,
)
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.auth import AclPolicy, PermissionAction
from agentic_rag.shared.schemas.common import PageRequest, PageResponse
from agentic_rag.shared.schemas.documents import (
    DocumentActionResponse,
    DocumentCreateRequest,
    DocumentListItem,
    DocumentRead,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentSourceType,
    DocumentUpdateRequest,
    DocumentUploadResponse,
    FileMetadata,
)
from agentic_rag.storage.object_store import ObjectStoreClient

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


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document_endpoint(
    file: UploadFile = File(...),
    workspace_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    idempotency_key: str | None = Form(default=None),
    db: Session = Depends(get_session),
    user_ctx: UserContext = Depends(require_scope("documents:write")),
) -> DocumentUploadResponse:
    safe_file_name = (file.filename or "").replace("\\", "/").split("/")[-1].strip()
    if not safe_file_name:
        raise HTTPException(status_code=400, detail="Uploaded file name is required.")

    logger.info(
        f"[DocumentAPI] Upload document tenant={user_ctx.tenant_id}, "
        f"user={user_ctx.id}, file={safe_file_name}"
    )

    upload_data = file.file.read()
    if not upload_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(upload_data) > settings.document_upload_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file is larger than the configured limit.",
        )

    metadata = {}
    if metadata_json:
        try:
            metadata_payload = json.loads(metadata_json)
            if not isinstance(metadata_payload, dict):
                raise ValueError("metadata_json must be a JSON object")
            metadata = metadata_payload
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[DocumentAPI] Invalid upload metadata: {e}")
            raise HTTPException(
                status_code=400,
                detail="metadata_json must be a valid JSON object.",
            )

    content_type = file.content_type or "application/octet-stream"
    content_hash = hashlib.sha256(upload_data).hexdigest()
    workspace = workspace_id.strip() if workspace_id and workspace_id.strip() else None
    document_title = title.strip() if title and title.strip() else safe_file_name

    document_payload = DocumentCreateRequest(
        workspace_id=workspace,
        source_type=DocumentSourceType.UPLOAD,
        source_uri=f"upload://{safe_file_name}",
        title=document_title,
        file=FileMetadata(
            file_name=safe_file_name,
            mime_type=content_type,
            byte_size=len(upload_data),
            content_hash=content_hash,
        ),
        metadata=metadata,
        acl=AclPolicy(
            allowed_user_ids=[user_ctx.id],
            acl_version=user_ctx.acl_version,
        ),
        idempotency_key=idempotency_key,
    )

    document = create_document(
        user_context=user_ctx,
        db=db,
        obj_in=document_payload,
    )

    object_store = ObjectStoreClient()
    object_key = object_store.build_object_key(
        tenant_id=user_ctx.tenant_id,
        workspace_id=document.workspace_id,
        document_id=document.id,
        file_name=safe_file_name,
    )

    try:
        upload_result = object_store.put_bytes(
            object_key=object_key,
            data=upload_data,
            content_type=content_type,
            metadata={
                "tenant_id": user_ctx.tenant_id,
                "document_id": str(document.id),
                "content_hash": content_hash,
            },
        )
    except Exception as e:
        mark_document_failed(db=db, db_obj=document)
        logger.exception(f"[DocumentAPI] Object upload failed for document {document.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload document object.",
        )

    try:
        document = attach_document_object(
            db=db,
            db_obj=document,
            object_key=object_key,
        )
        ingestion_job = create_ingestion_job_for_document(
            user_context=user_ctx,
            db=db,
            document=document,
            idempotency_key=idempotency_key,
            metadata={
                "file_name": safe_file_name,
                "mime_type": content_type,
                "content_hash": content_hash,
            },
        )
    except HTTPException:
        try:
            object_store.delete_object(object_key)
            logger.info(f"[DocumentAPI] Deleted orphaned object {object_key}")
        except Exception as cleanup_exc:
            logger.warning(
                f"[DocumentAPI] Failed to delete orphaned object {object_key}: "
                f"{cleanup_exc}"
            )
        mark_document_failed(db=db, db_obj=document)
        raise

    logger.info(
        f"[DocumentAPI] Uploaded document {document.id} "
        f"ingestion_job={ingestion_job.id}"
    )
    return DocumentUploadResponse(
        document=DocumentRead.model_validate(document),
        ingestion_job_id=ingestion_job.id,
        ingestion_status=ingestion_job.status,
        ingestion_stage=ingestion_job.current_stage,
        bucket=upload_result["bucket"],
        object_key=upload_result["object_key"],
        content_hash=upload_result["content_hash"],
        byte_size=upload_result["byte_size"],
    )


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
