import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from agentic_rag.core.auth0_management import (
    Auth0ManagementError,
    create_auth0_organization_invitation,
    delete_auth0_organization_invitation,
)
from agentic_rag.core.dependencies import require_role
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.llm.gateway import generate_chat_completion, generate_embeddings
from agentic_rag.shared.db.crud.llm_providers import (
    create_llm_provider,
    delete_llm_provider,
    get_llm_provider,
    list_llm_providers,
    update_llm_provider,
)
from agentic_rag.shared.db.crud.users import (
    create_invited_tenant_user,
    delete_incomplete_tenant_user,
    get_tenant_user_by_email,
    list_tenant_users,
)
from agentic_rag.shared.db.models import LLMProvider, Tenant, User
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.auth import (
    AuthContext,
    TenantUserListResponse,
    TenantUserRead,
    UserInvitationRequest,
    UserInvitationResponse,
)
from agentic_rag.shared.schemas.common import PageResponse
from agentic_rag.shared.schemas.llm import (
    ChatCompletionRequest,
    EmbeddingRequest,
    LLMMessage,
    LLMProviderCreate,
    LLMProviderListResponse,
    LLMProviderRead,
    LLMProviderType,
    LLMProviderUpdate,
    LLMProviderValidationRequest,
    LLMProviderValidationResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _build_tenant_user_response(user: User) -> TenantUserRead:
    roles = sorted(
        {
            link.role.name
            for link in user.role_links
            if link.role is not None
        }
    )
    workspace_id = user.metadata_.get("workspace_id")
    return TenantUserRead(
        id=user.id,
        tenant_id=user.tenant_id,
        external_subject=user.external_subject,
        email=user.email,
        display_name=user.display_name,
        status=user.status,
        roles=roles,
        workspace_id=str(workspace_id) if workspace_id else None,
        acl_version=user.acl_version,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def _build_llm_provider_response(provider: LLMProvider) -> LLMProviderRead:
    return LLMProviderRead(
        id=provider.id,
        tenant_id=provider.tenant_id,
        name=provider.name,
        provider_type=LLMProviderType(provider.provider_type),
        chat_model=provider.chat_model,
        embedding_model=provider.embedding_model,
        embedding_dimension=provider.embedding_dimension,
        base_url=provider.base_url,
        has_api_key=bool(provider.encrypted_api_key),
        config=dict(provider.config or {}),
        is_active=provider.is_active,
        is_default_chat=provider.is_default_chat,
        is_default_embedding=provider.is_default_embedding,
        created_by=provider.created_by,
        updated_by=provider.updated_by,
        created_at=provider.created_at,
        updated_at=provider.updated_at,
    )


@router.get("/users", response_model=TenantUserListResponse)
def list_tenant_users_endpoint(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> TenantUserListResponse:
    users, total = list_tenant_users(
        db,
        tenant_id=user_context.tenant_id,
        page=page,
        size=size,
    )
    logger.info(
        f"[AdminAPI] Listed tenant users tenant={user_context.tenant_id} "
        f"admin={user_context.id} total={total}"
    )
    return TenantUserListResponse(
        items=[_build_tenant_user_response(user) for user in users],
        page=PageResponse(page=page, size=size, total=total),
    )


@router.post(
    "/users/invitations",
    response_model=UserInvitationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def invite_tenant_user_endpoint(
    invitation: UserInvitationRequest,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> UserInvitationResponse:
    normalized_email = invitation.email.strip().lower()
    existing_user = get_tenant_user_by_email(
        db,
        tenant_id=user_context.tenant_id,
        email=normalized_email,
    )
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email is already a member of the tenant.",
        )

    role_name = invitation.role.value
    tenant = (
        db.query(Tenant)
        .filter(
            Tenant.tenant_id == user_context.tenant_id,
            Tenant.status == "active",
        )
        .first()
    )
    if (
        tenant is None
        or tenant.identity_provider != "auth0"
        or not tenant.external_organization_id
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This tenant is not connected to an Auth0 organization.",
        )

    auth0_invitation_id: str | None = None
    tenant_user: User | None = None

    try:
        auth0_invitation = await create_auth0_organization_invitation(
            organization_id=tenant.external_organization_id,
            email=normalized_email,
            display_name=invitation.display_name,
            tenant_id=user_context.tenant_id,
            workspace_id=invitation.workspace_id,
            role_name=role_name,
        )
        auth0_invitation_id = auth0_invitation.id
        tenant_user = create_invited_tenant_user(
            db,
            tenant_id=user_context.tenant_id,
            external_subject=f"auth0-invitation:{auth0_invitation.id}",
            email=normalized_email,
            display_name=invitation.display_name,
            role_name=role_name,
            workspace_id=invitation.workspace_id,
            invited_by=user_context.id,
            identity_invitation_id=auth0_invitation.id,
        )

    except Auth0ManagementError as error:
        if tenant_user is not None:
            try:
                delete_incomplete_tenant_user(db, tenant_user)
            except Exception as cleanup_error:
                logger.exception(
                    f"[AdminAPI] Database invitation rollback failed "
                    f"user={tenant_user.id}: {cleanup_error}"
                )
        if auth0_invitation_id is not None:
            try:
                await delete_auth0_organization_invitation(
                    tenant.external_organization_id,
                    auth0_invitation_id,
                )
            except Auth0ManagementError as cleanup_error:
                logger.exception(
                    f"[AdminAPI] Identity invitation rollback failed "
                    f"invitation={auth0_invitation_id}: {cleanup_error}"
                )
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error),
        ) from error

    except HTTPException:
        if auth0_invitation_id is not None:
            try:
                await delete_auth0_organization_invitation(
                    tenant.external_organization_id,
                    auth0_invitation_id,
                )
            except Auth0ManagementError as cleanup_error:
                logger.exception(
                    f"[AdminAPI] Identity invitation rollback failed "
                    f"invitation={auth0_invitation_id}: {cleanup_error}"
                )
        raise

    except Exception:
        if auth0_invitation_id is not None:
            try:
                await delete_auth0_organization_invitation(
                    tenant.external_organization_id,
                    auth0_invitation_id,
                )
            except Auth0ManagementError as cleanup_error:
                logger.exception(
                    f"[AdminAPI] Identity invitation rollback failed "
                    f"invitation={auth0_invitation_id}: {cleanup_error}"
                )
        raise

    if tenant_user is None or auth0_invitation_id is None:
        raise RuntimeError("Tenant invitation did not complete.")

    logger.info(
        f"[AdminAPI] Invited tenant user user={tenant_user.id} "
        f"tenant={user_context.tenant_id} role={role_name} "
        f"admin={user_context.id}"
    )
    return UserInvitationResponse(
        user=_build_tenant_user_response(tenant_user),
        invitation_email_sent=True,
        identity_invitation_id=auth0_invitation_id,
    )


@router.get("/llm-providers", response_model=LLMProviderListResponse)
def list_llm_providers_endpoint(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> LLMProviderListResponse:
    providers, total = list_llm_providers(
        db,
        tenant_id=user_context.tenant_id,
        page=page,
        size=size,
    )
    logger.info(
        f"[AdminAPI] Listed LLM providers tenant={user_context.tenant_id} "
        f"admin={user_context.id} total={total}"
    )
    return LLMProviderListResponse(
        items=[_build_llm_provider_response(provider) for provider in providers],
        page=PageResponse(page=page, size=size, total=total),
    )


@router.post(
    "/llm-providers",
    response_model=LLMProviderRead,
    status_code=status.HTTP_201_CREATED,
)
def create_llm_provider_endpoint(
    data: LLMProviderCreate,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> LLMProviderRead:
    provider = create_llm_provider(
        db,
        tenant_id=user_context.tenant_id,
        created_by=user_context.id,
        data=data,
    )
    logger.info(
        f"[AdminAPI] Created LLM provider provider={provider.id} "
        f"tenant={user_context.tenant_id} admin={user_context.id}"
    )
    return _build_llm_provider_response(provider)


@router.get("/llm-providers/{provider_id}", response_model=LLMProviderRead)
def get_llm_provider_endpoint(
    provider_id: UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> LLMProviderRead:
    provider = get_llm_provider(
        db,
        tenant_id=user_context.tenant_id,
        provider_id=provider_id,
    )
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found.",
        )
    return _build_llm_provider_response(provider)


@router.patch("/llm-providers/{provider_id}", response_model=LLMProviderRead)
def update_llm_provider_endpoint(
    provider_id: UUID,
    data: LLMProviderUpdate,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> LLMProviderRead:
    provider = get_llm_provider(
        db,
        tenant_id=user_context.tenant_id,
        provider_id=provider_id,
    )
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found.",
        )

    updated_provider = update_llm_provider(
        db,
        provider=provider,
        updated_by=user_context.id,
        data=data,
    )
    logger.info(
        f"[AdminAPI] Updated LLM provider provider={provider.id} "
        f"tenant={user_context.tenant_id} admin={user_context.id}"
    )
    return _build_llm_provider_response(updated_provider)


@router.delete(
    "/llm-providers/{provider_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_llm_provider_endpoint(
    provider_id: UUID,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> None:
    provider = get_llm_provider(
        db,
        tenant_id=user_context.tenant_id,
        provider_id=provider_id,
    )
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found.",
        )
    delete_llm_provider(
        db,
        provider=provider,
        deleted_by=user_context.id,
    )
    logger.info(
        f"[AdminAPI] Deleted LLM provider provider={provider_id} "
        f"tenant={user_context.tenant_id} admin={user_context.id}"
    )


@router.post(
    "/llm-providers/{provider_id}/validate",
    response_model=LLMProviderValidationResponse,
)
def validate_llm_provider_endpoint(
    provider_id: UUID,
    data: LLMProviderValidationRequest,
    db: Session = Depends(get_session),
    user_context: UserContext = Depends(require_role("admin")),
) -> LLMProviderValidationResponse:
    provider = get_llm_provider(
        db,
        tenant_id=user_context.tenant_id,
        provider_id=provider_id,
    )
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found.",
        )
    if not provider.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Activate this provider before validating it.",
        )

    auth_context = AuthContext(
        user_id=user_context.id,
        tenant_id=user_context.tenant_id,
        workspace_id=user_context.workspace_id,
        roles=user_context.roles,
        group_ids=user_context.group_ids,
        scopes=user_context.scopes,
        acl_version=user_context.acl_version,
    )
    try:
        if data.capability == "chat":
            if not provider.chat_model:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="This provider does not have a chat model.",
                )
            response = generate_chat_completion(
                ChatCompletionRequest(
                    auth=auth_context,
                    provider_id=provider.id,
                    messages=[
                        LLMMessage(
                            role="user",
                            content="Reply with exactly: healthy",
                        )
                    ],
                    temperature=0.0,
                    max_tokens=8,
                    metadata={"task": "provider_validation"},
                )
            )
            response_model = response.model
            response_provider = response.provider
            latency_ms = response.latency_ms
        else:
            if not provider.embedding_model:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="This provider does not have an embedding model.",
                )
            embedding_response = generate_embeddings(
                EmbeddingRequest(
                    auth=auth_context,
                    provider_id=provider.id,
                    texts=["Agentic RAG provider validation"],
                    metadata={"task": "provider_validation"},
                )
            )
            response_model = embedding_response.model
            response_provider = embedding_response.provider
            latency_ms = embedding_response.latency_ms

    except HTTPException:
        raise
    except Exception as error:
        logger.exception(
            f"[AdminAPI] LLM provider validation failed provider={provider.id} "
            f"tenant={user_context.tenant_id} "
            f"capability={data.capability} error_type={type(error).__name__}"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Provider validation failed. Check the model, base URL, "
                "credential, and provider availability."
            ),
        ) from error

    logger.info(
        f"[AdminAPI] LLM provider validation passed provider={provider.id} "
        f"tenant={user_context.tenant_id} capability={data.capability}"
    )
    return LLMProviderValidationResponse(
        status="healthy",
        capability=data.capability,
        provider=response_provider,
        model=response_model,
        latency_ms=latency_ms,
    )
