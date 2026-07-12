import logging

from fastapi import APIRouter, Depends, HTTPException, status

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings
from agentic_rag.shared.schemas.auth import (
    AuthConfigurationResponse,
    AuthSessionResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/config", response_model=AuthConfigurationResponse)
def get_auth_configuration_endpoint() -> AuthConfigurationResponse:
    provider = settings.auth_provider.strip().lower()
    if provider == "local":
        return AuthConfigurationResponse(
            mode="local",
            provider=provider,
        )

    if provider == "supertokens":
        social_providers = []
        if settings.google_client_id and settings.google_client_secret:
            social_providers.append("google")
        if settings.github_client_id and settings.github_client_secret:
            social_providers.append("github")
        return AuthConfigurationResponse(
            mode="supertokens",
            provider=provider,
            api_base_path=settings.supertokens_api_base_path,
            website_base_path=settings.supertokens_website_base_path,
            public_tenant_signup_enabled=settings.public_tenant_signup_enabled,
            social_providers=social_providers,
        )

    if provider != "auth0":
        logger.error(f"[AuthAPI] Unsupported auth provider configured provider={provider}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication provider is not configured correctly.",
        )

    if (
        not settings.auth0_issuer_url
        or not settings.auth0_api_audience
        or not settings.auth0_frontend_client_id
    ):
        logger.error(
            f"[AuthAPI] Incomplete Auth0 frontend configuration provider={provider}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth0 frontend configuration is incomplete.",
        )

    return AuthConfigurationResponse(
        mode="auth0",
        provider=provider,
        issuer_url=settings.auth0_issuer_url,
        audience=settings.auth0_api_audience,
        client_id=settings.auth0_frontend_client_id,
        scope=settings.auth0_frontend_scope,
        identity_connections=settings.auth0_identity_connections,
    )


@router.get("/session", response_model=AuthSessionResponse)
def get_auth_session_endpoint(
    user_context: UserContext = Depends(get_current_user),
) -> AuthSessionResponse:
    logger.info(
        f"[AuthAPI] Session loaded user={user_context.id} "
        f"tenant={user_context.tenant_id} workspace={user_context.workspace_id}"
    )
    return AuthSessionResponse(
        user_id=user_context.id,
        tenant_id=user_context.tenant_id,
        tenant_uuid=user_context.tenant_uuid,
        department_id=user_context.department_id,
        workspace_id=user_context.workspace_id,
        email=user_context.email,
        roles=sorted(set(user_context.roles or [])),
        group_ids=sorted(set(user_context.group_ids or [])),
        scopes=sorted(set(user_context.scopes or [])),
        tenant_permissions=sorted(set(user_context.tenant_permissions)),
        department_permissions=user_context.department_permissions,
        must_change_password=user_context.must_change_password,
        acl_version=user_context.acl_version,
        auth_provider=settings.auth_provider,
    )
