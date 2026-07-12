import asyncio
import logging
import time
from typing import Any
from uuid import UUID

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt  # type: ignore[import-untyped]
from sqlalchemy import text
from sqlalchemy.orm import Session
from supertokens_python.asyncio import get_user as get_supertokens_user
from supertokens_python.recipe.session.framework.fastapi import verify_session

from agentic_rag.core.auth0_management import (
    Auth0ManagementError,
    get_auth0_user_identity,
)
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.identity import (
    build_user_context,
    reconcile_authenticated_identity,
)
from agentic_rag.shared.db.crud.users import (
    activate_tenant_user,
    bind_invited_tenant_user_identity,
    get_tenant_by_identity_organization,
    get_tenant_user_by_email,
    get_tenant_user_by_subject,
)
from agentic_rag.shared.db.session import get_session
from agentic_rag.shared.schemas.auth import TENANT_ROLE_SCOPES, TenantUserRole


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
logger = logging.getLogger(__name__)

_auth0_jwks: list[dict[str, Any]] = []
_auth0_jwks_expires_at = 0.0
_auth0_jwks_lock = asyncio.Lock()
_supertokens_session_verifier = verify_session(
    session_required=True,
    check_database=True,
)


async def get_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_session),
) -> UserContext:
    provider = settings.auth_provider.strip().lower()
    if provider == "local":
        if token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required.",
            )
        if token != settings.local_auth_token:
            logger.warning("Invalid local auth token")
            raise HTTPException(status_code=401, detail="Invalid local auth token")

        logger.info(
            "Local auth token verified for user_id=%s tenant_id=%s",
            settings.local_user_id,
            settings.local_tenant_id,
        )
        return UserContext(
            id=settings.local_user_id,
            customer_id=settings.local_tenant_id,
            tenant_id=settings.local_tenant_id,
            workspace_id=settings.local_workspace_id or None,
            roles=settings.local_roles,
            group_ids=settings.local_groups,
            scopes=settings.local_scopes,
            acl_version=settings.local_acl_version,
        )

    if provider == "supertokens":
        session_container = await _supertokens_session_verifier(request)
        if session_container is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required.",
            )

        supertokens_user_id = session_container.get_user_id()
        request.state.supertokens_session_handle = session_container.get_handle()
        supertokens_user = await get_supertokens_user(supertokens_user_id)
        if supertokens_user is None:
            logger.warning(
                f"[Auth] Session references an unknown SuperTokens user "
                f"user={supertokens_user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="The authenticated account no longer exists.",
            )

        recipe_user_id = session_container.get_recipe_user_id().get_as_string()
        login_method = next(
            (
                method
                for method in supertokens_user.login_methods
                if method.recipe_user_id.get_as_string() == recipe_user_id
            ),
            None,
        )
        if login_method is None and supertokens_user.login_methods:
            login_method = supertokens_user.login_methods[0]
        if login_method is None or not login_method.email:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="A verified recovery email is required for this account.",
            )

        identity_provider = "supertokens_email_password"
        if login_method.third_party is not None:
            identity_provider = login_method.third_party.id.strip().lower()
        identity_email_verified = bool(login_method.verified)
        if not identity_email_verified:
            if identity_provider == "supertokens_email_password":
                detail = "Verify your email address before accessing Agentic RAG."
            else:
                detail = (
                    "The social provider must return a verified email address. "
                    "Use another verified sign-in method."
                )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=detail,
            )

        application_user = reconcile_authenticated_identity(
            db,
            supertokens_user_id=supertokens_user_id,
            provider=identity_provider,
            email=login_method.email,
            email_verified=identity_email_verified,
        )
        db.commit()
        db.refresh(application_user)
        if db.get_bind().dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.user_id', :user_id, true)"),
                {"user_id": str(application_user.id)},
            )

        requested_tenant_id = None
        requested_tenant_header = request.headers.get("X-Tenant-ID", "").strip()
        if requested_tenant_header:
            try:
                requested_tenant_id = UUID(requested_tenant_header)
            except ValueError as error:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="X-Tenant-ID must be a valid UUID.",
                ) from error

        requested_department_id = None
        requested_department_header = request.headers.get("X-Department-ID", "").strip()
        if requested_department_header:
            try:
                requested_department_id = UUID(requested_department_header)
            except ValueError as error:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="X-Department-ID must be a valid UUID.",
                ) from error

        user_context = build_user_context(
            db,
            user=application_user,
            supertokens_user_id=supertokens_user_id,
            requested_tenant_id=requested_tenant_id,
            requested_department_id=requested_department_id,
            requested_workspace_id=request.headers.get("X-Workspace-ID") or None,
        )
        if user_context.tenant_id and db.get_bind().dialect.name == "postgresql":
            db.execute(
                text(
                    "SELECT set_config('app.tenant_id', :tenant_id, true), "
                    "set_config('app.tenant_uuid', :tenant_uuid, true), "
                    "set_config('app.user_id', :user_id, true)"
                ),
                {
                    "tenant_id": user_context.tenant_id,
                    "tenant_uuid": str(user_context.tenant_uuid or ""),
                    "user_id": str(application_user.id),
                },
            )
        logger.info(
            f"[AuthZ] SuperTokens session authorized user={application_user.id} "
            f"tenant={user_context.tenant_uuid} department={user_context.department_id}"
        )
        return user_context

    if provider != "auth0":
        logger.error(f"[Auth] Unsupported authentication provider provider={provider}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unsupported authentication provider.",
        )

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
        )

    identity_context = await _verify_auth0_token(token)
    organization_id = identity_context.identity_organization_id
    if not organization_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing the Auth0 organization claim.",
        )

    tenant = get_tenant_by_identity_organization(
        db,
        identity_provider="auth0",
        external_organization_id=organization_id,
    )
    if tenant is None:
        logger.warning(
            f"[AuthZ] Unknown Auth0 organization user={identity_context.id} "
            f"organization={organization_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This Auth0 organization is not provisioned in Agentic RAG.",
        )

    membership = get_tenant_user_by_subject(
        db,
        tenant_id=tenant.tenant_id,
        external_subject=identity_context.id,
    )
    if membership is None:
        identity_email = identity_context.email
        email_verified = identity_context.email_verified
        display_name: str | None = None

        if not identity_email or not email_verified:
            try:
                auth0_identity = await get_auth0_user_identity(identity_context.id)
            except Auth0ManagementError as error:
                raise HTTPException(
                    status_code=error.status_code,
                    detail=str(error),
                ) from error
            identity_email = auth0_identity.email
            email_verified = auth0_identity.email_verified
            display_name = auth0_identity.display_name

        if not identity_email or not email_verified:
            logger.warning(
                f"[AuthZ] Invitation reconciliation requires a verified email "
                f"user={identity_context.id} tenant={tenant.tenant_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Verify the invited email address before signing in.",
            )

        invited_membership = get_tenant_user_by_email(
            db,
            tenant_id=tenant.tenant_id,
            email=identity_email,
        )
        if invited_membership is None:
            logger.warning(
                f"[AuthZ] Auth0 subject has no tenant invitation "
                f"user={identity_context.id} tenant={tenant.tenant_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This account has not been invited to the tenant.",
            )

        membership = bind_invited_tenant_user_identity(
            db,
            user=invited_membership,
            external_subject=identity_context.id,
            display_name=display_name,
        )
    elif membership.status == "invited":
        membership = activate_tenant_user(db, membership)
    elif membership.status != "active":
        logger.warning(
            f"[AuthZ] Tenant membership is not active user={identity_context.id} "
            f"tenant={tenant.tenant_id} status={membership.status}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant membership is not active.",
        )

    membership_roles = sorted(
        {link.role.name for link in membership.role_links if link.role is not None}
    )
    membership_groups = sorted(
        {link.group.name for link in membership.group_links if link.group is not None}
    )
    effective_scopes: set[str] = set()
    for role_name in membership_roles:
        try:
            tenant_role = TenantUserRole(role_name)
        except ValueError:
            logger.warning(
                f"[AuthZ] Ignoring unknown tenant role user={identity_context.id} "
                f"tenant={tenant.tenant_id} role={role_name}"
            )
            continue
        effective_scopes.update(TENANT_ROLE_SCOPES[tenant_role])

    workspace_id = membership.metadata_.get("workspace_id")
    logger.info(
        f"[AuthZ] Tenant membership allowed user={identity_context.id} "
        f"tenant={tenant.tenant_id} organization={organization_id} "
        f"roles={membership_roles}"
    )
    return UserContext(
        id=identity_context.id,
        customer_id=tenant.tenant_id,
        tenant_id=tenant.tenant_id,
        workspace_id=str(workspace_id) if workspace_id else None,
        identity_organization_id=organization_id,
        email=membership.email or identity_context.email,
        email_verified=True,
        roles=membership_roles,
        group_ids=membership_groups,
        scopes=sorted(effective_scopes),
        acl_version=membership.acl_version,
    )


async def _load_auth0_jwks(force_refresh: bool = False) -> list[dict[str, Any]]:
    global _auth0_jwks
    global _auth0_jwks_expires_at

    now = time.monotonic()
    if not force_refresh and _auth0_jwks and _auth0_jwks_expires_at > now:
        return _auth0_jwks

    async with _auth0_jwks_lock:
        now = time.monotonic()
        if not force_refresh and _auth0_jwks and _auth0_jwks_expires_at > now:
            return _auth0_jwks

        issuer_url = settings.auth0_issuer_url
        if not issuer_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Auth0 domain is not configured.",
            )

        jwks_url = f"{issuer_url}.well-known/jwks.json"
        try:
            async with httpx.AsyncClient(
                timeout=settings.auth0_jwks_timeout_seconds
            ) as client:
                response = await client.get(jwks_url)
                response.raise_for_status()
        except httpx.HTTPError as error:
            logger.exception(f"[Auth0] JWKS request failed: {error}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Auth0 signing keys are temporarily unavailable.",
            ) from error

        keys = response.json().get("keys")
        if not isinstance(keys, list) or not keys:
            logger.error("[Auth0] JWKS response did not contain signing keys")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Auth0 returned an invalid signing key response.",
            )

        _auth0_jwks = [key for key in keys if isinstance(key, dict)]
        _auth0_jwks_expires_at = now + settings.auth0_jwks_cache_seconds
        return _auth0_jwks


async def _verify_auth0_token(token: str) -> UserContext:
    if token.count(".") != 2:
        logger.warning("Malformed Auth0 token: does not have three parts")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed JWT token.",
        )
    if not settings.auth0_issuer_url or not settings.auth0_api_audience:
        logger.error("[Auth0] Domain or API audience is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth0 API validation is not configured.",
        )

    try:
        unverified_header = jwt.get_unverified_header(token)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed JWT token.",
        ) from error

    if unverified_header.get("alg") != "RS256":
        logger.warning(
            f"[Auth0] Rejected unsupported signing algorithm "
            f"algorithm={unverified_header.get('alg')}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unsupported JWT signing algorithm.",
        )

    key_id = unverified_header.get("kid")
    signing_keys = await _load_auth0_jwks()
    rsa_key = next((key for key in signing_keys if key.get("kid") == key_id), None)
    if rsa_key is None:
        signing_keys = await _load_auth0_jwks(force_refresh=True)
        rsa_key = next(
            (key for key in signing_keys if key.get("kid") == key_id),
            None,
        )
    if rsa_key is None:
        logger.warning(f"[Auth0] Signing key not found kid={key_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT signing key was not found.",
        )

    try:
        payload = jwt.decode(
            token,
            key={
                "kty": rsa_key["kty"],
                "kid": rsa_key["kid"],
                "use": rsa_key.get("use", "sig"),
                "n": rsa_key["n"],
                "e": rsa_key["e"],
            },
            algorithms=["RS256"],
            issuer=settings.auth0_issuer_url,
            audience=settings.auth0_api_audience,
        )
    except Exception as error:
        logger.warning(f"[Auth0] Token validation failed error={type(error).__name__}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Auth0 access token validation failed.",
        ) from error

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing the subject claim.",
        )

    organization_id = payload.get("org_id")
    if not isinstance(organization_id, str) or not organization_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing the Auth0 organization claim.",
        )

    email = payload.get("email") or payload.get("https://agentic-rag.ai/email")
    if not isinstance(email, str) or not email:
        email = None
    else:
        email = email.strip().lower()

    email_verified_claim = payload.get("email_verified")
    if email_verified_claim is None:
        email_verified_claim = payload.get("https://agentic-rag.ai/email_verified")

    logger.info(
        f"[Auth0] Access token verified user={subject} organization={organization_id}"
    )
    return UserContext(
        id=subject,
        customer_id=str(organization_id or ""),
        tenant_id=str(organization_id or ""),
        identity_organization_id=(
            organization_id if isinstance(organization_id, str) else None
        ),
        email=email,
        email_verified=email_verified_claim is True,
    )
