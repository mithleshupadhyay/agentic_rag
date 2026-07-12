import logging
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from agentic_rag.shared.config import settings


logger = logging.getLogger(__name__)


class Auth0ManagementError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Auth0Organization:
    id: str
    name: str


@dataclass(frozen=True)
class Auth0OrganizationInvitation:
    id: str
    organization_id: str
    email: str
    email_sent: bool


@dataclass(frozen=True)
class Auth0UserIdentity:
    subject: str
    email: str | None
    email_verified: bool
    display_name: str | None


_management_access_token: str | None = None
_management_access_token_expires_at = 0.0


def _get_auth0_management_configuration() -> tuple[str, str]:
    if settings.auth_provider.strip().lower() != "auth0":
        raise Auth0ManagementError(
            "User and organization management requires AUTH_PROVIDER=auth0.",
            status_code=503,
        )

    issuer_url = settings.auth0_issuer_url
    if not issuer_url:
        raise Auth0ManagementError(
            "Auth0 domain is not configured.",
            status_code=503,
        )
    if (
        not settings.auth0_management_client_id
        or not settings.auth0_management_client_secret
    ):
        raise Auth0ManagementError(
            "Auth0 Management API client credentials are not configured.",
            status_code=503,
        )

    return issuer_url.rstrip("/"), settings.auth0_management_audience


async def _get_auth0_management_access_token(
    client: httpx.AsyncClient,
    issuer_url: str,
    audience: str,
) -> str:
    global _management_access_token
    global _management_access_token_expires_at

    now = time.monotonic()
    if (
        _management_access_token
        and _management_access_token_expires_at > now + 30
    ):
        return _management_access_token

    try:
        response = await client.post(
            f"{issuer_url}/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.auth0_management_client_id,
                "client_secret": settings.auth0_management_client_secret,
                "audience": audience,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except httpx.HTTPError as error:
        logger.exception(f"[Auth0Management] Token request failed: {error}")
        raise Auth0ManagementError(
            "The identity service could not be reached.",
            status_code=503,
        ) from error

    if response.status_code != 200:
        logger.error(
            f"[Auth0Management] Token request rejected "
            f"status={response.status_code}"
        )
        raise Auth0ManagementError(
            "The identity service rejected the Management API credentials.",
            status_code=503,
        )

    payload = response.json()
    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access_token, str) or not access_token:
        logger.error("[Auth0Management] Token response did not contain an access token")
        raise Auth0ManagementError(
            "The identity service returned an invalid management token response."
        )

    _management_access_token = access_token
    _management_access_token_expires_at = now + max(int(expires_in or 60), 60)
    return access_token


async def create_auth0_organization(
    tenant_id: str,
    name: str,
    display_name: str,
) -> Auth0Organization:
    issuer_url, audience = _get_auth0_management_configuration()
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise Auth0ManagementError("Auth0 organization name is required.", 400)

    enabled_connections = []
    for connection_id in settings.auth0_organization_connection_ids:
        enabled_connections.append(
            {
                "connection_id": connection_id,
                "assign_membership_on_login": False,
            }
        )

    async with httpx.AsyncClient(
        timeout=settings.auth0_management_timeout_seconds
    ) as client:
        access_token = await _get_auth0_management_access_token(
            client,
            issuer_url,
            audience,
        )
        try:
            response = await client.post(
                f"{issuer_url}/api/v2/organizations",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "name": normalized_name,
                    "display_name": display_name.strip(),
                    "metadata": {"agentic_rag_tenant_id": tenant_id},
                    "enabled_connections": enabled_connections,
                },
            )
        except httpx.HTTPError as error:
            logger.exception(
                f"[Auth0Management] Organization request failed tenant={tenant_id}: "
                f"{error}"
            )
            raise Auth0ManagementError(
                "The identity service could not create the tenant organization.",
                status_code=503,
            ) from error

    if response.status_code == 409:
        raise Auth0ManagementError(
            "An Auth0 organization with this tenant slug already exists.",
            status_code=409,
        )
    if response.status_code != 201:
        logger.error(
            f"[Auth0Management] Organization creation rejected "
            f"tenant={tenant_id} status={response.status_code}"
        )
        raise Auth0ManagementError(
            "The identity service rejected the tenant organization.",
            status_code=502,
        )

    payload = response.json()
    organization_id = payload.get("id")
    organization_name = payload.get("name")
    if not isinstance(organization_id, str) or not organization_id:
        raise Auth0ManagementError(
            "The identity service returned an invalid organization response."
        )

    logger.info(
        f"[Auth0Management] Created organization tenant={tenant_id} "
        f"organization={organization_id}"
    )
    return Auth0Organization(
        id=organization_id,
        name=(
            organization_name
            if isinstance(organization_name, str) and organization_name
            else normalized_name
        ),
    )


async def delete_auth0_organization(organization_id: str) -> None:
    issuer_url, audience = _get_auth0_management_configuration()
    organization_path = quote(organization_id, safe="")

    async with httpx.AsyncClient(
        timeout=settings.auth0_management_timeout_seconds
    ) as client:
        access_token = await _get_auth0_management_access_token(
            client,
            issuer_url,
            audience,
        )
        try:
            response = await client.delete(
                f"{issuer_url}/api/v2/organizations/{organization_path}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except httpx.HTTPError as error:
            logger.exception(
                f"[Auth0Management] Organization deletion failed "
                f"organization={organization_id}: {error}"
            )
            raise Auth0ManagementError(
                "The identity service could not remove the tenant organization.",
                status_code=503,
            ) from error

    if response.status_code not in {204, 404}:
        logger.error(
            f"[Auth0Management] Organization deletion rejected "
            f"organization={organization_id} status={response.status_code}"
        )
        raise Auth0ManagementError(
            "The identity service rejected organization cleanup."
        )

    logger.warning(
        f"[Auth0Management] Removed organization organization={organization_id}"
    )


async def create_auth0_organization_invitation(
    organization_id: str,
    email: str,
    display_name: str | None,
    tenant_id: str,
    workspace_id: str | None,
    role_name: str,
) -> Auth0OrganizationInvitation:
    issuer_url, audience = _get_auth0_management_configuration()
    if not settings.auth0_frontend_client_id:
        raise Auth0ManagementError(
            "Auth0 frontend application client ID is not configured.",
            status_code=503,
        )

    normalized_email = email.strip().lower()
    organization_path = quote(organization_id, safe="")
    app_metadata = {
        "agentic_rag_tenant_id": tenant_id,
        "agentic_rag_role": role_name,
    }
    if workspace_id:
        app_metadata["agentic_rag_workspace_id"] = workspace_id

    request_payload: dict[str, object] = {
        "inviter": {"name": settings.auth0_inviter_name},
        "invitee": {"email": normalized_email},
        "client_id": settings.auth0_frontend_client_id,
        "app_metadata": app_metadata,
        "user_metadata": {
            "display_name": display_name.strip() if display_name else "",
        },
        "ttl_sec": settings.auth0_invitation_ttl_seconds,
        "send_invitation_email": True,
    }
    if settings.auth0_database_connection_id:
        request_payload["connection_id"] = settings.auth0_database_connection_id

    async with httpx.AsyncClient(
        timeout=settings.auth0_management_timeout_seconds
    ) as client:
        access_token = await _get_auth0_management_access_token(
            client,
            issuer_url,
            audience,
        )
        try:
            response = await client.post(
                f"{issuer_url}/api/v2/organizations/"
                f"{organization_path}/invitations",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
        except httpx.HTTPError as error:
            logger.exception(
                f"[Auth0Management] Invitation request failed "
                f"tenant={tenant_id} organization={organization_id}: {error}"
            )
            raise Auth0ManagementError(
                "The identity service could not create the user invitation.",
                status_code=503,
            ) from error

    if response.status_code == 409:
        raise Auth0ManagementError(
            "This email already has a pending organization invitation.",
            status_code=409,
        )
    if response.status_code != 200:
        logger.error(
            f"[Auth0Management] Invitation rejected tenant={tenant_id} "
            f"organization={organization_id} status={response.status_code}"
        )
        raise Auth0ManagementError(
            "The identity service rejected the user invitation.",
            status_code=502,
        )

    payload = response.json()
    invitation_id = payload.get("id")
    returned_organization_id = payload.get("organization_id")
    if not isinstance(invitation_id, str) or not invitation_id:
        raise Auth0ManagementError(
            "The identity service returned an invalid invitation response."
        )
    if returned_organization_id != organization_id:
        raise Auth0ManagementError(
            "The identity service returned an invitation for another organization."
        )

    logger.info(
        f"[Auth0Management] Created organization invitation tenant={tenant_id} "
        f"organization={organization_id} invitation={invitation_id}"
    )
    return Auth0OrganizationInvitation(
        id=invitation_id,
        organization_id=organization_id,
        email=normalized_email,
        email_sent=True,
    )


async def delete_auth0_organization_invitation(
    organization_id: str,
    invitation_id: str,
) -> None:
    issuer_url, audience = _get_auth0_management_configuration()
    organization_path = quote(organization_id, safe="")
    invitation_path = quote(invitation_id, safe="")

    async with httpx.AsyncClient(
        timeout=settings.auth0_management_timeout_seconds
    ) as client:
        access_token = await _get_auth0_management_access_token(
            client,
            issuer_url,
            audience,
        )
        try:
            response = await client.delete(
                f"{issuer_url}/api/v2/organizations/{organization_path}/"
                f"invitations/{invitation_path}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except httpx.HTTPError as error:
            logger.exception(
                f"[Auth0Management] Invitation deletion failed "
                f"organization={organization_id} invitation={invitation_id}: {error}"
            )
            raise Auth0ManagementError(
                "The identity service could not remove the invitation.",
                status_code=503,
            ) from error

    if response.status_code not in {204, 404}:
        logger.error(
            f"[Auth0Management] Invitation deletion rejected "
            f"organization={organization_id} invitation={invitation_id} "
            f"status={response.status_code}"
        )
        raise Auth0ManagementError(
            "The identity service rejected invitation cleanup."
        )

    logger.warning(
        f"[Auth0Management] Removed invitation organization={organization_id} "
        f"invitation={invitation_id}"
    )


async def get_auth0_user_identity(subject: str) -> Auth0UserIdentity:
    issuer_url, audience = _get_auth0_management_configuration()
    subject_path = quote(subject, safe="")

    async with httpx.AsyncClient(
        timeout=settings.auth0_management_timeout_seconds
    ) as client:
        access_token = await _get_auth0_management_access_token(
            client,
            issuer_url,
            audience,
        )
        try:
            response = await client.get(
                f"{issuer_url}/api/v2/users/{subject_path}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except httpx.HTTPError as error:
            logger.exception(
                f"[Auth0Management] User lookup failed subject={subject}: {error}"
            )
            raise Auth0ManagementError(
                "The identity service could not resolve the signed-in account.",
                status_code=503,
            ) from error

    if response.status_code == 404:
        raise Auth0ManagementError(
            "The signed-in identity no longer exists.",
            status_code=401,
        )
    if response.status_code != 200:
        logger.error(
            f"[Auth0Management] User lookup rejected subject={subject} "
            f"status={response.status_code}"
        )
        raise Auth0ManagementError(
            "The identity service rejected the signed-in account lookup.",
            status_code=502,
        )

    payload = response.json()
    user_id = payload.get("user_id")
    if user_id != subject:
        raise Auth0ManagementError(
            "The identity service returned an invalid user response."
        )

    email = payload.get("email")
    display_name = payload.get("name")
    logger.info(f"[Auth0Management] Resolved signed-in identity subject={subject}")
    return Auth0UserIdentity(
        subject=subject,
        email=(email.strip().lower() if isinstance(email, str) and email else None),
        email_verified=payload.get("email_verified") is True,
        display_name=(
            display_name.strip()
            if isinstance(display_name, str) and display_name.strip()
            else None
        ),
    )
