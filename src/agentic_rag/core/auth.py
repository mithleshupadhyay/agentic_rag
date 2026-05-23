import logging

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt  # type: ignore[import-untyped]

from agentic_rag.core.models.user_context import UserContext
from agentic_rag.shared.config import settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserContext:
    if settings.auth_provider == "local":
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

    if settings.auth_provider in ("auth0", "keycloak", "oidc"):
        return await _verify_oidc_token(token)

    raise HTTPException(status_code=500, detail="Unsupported AUTH_PROVIDER")


async def _verify_oidc_token(token: str) -> UserContext:
    if token.count(".") != 2:
        logger.warning("Malformed token: does not have 3 parts")
        raise HTTPException(status_code=401, detail="Malformed JWT token")

    try:
        unverified_header = jwt.get_unverified_header(token)

        jwks_urls = []
        if settings.oidc_jwks_url:
            jwks_urls.append(settings.oidc_jwks_url)
        elif settings.oidc_issuer_url:
            issuer_url = settings.oidc_issuer_url.rstrip("/")
            if settings.auth_provider == "keycloak":
                jwks_urls.append(f"{issuer_url}/protocol/openid-connect/certs")
            else:
                jwks_urls.append(f"{issuer_url}/.well-known/jwks.json")

        if not jwks_urls:
            raise HTTPException(status_code=500, detail="OIDC JWKS URL is not configured")

        for jwks_url in jwks_urls:
            try:
                logger.debug("Trying JWKS from: %s", jwks_url)

                async with httpx.AsyncClient() as client:
                    res = await client.get(jwks_url)
                    res.raise_for_status()
                    jwks = res.json()["keys"]

                rsa_key = next(
                    (key for key in jwks if key["kid"] == unverified_header["kid"]),
                    None,
                )
                if rsa_key is None:
                    logger.warning(
                        "RSA key not found for kid=%s from url=%s",
                        unverified_header["kid"],
                        jwks_url,
                    )
                    continue

                decode_kwargs = {
                    "key": {
                        "kty": rsa_key["kty"],
                        "kid": rsa_key["kid"],
                        "use": rsa_key.get("use", "sig"),
                        "n": rsa_key["n"],
                        "e": rsa_key["e"],
                    },
                    "algorithms": ["RS256"],
                }

                if settings.oidc_issuer_url:
                    decode_kwargs["issuer"] = settings.oidc_issuer_url.rstrip("/")
                if settings.oidc_audience:
                    decode_kwargs["audience"] = settings.oidc_audience
                else:
                    decode_kwargs["options"] = {"verify_aud": False}

                payload = jwt.decode(token, **decode_kwargs)

                user_id = payload.get("sub")
                tenant_id = (
                    payload.get("tenant_id")
                    or payload.get("org_id")
                    or payload.get("organization_id")
                    or payload.get("https://agentic-rag.ai/tenant_id")
                    or payload.get("https://agentic-rag.ai/org_id")
                    or user_id
                )

                if not user_id:
                    raise HTTPException(
                        status_code=401,
                        detail="Token payload missing 'sub' claim",
                    )

                roles = []
                if isinstance(payload.get("roles"), list):
                    roles.extend(payload["roles"])

                realm_access = payload.get("realm_access")
                if isinstance(realm_access, dict) and isinstance(
                    realm_access.get("roles"), list
                ):
                    roles.extend(realm_access["roles"])

                resource_access = payload.get("resource_access")
                if isinstance(resource_access, dict):
                    for resource in resource_access.values():
                        if isinstance(resource, dict) and isinstance(
                            resource.get("roles"), list
                        ):
                            roles.extend(resource["roles"])

                groups = payload.get("groups") if isinstance(payload.get("groups"), list) else []
                scopes = []
                if isinstance(payload.get("scopes"), list):
                    scopes.extend(payload["scopes"])
                if isinstance(payload.get("scope"), str):
                    scopes.extend(payload["scope"].split())

                logger.info(
                    "OIDC token verified for user_id=%s via jwks_url=%s",
                    user_id,
                    jwks_url,
                )
                return UserContext(
                    id=user_id,
                    customer_id=tenant_id,
                    tenant_id=tenant_id,
                    workspace_id=payload.get("workspace_id"),
                    roles=sorted(set(roles)),
                    group_ids=groups,
                    scopes=sorted(set(scopes)),
                    acl_version=int(payload.get("acl_version", 1)),
                )

            except HTTPException:
                raise
            except Exception as inner_exc:
                logger.warning("Token validation failed for jwks_url=%s: %s", jwks_url, inner_exc)
                continue

        raise HTTPException(status_code=401, detail="Token validation failed for all configured OIDC issuers")

    except Exception as e:
        logger.exception("Token verification failed")
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")
