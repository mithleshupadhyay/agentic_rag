from fastapi import Depends, HTTPException

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.authorization import require_tenant_permission
from agentic_rag.core.models.user_context import UserContext


def require_role(role: str):
    async def dependency(
        user_context: UserContext = Depends(get_current_user),
    ) -> UserContext:
        if role == "admin" and "tenant.update" in user_context.tenant_permissions:
            return user_context
        if role not in (user_context.roles or []):
            raise HTTPException(
                status_code=403, detail=f"Missing required role: {role}"
            )
        return user_context

    return dependency


def require_scope(scope: str):
    async def dependency(
        user_context: UserContext = Depends(get_current_user),
    ) -> UserContext:
        if scope not in (user_context.scopes or []):
            raise HTTPException(
                status_code=403, detail=f"Missing required scope: {scope}"
            )
        return user_context

    return dependency


def require_tenant_context(
    user_context: UserContext = Depends(get_current_user),
) -> UserContext:
    if not user_context.tenant_id:
        raise HTTPException(status_code=403, detail="Missing tenant context")
    if user_context.must_change_password:
        raise HTTPException(
            status_code=403,
            detail="Change the temporary password before accessing company data.",
        )
    return user_context


def require_permission(permission_code: str):
    async def dependency(
        user_context: UserContext = Depends(get_current_user),
    ) -> UserContext:
        if user_context.tenant_uuid is None:
            raise HTTPException(status_code=403, detail="Missing company context")
        return require_tenant_permission(
            user_context,
            user_context.tenant_uuid,
            permission_code,
        )

    return dependency
