from fastapi import Depends, HTTPException

from agentic_rag.core.auth import get_current_user
from agentic_rag.core.models.user_context import UserContext


def require_role(role: str):
    async def dependency(user_context: UserContext = Depends(get_current_user)) -> UserContext:
        if role not in (user_context.roles or []):
            raise HTTPException(status_code=403, detail=f"Missing required role: {role}")
        return user_context

    return dependency


def require_scope(scope: str):
    async def dependency(user_context: UserContext = Depends(get_current_user)) -> UserContext:
        if scope not in (user_context.scopes or []):
            raise HTTPException(status_code=403, detail=f"Missing required scope: {scope}")
        return user_context

    return dependency


def require_tenant_context(
    user_context: UserContext = Depends(get_current_user),
) -> UserContext:
    if not user_context.tenant_id:
        raise HTTPException(status_code=403, detail="Missing tenant context")
    return user_context
