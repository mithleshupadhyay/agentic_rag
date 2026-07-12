from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    id: str
    customer_id: str
    tenant_id: str
    app_user_id: Optional[UUID] = None
    supertokens_user_id: Optional[str] = None
    tenant_uuid: Optional[UUID] = None
    department_id: Optional[UUID] = None
    workspace_id: Optional[str] = None
    identity_organization_id: Optional[str] = None
    email: Optional[str] = None
    email_verified: bool = False
    roles: List[str] = Field(default_factory=list)
    group_ids: List[str] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)
    tenant_permissions: List[str] = Field(default_factory=list)
    department_permissions: dict[UUID, List[str]] = Field(default_factory=dict)
    accessible_department_ids: List[UUID] = Field(default_factory=list)
    must_change_password: bool = False
    acl_version: int = 1
    default_sort: List[str] = Field(default_factory=lambda: ["created_at"])
    default_sort_dir: List[str] = Field(default_factory=lambda: ["desc"])

    def has_tenant_permission(self, permission_code: str) -> bool:
        return permission_code in self.tenant_permissions

    def has_department_permission(
        self,
        department_id: UUID,
        permission_code: str,
    ) -> bool:
        if "tenant.data.manage_all" in self.tenant_permissions:
            return True
        if (
            permission_code in {"department.view", "documents.view", "rag.query"}
            and "tenant.data.view_all" in self.tenant_permissions
        ):
            return True
        return permission_code in self.department_permissions.get(department_id, [])
