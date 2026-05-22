from typing import List, Optional

from pydantic import BaseModel


class UserContext(BaseModel):
    id: str
    customer_id: str
    tenant_id: str
    workspace_id: Optional[str] = None
    roles: Optional[List[str]] = []
    group_ids: Optional[List[str]] = []
    scopes: Optional[List[str]] = []
    acl_version: int = 1
    default_sort: Optional[List[str]] = ["created_at"]
    default_sort_dir: Optional[List[str]] = ["desc"]
