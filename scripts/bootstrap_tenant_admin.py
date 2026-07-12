import argparse
import asyncio
import logging
import re

from sqlalchemy.exc import IntegrityError

from agentic_rag.core.auth0_management import (
    Auth0ManagementError,
    create_auth0_organization,
    create_auth0_organization_invitation,
    delete_auth0_organization,
    delete_auth0_organization_invitation,
)
from agentic_rag.shared.db.crud.users import (
    create_invited_tenant_user,
    delete_incomplete_tenant_user,
    get_tenant_user_by_email,
)
from agentic_rag.shared.db.models import Tenant, User
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.shared.schemas.auth import TenantUserRole


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Provision an Agentic RAG tenant, Auth0 organization, and first "
            "administrator invitation."
        ),
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--tenant-name", required=True)
    parser.add_argument("--tenant-slug")
    parser.add_argument("--admin-email", required=True)
    parser.add_argument("--admin-name")
    parser.add_argument("--workspace-id")
    parser.add_argument(
        "--auth0-organization-id",
        help="Attach an existing Auth0 organization instead of creating one.",
    )
    return parser.parse_args()


async def bootstrap_tenant_admin(arguments: argparse.Namespace) -> None:
    tenant_id = arguments.tenant_id.strip()
    tenant_name = arguments.tenant_name.strip()
    admin_email = arguments.admin_email.strip().lower()
    tenant_slug = (arguments.tenant_slug or tenant_name).strip().lower()
    tenant_slug = re.sub(r"[^a-z0-9_-]+", "-", tenant_slug).strip("-_")

    if not tenant_id or not tenant_name or not tenant_slug or not admin_email:
        raise ValueError(
            "Tenant ID, tenant name, tenant slug, and admin email are required."
        )
    if len(tenant_slug) > 50:
        raise ValueError("Tenant slug must be 50 characters or fewer for Auth0.")

    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        tenant = db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
        tenant_created = False
        organization_created = False
        mapping_created = False
        organization_id = (
            arguments.auth0_organization_id.strip()
            if arguments.auth0_organization_id
            else None
        )
        invitation_id: str | None = None
        tenant_user: User | None = None

        if tenant is None:
            try:
                tenant = Tenant(
                    tenant_id=tenant_id,
                    name=tenant_name,
                    slug=tenant_slug,
                    status="active",
                    metadata_={"provisioned_by": "bootstrap_tenant_admin"},
                )
                db.add(tenant)
                db.commit()
                tenant_created = True
                logger.info(f"[DB] Bootstrapped tenant tenant={tenant_id}")
            except IntegrityError as error:
                db.rollback()
                raise ValueError(
                    "The tenant ID or slug is already used by another tenant."
                ) from error
        elif tenant.status != "active":
            raise ValueError("The existing tenant is not active.")

        existing_membership = get_tenant_user_by_email(
            db,
            tenant_id=tenant_id,
            email=admin_email,
        )
        if existing_membership is not None:
            raise ValueError("This administrator is already a member of the tenant.")

        try:
            if tenant.external_organization_id:
                if tenant.identity_provider != "auth0":
                    raise ValueError(
                        "The existing tenant is connected to another identity provider."
                    )
                if organization_id and organization_id != tenant.external_organization_id:
                    raise ValueError(
                        "The supplied Auth0 organization does not match the tenant mapping."
                    )
                organization_id = tenant.external_organization_id
            elif not organization_id:
                organization = await create_auth0_organization(
                    tenant_id=tenant_id,
                    name=tenant_slug,
                    display_name=tenant_name,
                )
                organization_id = organization.id
                organization_created = True

            if not organization_id:
                raise RuntimeError("Auth0 organization provisioning did not complete.")

            if not tenant.external_organization_id:
                tenant.identity_provider = "auth0"
                tenant.external_organization_id = organization_id
                db.commit()
                mapping_created = True
                logger.info(
                    f"[DB] Connected tenant to Auth0 tenant={tenant_id} "
                    f"organization={organization_id}"
                )

            auth0_invitation = await create_auth0_organization_invitation(
                organization_id=organization_id,
                email=admin_email,
                display_name=arguments.admin_name,
                tenant_id=tenant_id,
                workspace_id=arguments.workspace_id,
                role_name=TenantUserRole.ADMIN.value,
            )
            invitation_id = auth0_invitation.id
            tenant_user = create_invited_tenant_user(
                db,
                tenant_id=tenant_id,
                external_subject=f"auth0-invitation:{invitation_id}",
                email=admin_email,
                display_name=arguments.admin_name,
                role_name=TenantUserRole.ADMIN.value,
                workspace_id=arguments.workspace_id,
                invited_by="tenant-bootstrap",
                identity_invitation_id=invitation_id,
            )

        except Exception:
            if tenant_user is not None:
                try:
                    delete_incomplete_tenant_user(db, tenant_user)
                except Exception as cleanup_error:
                    logger.exception(
                        f"[Bootstrap] Membership cleanup failed user={tenant_user.id}: "
                        f"{cleanup_error}"
                    )
            if invitation_id and organization_id:
                try:
                    await delete_auth0_organization_invitation(
                        organization_id,
                        invitation_id,
                    )
                except Auth0ManagementError as cleanup_error:
                    logger.exception(
                        f"[Bootstrap] Invitation cleanup failed "
                        f"invitation={invitation_id}: {cleanup_error}"
                    )
            if organization_created and organization_id:
                try:
                    await delete_auth0_organization(organization_id)
                except Auth0ManagementError as cleanup_error:
                    logger.exception(
                        f"[Bootstrap] Organization cleanup failed "
                        f"organization={organization_id}: {cleanup_error}"
                    )
            if tenant_created:
                tenant_to_delete = (
                    db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
                )
                if tenant_to_delete is not None:
                    db.delete(tenant_to_delete)
                    db.commit()
            elif mapping_created:
                tenant.identity_provider = None
                tenant.external_organization_id = None
                db.commit()
            raise

        if tenant_user is None or invitation_id is None:
            raise RuntimeError("Administrator invitation did not complete.")

        logger.info(
            f"[Bootstrap] Tenant administrator invited user={tenant_user.id} "
            f"tenant={tenant_id} organization={organization_id}"
        )
        print(
            f"Auth0 invitation sent to {admin_email} for tenant {tenant_id}. "
            "The administrator can accept it with the configured database, "
            "Google, or GitHub connection."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(bootstrap_tenant_admin(parse_arguments()))
