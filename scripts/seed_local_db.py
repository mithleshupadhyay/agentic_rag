import logging

from sqlalchemy.exc import IntegrityError

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.models import Tenant, User
from agentic_rag.shared.db.session import get_sync_session_factory


logger = logging.getLogger(__name__)


def seed_local_development_data() -> None:
    if settings.auth_provider != "local":
        logger.info("[DB] Local seed skipped for auth_provider=%s", settings.auth_provider)
        return

    SessionLocal = get_sync_session_factory()
    with SessionLocal() as session:
        try:
            tenant = (
                session.query(Tenant)
                .filter(Tenant.tenant_id == settings.local_tenant_id)
                .first()
            )
            if not tenant:
                tenant = Tenant(
                    tenant_id=settings.local_tenant_id,
                    name=settings.local_tenant_id,
                    slug=settings.local_tenant_id,
                    status="active",
                    metadata_={"source": "local-seed"},
                )
                session.add(tenant)
                logger.info("[DB] Local tenant seeded: %s", settings.local_tenant_id)

            user = (
                session.query(User)
                .filter(
                    User.tenant_id == settings.local_tenant_id,
                    User.external_subject == settings.local_user_id,
                )
                .first()
            )
            if not user:
                user = User(
                    tenant_id=settings.local_tenant_id,
                    external_subject=settings.local_user_id,
                    display_name=settings.local_user_id,
                    status="active",
                    acl_version=settings.local_acl_version,
                    metadata_={"source": "local-seed"},
                )
                session.add(user)
                logger.info("[DB] Local user seeded: %s", settings.local_user_id)

            session.commit()
            logger.info("[DB] Local development seed completed")

        except IntegrityError as e:
            session.rollback()
            logger.exception("[DB] Local development seed failed: %s", e)
            raise


if __name__ == "__main__":
    seed_local_development_data()
