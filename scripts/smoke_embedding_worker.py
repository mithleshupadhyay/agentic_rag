import logging
import os

from sqlalchemy import text

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.models import Tenant
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.workers.embedding import process_embedding_batches


logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )

    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
        active_tenant_count = db.query(Tenant).filter(Tenant.status == "active").count()

    logger.info(
        f"[EmbeddingSmoke] Embedding worker import and DB smoke ok "
        f"model={settings.embedding_model_name} "
        f"vector_version={settings.embedding_vector_version} "
        f"active_tenants={active_tenant_count}"
    )

    _ = process_embedding_batches
    print("embedding-worker smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
