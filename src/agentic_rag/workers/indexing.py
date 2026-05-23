import logging
import os
import time
from typing import Optional

from sqlalchemy.orm import Session

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.indexing import (
    list_chunks_pending_bm25_index,
    mark_chunk_bm25_failed,
    mark_chunk_bm25_indexed,
)
from agentic_rag.shared.db.session import get_sync_session_factory
from agentic_rag.search.opensearch import OpenSearchClient


logger = logging.getLogger(__name__)


def process_bm25_index_batch(
    db: Session,
    search_client: Optional[OpenSearchClient] = None,
    limit: Optional[int] = None,
) -> int:
    logger.info("[IndexingWorker] Processing BM25 index batch")
    search_client = search_client or OpenSearchClient()
    index_name = search_client.index_name

    chunks = list_chunks_pending_bm25_index(
        db=db,
        limit=limit or settings.bm25_index_batch_size,
        index_name=index_name,
    )
    if not chunks:
        logger.info("[IndexingWorker] No chunks pending BM25 index")
        return 0

    try:
        search_client.ensure_chunk_index(index_name)
        indexed_count = search_client.bulk_index_chunks(chunks, index_name=index_name)
        for chunk in chunks:
            mark_chunk_bm25_indexed(
                db=db,
                chunk=chunk,
                index_name=index_name,
            )

        logger.info(f"[IndexingWorker] Indexed {indexed_count} chunks into BM25")
        return indexed_count

    except Exception as e:
        logger.exception(f"[IndexingWorker] BM25 indexing batch failed: {e}")
        for chunk in chunks:
            mark_chunk_bm25_failed(
                db=db,
                chunk=chunk,
                error_message=str(e),
            )
        return 0


def run_indexing_worker_once(
    search_client: Optional[OpenSearchClient] = None,
) -> bool:
    SessionLocal = get_sync_session_factory()
    with SessionLocal() as db:
        indexed_count = process_bm25_index_batch(
            db=db,
            search_client=search_client,
        )
        return indexed_count > 0


def run_indexing_worker_loop() -> None:
    logger.info("[IndexingWorker] Worker loop started")

    while True:
        try:
            processed = run_indexing_worker_once()
            if not processed:
                time.sleep(settings.indexing_worker_poll_seconds)
        except KeyboardInterrupt:
            logger.info("[IndexingWorker] Worker loop stopped")
            break
        except Exception as e:
            logger.exception(f"[IndexingWorker] Worker loop error: {e}")
            time.sleep(settings.indexing_worker_poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )
    run_indexing_worker_loop()
