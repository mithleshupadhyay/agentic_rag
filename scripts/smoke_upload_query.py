import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

import httpx

from agentic_rag.shared.config import settings


logger = logging.getLogger(__name__)


def _response_preview(response: httpx.Response) -> str:
    text = response.text.strip()
    if not text:
        return ""
    return text[:1000]


def _log_http_failure(action: str, response: httpx.Response) -> None:
    logger.error(
        f"[UploadQuerySmoke] {action} failed status={response.status_code} "
        f"body={_response_preview(response)}"
    )


def _wait_for_api(
    client: httpx.Client,
    timeout_seconds: float,
    poll_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""

    while time.monotonic() < deadline:
        try:
            response = client.get("/readiness")
            if response.status_code == 200:
                payload = response.json()
                if payload.get("status") == "healthy":
                    logger.info("[UploadQuerySmoke] API readiness check passed")
                    return True
                last_error = f"unexpected readiness payload={payload}"
            else:
                last_error = (
                    f"status={response.status_code} "
                    f"body={_response_preview(response)}"
                )
        except (httpx.HTTPError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"

        time.sleep(poll_seconds)

    logger.error(
        f"[UploadQuerySmoke] API readiness check timed out "
        f"timeout_seconds={timeout_seconds} last_error={last_error}"
    )
    return False


def _upload_document(
    client: httpx.Client,
    workspace_id: str,
    smoke_id: str,
) -> dict[str, Any] | None:
    marker = f"upload query smoke marker {smoke_id}"
    content = (
        f"{marker}\n\n"
        "The uploaded smoke document says tenant-safe retrieval should return "
        "this exact document after ingestion and BM25 indexing.\n"
        "This content is unique to the upload-to-query smoke check.\n"
    ).encode("utf-8")

    files = {
        "file": (
            f"upload-query-smoke-{smoke_id}.txt",
            content,
            "text/plain",
        )
    }
    data = {
        "workspace_id": workspace_id,
        "title": f"Upload Query Smoke {smoke_id}",
        "metadata_json": json.dumps(
            {
                "smoke": True,
                "smoke_id": smoke_id,
                "smoke_type": "upload_query",
            },
            sort_keys=True,
        ),
        "idempotency_key": f"upload-query-smoke-{smoke_id}",
    }

    response = client.post("/documents/upload", files=files, data=data)
    if response.status_code != 201:
        _log_http_failure("Document upload", response)
        return None

    payload = response.json()
    document = payload.get("document") or {}
    logger.info(
        f"[UploadQuerySmoke] Uploaded document document={document.get('id')} "
        f"ingestion_job={payload.get('ingestion_job_id')} workspace={workspace_id}"
    )
    return payload


def _wait_for_ingestion_job(
    client: httpx.Client,
    document_id: str,
    job_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""

    while time.monotonic() < deadline:
        try:
            response = client.get(f"/documents/{document_id}/ingestion-jobs/{job_id}")
            if response.status_code != 200:
                last_error = (
                    f"status={response.status_code} "
                    f"body={_response_preview(response)}"
                )
                time.sleep(poll_seconds)
                continue

            payload = response.json()
            actual_job_id = str(payload.get("id"))
            actual_document_id = str(payload.get("document_id"))
            status = payload.get("status")
            current_stage = payload.get("current_stage")
            if actual_job_id != job_id or actual_document_id != document_id:
                logger.error(
                    f"[UploadQuerySmoke] Ingestion job response mismatch "
                    f"expected_job={job_id} actual_job={actual_job_id} "
                    f"expected_document={document_id} actual_document={actual_document_id}"
                )
                return None

            if status == "completed":
                logger.info(
                    f"[UploadQuerySmoke] Ingestion job completed "
                    f"job={job_id} document={document_id} stage={current_stage}"
                )
                return payload

            if status in {"failed", "cancelled"}:
                logger.error(
                    f"[UploadQuerySmoke] Ingestion job reached terminal failure "
                    f"job={job_id} document={document_id} status={status} "
                    f"stage={current_stage} error_type={payload.get('error_type')} "
                    f"error_message={payload.get('error_message')}"
                )
                return None

            last_error = (
                f"status={status} stage={current_stage} "
                f"retry_count={payload.get('retry_count')} "
                f"next_retry_at={payload.get('next_retry_at')}"
            )

        except (httpx.HTTPError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"

        time.sleep(poll_seconds)

    logger.error(
        f"[UploadQuerySmoke] Ingestion job did not complete "
        f"job={job_id} document={document_id} timeout_seconds={timeout_seconds} "
        f"last_error={last_error}"
    )
    return None


def _wait_for_query_result(
    client: httpx.Client,
    workspace_id: str,
    document_id: str,
    smoke_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    query_text = f"upload query smoke marker {smoke_id}"
    last_error = ""

    request_payload = {
        "query": query_text,
        "workspace_id": workspace_id,
        "filters": {
            "workspace_id": workspace_id,
            "document_ids": [document_id],
        },
        "retrieval_limit": 5,
        "max_context_chunks": 3,
        "max_context_tokens": 500,
    }

    while time.monotonic() < deadline:
        try:
            response = client.post("/query", json=request_payload)
            if response.status_code != 200:
                last_error = (
                    f"status={response.status_code} "
                    f"body={_response_preview(response)}"
                )
                time.sleep(poll_seconds)
                continue

            payload = response.json()
            context = payload.get("context") or []
            citations = payload.get("citations") or []
            candidate_document_ids = {
                str(candidate.get("document_id"))
                for candidate in payload.get("candidates") or []
                if isinstance(candidate, dict)
            }
            context_document_ids = {
                str(item.get("document_id"))
                for item in context
                if isinstance(item, dict)
            }
            citation_document_ids = {
                str(item.get("document_id"))
                for item in citations
                if isinstance(item, dict)
            }
            if (
                document_id in candidate_document_ids
                and document_id in context_document_ids
                and document_id in citation_document_ids
            ):
                logger.info(
                    f"[UploadQuerySmoke] Query found uploaded document "
                    f"document={document_id} context_chunks={len(context)}"
                )
                return payload

            last_error = (
                f"candidates={sorted(candidate_document_ids)} "
                f"context={sorted(context_document_ids)} "
                f"citations={sorted(citation_document_ids)}"
            )

        except (httpx.HTTPError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"

        time.sleep(poll_seconds)

    logger.error(
        f"[UploadQuerySmoke] Query did not return uploaded document "
        f"document={document_id} timeout_seconds={timeout_seconds} "
        f"last_error={last_error}"
    )
    return None


def main() -> int:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )

    base_url = os.getenv("SMOKE_API_BASE_URL", "http://localhost:8000").rstrip("/")
    timeout_seconds = float(os.getenv("SMOKE_UPLOAD_QUERY_TIMEOUT_SECONDS", "120"))
    poll_seconds = float(os.getenv("SMOKE_UPLOAD_QUERY_POLL_SECONDS", "2"))
    smoke_id = str(uuid4())
    workspace_id = settings.local_workspace_id or f"upload-query-smoke-{smoke_id[:8]}"

    headers = {
        "Authorization": f"Bearer {settings.local_auth_token}",
        "X-Request-ID": f"upload-query-smoke-{smoke_id}",
    }

    logger.info(
        f"[UploadQuerySmoke] Starting upload-to-query smoke "
        f"base_url={base_url} tenant={settings.local_tenant_id} "
        f"workspace={workspace_id} smoke_id={smoke_id}"
    )

    with httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=15,
    ) as client:
        if not _wait_for_api(client, timeout_seconds=30, poll_seconds=poll_seconds):
            return 1

        upload_payload = _upload_document(
            client=client,
            workspace_id=workspace_id,
            smoke_id=smoke_id,
        )
        if upload_payload is None:
            return 1

        document = upload_payload.get("document") or {}
        document_id = document.get("id")
        if not document_id:
            logger.error("[UploadQuerySmoke] Upload response did not include document.id")
            return 1
        ingestion_job_id = upload_payload.get("ingestion_job_id")
        if not ingestion_job_id:
            logger.error(
                "[UploadQuerySmoke] Upload response did not include ingestion_job_id"
            )
            return 1

        ingestion_payload = _wait_for_ingestion_job(
            client=client,
            document_id=str(document_id),
            job_id=str(ingestion_job_id),
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        if ingestion_payload is None:
            return 1

        query_payload = _wait_for_query_result(
            client=client,
            workspace_id=workspace_id,
            document_id=str(document_id),
            smoke_id=smoke_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        if query_payload is None:
            return 1

        if query_payload.get("retrieval_strategy") != "bm25":
            logger.error(
                f"[UploadQuerySmoke] Unexpected retrieval strategy "
                f"strategy={query_payload.get('retrieval_strategy')}"
            )
            return 1

        if query_payload.get("synthesis_enabled") is True:
            logger.error("[UploadQuerySmoke] Smoke expected deterministic retrieval-only query")
            return 1

    print("upload-query smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
