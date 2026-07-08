import json
import logging
import mimetypes
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr
import httpx


logger = logging.getLogger(__name__)


DEFAULT_API_URL = os.getenv("AGENTIC_RAG_API_URL", "http://localhost:8100")
DEFAULT_AUTH_TOKEN = os.getenv("AGENTIC_RAG_AUTH_TOKEN", "local-dev-token")
DEFAULT_WORKSPACE_ID = os.getenv("AGENTIC_RAG_WORKSPACE_ID", "local-workspace")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("AGENTIC_RAG_DEMO_REQUEST_TIMEOUT", "30"))
DEFAULT_INGESTION_TIMEOUT_SECONDS = int(
    os.getenv("AGENTIC_RAG_DEMO_INGESTION_TIMEOUT", "120")
)
DEFAULT_QUERY_TIMEOUT_SECONDS = int(os.getenv("AGENTIC_RAG_DEMO_QUERY_TIMEOUT", "90"))

TERMINAL_INGESTION_STATUSES = {"completed", "failed", "cancelled"}


class DemoAPIError(Exception):
    pass


def _request_json(
    method: str,
    api_url: str,
    path: str,
    auth_token: str,
    **kwargs: Any,
) -> dict[str, Any]:
    base_url = (api_url or "").strip().rstrip("/")
    if not base_url:
        raise DemoAPIError("API URL is required.")

    url = f"{base_url}{path}"
    headers = {
        "Accept": "application/json",
        "X-Request-ID": f"agentic-rag-demo-{uuid4()}",
    }
    token = (auth_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                **kwargs,
            )
    except httpx.HTTPError as e:
        logger.warning(f"[Demo] API request failed method={method} path={path}: {e}")
        raise DemoAPIError(f"API request failed: {e}") from e

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except ValueError:
            detail = response.text[:1000]
        else:
            if isinstance(error_payload, dict):
                detail_value = error_payload.get("detail")
                if isinstance(detail_value, str):
                    detail = detail_value
                elif detail_value is not None:
                    detail = json.dumps(detail_value, default=str)[:1000]
                else:
                    detail = json.dumps(error_payload, default=str)[:1000]
            else:
                detail = str(error_payload)[:1000]

        logger.warning(
            f"[Demo] API returned error method={method} path={path} "
            f"status={response.status_code}"
        )
        raise DemoAPIError(f"API returned {response.status_code}: {detail}")

    try:
        payload = response.json()
    except ValueError as e:
        raise DemoAPIError("API returned a non-JSON response.") from e

    if not isinstance(payload, dict):
        raise DemoAPIError("API returned an unexpected JSON response.")
    return payload


def _citation_rows(payload: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for citation in payload.get("citations") or []:
        quote_text = "" if citation.get("quote") is None else str(citation.get("quote"))
        if len(quote_text) > 320:
            quote_text = f"{quote_text[:320].rstrip()}..."

        rows.append(
            [
                citation.get("document_id"),
                citation.get("chunk_id"),
                citation.get("title"),
                citation.get("score"),
                quote_text,
            ]
        )
    return rows


def _context_rows(payload: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for context_chunk in payload.get("context") or []:
        citation = context_chunk.get("citation") or {}
        content_text = (
            ""
            if context_chunk.get("content") is None
            else str(context_chunk.get("content"))
        )
        if len(content_text) > 900:
            content_text = f"{content_text[:900].rstrip()}..."

        rows.append(
            [
                context_chunk.get("document_id"),
                context_chunk.get("chunk_id"),
                context_chunk.get("token_count"),
                citation.get("section_path"),
                content_text,
            ]
        )
    return rows


def _query_once(
    api_url: str,
    auth_token: str,
    workspace_id: str,
    question: str,
    retrieval_limit: int,
    max_context_chunks: int,
    max_context_tokens: int,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workspace = (workspace_id or "").strip() or None
    document = (document_id or "").strip()
    filters: dict[str, Any] = {}
    if workspace:
        filters["workspace_id"] = workspace
    if document:
        filters["document_ids"] = [document]
    elif document_ids:
        clean_document_ids = []
        for candidate_document_id in document_ids:
            clean_document_id = str(candidate_document_id or "").strip()
            if clean_document_id:
                clean_document_ids.append(clean_document_id)
        if clean_document_ids:
            filters["document_ids"] = clean_document_ids

    payload: dict[str, Any] = {
        "query": question.strip(),
        "retrieval_limit": int(retrieval_limit),
        "max_context_chunks": int(max_context_chunks),
        "max_context_tokens": int(max_context_tokens),
        "filters": filters,
    }
    if workspace:
        payload["workspace_id"] = workspace
    if history:
        payload["history"] = history[-10:]

    return _request_json("POST", api_url, "/query", auth_token, json=payload)


def check_api_health(api_url: str, auth_token: str) -> tuple[str, dict[str, Any]]:
    try:
        payload = _request_json("GET", api_url, "/readiness", auth_token)
    except DemoAPIError as e:
        return f"API check failed: {e}", {}

    status = payload.get("status", "unknown")
    service = payload.get("service", "Agentic RAG")
    version = payload.get("version", "unknown")
    return f"{service} is {status} on version {version}.", payload


def upload_document(
    api_url: str,
    auth_token: str,
    workspace_id: str,
    title: str,
    metadata_json: str,
    uploaded_file: str | None,
) -> tuple[str, str, str, str, dict[str, Any]]:
    try:
        if not uploaded_file:
            raise DemoAPIError("Select a document before uploading.")

        file_path = Path(str(uploaded_file))
        if not file_path.exists() or not file_path.is_file():
            raise DemoAPIError("Selected document is not readable.")

        if metadata_json and metadata_json.strip():
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as e:
                raise DemoAPIError(f"Metadata must be valid JSON: {e}") from e
            if not isinstance(metadata, dict):
                raise DemoAPIError("Metadata must be a JSON object.")
        else:
            metadata = {}

        metadata.setdefault("uploaded_from", "agentic_rag_gradio_demo")
        metadata.setdefault("uploaded_at", datetime.now(UTC).isoformat())

        workspace = (workspace_id or "").strip()
        document_title = (title or "").strip() or file_path.name
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        with file_path.open("rb") as file_obj:
            payload = _request_json(
                "POST",
                api_url,
                "/documents/upload",
                auth_token,
                files={"file": (file_path.name, file_obj, mime_type)},
                data={
                    "workspace_id": workspace,
                    "title": document_title,
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                    "idempotency_key": f"agentic-rag-demo-upload-{uuid4()}",
                },
            )
    except DemoAPIError as e:
        return f"Upload failed: {e}", "", "", "", {}

    document = payload.get("document") or {}
    document_id = str(document.get("id") or "")
    job_id = str(payload.get("ingestion_job_id") or "")
    status = str(payload.get("ingestion_status") or document.get("status") or "")
    stage = str(payload.get("ingestion_stage") or "")
    message = (
        f"Uploaded document {document_id}. "
        f"Ingestion job {job_id} is {status} at stage {stage}."
    )
    return message, document_id, job_id, status, payload


def wait_for_ingestion(
    api_url: str,
    auth_token: str,
    document_id: str,
    job_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> tuple[str, dict[str, Any]]:
    document = (document_id or "").strip()
    job = (job_id or "").strip()
    if not document or not job:
        return "Document ID and ingestion job ID are required.", {}

    deadline = time.monotonic() + max(1, int(timeout_seconds))
    interval = max(0.5, float(poll_interval_seconds))
    last_payload: dict[str, Any] = {}
    last_error = ""

    while time.monotonic() < deadline:
        try:
            last_payload = _request_json(
                "GET",
                api_url,
                f"/documents/{document}/ingestion-jobs/{job}",
                auth_token,
            )
            status = str(last_payload.get("status") or "")
            stage = str(last_payload.get("current_stage") or "")
            if status in TERMINAL_INGESTION_STATUSES:
                return (
                    f"Ingestion job {job} finished with status {status} "
                    f"at stage {stage}.",
                    last_payload,
                )
        except DemoAPIError as e:
            last_error = str(e)

        time.sleep(interval)

    if last_payload:
        status = str(last_payload.get("status") or "unknown")
        stage = str(last_payload.get("current_stage") or "unknown")
        return (
            f"Ingestion did not finish before timeout. "
            f"Last status was {status} at stage {stage}.",
            last_payload,
        )
    return f"Ingestion status check timed out. Last error: {last_error}", {}


def ask_question(
    api_url: str,
    auth_token: str,
    workspace_id: str,
    question: str,
    retrieval_limit: int,
    max_context_chunks: int,
    max_context_tokens: int,
) -> tuple[str, list[list[Any]], list[list[Any]], dict[str, Any]]:
    try:
        if not question or not question.strip():
            raise DemoAPIError("Question is required.")
        payload = _query_once(
            api_url=api_url,
            auth_token=auth_token,
            workspace_id=workspace_id,
            question=question,
            retrieval_limit=retrieval_limit,
            max_context_chunks=max_context_chunks,
            max_context_tokens=max_context_tokens,
        )
    except DemoAPIError as e:
        return f"Query failed: {e}", [], [], {}

    answer = payload.get("answer") or ""
    confidence = payload.get("confidence_score")
    latency = payload.get("latency_ms")
    status = f"Confidence: {confidence} | Latency: {latency} ms"
    return f"{answer}\n\n{status}", _citation_rows(payload), _context_rows(payload), payload


def upload_and_index_for_chat(
    api_url: str,
    auth_token: str,
    workspace_id: str,
    title: str,
    metadata_json: str,
    uploaded_file: str | None,
    ingestion_timeout_seconds: int,
    documents: list[dict[str, Any]] | None,
    chat_history: list[dict[str, Any]] | None,
) -> tuple[
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    existing_documents = list(documents or [])
    messages = list(chat_history or [])

    upload_message, document_id, job_id, _, upload_payload = upload_document(
        api_url=api_url,
        auth_token=auth_token,
        workspace_id=workspace_id,
        title=title,
        metadata_json=metadata_json,
        uploaded_file=uploaded_file,
    )
    if not document_id or not job_id:
        messages.append(
            {
                "role": "assistant",
                "content": upload_message,
            }
        )
        return messages, upload_message, existing_documents, existing_documents, upload_payload

    ingestion_message, ingestion_payload = wait_for_ingestion(
        api_url=api_url,
        auth_token=auth_token,
        document_id=document_id,
        job_id=job_id,
        timeout_seconds=ingestion_timeout_seconds,
        poll_interval_seconds=2,
    )
    raw_payload = {
        "upload": upload_payload,
        "ingestion": ingestion_payload,
    }
    if ingestion_payload.get("status") != "completed":
        error_message = ingestion_payload.get("error_message")
        status_message = f"{upload_message}\n\n{ingestion_message}"
        if error_message:
            status_message = f"{status_message}\n\n{error_message}"
        messages.append(
            {
                "role": "assistant",
                "content": status_message,
            }
        )
        return messages, status_message, existing_documents, existing_documents, raw_payload

    document = upload_payload.get("document") or {}
    document_title = str(document.get("title") or title or document.get("file_name") or document_id)
    indexed_document = {
        "document_id": document_id,
        "title": document_title,
        "job_id": job_id,
        "status": "ready",
    }
    updated_documents = []
    for existing_document in existing_documents:
        if existing_document.get("document_id") != document_id:
            updated_documents.append(existing_document)
    updated_documents.append(indexed_document)

    status_message = (
        f"Indexed {document_title}. "
        "You can ask questions about this document now."
    )
    messages.append(
        {
            "role": "assistant",
            "content": status_message,
        }
    )
    return messages, status_message, updated_documents, updated_documents, raw_payload


def chat_with_documents(
    api_url: str,
    auth_token: str,
    workspace_id: str,
    question: str,
    chat_history: list[dict[str, Any]] | None,
    documents: list[dict[str, Any]] | None,
    retrieval_limit: int,
    max_context_chunks: int,
    max_context_tokens: int,
) -> tuple[
    list[dict[str, Any]],
    str,
    list[list[Any]],
    list[list[Any]],
    dict[str, Any],
]:
    messages = list(chat_history or [])
    clean_question = (question or "").strip()
    if not clean_question:
        return messages, "", [], [], {}

    messages.append(
        {
            "role": "user",
            "content": clean_question,
        }
    )

    indexed_document_ids = []
    for document in documents or []:
        if document.get("status") != "ready":
            continue
        document_id = str(document.get("document_id") or "").strip()
        if document_id:
            indexed_document_ids.append(document_id)

    if not indexed_document_ids:
        messages.append(
            {
                "role": "assistant",
                "content": "Upload and index a document first.",
            }
        )
        return messages, "", [], [], {}

    history_payload = []
    for message in messages[:-1]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            history_payload.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    try:
        payload = _query_once(
            api_url=api_url,
            auth_token=auth_token,
            workspace_id=workspace_id,
            question=clean_question,
            retrieval_limit=retrieval_limit,
            max_context_chunks=max_context_chunks,
            max_context_tokens=max_context_tokens,
            document_ids=indexed_document_ids,
            history=history_payload,
        )
    except DemoAPIError as e:
        messages.append(
            {
                "role": "assistant",
                "content": f"Query failed: {e}",
            }
        )
        return messages, "", [], [], {}

    answer = str(payload.get("answer") or "").strip()
    if payload.get("context") and not payload.get("synthesis_enabled"):
        context_sections = []
        for context_chunk in payload.get("context")[:3]:
            content_text = str(context_chunk.get("content") or "").strip()
            if not content_text:
                continue
            citation = context_chunk.get("citation") or {}
            title = str(citation.get("title") or "Uploaded document")
            if len(content_text) > 900:
                content_text = f"{content_text[:900].rstrip()}..."
            context_sections.append(f"**{title}**\n\n{content_text}")
        if context_sections:
            answer = "I found this in the uploaded document:\n\n" + "\n\n".join(
                context_sections
            )

    if not answer:
        answer = "I could not find relevant context in the indexed documents."

    confidence = payload.get("confidence_score")
    latency = payload.get("latency_ms")
    footer = f"\n\nConfidence: {confidence} | Latency: {latency} ms"
    messages.append(
        {
            "role": "assistant",
            "content": f"{answer}{footer}",
        }
    )
    return messages, "", _citation_rows(payload), _context_rows(payload), payload


def clear_chat_session() -> tuple[
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[list[Any]],
    list[list[Any]],
    dict[str, Any],
]:
    messages: list[dict[str, Any]] = []
    document_status = ""
    indexed_documents: list[dict[str, Any]] = []
    citation_rows: list[list[Any]] = []
    context_rows: list[list[Any]] = []
    raw_payload: dict[str, Any] = {}
    return (
        messages,
        document_status,
        indexed_documents,
        indexed_documents,
        citation_rows,
        context_rows,
        raw_payload,
    )


def run_end_to_end(
    api_url: str,
    auth_token: str,
    workspace_id: str,
    title: str,
    metadata_json: str,
    uploaded_file: str | None,
    question: str,
    retrieval_limit: int,
    max_context_chunks: int,
    max_context_tokens: int,
    ingestion_timeout_seconds: int,
    query_timeout_seconds: int,
) -> tuple[
    str,
    str,
    str,
    str,
    list[list[Any]],
    list[list[Any]],
    dict[str, Any],
]:
    upload_message, document_id, job_id, _, upload_payload = upload_document(
        api_url=api_url,
        auth_token=auth_token,
        workspace_id=workspace_id,
        title=title,
        metadata_json=metadata_json,
        uploaded_file=uploaded_file,
    )
    if not document_id or not job_id:
        return upload_message, "", "", "", [], [], upload_payload

    ingestion_message, ingestion_payload = wait_for_ingestion(
        api_url=api_url,
        auth_token=auth_token,
        document_id=document_id,
        job_id=job_id,
        timeout_seconds=ingestion_timeout_seconds,
        poll_interval_seconds=2,
    )
    if ingestion_payload.get("status") != "completed":
        return (
            f"{upload_message}\n\n{ingestion_message}",
            document_id,
            job_id,
            "",
            [],
            [],
            {"upload": upload_payload, "ingestion": ingestion_payload},
        )

    if not question or not question.strip():
        return (
            f"{upload_message}\n\n{ingestion_message}\n\nQuestion is required.",
            document_id,
            job_id,
            "",
            [],
            [],
            {"upload": upload_payload, "ingestion": ingestion_payload},
        )

    deadline = time.monotonic() + max(1, int(query_timeout_seconds))
    last_query_payload: dict[str, Any] = {}
    query_error = ""
    while time.monotonic() < deadline:
        try:
            last_query_payload = _query_once(
                api_url=api_url,
                auth_token=auth_token,
                workspace_id=workspace_id,
                question=question,
                retrieval_limit=retrieval_limit,
                max_context_chunks=max_context_chunks,
                max_context_tokens=max_context_tokens,
                document_id=document_id,
            )
            uploaded_document_returned = False
            for result_section in (
                last_query_payload.get("citations") or [],
                last_query_payload.get("context") or [],
                last_query_payload.get("candidates") or [],
            ):
                for result_item in result_section:
                    if str(result_item.get("document_id")) == document_id:
                        uploaded_document_returned = True
                        break
                if uploaded_document_returned:
                    break

            if uploaded_document_returned:
                answer = last_query_payload.get("answer") or ""
                status = (
                    f"{upload_message}\n\n{ingestion_message}\n\n"
                    f"Query returned the uploaded document."
                )
                return (
                    status,
                    document_id,
                    job_id,
                    answer,
                    _citation_rows(last_query_payload),
                    _context_rows(last_query_payload),
                    {
                        "upload": upload_payload,
                        "ingestion": ingestion_payload,
                        "query": last_query_payload,
                    },
                )
        except DemoAPIError as e:
            query_error = str(e)

        time.sleep(2)

    if last_query_payload:
        answer = last_query_payload.get("answer") or ""
        status = (
            f"{upload_message}\n\n{ingestion_message}\n\n"
            "Query completed, but the uploaded document was not returned "
            "before the query timeout."
        )
        return (
            status,
            document_id,
            job_id,
            answer,
            _citation_rows(last_query_payload),
            _context_rows(last_query_payload),
            {
                "upload": upload_payload,
                "ingestion": ingestion_payload,
                "query": last_query_payload,
            },
        )

    return (
        f"{upload_message}\n\n{ingestion_message}\n\nQuery failed: {query_error}",
        document_id,
        job_id,
        "",
        [],
        [],
        {"upload": upload_payload, "ingestion": ingestion_payload},
    )


def build_demo() -> gr.Blocks:
    citation_headers = ["Document ID", "Chunk ID", "Title", "Score", "Quote"]
    context_headers = ["Document ID", "Chunk ID", "Tokens", "Section", "Content"]
    default_metadata = json.dumps(
        {
            "demo": "true",
            "source": "gradio",
        },
        indent=2,
        default=str,
        sort_keys=True,
    )

    with gr.Blocks(
        title="Agentic RAG Demo",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as demo:
        gr.Markdown("# Agentic RAG Demo")
        chat_documents_state = gr.State([])

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    chat_file = gr.File(
                        label="Document",
                        file_count="single",
                        type="filepath",
                        file_types=[
                            ".pdf",
                            ".txt",
                            ".md",
                            ".markdown",
                            ".json",
                            ".jsonl",
                            ".csv",
                            ".tsv",
                        ],
                    )
                    chat_title = gr.Textbox(label="Title", value="")
                    chat_metadata = gr.Code(
                        label="Metadata JSON",
                        value=default_metadata,
                        language="json",
                    )
                    chat_upload_button = gr.Button("Upload & Index", variant="primary")
                    chat_upload_status = gr.Textbox(
                        label="Document status",
                        lines=5,
                        interactive=False,
                    )
                    chat_documents = gr.JSON(label="Indexed documents")

                    with gr.Accordion("Connection", open=False):
                        api_url = gr.Textbox(
                            label="API URL",
                            value=DEFAULT_API_URL,
                        )
                        auth_token = gr.Textbox(
                            label="Bearer token",
                            value=DEFAULT_AUTH_TOKEN,
                            type="password",
                        )
                        workspace_id = gr.Textbox(
                            label="Workspace",
                            value=DEFAULT_WORKSPACE_ID,
                        )
                        health_button = gr.Button("Check API", variant="secondary")
                        health_status = gr.Textbox(
                            label="API status",
                            interactive=False,
                        )

                    with gr.Accordion("Retrieval Settings", open=False):
                        chat_retrieval_limit = gr.Slider(
                            label="Retrieval limit",
                            minimum=1,
                            maximum=50,
                            value=5,
                            step=1,
                        )
                        chat_max_context_chunks = gr.Slider(
                            label="Max context chunks",
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1,
                        )
                        chat_max_context_tokens = gr.Slider(
                            label="Max context tokens",
                            minimum=500,
                            maximum=12000,
                            value=1000,
                            step=100,
                        )
                        chat_ingestion_timeout = gr.Slider(
                            label="Ingestion timeout seconds",
                            minimum=10,
                            maximum=600,
                            value=DEFAULT_INGESTION_TIMEOUT_SECONDS,
                            step=5,
                        )

                with gr.Column(scale=2, min_width=520):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        type="messages",
                        height=540,
                    )
                    chat_question = gr.Textbox(
                        label="Message",
                        placeholder="Ask a question about the uploaded document",
                        lines=3,
                    )
                    with gr.Row():
                        chat_send_button = gr.Button("Send", variant="primary")
                        chat_clear_button = gr.Button("Clear", variant="secondary")

                    with gr.Accordion("Latest Citations", open=False):
                        chat_citations = gr.Dataframe(
                            headers=citation_headers,
                            label="Citations",
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Accordion("Latest Retrieved Context", open=False):
                        chat_context = gr.Dataframe(
                            headers=context_headers,
                            label="Retrieved context",
                            interactive=False,
                            wrap=True,
                        )

            chat_raw = gr.JSON(label="Latest API payload", visible=False)

        with gr.Tab("Diagnostics"):
            with gr.Accordion("Readiness", open=False):
                diagnostics_health_payload = gr.JSON(label="Readiness payload")

            with gr.Tab("End-to-end"):
                with gr.Row():
                    with gr.Column(scale=1):
                        e2e_file = gr.File(
                            label="Document",
                            file_count="single",
                            type="filepath",
                        )
                        e2e_title = gr.Textbox(label="Title", value="")
                        e2e_metadata = gr.Code(
                            label="Metadata JSON",
                            value=default_metadata,
                            language="json",
                        )
                    with gr.Column(scale=1):
                        e2e_question = gr.Textbox(
                            label="Question",
                            value="What does this document describe?",
                            lines=5,
                        )
                        e2e_retrieval_limit = gr.Slider(
                            label="Retrieval limit",
                            minimum=1,
                            maximum=50,
                            value=5,
                            step=1,
                        )
                        e2e_max_context_chunks = gr.Slider(
                            label="Max context chunks",
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1,
                        )
                        e2e_max_context_tokens = gr.Slider(
                            label="Max context tokens",
                            minimum=500,
                            maximum=12000,
                            value=1000,
                            step=100,
                        )
                        e2e_ingestion_timeout = gr.Slider(
                            label="Ingestion timeout seconds",
                            minimum=10,
                            maximum=600,
                            value=DEFAULT_INGESTION_TIMEOUT_SECONDS,
                            step=5,
                        )
                        e2e_query_timeout = gr.Slider(
                            label="Query timeout seconds",
                            minimum=10,
                            maximum=300,
                            value=DEFAULT_QUERY_TIMEOUT_SECONDS,
                            step=5,
                        )
                e2e_button = gr.Button("Run End-to-End", variant="primary")
                e2e_status = gr.Textbox(label="Run status", lines=6, interactive=False)
                with gr.Row():
                    e2e_document_id = gr.Textbox(label="Document ID", interactive=False)
                    e2e_job_id = gr.Textbox(
                        label="Ingestion Job ID",
                        interactive=False,
                    )
                e2e_answer = gr.Markdown(label="Answer")
                e2e_citations = gr.Dataframe(
                    headers=citation_headers,
                    label="Citations",
                    interactive=False,
                    wrap=True,
                )
                e2e_context = gr.Dataframe(
                    headers=context_headers,
                    label="Retrieved context",
                    interactive=False,
                    wrap=True,
                )
                e2e_raw = gr.JSON(label="Raw API payload")

            with gr.Tab("Documents"):
                with gr.Row():
                    with gr.Column():
                        upload_file_input = gr.File(
                            label="Document",
                            file_count="single",
                            type="filepath",
                        )
                        upload_title = gr.Textbox(label="Title", value="")
                        upload_metadata = gr.Code(
                            label="Metadata JSON",
                            value=default_metadata,
                            language="json",
                        )
                        upload_button = gr.Button("Upload Document", variant="primary")
                    with gr.Column():
                        upload_status = gr.Textbox(
                            label="Upload status",
                            lines=5,
                            interactive=False,
                        )
                        upload_document_id = gr.Textbox(label="Document ID")
                        upload_job_id = gr.Textbox(label="Ingestion Job ID")
                        upload_ingestion_status = gr.Textbox(
                            label="Ingestion status",
                            interactive=False,
                        )
                upload_raw = gr.JSON(label="Upload payload")

                with gr.Row():
                    ingestion_timeout = gr.Slider(
                        label="Ingestion timeout seconds",
                        minimum=10,
                        maximum=600,
                        value=DEFAULT_INGESTION_TIMEOUT_SECONDS,
                        step=5,
                    )
                    ingestion_poll_interval = gr.Slider(
                        label="Poll interval seconds",
                        minimum=0.5,
                        maximum=10,
                        value=2,
                        step=0.5,
                    )
                    wait_button = gr.Button("Wait For Ingestion", variant="secondary")
                ingestion_status = gr.Textbox(
                    label="Ingestion result",
                    lines=4,
                    interactive=False,
                )
                ingestion_raw = gr.JSON(label="Ingestion payload")

            with gr.Tab("Query"):
                query_question = gr.Textbox(
                    label="Question",
                    value="What does this document describe?",
                    lines=5,
                )
                with gr.Row():
                    query_retrieval_limit = gr.Slider(
                        label="Retrieval limit",
                        minimum=1,
                        maximum=50,
                        value=5,
                        step=1,
                    )
                    query_max_context_chunks = gr.Slider(
                        label="Max context chunks",
                        minimum=1,
                        maximum=20,
                        value=3,
                        step=1,
                    )
                    query_max_context_tokens = gr.Slider(
                        label="Max context tokens",
                        minimum=500,
                        maximum=12000,
                        value=1000,
                        step=100,
                    )
                query_button = gr.Button("Ask Question", variant="primary")
                query_answer = gr.Markdown(label="Answer")
                query_citations = gr.Dataframe(
                    headers=citation_headers,
                    label="Citations",
                    interactive=False,
                    wrap=True,
                )
                query_context = gr.Dataframe(
                    headers=context_headers,
                    label="Retrieved context",
                    interactive=False,
                    wrap=True,
                )
                query_raw = gr.JSON(label="Query payload")

        health_button.click(
            fn=check_api_health,
            inputs=[api_url, auth_token],
            outputs=[health_status, diagnostics_health_payload],
        )
        chat_upload_button.click(
            fn=upload_and_index_for_chat,
            inputs=[
                api_url,
                auth_token,
                workspace_id,
                chat_title,
                chat_metadata,
                chat_file,
                chat_ingestion_timeout,
                chat_documents_state,
                chatbot,
            ],
            outputs=[
                chatbot,
                chat_upload_status,
                chat_documents_state,
                chat_documents,
                chat_raw,
            ],
        )
        chat_question.submit(
            fn=chat_with_documents,
            inputs=[
                api_url,
                auth_token,
                workspace_id,
                chat_question,
                chatbot,
                chat_documents_state,
                chat_retrieval_limit,
                chat_max_context_chunks,
                chat_max_context_tokens,
            ],
            outputs=[
                chatbot,
                chat_question,
                chat_citations,
                chat_context,
                chat_raw,
            ],
        )
        chat_send_button.click(
            fn=chat_with_documents,
            inputs=[
                api_url,
                auth_token,
                workspace_id,
                chat_question,
                chatbot,
                chat_documents_state,
                chat_retrieval_limit,
                chat_max_context_chunks,
                chat_max_context_tokens,
            ],
            outputs=[
                chatbot,
                chat_question,
                chat_citations,
                chat_context,
                chat_raw,
            ],
        )
        chat_clear_button.click(
            fn=clear_chat_session,
            inputs=[],
            outputs=[
                chatbot,
                chat_upload_status,
                chat_documents_state,
                chat_documents,
                chat_citations,
                chat_context,
                chat_raw,
            ],
        )
        e2e_button.click(
            fn=run_end_to_end,
            inputs=[
                api_url,
                auth_token,
                workspace_id,
                e2e_title,
                e2e_metadata,
                e2e_file,
                e2e_question,
                e2e_retrieval_limit,
                e2e_max_context_chunks,
                e2e_max_context_tokens,
                e2e_ingestion_timeout,
                e2e_query_timeout,
            ],
            outputs=[
                e2e_status,
                e2e_document_id,
                e2e_job_id,
                e2e_answer,
                e2e_citations,
                e2e_context,
                e2e_raw,
            ],
        )
        upload_button.click(
            fn=upload_document,
            inputs=[
                api_url,
                auth_token,
                workspace_id,
                upload_title,
                upload_metadata,
                upload_file_input,
            ],
            outputs=[
                upload_status,
                upload_document_id,
                upload_job_id,
                upload_ingestion_status,
                upload_raw,
            ],
        )
        wait_button.click(
            fn=wait_for_ingestion,
            inputs=[
                api_url,
                auth_token,
                upload_document_id,
                upload_job_id,
                ingestion_timeout,
                ingestion_poll_interval,
            ],
            outputs=[ingestion_status, ingestion_raw],
        )
        query_button.click(
            fn=ask_question,
            inputs=[
                api_url,
                auth_token,
                workspace_id,
                query_question,
                query_retrieval_limit,
                query_max_context_chunks,
                query_max_context_tokens,
            ],
            outputs=[query_answer, query_citations, query_context, query_raw],
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )
    app = build_demo()
    app.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
