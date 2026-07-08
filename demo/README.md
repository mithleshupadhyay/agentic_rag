# Agentic RAG Gradio Demo

This demo is a chat-first Gradio client for the Agentic RAG API. It lets you
upload a document, wait for indexing, and chat with the indexed document from a
single browser page.

## Prerequisites

- Poetry
- Docker and Docker Compose
- Running Agentic RAG backend stack

## Start Backend

Run these commands from the project root:

```bash
make docker-up-build
curl http://localhost:8100/readiness
```

## Run Demo Locally

Run these commands from the project root:

```bash
cd demo
poetry install
poetry run python gradio_app.py
```

Open:

```text
http://localhost:7860
```

The local defaults are:

```text
AGENTIC_RAG_API_URL=http://localhost:8100
AGENTIC_RAG_AUTH_TOKEN=local-dev-token
AGENTIC_RAG_WORKSPACE_ID=local-workspace
```

## Run Demo With Docker

Build the demo image:

```bash
docker build -t agentic-rag-demo ./demo
```

Run the demo container:

```bash
docker run --rm \
  -p 7860:7860 \
  --add-host=host.docker.internal:host-gateway \
  agentic-rag-demo
```

Open:

```text
http://localhost:7860
```

## Demo Flow

1. Open `http://localhost:7860`.
2. Upload a PDF or text document.
3. Click `Upload & Index`.
4. Ask questions in the chat box.
5. Open `Latest Citations` or `Latest Retrieved Context` when you need evidence.

The `Diagnostics` tab keeps the lower-level upload, ingestion, query, and raw
payload views for debugging.

## Dokploy Deployment

Deploy the API stack and demo as separate services.

Use these demo service environment variables when the demo container shares the
same Docker network as the API service:

```text
AGENTIC_RAG_API_URL=http://api:8000
AGENTIC_RAG_AUTH_TOKEN=<strong-demo-token>
AGENTIC_RAG_WORKSPACE_ID=local-workspace
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

Expose:

```text
https://demo.yourdomain.com
```

For public demos, do not use `local-dev-token`. Set a strong token in the API
environment and use the same value in `AGENTIC_RAG_AUTH_TOKEN`.
