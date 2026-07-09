# Agentic RAG Frontend

React frontend for the Agentic RAG API. The main screen is a document-scoped
chat interface with upload, ingestion polling, retrieval-backed answers,
citations, and context evidence.

## Start Backend

Run these commands from the project root:

```bash
make docker-up-build
curl http://localhost:8100/readiness
```

## Run Frontend Locally

The full Docker stack serves the frontend at port `5173`:

```bash
make docker-up-build
```

Open:

```text
http://localhost:5173
```

For frontend development, run these commands from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The local dev server proxies `/api` to `http://localhost:8100`.

## Build Check

```bash
cd frontend
npm run build
```

## Docker Build

```bash
docker build -t agentic-rag-frontend ./frontend
```

The production container serves the frontend with Nginx and proxies `/api` to
`http://api:8000` on the same Docker network.

## Environment

For local development, the frontend defaults are:

```text
VITE_API_BASE_URL=/api
VITE_AUTH_TOKEN=local-dev-token
VITE_WORKSPACE_ID=local-workspace
VITE_QUERY_STRATEGY=hybrid
```
