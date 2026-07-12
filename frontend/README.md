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

## Authentication

The frontend loads public authentication settings from `/api/auth/config`.
With `AUTH_PROVIDER=keycloak`, it uses Authorization Code with PKCE and obtains
the API bearer token from the OIDC session. Google, GitHub, LinkedIn, and other
providers are configured as Keycloak identity providers, not in React.

The login screen exposes configured social providers in the top-right and sends
email or username sign-in to Keycloak's hosted password form. Tenant admins use
the Users view to invite members and assign `viewer`, `user`, or `admin` roles.
Passwords and provider secrets never enter the frontend or Agentic RAG API.

The backend `/api/auth/session` response supplies the authoritative tenant,
workspace, roles, groups, scopes, and ACL version. The frontend uses these
values to hide unavailable actions, while the API enforces every permission.

See [Authentication And Tenant Access](../docs/AUTHENTICATION.md) for Keycloak,
claim mapper, social provider, and deployment configuration.
