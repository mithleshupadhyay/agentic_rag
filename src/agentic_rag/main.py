import logging
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator
from supertokens_python.framework.fastapi import get_middleware

from agentic_rag.api import admin, auth, documents, health, query, retrieval, tenancy
from agentic_rag.core.supertokens import initialize_supertokens
from agentic_rag.shared.config import settings


load_dotenv()

logger = logging.getLogger(__name__)

supertokens_enabled = initialize_supertokens()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready multi-tenant Agentic RAG API with JWT auth.",
)

# Prometheus metrics must be registered before startup.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

if supertokens_enabled:
    app.add_middleware(get_middleware())

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    started_at = time.perf_counter()

    logger.info(
        f"[Request] Started request_id={request_id} "
        f"method={request.method} path={request.url.path}"
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    duration_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    logger.info(
        f"[Request] Completed request_id={request_id} "
        f"method={request.method} path={request.url.path} "
        f"status_code={response.status_code} duration_ms={duration_ms}"
    )
    return response


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(tenancy.router)
app.include_router(admin.router)
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(query.router)


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema.setdefault("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    public_paths = {"/auth/config", "/health", "/liveness", "/metrics", "/readiness"}
    for path_name, path in openapi_schema["paths"].items():
        for operation in path.values():
            if isinstance(operation, dict):
                operation["security"] = (
                    [] if path_name in public_paths else [{"BearerAuth": []}]
                )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]
