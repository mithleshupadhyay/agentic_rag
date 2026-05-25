import logging
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator

from agentic_rag.api import documents, health, query, retrieval
from agentic_rag.shared.config import settings


load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready multi-tenant Agentic RAG API with JWT auth.",
)

# Prometheus metrics must be registered before startup.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

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

    for path in openapi_schema["paths"].values():
        for operation in path.values():
            operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]
