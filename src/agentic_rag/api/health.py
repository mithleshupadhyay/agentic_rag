import time

from fastapi import APIRouter

from agentic_rag.shared.config import settings
from agentic_rag.shared.schemas.common import DependencyStatus, HealthResponse


start_time = time.time()
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        service=settings.app_name,
        status="healthy",
        version=settings.app_version,
        dependencies={},
    )


@router.get("/liveness")
def liveness_check() -> dict[str, float | str]:
    return {"status": "alive", "uptime_seconds": round(time.time() - start_time, 2)}


@router.get("/readiness", response_model=HealthResponse)
def readiness_check() -> HealthResponse:
    dependencies = {
        "configuration": DependencyStatus(
            name="configuration",
            status="healthy",
            detail="Application configuration loaded",
        )
    }
    return HealthResponse(
        service=settings.app_name,
        status="healthy",
        version=settings.app_version,
        dependencies=dependencies,
    )

