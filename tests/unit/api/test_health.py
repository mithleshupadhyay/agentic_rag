import logging
from uuid import UUID

from fastapi.testclient import TestClient

from agentic_rag.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "Agentic RAG",
        "status": "healthy",
        "version": "0.1.0",
        "dependencies": {},
    }


def test_liveness_endpoint() -> None:
    response = client.get("/liveness")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert response.json()["uptime_seconds"] >= 0


def test_readiness_endpoint() -> None:
    response = client.get("/readiness")

    assert response.status_code == 200
    assert response.json()["dependencies"]["configuration"]["status"] == "healthy"


def test_openapi_has_bearer_auth() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    security_schemes = response.json()["components"]["securitySchemes"]
    assert security_schemes["BearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }


def test_request_id_header_is_generated() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    UUID(response.headers["X-Request-ID"])


def test_request_id_header_preserves_client_value() -> None:
    response = client.get(
        "/health",
        headers={"X-Request-ID": "request-id-from-client"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-id-from-client"


def test_request_logs_include_request_id_and_duration(caplog) -> None:
    caplog.set_level(logging.INFO, logger="agentic_rag.main")

    response = client.get(
        "/health",
        headers={"X-Request-ID": "request-id-from-log-test"},
    )

    assert response.status_code == 200
    assert "Started request_id=request-id-from-log-test" in caplog.text
    assert "Completed request_id=request-id-from-log-test" in caplog.text
    assert "method=GET" in caplog.text
    assert "path=/health" in caplog.text
    assert "status_code=200" in caplog.text
    assert "duration_ms=" in caplog.text
