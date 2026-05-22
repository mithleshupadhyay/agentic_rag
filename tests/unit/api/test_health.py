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
