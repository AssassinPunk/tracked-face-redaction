"""API tests for the health endpoint and generated docs."""
from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings
from app.main import create_app


def test_health_ok_when_model_loaded():
    # `with` runs the lifespan startup, so the model actually loads.
    with TestClient(create_app()) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "version": __version__}


def test_health_503_when_model_missing(tmp_path):
    settings = Settings(model_path=tmp_path / "missing.onnx")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_error_response_does_not_leak_server_paths(tmp_path):
    settings = Settings(model_path=tmp_path / "missing.onnx")
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/health").text
    assert "missing.onnx" not in body and str(tmp_path) not in body


def test_openapi_docs_are_generated():
    with TestClient(create_app()) as client:
        assert client.get("/docs").status_code == 200
        assert "/api/health" in client.get("/openapi.json").json()["paths"]