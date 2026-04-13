import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.unit
def test_health_check(client):
    """Verify that the health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data

@pytest.mark.unit
def test_video_query_docs_access(client):
    """Verify that API docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
