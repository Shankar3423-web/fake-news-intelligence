import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to sys.path so we can import 'app' module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
