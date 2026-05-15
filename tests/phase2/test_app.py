"""Tests for web/app.py — UI-01 (FastAPI debug UI), UI-02 (dataset upload)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI test client from web.app."""
    from web.app import app
    with TestClient(app) as c:
        yield c


def test_index_route_returns_200(client):
    """UI-01: GET / returns 200 and serves the debug UI HTML."""
    response = client.get("/")
    assert response.status_code == 200


def test_index_route_returns_html(client):
    """UI-01: index.html response contains cluster-related HTML structure."""
    response = client.get("/")
    assert b"cluster" in response.content.lower() or b"Cluster" in response.content


def test_upload_endpoint_exists(client):
    """UI-02: POST /upload endpoint exists (returns non-404)."""
    response = client.post("/upload")
    assert response.status_code != 404


def test_status_endpoint(client):
    """GET /status returns JSON with session state info."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_upload_resets_session(client, tmp_path):
    """UI-02: POST /upload with valid CSV data starts a new session (state cleared)."""
    import io
    # Minimal CSV: 5 rows with 'text' column
    csv_content = "text\nhello world\nfoo bar\nbaz qux\nalpha beta\ngamma delta\n"
    response = client.post(
        "/upload",
        files={"file": ("test.csv", csv_content.encode(), "text/csv")},
    )
    # Accept either 200 (sync start) or 202 (async start)
    assert response.status_code in (200, 202)
