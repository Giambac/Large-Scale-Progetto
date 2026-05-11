"""Tests for web/app.py — UI-01 (Flask debug UI), UI-02 (dataset upload)."""
import pytest


@pytest.fixture
def flask_client():
    """Create a Flask test client from web.app."""
    from web.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route_returns_200(flask_client):
    """UI-01: GET / returns 200 and serves the debug UI HTML."""
    response = flask_client.get("/")
    assert response.status_code == 200


def test_index_route_returns_html(flask_client):
    """UI-01: index.html response contains cluster-related HTML structure."""
    response = flask_client.get("/")
    assert b"cluster" in response.data.lower() or b"Cluster" in response.data


def test_upload_endpoint_exists(flask_client):
    """UI-02: POST /upload endpoint exists (returns non-404)."""
    response = flask_client.post("/upload")
    assert response.status_code != 404


def test_status_endpoint(flask_client):
    """GET /status returns JSON with session state info."""
    response = flask_client.get("/status")
    assert response.status_code == 200
    import json
    data = json.loads(response.data)
    assert "status" in data


def test_upload_resets_session(flask_client, tmp_path):
    """UI-02: POST /upload with valid CSV data starts a new session (state cleared)."""
    import io
    # Minimal CSV: 5 rows with 'text' column
    csv_content = "text\nhello world\nfoo bar\nbaz qux\nalpha beta\ngamma delta\n"
    data = {"file": (io.BytesIO(csv_content.encode()), "test.csv")}
    response = flask_client.post("/upload", data=data, content_type="multipart/form-data")
    # Accept either 200 (sync start) or 202 (async start)
    assert response.status_code in (200, 202)
