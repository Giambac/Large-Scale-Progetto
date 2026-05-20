"""
tests/test_study_ui.py — FastAPI endpoint tests for study routes (GEN-02, EXP-V2-01, 06-04).

Covers:
  - POST /study/sessions with valid dataset → 200 + session_id + study_url
  - POST /study/sessions with nonexistent dataset → 400
  - POST /study/sessions with invalid backend → 400
  - GET /study/{session_id} with known session → 200 with "study" in body
  - GET /study/{session_id} with unknown session → 404
  - STUDY_MAX_TURNS default value (15)
  - STUDY_MAX_TURNS env override parsing

No test makes a live Anthropic API call or a live DB connection.
"""
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_dataset(tmp_path) -> str:
    """Create a minimal JSONL dataset with 10 records. Returns the file path."""
    dataset_path = tmp_path / "dataset.jsonl"
    lines = [f'{{"text": "item text number {i}"}}' for i in range(10)]
    dataset_path.write_text("\n".join(lines), encoding="utf-8")
    return str(dataset_path)


@pytest.fixture
def study_client():
    """FastAPI TestClient using the FastAPI app object (not asgi_app)."""
    from web.app import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_post_study_sessions_valid(test_dataset, study_client, monkeypatch):
    """
    POST /study/sessions with a valid dataset and backend returns 200 with
    'session_id' and 'study_url' fields in the JSON response.

    The background worker is suppressed by patching _run_study_background to a
    no-op — this avoids patching threading.Thread.start globally, which would
    break TestClient's own internal threading.
    """
    # Route DB operations through an in-memory SQLite DB (D-01 pattern)
    mem_db = sqlite3.connect(":memory:", check_same_thread=False)
    mem_db.execute("PRAGMA foreign_keys=ON")
    from src.db.connection import init_schema
    init_schema(mem_db)

    # Patch connect() and init_schema() in the src.db.connection module
    # (create_study_session does: from src.db.connection import connect as _db_connect)
    monkeypatch.setattr("src.db.connection.connect", lambda: mem_db)
    monkeypatch.setattr("src.db.connection.init_schema", lambda db: None)

    # Patch experiments.create to return a fake ExperimentRead object
    fake_exp = MagicMock()
    fake_exp.id = 1
    monkeypatch.setattr("src.db.experiments.create", lambda db, ec: fake_exp)

    # Suppress the background worker without breaking TestClient's own thread
    # (TestClient uses threading.Thread internally — global Thread.start patch kills it)
    monkeypatch.setattr("web.app._run_study_background", lambda session_id: None)

    resp = study_client.post(
        "/study/sessions",
        json={"dataset_path": test_dataset, "backend": "hdbscan"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "study_url" in data
    assert data["study_url"].startswith("/study/")


def test_post_study_sessions_nonexistent_dataset(study_client):
    """POST /study/sessions with a nonexistent dataset_path returns 400."""
    resp = study_client.post(
        "/study/sessions",
        json={"dataset_path": "/nonexistent/path/dataset.jsonl", "backend": "hdbscan"},
    )
    assert resp.status_code == 400


def test_post_study_sessions_invalid_backend(test_dataset, study_client):
    """POST /study/sessions with an invalid backend name returns 400."""
    resp = study_client.post(
        "/study/sessions",
        json={"dataset_path": test_dataset, "backend": "invalid"},
    )
    assert resp.status_code == 400


def test_get_study_page_known_session(study_client, monkeypatch):
    """
    GET /study/{session_id} returns 200 and HTML containing 'study'
    when the session_id is present in _study_sessions.
    """
    import web.app as app_module
    # Pre-populate _study_sessions with a fake session
    app_module._study_sessions["test-session-99"] = {"state": None, "ended": False}
    resp = study_client.get("/study/test-session-99")
    assert resp.status_code == 200
    assert b"study" in resp.content.lower()


def test_get_study_page_unknown_session(study_client):
    """GET /study/{session_id} returns 404 for an unknown session_id."""
    resp = study_client.get("/study/nonexistent-session-xyz")
    assert resp.status_code == 404


def test_study_max_turns_env_default(monkeypatch):
    """
    STUDY_MAX_TURNS defaults to 15 when the environment variable is not set.
    Validates the module-level constant directly (module already loaded).
    """
    import web.app as app_module
    # Remove STUDY_MAX_TURNS from env if present, then check the constant.
    # The module constant was set at import time; verify it's 15.
    monkeypatch.delenv("STUDY_MAX_TURNS", raising=False)
    assert app_module.STUDY_MAX_TURNS == 15


def test_study_max_turns_env_override(monkeypatch):
    """STUDY_MAX_TURNS reflects the env var when web.app is re-imported."""
    monkeypatch.setenv("STUDY_MAX_TURNS", "5")
    import importlib
    import web.app as app_module
    importlib.reload(app_module)
    assert app_module.STUDY_MAX_TURNS == 5
