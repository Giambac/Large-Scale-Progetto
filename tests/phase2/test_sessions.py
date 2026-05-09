"""Tests for UI-V2-01: persistent sessions — directory structure, state.json write, endpoints."""
from __future__ import annotations
import json
import os
import tempfile
import pytest


@pytest.fixture
def client():
    """Flask test client with testing mode enabled."""
    from web.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def simple_state():
    """Minimal ClusteringState for serialization tests."""
    from src.state import Cluster, ClusteringState
    return ClusteringState(
        turn_index=3,
        timestamp="2026-05-08T14:32:00+00:00",
        clusters=[
            Cluster(id=0, name="A", description="desc A", item_ids=[0, 1, 2]),
            Cluster(id=1, name="B", description="desc B", item_ids=[3, 4]),
        ],
        assignments={0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
        soft_probs={
            0: [0.9, 0.1], 1: [0.8, 0.2], 2: [0.85, 0.15],
            3: [0.1, 0.9], 4: [0.15, 0.85],
        },
    )


def test_session_timestamp_format():
    from web.app import _make_session_timestamp
    ts = _make_session_timestamp()
    assert isinstance(ts, str)
    assert ":" not in ts  # colons replaced with hyphens (D-26)
    assert "T" in ts  # ISO-like format with T separator
    assert len(ts) == len("2026-05-08T14-32-00")


def test_write_session_state_creates_file(simple_state):
    from web.app import _write_session_state
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_session_state(simple_state, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "state.json"))


def test_write_session_state_is_valid_json(simple_state):
    from web.app import _write_session_state
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_session_state(simple_state, tmpdir)
        data = json.load(open(os.path.join(tmpdir, "state.json")))
        assert "turn_index" in data
        assert "clusters" in data
        assert data["turn_index"] == 3


def test_write_session_state_deserializes_correctly(simple_state):
    from web.app import _write_session_state
    from src.serialization import deserialize_state
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_session_state(simple_state, tmpdir)
        line = open(os.path.join(tmpdir, "state.json")).read()
        loaded = deserialize_state(line)
        assert loaded.turn_index == simple_state.turn_index
        assert len(loaded.clusters) == 2
        assert loaded.assignments[0] == 0


def test_get_sessions_endpoint_returns_200(client):
    response = client.get("/sessions")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)


def test_get_sessions_discovers_session_directory(client, simple_state, monkeypatch):
    from web.app import _write_session_state
    import web.app as app_module
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(app_module, "SESSIONS_DIR", tmpdir)
        session_dir = os.path.join(tmpdir, "2026-05-08T14-32-00")
        os.makedirs(session_dir, exist_ok=True)
        _write_session_state(simple_state, session_dir)
        response = client.get("/sessions")
        data = json.loads(response.data)
        assert len(data) >= 1
        assert any(s["session_id"] == "2026-05-08T14-32-00" for s in data)


def test_sessions_response_has_required_fields(client, simple_state, monkeypatch):
    from web.app import _write_session_state
    import web.app as app_module
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(app_module, "SESSIONS_DIR", tmpdir)
        session_dir = os.path.join(tmpdir, "2026-05-08T14-32-00")
        os.makedirs(session_dir, exist_ok=True)
        _write_session_state(simple_state, session_dir)
        response = client.get("/sessions")
        data = json.loads(response.data)
        session = data[0]
        assert "session_id" in session
        assert "timestamp" in session
        assert "cluster_count" in session
        assert "turn_count" in session
        assert session["cluster_count"] == 2  # from simple_state
        assert session["turn_count"] == 3  # turn_index from simple_state


def test_resume_endpoint_exists(client):
    response = client.post("/resume/nonexistent-session")
    assert response.status_code in (400, 404, 500)  # fails loudly on missing session (not 200)


def test_resume_endpoint_loads_valid_session(client, simple_state, monkeypatch):
    from web.app import _write_session_state
    import web.app as app_module
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(app_module, "SESSIONS_DIR", tmpdir)
        session_dir = os.path.join(tmpdir, "2026-05-08T14-32-00")
        os.makedirs(session_dir, exist_ok=True)
        _write_session_state(simple_state, session_dir)
        response = client.post("/resume/2026-05-08T14-32-00")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "resumed"
        assert data["turn_index"] == 3


def test_per_turn_write_via_callback(simple_state):
    from web.app import _write_session_state
    from src.serialization import deserialize_state
    from src.state import Cluster, ClusteringState
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate turn 1
        state_turn1 = simple_state  # turn_index=3 acts as turn-1 baseline

        # Simulate turn 2 with a higher turn_index
        state_turn2 = ClusteringState(
            turn_index=2,
            timestamp="2026-05-08T14:32:01+00:00",
            clusters=[
                Cluster(id=0, name="A", description="desc A", item_ids=[0, 1, 2]),
                Cluster(id=1, name="B", description="desc B", item_ids=[3, 4]),
            ],
            assignments={0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
            soft_probs={
                0: [0.9, 0.1], 1: [0.8, 0.2], 2: [0.85, 0.15],
                3: [0.1, 0.9], 4: [0.15, 0.85],
            },
        )

        # Simulate per-turn callback calls
        _write_session_state(state_turn1, tmpdir)
        _write_session_state(state_turn2, tmpdir)

        # Read back state.json — must contain most recent state (turn-2, not turn-1)
        state_path = os.path.join(tmpdir, "state.json")
        loaded = deserialize_state(open(state_path).read())
        assert loaded.turn_index == 2  # most recent state written (turn-2, not turn-1)
