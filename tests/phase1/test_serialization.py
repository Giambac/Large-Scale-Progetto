"""Tests for serialization.py — FOUND-04: JSONL round-trip and AuditLog."""
import json
import pytest

from src.serialization import serialize_state, deserialize_state, append_to_audit_log
from src.state import ClusteringState, Cluster


def _make_tiny_state() -> ClusteringState:
    return ClusteringState(
        turn_index=0,
        timestamp="2026-05-04T12:00:00",
        clusters=[
            Cluster(id=0, name="Positive", description="Happy reviews.", item_ids=[0, 2, 4]),
            Cluster(id=1, name="Negative", description="Unhappy reviews.", item_ids=[1, 3]),
        ],
        assignments={0: 0, 1: 1, 2: 0, 3: 1, 4: 0},
        soft_probs={0: [0.9, 0.1], 1: [0.1, 0.9], 2: [0.8, 0.2], 3: [0.2, 0.8], 4: [0.85, 0.15]},
    )


def test_roundtrip_turn_index():
    """Deserialized state has same turn_index as original."""
    state = _make_tiny_state()
    line = serialize_state(state)
    recovered = deserialize_state(line)
    assert recovered.turn_index == state.turn_index


def test_roundtrip_timestamp():
    """Deserialized state has same timestamp as original."""
    state = _make_tiny_state()
    line = serialize_state(state)
    recovered = deserialize_state(line)
    assert recovered.timestamp == state.timestamp


def test_roundtrip_cluster_count():
    """Deserialized state has same number of clusters."""
    state = _make_tiny_state()
    recovered = deserialize_state(serialize_state(state))
    assert len(recovered.clusters) == len(state.clusters)


def test_roundtrip_cluster_fields():
    """Deserialized clusters have same id, name, description, item_ids."""
    state = _make_tiny_state()
    recovered = deserialize_state(serialize_state(state))
    for orig, rec in zip(state.clusters, recovered.clusters):
        assert rec.id == orig.id
        assert rec.name == orig.name
        assert rec.description == orig.description
        assert rec.item_ids == orig.item_ids


def test_roundtrip_assignments():
    """Deserialized assignments dict is identical to original."""
    state = _make_tiny_state()
    recovered = deserialize_state(serialize_state(state))
    assert recovered.assignments == state.assignments


def test_roundtrip_soft_probs():
    """Deserialized soft_probs dict is identical to original."""
    state = _make_tiny_state()
    recovered = deserialize_state(serialize_state(state))
    assert recovered.soft_probs == state.soft_probs


def test_int_keys_assignments():
    """assignments keys are int after deserialization (not str) — JSON key type loss pitfall."""
    state = _make_tiny_state()
    recovered = deserialize_state(serialize_state(state))
    for key in recovered.assignments:
        assert isinstance(key, int), f"assignments key {key!r} is type {type(key).__name__}, expected int"


def test_int_keys_soft_probs():
    """soft_probs keys are int after deserialization (not str)."""
    state = _make_tiny_state()
    recovered = deserialize_state(serialize_state(state))
    for key in recovered.soft_probs:
        assert isinstance(key, int), f"soft_probs key {key!r} is type {type(key).__name__}, expected int"


def test_serialize_produces_valid_json():
    """serialize_state output parses as valid JSON."""
    state = _make_tiny_state()
    line = serialize_state(state)
    parsed = json.loads(line)  # must not raise
    assert "turn_index" in parsed


def test_append_to_audit_log_creates_file(tmp_path):
    """append_to_audit_log creates the log file if it does not exist."""
    log_path = str(tmp_path / "audit_log.jsonl")
    state = _make_tiny_state()
    append_to_audit_log(state, log_path)
    import os
    assert os.path.exists(log_path)


def test_append_to_audit_log_one_line_per_call(tmp_path):
    """Each call to append_to_audit_log adds exactly one line."""
    log_path = str(tmp_path / "audit_log.jsonl")
    state = _make_tiny_state()
    append_to_audit_log(state, log_path)
    append_to_audit_log(state, log_path)
    with open(log_path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) == 2


def test_append_to_audit_log_each_line_is_valid_json(tmp_path):
    """Every appended line is valid JSON."""
    log_path = str(tmp_path / "audit_log.jsonl")
    state = _make_tiny_state()
    append_to_audit_log(state, log_path)
    append_to_audit_log(state, log_path)
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                json.loads(line)  # must not raise
