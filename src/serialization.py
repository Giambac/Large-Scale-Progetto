"""
serialization.py — JSONL serialization for ClusteringState (FOUND-04).

The AuditLog is append-only JSONL: one JSON object per line, one line per turn.
Each line is a complete ClusteringState snapshot.

Key serialization risks:
- numpy types: float32 raises TypeError with json.dumps. Custom encoder handles this.
- JSON dict key type loss: json.loads always deserializes keys as str.
  deserialize_state MUST cast assignments and soft_probs dict keys to int with int(k).

D-14: Deserialization must reproduce the exact same object — no data loss.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import numpy as np

from src.state import Cluster, ClusteringState


class _StateEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles:
    - dataclasses (via dataclasses.asdict)
    - numpy scalar types (np.integer, np.floating)
    - numpy arrays (via .tolist())

    Applied by serialize_state via cls=_StateEncoder.
    """

    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def serialize_state(state: ClusteringState) -> str:
    """
    Serialize a ClusteringState to a single JSON line.

    Returns a string with no leading/trailing whitespace and no trailing newline.
    The string is valid JSON parseable by json.loads().

    Raises TypeError (re-raised from json.dumps) if state contains an
    unhandled non-serializable type — this is intentional (fail loudly).
    """
    assert isinstance(state, ClusteringState), (
        f"serialize_state expects ClusteringState, got {type(state)}"
    )
    line = json.dumps(dataclasses.asdict(state), cls=_StateEncoder)
    assert "\n" not in line, "BUG: serialized state contains newline (would break JSONL)"
    return line


def deserialize_state(line: str) -> ClusteringState:
    """
    Reconstruct a ClusteringState from a JSONL line.

    CRITICAL: JSON object keys are always strings. This function explicitly
    casts assignments and soft_probs dict keys to int with int(k).
    Failing to do this causes KeyError when code does state.assignments[0].

    Args:
        line: A single JSON line (from the AuditLog or serialize_state output).

    Returns:
        ClusteringState with int keys in assignments and soft_probs.

    Raises:
        json.JSONDecodeError: if line is not valid JSON.
        KeyError/TypeError: if required fields are missing (fail loudly).
    """
    d = json.loads(line)

    assert "turn_index" in d, f"Missing 'turn_index' in deserialized state: {list(d.keys())}"
    assert "timestamp" in d, f"Missing 'timestamp' in deserialized state"
    assert "clusters" in d, f"Missing 'clusters' in deserialized state"
    assert "assignments" in d, f"Missing 'assignments' in deserialized state"
    assert "soft_probs" in d, f"Missing 'soft_probs' in deserialized state"

    clusters = [
        Cluster(
            id=int(c["id"]),
            name=c["name"],
            description=c["description"],
            item_ids=[int(i) for i in c["item_ids"]],
        )
        for c in d["clusters"]
    ]

    # Cast dict keys from str back to int — JSON key type loss invariant
    assignments: dict[int, int] = {int(k): int(v) for k, v in d["assignments"].items()}
    soft_probs: dict[int, list[float]] = {int(k): [float(p) for p in v] for k, v in d["soft_probs"].items()}

    return ClusteringState(
        turn_index=d["turn_index"],
        timestamp=d["timestamp"],
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )


def append_to_audit_log(state: ClusteringState, log_path: str) -> None:
    """
    Append one ClusteringState as a JSON line to the AuditLog file.

    Creates the file (and parent directories) if it does not exist (append mode).
    Each call writes exactly one line (serialize_state(state) + newline).

    Args:
        state: The ClusteringState to append.
        log_path: Path to the JSONL AuditLog file (e.g. "audit_log.jsonl").
    """
    assert isinstance(log_path, str) and log_path, "log_path must be a non-empty string"
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = serialize_state(state)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_audit_log(log_path: str) -> list[ClusteringState]:
    """
    Load all turns from the AuditLog.

    Reads each non-empty line and deserializes to ClusteringState.
    Asserts that at least one turn exists (empty AuditLog is an error).
    """
    assert os.path.exists(log_path), f"AuditLog not found: {log_path}"
    states = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(deserialize_state(line))
    assert len(states) > 0, f"AuditLog at {log_path} is empty — no turns recorded"
    return states
