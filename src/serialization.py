"""
serialization.py — JSONL serialization for ClusteringState (FOUND-04).

serialize_state(state) -> single-line JSON string (no trailing newline).
deserialize_state(line) -> ClusteringState with int keys (not str).
append_to_audit_log(state, path) -> appends one JSON line to the JSONL file.

NOTE: JSON serializes dict keys as strings. Deserialization casts them back
to int with int(k) to preserve the dict[int, int] contract from D-13.
"""
from __future__ import annotations

import dataclasses
import json
import os

from src.state import Cluster, ClusteringState


class _StateEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses and numpy scalar types."""

    def default(self, obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        # numpy int/float types surface as Python ints/floats via .tolist(),
        # but handle them defensively here in case any slip through.
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def serialize_state(state: ClusteringState) -> str:
    """
    Serialize a ClusteringState to a single-line JSON string.

    Returns a compact JSON string with no trailing newline.
    """
    assert isinstance(state, ClusteringState), (
        f"serialize_state expects ClusteringState, got {type(state).__name__}"
    )
    d = dataclasses.asdict(state)
    return json.dumps(d, cls=_StateEncoder, separators=(",", ":"))


def deserialize_state(line: str) -> ClusteringState:
    """
    Deserialize a JSON line (from serialize_state) back to ClusteringState.

    Casts dict keys for assignments and soft_probs back to int — JSON always
    produces string keys, but D-13 requires dict[int, int] and dict[int, list[float]].
    """
    assert isinstance(line, str) and line.strip(), "deserialize_state expects a non-empty string"
    d = json.loads(line)

    clusters = [
        Cluster(
            id=int(c["id"]),
            name=c["name"],
            description=c["description"],
            item_ids=[int(i) for i in c["item_ids"]],
        )
        for c in d["clusters"]
    ]

    # JSON keys are always strings — cast back to int (D-13 contract)
    assignments = {int(k): int(v) for k, v in d["assignments"].items()}
    soft_probs = {int(k): [float(p) for p in v] for k, v in d["soft_probs"].items()}

    return ClusteringState(
        turn_index=int(d["turn_index"]),
        timestamp=d["timestamp"],
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )


def append_to_audit_log(state: ClusteringState, log_path: str) -> None:
    """
    Append one serialized ClusteringState line to the JSONL audit log.

    Creates the file (and parent directories) if they do not exist.
    Each call writes exactly one line followed by a newline character.
    """
    assert isinstance(log_path, str) and log_path, "log_path must be a non-empty string"
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = serialize_state(state)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
