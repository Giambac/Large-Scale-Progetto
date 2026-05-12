"""
cognitive_load.py — f_cognitive_load() pure function + named constants (ORC-03, D-06, D-07, D-08).

No I/O. No global state. Pure function only.

Formula: load = (len(state.clusters)/MAX_K)*w1 + (items_shown/total_items)*w2 + (len(message)/MAX_MSG_LEN)*w3
Default weights w1=w2=w3=1/3. items_shown = len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER.
Result clamped to [0, 1].
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.state import ClusteringState

# Named constants — not magic numbers inline (per D-07 and PATTERNS.md)
MAX_K: int = 20
MAX_MSG_LEN: int = 500
TOP_K_ITEMS_PER_CLUSTER: int = 5
COG_LOAD_THRESHOLD: float = 0.7


def f_cognitive_load(state: "ClusteringState", message: str) -> float:
    """
    Pure function. No I/O. No global state.

    Compute a normalized cognitive load score in [0, 1].

    Formula (D-07):
        load = (clusters/MAX_K)*w1 + (items_shown/total_items)*w2 + (msg_len/MAX_MSG_LEN)*w3
    Default weights: w1=w2=w3=1/3.
    items_shown = len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER (matching _format_message top-5 default).

    Each term is clamped to [0, 1] before weighting to prevent overflow.
    """
    assert len(state.clusters) > 0, "f_cognitive_load: no clusters in state"
    total_items = len(state.assignments)
    assert total_items > 0, "f_cognitive_load: no items in state"

    w = 1.0 / 3.0
    cluster_term = min(len(state.clusters) / MAX_K, 1.0) * w
    items_shown = len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER
    items_term = min(items_shown / total_items, 1.0) * w
    msg_term = min(len(message) / MAX_MSG_LEN, 1.0) * w
    return cluster_term + items_term + msg_term
