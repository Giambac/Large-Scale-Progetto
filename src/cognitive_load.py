"""
cognitive_load.py — f_cognitive_load pure function (ORC-03, D-06, D-07, D-08).

Computes a normalized cognitive load score in [0, 1] that the conversation loop
passes to OracleAgent.reply() before each turn. The oracle uses this score to
decide whether to inject the OVERLOAD instruction into its system prompt.

No I/O. No global state. Pure function only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.state import ClusteringState

# ── Named constants (D-07) — not magic numbers inline ─────────────────────────
MAX_K: int = 20                   # cap for cluster-count normalization term
MAX_MSG_LEN: int = 500            # cap for message-length normalization term
TOP_K_ITEMS_PER_CLUSTER: int = 5  # matches _format_message default (top-5 items shown)
COG_LOAD_THRESHOLD: float = 0.7   # D-08: load > COG_LOAD_THRESHOLD → OVERLOAD instruction in system prompt


def f_cognitive_load(state: "ClusteringState", message: str) -> float:
    """
    Compute a normalized cognitive load score in [0, 1].

    Formula (D-07):
        load = (clusters / MAX_K) * w1
             + (items_shown / total_items) * w2
             + (msg_len / MAX_MSG_LEN) * w3
    where w1 = w2 = w3 = 1/3 and each term is clamped to [0, 1].
    items_shown = len(clusters) * TOP_K_ITEMS_PER_CLUSTER (what _format_message shows).

    Args:
        state:   Current ClusteringState. Must have at least one cluster and one item.
        message: The formatted message string sent to the oracle this turn.

    Returns:
        float in [0.0, 1.0]. Compare against COG_LOAD_THRESHOLD to decide prompt injection.

    Raises:
        AssertionError: if state has no clusters or no items (fail-loudly per CLAUDE.md).
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
