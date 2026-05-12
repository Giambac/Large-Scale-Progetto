"""
test_cognitive_load.py — Unit tests for f_cognitive_load pure function (ORC-03).

TDD RED phase: these tests must fail before src/cognitive_load.py exists.
"""
import pytest

from src.state import Cluster, ClusteringState


def _make_state(n_clusters: int, n_items: int) -> ClusteringState:
    """Helper: build a minimal ClusteringState with n_clusters clusters and n_items items."""
    assert n_clusters > 0
    assert n_items >= n_clusters  # each cluster gets at least one item

    clusters = []
    assignments: dict[int, int] = {}
    soft_probs: dict[int, list[float]] = {}

    # Distribute items round-robin across clusters
    for i in range(n_clusters):
        clusters.append(Cluster(id=i, name=f"C{i}", description=".", item_ids=[]))

    for item_id in range(n_items):
        cluster_id = item_id % n_clusters
        clusters[cluster_id].item_ids.append(item_id)
        assignments[item_id] = cluster_id
        probs = [0.0] * n_clusters
        probs[cluster_id] = 1.0
        soft_probs[item_id] = probs

    return ClusteringState(
        turn_index=0,
        timestamp="2026-05-12T00:00:00",
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )


# ── Test 1: return value is always in [0.0, 1.0] ─────────────────────────────

def test_load_in_range():
    """f_cognitive_load must return a float in [0.0, 1.0] for any valid state."""
    from src.cognitive_load import f_cognitive_load

    state = _make_state(n_clusters=3, n_items=30)
    load = f_cognitive_load(state, "Hello, here are your clusters.")
    assert isinstance(load, float)
    assert 0.0 <= load <= 1.0, f"load={load} out of [0, 1]"


# ── Test 2: high-stress state exceeds COG_LOAD_THRESHOLD ─────────────────────

def test_load_above_threshold():
    """State with MAX_K clusters and MAX_MSG_LEN message must exceed COG_LOAD_THRESHOLD."""
    from src.cognitive_load import (
        COG_LOAD_THRESHOLD,
        MAX_K,
        MAX_MSG_LEN,
        TOP_K_ITEMS_PER_CLUSTER,
        f_cognitive_load,
    )

    # Use MAX_K clusters with enough items so items_shown / total_items term is also high
    n_items = MAX_K * TOP_K_ITEMS_PER_CLUSTER  # exactly saturates items_term
    state = _make_state(n_clusters=MAX_K, n_items=n_items)
    message = "x" * MAX_MSG_LEN  # saturates message term
    load = f_cognitive_load(state, message)
    assert load > COG_LOAD_THRESHOLD, (
        f"Expected load > {COG_LOAD_THRESHOLD} for MAX_K clusters + MAX_MSG_LEN message, got {load}"
    )


# ── Test 3: minimal state produces low (but valid) load ───────────────────────

def test_load_zero_for_minimal_state():
    """Single cluster, 1 item, empty message → lowest possible load (should be low, in [0,1])."""
    from src.cognitive_load import MAX_K, MAX_MSG_LEN, TOP_K_ITEMS_PER_CLUSTER, f_cognitive_load

    state = _make_state(n_clusters=1, n_items=1)
    load = f_cognitive_load(state, "")
    assert 0.0 <= load <= 1.0

    # Verify the individual term contributions are small
    w = 1.0 / 3.0
    expected_cluster_term = (1 / MAX_K) * w
    expected_items_term = min((1 * TOP_K_ITEMS_PER_CLUSTER) / 1, 1.0) * w  # clamped to 1.0 → 1/3
    expected_msg_term = (0 / MAX_MSG_LEN) * w
    expected = expected_cluster_term + expected_items_term + expected_msg_term
    assert abs(load - expected) < 1e-9, f"load={load} expected={expected}"


# ── Test 4: AssertionError on empty state ─────────────────────────────────────

def test_f_cognitive_load_crashes_on_empty_state():
    """f_cognitive_load must raise AssertionError on a state with no clusters or no items."""
    from src.cognitive_load import f_cognitive_load

    # No clusters
    empty_no_clusters = ClusteringState(
        turn_index=0,
        timestamp="2026-05-12T00:00:00",
        clusters=[],
        assignments={},
        soft_probs={},
    )
    with pytest.raises(AssertionError, match="no clusters"):
        f_cognitive_load(empty_no_clusters, "some message")

    # Has cluster but no items assigned
    empty_no_items = ClusteringState(
        turn_index=0,
        timestamp="2026-05-12T00:00:00",
        clusters=[Cluster(id=0, name="C", description=".", item_ids=[])],
        assignments={},  # empty — no items
        soft_probs={},
    )
    with pytest.raises(AssertionError, match="no items"):
        f_cognitive_load(empty_no_items, "some message")


# ── Test 5: pure function — same inputs → same output ─────────────────────────

def test_f_cognitive_load_pure():
    """f_cognitive_load called twice with same inputs must return identical results."""
    from src.cognitive_load import f_cognitive_load

    state = _make_state(n_clusters=5, n_items=50)
    message = "Here are 5 clusters with their top items."

    result1 = f_cognitive_load(state, message)
    result2 = f_cognitive_load(state, message)
    assert result1 == result2, f"Non-deterministic: {result1} != {result2}"
