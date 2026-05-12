"""Unit tests for f_cognitive_load pure function."""
import pytest
from src.state import Cluster, ClusteringState


def _make_state(n_clusters: int, n_items: int) -> ClusteringState:
    """Build a minimal ClusteringState with n_clusters clusters and n_items items."""
    items_per = max(1, n_items // n_clusters)
    clusters = []
    assignments = {}
    soft_probs = {}
    item_id = 0
    for cid in range(n_clusters):
        batch = list(range(item_id, item_id + items_per))
        clusters.append(Cluster(id=cid, name=f"C{cid}", description=".", item_ids=batch))
        for i in batch:
            assignments[i] = cid
            soft_probs[i] = [1.0 / n_clusters] * n_clusters
        item_id += items_per
    return ClusteringState(
        turn_index=0, timestamp="2026-05-11T00:00:00",
        clusters=clusters, assignments=assignments, soft_probs=soft_probs,
    )


def test_load_in_range(tiny_state_3cluster):
    """f_cognitive_load returns float in [0.0, 1.0]."""
    from src.cognitive_load import f_cognitive_load
    load = f_cognitive_load(tiny_state_3cluster, "show clusters")
    assert isinstance(load, float), f"Expected float, got {type(load)}"
    assert 0.0 <= load <= 1.0, f"load {load} outside [0, 1]"


def test_load_above_threshold(tiny_state_3cluster):
    """f_cognitive_load > COG_LOAD_THRESHOLD for max-stress state (20 clusters, long message)."""
    from src.cognitive_load import f_cognitive_load, COG_LOAD_THRESHOLD, MAX_MSG_LEN
    # Max-stress: 20 clusters (at MAX_K=20 cap), 100 items, max-length message
    state = _make_state(20, 100)
    message = "x" * MAX_MSG_LEN
    load = f_cognitive_load(state, message)
    assert load > COG_LOAD_THRESHOLD, \
        f"Expected load > {COG_LOAD_THRESHOLD} for max-stress state, got {load}"


def test_load_zero_for_minimal_state():
    """Single-cluster state with 1 item and short message yields load < COG_LOAD_THRESHOLD."""
    from src.cognitive_load import f_cognitive_load, COG_LOAD_THRESHOLD
    state = _make_state(1, 1)
    load = f_cognitive_load(state, "hi")
    assert load < COG_LOAD_THRESHOLD, \
        f"Expected load < {COG_LOAD_THRESHOLD} for minimal state, got {load}"


def test_f_cognitive_load_crashes_on_empty_state():
    """f_cognitive_load raises AssertionError for empty ClusteringState."""
    from src.cognitive_load import f_cognitive_load
    from src.state import ClusteringState
    empty = ClusteringState(
        turn_index=0, timestamp="2026-05-11T00:00:00",
        clusters=[], assignments={}, soft_probs={},
    )
    with pytest.raises(AssertionError):
        f_cognitive_load(empty, "any message")


def test_f_cognitive_load_pure(tiny_state_3cluster):
    """f_cognitive_load is a pure function — calling twice yields same result."""
    from src.cognitive_load import f_cognitive_load
    load1 = f_cognitive_load(tiny_state_3cluster, "show clusters")
    load2 = f_cognitive_load(tiny_state_3cluster, "show clusters")
    assert load1 == load2, f"f_cognitive_load not pure: got {load1} then {load2}"
