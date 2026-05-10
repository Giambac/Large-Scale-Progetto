"""Tests for VIZ-V2-01: UMAP projection computation and SocketIO event payload."""
from __future__ import annotations
import numpy as np
import pytest

pytest.importorskip("umap", reason="umap-learn not installed — pip install umap-learn to run projection tests")


@pytest.fixture
def tiny_embeddings():
    """12 points in 2 clusters (768-dim). Fast UMAP fit for CI."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 768)) * 0.1 + np.array([5.0] + [0.0] * 767)
    b = rng.standard_normal((6, 768)) * 0.1 - np.array([5.0] + [0.0] * 767)
    return np.vstack([a, b]).astype(np.float32)


@pytest.fixture
def fake_state_k2():
    """ClusteringState with 12 items in 2 clusters."""
    from src.state import Cluster, ClusteringState
    clusters = [
        Cluster(id=0, name="A", description="", item_ids=list(range(6))),
        Cluster(id=1, name="B", description="", item_ids=list(range(6, 12))),
    ]
    assignments = {i: 0 for i in range(6)}
    assignments.update({i: 1 for i in range(6, 12)})
    soft_probs = {i: [0.9, 0.1] if i < 6 else [0.1, 0.9] for i in range(12)}
    return ClusteringState(
        turn_index=0, timestamp="2026-01-01T00:00:00+00:00",
        clusters=clusters, assignments=assignments, soft_probs=soft_probs,
    )


def test_compute_and_emit_projection_importable():
    """compute_and_emit_projection can be imported from web.app — no ImportError."""
    from web.app import compute_and_emit_projection  # noqa: F401


def test_compute_projection_coords_shape(tiny_embeddings):
    """_compute_projection returns shape (N, 2) — one 2D point per item."""
    from web.app import _compute_projection
    coords = _compute_projection(tiny_embeddings)
    assert coords.shape == (12, 2)
    assert coords.dtype in (np.float32, np.float64)


def test_compute_projection_reproducible(tiny_embeddings):
    """_compute_projection is reproducible across calls (fixed random_state=42)."""
    from web.app import _compute_projection
    coords_a = _compute_projection(tiny_embeddings)
    coords_b = _compute_projection(tiny_embeddings)
    assert np.allclose(coords_a, coords_b, atol=1e-4)


def test_projection_payload_structure(fake_state_k2):
    """_build_projection_payload returns dict with required keys and correct lengths."""
    from web.app import _build_projection_payload
    coords = np.array([[1.0, 2.0]] * 12, dtype=np.float32)
    payload = _build_projection_payload(coords, fake_state_k2)
    assert "coords" in payload
    assert "cluster_ids" in payload
    assert "max_probs" in payload
    assert "cluster_colors" in payload
    assert len(payload["coords"]) == 12
    assert len(payload["cluster_ids"]) == 12
    assert len(payload["max_probs"]) == 12
    assert len(payload["cluster_colors"]) == 2
    assert all(
        isinstance(c, str) and c.startswith("#")
        for c in payload["cluster_colors"].values()
    )


def test_projection_cluster_ids_match_state(fake_state_k2):
    """cluster_ids in payload match state assignments ordered by item_id."""
    from web.app import _build_projection_payload
    coords = np.array([[1.0, 2.0]] * 12, dtype=np.float32)
    payload = _build_projection_payload(coords, fake_state_k2)
    assert payload["cluster_ids"][:6] == [0] * 6
    assert payload["cluster_ids"][6:] == [1] * 6


def test_projection_max_probs_range(fake_state_k2):
    """All max_probs are in [0.0, 1.0]."""
    from web.app import _build_projection_payload
    coords = np.array([[1.0, 2.0]] * 12, dtype=np.float32)
    payload = _build_projection_payload(coords, fake_state_k2)
    assert all(0.0 <= p <= 1.0 for p in payload["max_probs"])


def test_should_recompute_projection_split():
    """_should_recompute_projection returns True for SplitFeedback."""
    from web.app import _should_recompute_projection
    from src.feedback import SplitFeedback
    assert _should_recompute_projection([SplitFeedback(cluster_id=0, seed_item_ids=[])]) is True


def test_should_recompute_projection_merge():
    """_should_recompute_projection returns True for MergeFeedback."""
    from web.app import _should_recompute_projection
    from src.feedback import MergeFeedback
    assert _should_recompute_projection([MergeFeedback(cluster_a_id=0, cluster_b_id=1)]) is True


def test_should_not_recompute_projection_move():
    """_should_recompute_projection returns False for MoveItemFeedback."""
    from web.app import _should_recompute_projection
    from src.feedback import MoveItemFeedback
    assert _should_recompute_projection([MoveItemFeedback(item_id=0, target_cluster_id=1)]) is False


def test_should_not_recompute_projection_global():
    """_should_recompute_projection returns False for GlobalFeedback."""
    from web.app import _should_recompute_projection
    from src.feedback import GlobalFeedback
    assert _should_recompute_projection([GlobalFeedback(instruction_text="test")]) is False
