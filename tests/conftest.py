"""Shared fixtures for Phase 1 tests."""
import pytest
import numpy as np


@pytest.fixture
def sample_texts():
    """Five short review-like texts for testing data loader and embedding."""
    return [
        "Great product, very happy with quality.",
        "Poor stitching, fell apart after two uses.",
        "Perfect for my crafts, highly recommended.",
        "Color was not as described, disappointed.",
        "Fast shipping, item exactly as shown.",
    ]


@pytest.fixture
def tiny_clustering_state_dict():
    """
    A minimal ClusteringState as a plain dict (pre-construction).
    Matches D-13 schema exactly. Two clusters, five items.
    """
    return {
        "turn_index": 0,
        "timestamp": "2026-05-04T12:00:00",
        "clusters": [
            {"id": 0, "name": "Positive Reviews", "description": "Happy customers.", "item_ids": [0, 2, 4]},
            {"id": 1, "name": "Negative Reviews", "description": "Unhappy customers.", "item_ids": [1, 3]},
        ],
        "assignments": {0: 0, 1: 1, 2: 0, 3: 1, 4: 0},
        "soft_probs": {
            0: [0.9, 0.1],
            1: [0.1, 0.9],
            2: [0.8, 0.2],
            3: [0.2, 0.8],
            4: [0.85, 0.15],
        },
    }


@pytest.fixture
def mock_hdbscan_output():
    """
    Synthetic HDBSCAN output for 5 items, 2 clusters.
    labels_: item 0,2,4 -> cluster 0; item 1,3 -> cluster 1. No noise.
    soft_probs: shape (5, 2), rows sum to 1.0.
    """
    labels = np.array([0, 1, 0, 1, 0])
    soft_probs = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.85, 0.15],
    ], dtype=np.float32)
    return labels, soft_probs


@pytest.fixture
def mock_embeddings():
    """
    Tiny embedding array: 5 items x 768 dims (float32).
    Values are deterministic (seeded) for reproducibility.
    """
    rng = np.random.default_rng(seed=42)
    return rng.random((5, 768)).astype(np.float32)


# ──────────────────────────────────────────
# Phase 2 fixtures
# ──────────────────────────────────────────

@pytest.fixture
def tiny_state_3cluster():
    """
    A 3-cluster ClusteringState with 6 items.
    Used by Phase 2 tests for split/merge/move/uncertainty operations.
    Cluster IDs: 0, 1, 2. Items: 0-5.
    soft_probs rows sum to 1.0.
    """
    from src.state import Cluster, ClusteringState
    return ClusteringState(
        turn_index=0,
        timestamp="2026-05-07T10:00:00",
        clusters=[
            Cluster(id=0, name="Alpha", description="Cluster alpha.", item_ids=[0, 1]),
            Cluster(id=1, name="Beta",  description="Cluster beta.",  item_ids=[2, 3]),
            Cluster(id=2, name="Gamma", description="Cluster gamma.", item_ids=[4, 5]),
        ],
        assignments={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2},
        soft_probs={
            0: [0.80, 0.10, 0.10],
            1: [0.70, 0.20, 0.10],
            2: [0.10, 0.80, 0.10],
            3: [0.15, 0.70, 0.15],
            4: [0.05, 0.05, 0.90],
            5: [0.10, 0.15, 0.75],
        },
    )


@pytest.fixture
def mock_embeddings_3cluster():
    """
    6 embeddings of dim 768 (float32), deterministic seed=7.
    Items 0-1 belong to cluster 0, items 2-3 to cluster 1, items 4-5 to cluster 2.
    Used by f_next_state split path (D-08) tests.
    """
    rng = np.random.default_rng(seed=7)
    return rng.random((6, 768)).astype(np.float32)


@pytest.fixture
def mock_oracle_factory():
    """
    Factory for MockOracle with a scripted OracleReply sequence.
    Usage: mock_oracle_factory([reply1, reply2, ...])
    After script exhausted, defaults to OracleReply(raw_text='', satisfied=False).
    Requires src/oracle_protocol.py to exist (will ImportError otherwise).
    """
    def _factory(replies):
        from src.oracle_protocol import MockOracle
        return MockOracle(script=replies)
    return _factory
