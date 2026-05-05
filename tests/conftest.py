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
