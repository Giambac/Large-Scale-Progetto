"""Tests for BACK-V2-01: ClusteringBackend Protocol, HDBSCANBackend, KMeansBackend."""
from __future__ import annotations
import math
import numpy as np
import pytest


@pytest.fixture
def small_embeddings():
    """60 points in 3 tight clusters (768-dim). Fast enough for BIC loop."""
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((3, 768)) * 5.0
    parts = [centers[i] + rng.standard_normal((20, 768)) * 0.3 for i in range(3)]
    return np.vstack(parts).astype(np.float32)


def test_clustering_backend_protocol_importable():
    """ClusteringBackend can be imported from src.clustering without error."""
    from src.clustering import ClusteringBackend  # noqa: F401


def test_hdbscan_backend_fit_returns_labels_and_soft_probs(small_embeddings):
    """HDBSCANBackend.fit() returns (labels, soft_probs) with correct shapes and normalization."""
    from src.clustering import HDBSCANBackend
    backend = HDBSCANBackend(min_cluster_size=5, min_samples=3)
    labels, soft_probs = backend.fit(small_embeddings)
    assert labels.shape == (60,)
    assert soft_probs.shape[0] == 60
    assert soft_probs.shape[1] > 0
    assert np.allclose(soft_probs.sum(axis=1), 1.0, atol=1e-4)


def test_kmeans_backend_bic_selects_k(small_embeddings):
    """KMeansBackend() selects K via BIC; K is within the valid range [2, sqrt(N)+1]."""
    from src.clustering import KMeansBackend
    backend = KMeansBackend()
    backend.fit(small_embeddings)  # K is set during fit (lazy init)
    assert 2 <= backend.k <= int(math.sqrt(60)) + 1


def test_kmeans_backend_fit_returns_correct_shape(small_embeddings):
    """KMeansBackend.fit() returns (labels, soft_probs) with shape (N,) and (N, K)."""
    from src.clustering import KMeansBackend
    backend = KMeansBackend()
    backend._k = 3  # override to skip slow BIC in this test
    labels, soft_probs = backend.fit(small_embeddings)
    assert labels.shape == (60,)
    assert soft_probs.shape == (60, 3)
    assert np.all(labels >= 0), "K-means must not produce -1 noise labels"
    assert np.allclose(soft_probs.sum(axis=1), 1.0, atol=1e-4)


def test_kmeans_soft_probs_boundary_symmetry():
    """
    A point equidistant from two centroids should have soft_probs ~[0.5, 0.5].
    Tests _compute_soft_probs directly with a controlled geometry.
    """
    from src.clustering import KMeansBackend, KMEANS_SOFTMAX_TEMP
    # Two centroids at +1 and -1 on dim 0, zeros elsewhere (768-dim)
    centroids = np.zeros((2, 768), dtype=np.float32)
    centroids[0, 0] = 1.0
    centroids[1, 0] = -1.0
    # Test point at origin — equidistant from both centroids
    embeddings = np.zeros((1, 768), dtype=np.float32)
    soft_probs = KMeansBackend._compute_soft_probs(embeddings, centroids, KMEANS_SOFTMAX_TEMP)
    assert soft_probs.shape == (1, 2)
    assert abs(soft_probs[0, 0] - 0.5) < 0.01
    assert abs(soft_probs[0, 1] - 0.5) < 0.01


def test_kmeans_softmax_temp_constant():
    """KMEANS_SOFTMAX_TEMP must be exactly 1.0 (D-20)."""
    from src.clustering import KMEANS_SOFTMAX_TEMP
    assert KMEANS_SOFTMAX_TEMP == 1.0


def test_build_initial_clustering_state_with_kmeans_backend():
    """build_initial_clustering_state works with KMeansBackend as backend parameter."""
    from src.clustering import KMeansBackend, build_initial_clustering_state

    rng = np.random.default_rng(99)
    embeddings = rng.standard_normal((10, 768)).astype(np.float32)
    records = [{"item_id": i, "text": f"item text {i}"} for i in range(10)]

    class MockNamer:
        def name_cluster(self, texts, cluster_id):
            return {"name": "Test", "description": "desc"}

    namer = MockNamer()
    backend = KMeansBackend()
    backend._k = 2  # override to skip BIC for this integration test

    state = build_initial_clustering_state(embeddings, records, namer, backend=backend)
    assert len(state.assignments) == 10
    assert len(state.soft_probs) == 10
    assert len(state.clusters) >= 1


def test_build_initial_clustering_state_with_hdbscan_backend():
    """build_initial_clustering_state works with HDBSCANBackend as backend parameter."""
    from src.clustering import HDBSCANBackend, build_initial_clustering_state

    rng = np.random.default_rng(42)
    centers = rng.standard_normal((2, 768)) * 5.0
    parts = [centers[i] + rng.standard_normal((10, 768)) * 0.3 for i in range(2)]
    embeddings = np.vstack(parts).astype(np.float32)
    records = [{"item_id": i, "text": f"item text {i}"} for i in range(20)]

    class MockNamer:
        def name_cluster(self, texts, cluster_id):
            return {"name": "Test", "description": "desc"}

    namer = MockNamer()
    backend = HDBSCANBackend(min_cluster_size=3, min_samples=2)

    state = build_initial_clustering_state(embeddings, records, namer, backend=backend)
    assert len(state.assignments) == 20
    assert len(state.soft_probs) == 20
    assert len(state.clusters) >= 1
