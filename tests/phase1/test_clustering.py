"""Tests for clustering.py — FOUND-02 (initial clustering) and FOUND-03 (soft probs)."""
import numpy as np
import pytest

from src.clustering import run_hdbscan, assign_noise_to_nearest


def test_run_hdbscan_returns_labels_and_soft_probs(mock_embeddings):
    """run_hdbscan returns (labels, soft_probs) tuple on valid input."""
    # Use tiny synthetic data with clear clusters to avoid all-noise result
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0.0] * 10, scale=0.1, size=(30, 10)).astype(np.float32)
    cluster_b = rng.normal(loc=[5.0] * 10, scale=0.1, size=(30, 10)).astype(np.float32)
    embeddings = np.vstack([cluster_a, cluster_b])
    labels, soft_probs = run_hdbscan(embeddings)
    assert labels.shape == (60,)
    assert soft_probs.ndim == 2
    assert soft_probs.shape[0] == 60


def test_soft_probs_shape_matches_n(mock_embeddings):
    """soft_probs has exactly N rows (one per item)."""
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=[0.0] * 10, scale=0.05, size=(40, 10)).astype(np.float32)
    cluster_b = rng.normal(loc=[8.0] * 10, scale=0.05, size=(40, 10)).astype(np.float32)
    embeddings = np.vstack([cluster_a, cluster_b])
    labels, soft_probs = run_hdbscan(embeddings)
    assert soft_probs.shape[0] == len(embeddings)


def test_soft_probs_rows_sum_to_one():
    """Each row of soft_probs sums to approximately 1.0."""
    rng = np.random.default_rng(2)
    cluster_a = rng.normal(loc=[0.0] * 10, scale=0.05, size=(40, 10)).astype(np.float32)
    cluster_b = rng.normal(loc=[8.0] * 10, scale=0.05, size=(40, 10)).astype(np.float32)
    embeddings = np.vstack([cluster_a, cluster_b])
    labels, soft_probs = run_hdbscan(embeddings)
    row_sums = soft_probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(embeddings)), atol=1e-5)


def test_assign_noise_to_nearest_no_minus_one(mock_hdbscan_output):
    """assign_noise_to_nearest returns assignments with no -1 values."""
    labels, soft_probs = mock_hdbscan_output
    # Inject one noise point
    labels_with_noise = labels.copy()
    labels_with_noise[2] = -1
    assignments = assign_noise_to_nearest(labels_with_noise, soft_probs)
    assert -1 not in assignments.values(), "No item should be assigned to cluster -1"


def test_assign_noise_to_nearest_all_items_present(mock_hdbscan_output):
    """assign_noise_to_nearest returns one assignment per item (N items total)."""
    labels, soft_probs = mock_hdbscan_output
    assignments = assign_noise_to_nearest(labels, soft_probs)
    assert len(assignments) == len(labels)
    assert set(assignments.keys()) == set(range(len(labels)))


def test_all_items_assigned(mock_hdbscan_output):
    """Every item_id from 0 to N-1 has an assignment."""
    labels, soft_probs = mock_hdbscan_output
    assignments = assign_noise_to_nearest(labels, soft_probs)
    for i in range(len(labels)):
        assert i in assignments, f"item_id {i} missing from assignments"
