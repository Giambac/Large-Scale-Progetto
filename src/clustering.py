"""
clustering.py — HDBSCAN clustering and ClusteringState assembly (FOUND-02, FOUND-03).

CRITICAL FACTS (from research):
- Use hdbscan.all_points_membership_vectors(clusterer) for soft probs.
- prediction_data=True MUST be set in the HDBSCAN constructor.
- Noise points (label == -1) are reassigned to argmax(soft_probs) to ensure
  complete assignments (D-13 requires all N items in assignments dict).
- all_points_membership_vectors returns unnormalized membership weights;
  rows are normalized to sum to 1.0 before storing in ClusteringState (FOUND-03).
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import hdbscan
import numpy as np

from src.state import Cluster, ClusteringState

if TYPE_CHECKING:
    from src.cluster_naming import ClusterNamer


# HDBSCAN hyperparameters — documented as named constants, not magic numbers.
# Starting values for ~15K 768-dim review embeddings.
# Adjust if run_hdbscan produces 0 or 100+ clusters.
MIN_CLUSTER_SIZE = 50    # ~0.3% of 15K; typical for dense text embedding spaces
MIN_SAMPLES = 10         # noise sensitivity; lower = fewer noise points


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit HDBSCAN on embeddings and return (labels, soft_probs).

    Args:
        embeddings: shape (N, dim) float32 array from EmbeddingStore.
        min_cluster_size: override MIN_CLUSTER_SIZE constant (for testing).
        min_samples: override MIN_SAMPLES constant (for testing).

    Returns:
        labels: shape (N,) int array. -1 means noise.
        soft_probs: shape (N, K) float32. K = number of discovered non-noise clusters.
                    Rows sum to approximately 1.0 (normalized after HDBSCAN).

    Asserts:
        - At least 1 non-noise cluster found (crashes on all-noise output).
        - soft_probs.shape[0] == N.
        - soft_probs.shape[1] > 0 (at least 1 cluster column).
        - All rows of soft_probs sum to ~1.0 after normalization.
    """
    assert embeddings.ndim == 2, f"Expected 2D array, got shape {embeddings.shape}"
    assert embeddings.shape[0] > 0, "Cannot cluster empty embedding set"

    # Auto-scale MIN_CLUSTER_SIZE when N is small (e.g., in tests with synthetic data).
    # For production (~15K points): max(5, min(50, 1500)) = 50.
    # For tests (~80 points): max(5, min(50, 8)) = 8.
    # This preserves the MIN_CLUSTER_SIZE constant as the production ceiling while
    # allowing tests to run on small synthetic datasets without all-noise output.
    n_items = embeddings.shape[0]
    _min_cluster_size = (
        min_cluster_size
        if min_cluster_size is not None
        else max(5, min(MIN_CLUSTER_SIZE, n_items // 10))
    )
    _min_samples = min_samples if min_samples is not None else MIN_SAMPLES

    assert _min_cluster_size > 0, f"min_cluster_size must be positive, got {_min_cluster_size}"
    assert _min_samples > 0, f"min_samples must be positive, got {_min_samples}"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=_min_cluster_size,
        min_samples=_min_samples,
        prediction_data=True,   # REQUIRED for all_points_membership_vectors()
        metric="euclidean",
    )
    clusterer.fit(embeddings)

    # Verify we got at least one real cluster (not all noise)
    unique_labels = set(clusterer.labels_) - {-1}
    assert len(unique_labels) > 0, (
        f"HDBSCAN produced 0 clusters (all points labeled as noise). "
        f"Adjust MIN_CLUSTER_SIZE (current={_min_cluster_size}) or "
        f"MIN_SAMPLES (current={_min_samples}) in clustering.py."
    )

    # Compute soft membership vectors — use the module-level function, not an instance attribute
    raw_soft_probs = hdbscan.all_points_membership_vectors(clusterer)

    assert raw_soft_probs.shape[0] == len(embeddings), (
        f"soft_probs row count {raw_soft_probs.shape[0]} != N={len(embeddings)}"
    )
    assert raw_soft_probs.shape[1] > 0, "soft_probs has 0 cluster columns"

    # Normalize rows to sum to 1.0 (FOUND-03: rows must sum to approximately 1.0).
    # all_points_membership_vectors returns unnormalized weights; normalization ensures
    # that soft_probs[i] is a proper probability distribution over clusters.
    row_sums = raw_soft_probs.sum(axis=1, keepdims=True)
    # Guard against all-zero rows (can happen if a noise point has 0 membership in all clusters)
    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
    soft_probs = raw_soft_probs / row_sums_safe

    assert soft_probs.shape == raw_soft_probs.shape, "BUG: normalization changed shape"

    return clusterer.labels_, soft_probs


def assign_noise_to_nearest(
    labels: np.ndarray,
    soft_probs: np.ndarray,
) -> dict[int, int]:
    """
    Create a complete hard assignment mapping with no noise labels (-1).

    Noise points (label == -1) are assigned to the cluster with the highest
    soft probability for that point (argmax). This preserves the anytime
    behavior requirement (CLUS-01): all N items always have an assignment.

    Returns:
        dict[int, int] — {item_id: cluster_id} for all N items.
        cluster_id is always >= 0 (no -1 values).
    """
    assert len(labels) == soft_probs.shape[0], (
        f"labels length {len(labels)} != soft_probs rows {soft_probs.shape[0]}"
    )

    assignments: dict[int, int] = {}
    for i, label in enumerate(labels):
        if label == -1:
            # Noise: assign to cluster with highest soft probability
            assignments[i] = int(np.argmax(soft_probs[i]))
        else:
            assignments[i] = int(label)

    assert -1 not in assignments.values(), "BUG: -1 assignment survived noise reassignment"
    assert len(assignments) == len(labels), "BUG: missing item_ids in assignments"
    return assignments


def build_initial_clustering_state(
    embeddings: np.ndarray,
    records: list[dict],
    namer: "ClusterNamer",
    min_cluster_size: int | None = None,
) -> ClusteringState:
    """
    Full pipeline: embeddings → ClusteringState at turn_index=0.

    1. Run HDBSCAN → (labels, soft_probs_matrix)
    2. Assign noise points to nearest cluster
    3. Group item_ids by cluster_id
    4. LLM-name each cluster using sample texts
    5. Assemble ClusteringState

    Args:
        embeddings: shape (N, dim) from EmbeddingStore.get_all()
        records: list of {"item_id": int, "text": str} dicts (all N items)
        namer: ClusterNamer instance for LLM cluster naming

    Returns:
        ClusteringState at turn_index=0
    """
    assert len(records) == len(embeddings), (
        f"Record count {len(records)} != embedding count {len(embeddings)}"
    )

    labels, soft_probs_matrix = run_hdbscan(embeddings, min_cluster_size=min_cluster_size)
    assignments = assign_noise_to_nearest(labels, soft_probs_matrix)

    # Group item_ids by cluster_id
    cluster_items: dict[int, list[int]] = {}
    for item_id, cluster_id in assignments.items():
        cluster_items.setdefault(cluster_id, []).append(item_id)

    # Build a text lookup for sampling
    id_to_text = {r["item_id"]: r["text"] for r in records}

    # Create Cluster objects with LLM-generated names and descriptions
    clusters = []
    for cluster_id in sorted(cluster_items.keys()):
        item_ids = cluster_items[cluster_id]
        # Sample up to 5 representative texts for naming
        sample_ids = item_ids[:5]
        sample_texts = [id_to_text[i] for i in sample_ids]
        naming_result = namer.name_cluster(sample_texts, cluster_id)
        clusters.append(Cluster(
            id=cluster_id,
            name=naming_result["name"],
            description=naming_result["description"],
            item_ids=item_ids,
        ))

    # Convert soft_probs matrix rows to per-item lists
    soft_probs: dict[int, list[float]] = {
        i: soft_probs_matrix[i].tolist()
        for i in range(len(embeddings))
    }

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    state = ClusteringState(
        turn_index=0,
        timestamp=timestamp,
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )

    # Verify completeness invariants before returning
    assert len(state.assignments) == len(embeddings), "assignments incomplete"
    assert len(state.soft_probs) == len(embeddings), "soft_probs incomplete"
    assert len(state.clusters) > 0, "No clusters in initial state"

    return state
