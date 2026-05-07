"""
uncertainty.py — f_uncertainty pure function + UncertaintyReport dataclass (CLUS-02).

Computes normalized Shannon entropy per item and aggregates into ranked lists for
boundary detection (split candidates) and proximity detection (merge candidates).
No I/O. No global state. Pure function only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.state import ClusteringState


@dataclass
class UncertaintyReport:
    """
    Output of f_uncertainty. Three ranked views on assignment uncertainty.

    boundary_items:    (item_id, normalized_entropy) pairs, descending by entropy.
    split_candidates:  (cluster_id, mean_entropy) pairs, descending — most uncertain cluster first.
    merge_candidates:  (cluster_a_id, cluster_b_id, distance) triples, ascending — closest pair first.
    """
    boundary_items: list[tuple[int, float]]        # (item_id, normalized_entropy), descending
    split_candidates: list[tuple[int, float]]      # (cluster_id, mean_entropy), descending
    merge_candidates: list[tuple[int, int, float]] # (cluster_a_id, cluster_b_id, distance), ascending


def f_uncertainty(state: ClusteringState) -> UncertaintyReport:
    """
    Pure function. No I/O. No global state.

    Computes normalized Shannon entropy for every item in state.soft_probs,
    then assembles three ranked views:
      - boundary_items: all items sorted by entropy (descending)
      - split_candidates: clusters sorted by mean item entropy (descending)
      - merge_candidates: cluster pairs sorted by soft_probs centroid distance (ascending)
    """
    K = len(state.clusters)
    assert K > 0, "f_uncertainty called on empty ClusteringState (no clusters)"
    log_K = np.log(K) if K > 1 else 1.0  # avoid div-by-zero for K=1

    # Step 2: Per-item normalized Shannon entropy
    item_entropy: dict[int, float] = {}
    for item_id, probs in state.soft_probs.items():
        p = np.array(probs, dtype=np.float64)
        mask = p > 0
        h = -np.sum(p[mask] * np.log(p[mask]))
        item_entropy[item_id] = float(h / log_K)  # normalized to [0, 1]

    # Step 3: boundary_items — all items sorted by entropy descending
    boundary_items = sorted(item_entropy.items(), key=lambda x: x[1], reverse=True)

    # Step 4: split_candidates — per-cluster mean entropy, sorted descending
    cluster_entropy: dict[int, float] = {}
    for cluster in state.clusters:
        if cluster.item_ids:
            cluster_entropy[cluster.id] = float(
                np.mean([item_entropy[i] for i in cluster.item_ids])
            )
    split_candidates = sorted(cluster_entropy.items(), key=lambda x: x[1], reverse=True)

    # Step 5: merge_candidates — cluster pairs ranked by soft_probs centroid euclidean distance
    cluster_centroids: dict[int, np.ndarray] = {}
    for cluster in state.clusters:
        assert len(cluster.item_ids) > 0, (
            f"f_uncertainty: cluster {cluster.id} has no items — "
            "empty clusters must be removed before f_uncertainty is called"
        )
        vecs = np.array([state.soft_probs[i] for i in cluster.item_ids])
        cluster_centroids[cluster.id] = vecs.mean(axis=0)

    merge_candidates = []
    cluster_ids = [c.id for c in state.clusters]
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            a, b = cluster_ids[i], cluster_ids[j]
            dist = float(np.linalg.norm(cluster_centroids[a] - cluster_centroids[b]))
            merge_candidates.append((a, b, dist))
    merge_candidates.sort(key=lambda x: x[2])  # ascending: closest pair first

    return UncertaintyReport(
        boundary_items=boundary_items,
        split_candidates=split_candidates,
        merge_candidates=merge_candidates,
    )
