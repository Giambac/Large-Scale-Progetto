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
import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.state import Cluster, ClusteringState

if TYPE_CHECKING:
    from src.cluster_naming import ClusterNamer


# HDBSCAN hyperparameters — documented as named constants, not magic numbers.
# Starting values for ~15K 768-dim review embeddings.
# Adjust if run_hdbscan produces 0 or 100+ clusters.
MIN_CLUSTER_SIZE = 50    # ~0.3% of 15K; typical for dense text embedding spaces
MIN_SAMPLES = 10         # noise sensitivity; lower = fewer noise points

# Temperature for KMeans soft probability computation (D-20).
# softmax(-dist / T): T=1.0 means raw L2 distances, no scaling.
# Named constant — not a CLI parameter.
KMEANS_SOFTMAX_TEMP = 1.0


@runtime_checkable
class ClusteringBackend(Protocol):
    """
    Protocol for clustering backends (D-16).

    Any backend must implement fit() with this exact signature.
    HDBSCANBackend and KMeansBackend both satisfy this Protocol.
    Adding future backends (LLM-first, etc.) requires only implementing fit().
    """

    def fit(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the backend to embeddings and return (labels, soft_probs).

        Args:
            embeddings: shape (N, dim) float32 array

        Returns:
            labels: shape (N,) int array. No -1 noise labels — every item assigned.
            soft_probs: shape (N, K) float32 array. Rows sum to 1.0.
        """
        ...


class HDBSCANBackend:
    """
    HDBSCAN clustering backend implementing ClusteringBackend Protocol (D-16).

    Wraps run_hdbscan() and assign_noise_to_nearest() into the fit() interface.
    Noise points (-1) are reassigned to argmax(soft_probs) to guarantee no -1 labels.
    """

    def __init__(
        self,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
    ) -> None:
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples

    def fit(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit HDBSCAN and return (labels, soft_probs).
        Noise labels (-1) are resolved to the nearest cluster via argmax(soft_probs).
        """
        labels, soft_probs = run_hdbscan(
            embeddings,
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
        )
        # Resolve noise points so labels has no -1 values
        assignments = assign_noise_to_nearest(labels, soft_probs)
        resolved_labels = np.array(
            [assignments[i] for i in range(len(labels))], dtype=np.intp
        )
        assert np.all(resolved_labels >= 0), "BUG: HDBSCANBackend.fit produced -1 labels"
        return resolved_labels, soft_probs


class KMeansBackend:
    """
    K-means clustering backend implementing ClusteringBackend Protocol (D-16).

    K is selected once at first fit() call via BIC on Gaussian Mixture Models for K=2..sqrt(N)
    (D-17). K is then FIXED — changes only through oracle intent (no auto-reoptimization).

    Soft probabilities use softmax of negative centroid distances (D-19):
        soft_probs[i][c] = softmax(-dist(item_i, centroid_c) / KMEANS_SOFTMAX_TEMP)
    Temperature T = KMEANS_SOFTMAX_TEMP = 1.0 (D-20).
    """

    def __init__(self) -> None:
        # K is set lazily on first fit() call (we need N to compute sqrt(N)).
        # Can be overridden directly for testing: backend._k = 3
        self._k: int | None = None

    @property
    def k(self) -> int | None:
        """The chosen K. None until fit() is called for the first time."""
        return self._k

    def fit(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit k-means on embeddings. Selects K via BIC if not yet set.

        Returns:
            labels: shape (N,) int array, values in [0, K). No -1 values.
            soft_probs: shape (N, K) float32 array. Rows sum to 1.0.
        """
        assert embeddings.ndim == 2, f"Expected 2D array, got shape {embeddings.shape}"
        N = embeddings.shape[0]
        assert N >= 2, f"KMeansBackend.fit: need at least 2 points, got {N}"

        if self._k is None:
            self._k = self._select_k_via_bic(embeddings)

        assert self._k >= 2, f"KMeansBackend: k must be >= 2, got {self._k}"
        assert self._k <= N, f"KMeansBackend: k={self._k} exceeds N={N}"

        km = KMeans(
            n_clusters=self._k,
            init="k-means++",
            n_init=10,
            random_state=0,
        )
        km.fit(embeddings)
        labels = km.labels_.astype(np.intp)
        assert labels.shape == (N,), f"KMeans labels shape mismatch: {labels.shape}"
        assert np.all(labels >= 0), "BUG: KMeans produced negative labels"

        soft_probs = self._compute_soft_probs(
            embeddings, km.cluster_centers_, KMEANS_SOFTMAX_TEMP
        )
        assert soft_probs.shape == (N, self._k), (
            f"soft_probs shape {soft_probs.shape} != ({N}, {self._k})"
        )
        assert np.allclose(soft_probs.sum(axis=1), 1.0, atol=1e-4), (
            "KMeansBackend.fit: soft_probs rows do not sum to 1.0"
        )
        return labels, soft_probs

    def _select_k_via_bic(self, embeddings: np.ndarray) -> int:
        """
        Fit GMM for K=2..int(sqrt(N)) and return K that minimizes BIC (D-17).

        For N=12000 this is K=2..109. Each GMM fit uses 'diag' covariance for speed
        (full covariance on 768-dim data would be prohibitively slow).
        K is logged externally by the caller (web/app.py) for AuditLog reproducibility.
        """
        N = embeddings.shape[0]
        k_max = max(2, int(math.sqrt(N)))
        best_k = 2
        best_bic = float("inf")
        for k in range(2, k_max + 1):
            gm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=0,
                max_iter=50,      # limit iterations for speed
                n_init=1,
            )
            gm.fit(embeddings)
            bic = gm.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        assert best_k >= 2, f"BIC selected k={best_k} < 2 — impossible"
        return best_k

    @staticmethod
    def _compute_soft_probs(
        embeddings: np.ndarray,
        centroids: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """
        Compute soft assignment probabilities via softmax of negative centroid distances (D-19).

        soft_probs[i][c] = softmax(-dist(item_i, centroid_c) / temperature)

        Uses L2 distance. Numerically stable via max-subtraction before exp.

        Args:
            embeddings: shape (N, dim)
            centroids:  shape (K, dim)
            temperature: float > 0 (KMEANS_SOFTMAX_TEMP = 1.0)

        Returns:
            soft_probs: shape (N, K) float32, rows sum to 1.0
        """
        assert temperature > 0, f"temperature must be positive, got {temperature}"
        # Compute pairwise L2 distances: shape (N, K)
        # dist[i,c] = ||embedding[i] - centroid[c]||_2
        diff = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, K, dim)
        distances = np.sqrt((diff ** 2).sum(axis=2))  # (N, K)

        # Softmax of negative distances scaled by temperature
        logits = -distances / temperature  # (N, K)
        # Numerically stable: subtract row max before exp
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        soft_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return soft_probs.astype(np.float32)


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
    import hdbscan  # lazy import — hdbscan is optional; fails loudly here if not installed
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
    backend: "ClusteringBackend | None" = None,
) -> ClusteringState:
    """
    Full pipeline: embeddings → ClusteringState at turn_index=0.

    1. Run clustering backend (default: HDBSCANBackend) → (labels, soft_probs_matrix)
    2. Group item_ids by cluster_id
    3. LLM-name each cluster using sample texts
    4. Assemble ClusteringState

    Args:
        embeddings: shape (N, dim) from EmbeddingStore.get_all()
        records: list of {"item_id": int, "text": str} dicts (all N items)
        namer: ClusterNamer instance for LLM cluster naming
        min_cluster_size: passed to HDBSCANBackend when backend is None (default HDBSCAN path)
        backend: ClusteringBackend instance. Defaults to HDBSCANBackend when None.

    Returns:
        ClusteringState at turn_index=0
    """
    assert len(records) == len(embeddings), (
        f"Record count {len(records)} != embedding count {len(embeddings)}"
    )

    if backend is None:
        backend = HDBSCANBackend(min_cluster_size=min_cluster_size)
    labels, soft_probs_matrix = backend.fit(embeddings)
    # backend.fit() guarantees no -1 labels — skip assign_noise_to_nearest
    assignments: dict[int, int] = {i: int(labels[i]) for i in range(len(labels))}
    assert -1 not in assignments.values(), "BUG: backend.fit returned -1 labels"

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
