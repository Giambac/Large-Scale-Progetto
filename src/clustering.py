"""
clustering.py — Esegue il clustering sugli embeddings e produce il primo stato.

Questo file fa due cose:
    1. Esegue l'algoritmo di clustering (HDBSCAN o KMeans) sugli embeddings e produce le assegnazioni e le probabilità morbide.
    2. Assembla il ClusteringState iniziale al turno 0 — quello da cui parte tutta la conversazione.

Due backend disponibili, entrambi con la stessa interfaccia:
    - HDBSCANBackend : trova automaticamente quanti cluster esistono nei dati.
    - KMeansBackend : richiede di sapere quanti cluster creare, ma lo sceglie automaticamente via BIC.

La funzione principale è build_initial_clustering_state — prende gli embeddings, chiama il backend, nomina i cluster con l'LLM, e restituisce il ClusteringState
al turno 0 pronto per iniziare la conversazione.
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


# Parametri HDBSCAN per ~15K recensioni con embeddings da 768 dimensioni.
# Se il clustering produce 0 o troppi cluster, regolare questi valori.
MIN_CLUSTER_SIZE = 50    # circa 0.3% di 15K — soglia minima per formare un cluster
MIN_SAMPLES = 10         # sensibilità al rumore — valore più basso = meno punti rumore

# Temperatura per il calcolo delle probabilità morbide di KMeans.
# Con 1.0 le distanze dai centroidi vengono usate direttamente senza scalatura.
KMEANS_SOFTMAX_TEMP = 1.0

"""
class ClusteringBackend(Protocol):
    L'interfaccia che qualsiasi backend di clustering deve rispettare.

    Basta implementare fit(embeddings) che restituisce (labels, soft_probs).
    Non serve ereditare da questa classe.

    Garanzia: fit() non restituisce mai etichette -1 — ogni recensione ha sempre un cluster assegnato.
"""
@runtime_checkable
class ClusteringBackend(Protocol):
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

"""
class HDBSCANBackend:
    Backend che usa HDBSCAN — trova automaticamente quanti cluster esistono.

    Avvolge run_hdbscan() e assign_noise_to_nearest() nell'interfaccia fit().
    I punti rumore (etichetta -1) vengono sempre forzati nel cluster a cui assomigliano di più prima di restituire il risultato.
"""
class HDBSCANBackend:
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

"""
class KMeansBackend:
    Backend che usa KMeans — richiede K ma lo sceglie automaticamente.

    K viene scelto al primo fit() usando il BIC (Bayesian Information Criterion): prova K=2, K=3, ... fino a K=√N e sceglie quello che minimizza il BIC.
    Una volta scelto, K rimane fisso per tutta la sessione — cambia solo se l'oracle fa un'operazione di split o merge.

    Le probabilità morbide vengono calcolate con softmax sulle distanze dai centroidi: più una recensione è vicina a un centroide, più alta è la sua probabilità per quel cluster.
"""
class KMeansBackend:
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
        Prova GMM per K=2..√N e restituisce il K che minimizza il BIC.

        Usa covarianza diagonale per velocità — la covarianza piena su 768 dimensioni sarebbe troppo lenta. Massimo 50 iterazioni per K.
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

    """
    def _compute_soft_probs( )
        Calcola le probabilità morbide con softmax sulle distanze negative dai centroidi.

        Formula: soft_probs[i][c] = softmax(-distanza(recensione_i, centroide_c) / temperatura)

        Più una recensione è vicina a un centroide, più alta è la sua probabilità per quel cluster. Usa la sottrazione del massimo di riga per stabilità numerica
        (evita overflow nell'exp).
    """
    @staticmethod
    def _compute_soft_probs(
        embeddings: np.ndarray,
        centroids: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
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

"""
def run_hdbscan( )
    Esegue HDBSCAN sugli embeddings e restituisce (labels, soft_probs).

    HDBSCAN trova automaticamente quanti cluster esistono — non bisogna specificare K a priori. I punti che non appartengono chiaramente a
    nessun cluster ricevono l'etichetta -1 (rumore).

    Due dettagli tecnici importanti:
        - prediction_data=True DEVE essere impostato nel costruttore, altrimenti all_points_membership_vectors() non funziona.
        - Le righe di soft_probs vengono normalizzate a 1.0 perché HDBSCAN le restituisce non normalizzate.

    Auto-scaling di min_cluster_size: se il dataset è piccolo (es. nei test), il valore viene scalato automaticamente per evitare output tutto-rumore.
    Con 15K recensioni il valore rimane 50. Con 80 recensioni nei test diventa 8.
"""
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import hdbscan  # lazy import — hdbscan is optional; fails loudly here if not installed
    assert embeddings.ndim == 2, f"Expected 2D array, got shape {embeddings.shape}"
    assert embeddings.shape[0] > 0, "Cannot cluster empty embedding set"

    # Auto-scale MIN_CLUSTER_SIZE when N is small (e.g., in tests with synthetic data).
    # For production (~15K points): max(5, min(50, 1500)) = 50.
    # For tests (~80 points): max(5, min(50, 8)) = 8.
    # This preserves the MIN_CLUSTER_SIZE constant as the production ceiling while allowing tests to run on small synthetic datasets without all-noise output.
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

"""
def assign_noise_to_nearest(
    Risolve i punti rumore (-1) assegnandoli al cluster con probabilità più alta.

    HDBSCAN assegna -1 alle recensioni che non appartengono chiaramente a nessun cluster. Questa funzione li forza nel cluster a cui assomigliano 
    di più guardando l'argmax delle soft_probs.

    Restituisce un dizionario item_id -> cluster_id completo, senza -1.
"""
def assign_noise_to_nearest(
    labels: np.ndarray,
    soft_probs: np.ndarray,
) -> dict[int, int]:
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

"""
def build_initial_clustering_state( )
    Pipeline completa: embeddings → ClusteringState al turno 0.

    È il punto di ingresso del sistema — da qui parte tutta la conversazione.

    Quattro passi in sequenza:
        1. Esegue il backend di clustering (default: HDBSCANBackend) sugli embeddings.
        2. Raggruppa le recensioni per cluster_id.
        3. Chiama il namer per dare un nome e una descrizione a ogni cluster.
        4. Assembla il ClusteringState e verifica che sia completo.

    Se backend=None usa HDBSCANBackend. Passare un KMeansBackend usa KMeans.
    Il codice che chiama questa funzione senza backend continua a funzionare come prima — compatibilità garantita.
"""
def build_initial_clustering_state(
    embeddings: np.ndarray,
    records: list[dict],
    namer: "ClusterNamer",
    min_cluster_size: int | None = None,
    backend: "ClusteringBackend | None" = None,
) -> ClusteringState:
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
