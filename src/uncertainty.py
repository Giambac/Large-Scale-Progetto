"""
uncertainty.py — Calcola quanto il sistema è incerto su ogni recensione e ogni cluster.

Questo file risponde a una domanda fondamentale: "cosa non sa il sistema?"
Produce tre liste ordinate che dicono:
    - quali recensioni sono ambigue (non si capisce bene dove metterle)
    - quali cluster sono i più "confusi" internamente (buoni candidati per uno split)
    - quali coppie di cluster si assomigliano di più (buoni candidati per un merge)

Queste informazioni vengono usate dal loop per decidere cosa mostrare o chiedere all'oracle al turno successivo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.state import ClusteringState

"""
class UncertaintyReport:
    Il risultato di f_uncertainty — tre viste ordinate sull'incertezza del sistema.

    boundary_items      — tutte le recensioni ordinate dalla più ambigua alla meno. Ogni elemento è (item_id, entropia). 
                          Una recensione con entropia 1.0 è completamente ambigua, con 0.0 è certissima.

    split_candidates    — i cluster ordinati dal più "confuso" al più compatto. Ogni elemento è (cluster_id, entropia_media). 
                          Un cluster con entropia media alta contiene recensioni molto diverse tra loro — è il primo candidato per uno split.

    merge_candidates    — tutte le coppie di cluster ordinate dalla più simile alla più diversa. Ogni elemento è (cluster_a_id, cluster_b_id, distanza). 
                          La coppia con distanza minore è il primo candidato per un merge.
"""
@dataclass
class UncertaintyReport:
    boundary_items: list[tuple[int, float]]        # (item_id, normalized_entropy), descending
    split_candidates: list[tuple[int, float]]      # (cluster_id, mean_entropy), descending
    merge_candidates: list[tuple[int, int, float]] # (cluster_a_id, cluster_b_id, distance), ascending


"""
def f_uncertainty(state: ClusteringState) -> UncertaintyReport:
    Calcola l'incertezza del sistema a partire dalle probabilità morbide (soft_probs).

    Come funziona:
        1. Per ogni recensione calcola l'entropia di Shannon normalizzata. 
        L'entropia misura quanto sono "spalmati" i valori di soft_probs: [0.95, 0.05] → quasi zero (molto certa), [0.5, 0.5] → massima (ambigua).
        Dividere per log(K) normalizza il risultato tra 0 e 1 indipendentemente da quanti cluster ci sono.
      2. Ordina tutte le recensioni per entropia decrescente → boundary_items.
      3. Per ogni cluster calcola la media delle entropie delle sue recensioni, poi ordina i cluster → split_candidates.
      4. Calcola il centroide di ogni cluster nello spazio delle probabilità, poi calcola la distanza tra ogni coppia di centroidi → merge_candidates.

    Crasha subito se un cluster non ha recensioni — cluster vuoti non devono mai arrivare qui.
"""
def f_uncertainty(state: ClusteringState) -> UncertaintyReport:
    K = len(state.clusters)
    assert K > 0, "f_uncertainty called on empty ClusteringState (no clusters)"
    log_K = np.log(K) if K > 1 else 1.0  # avoid div-by-zero for K=1

    # Passo 1: calcolo dell'entropia normalizzata per ogni recensione.
    # Vengono usati solo i valori > 0 per evitare log(0).
    item_entropy: dict[int, float] = {}
    for item_id, probs in state.soft_probs.items():
        p = np.array(probs, dtype=np.float64)
        mask = p > 0
        h = -np.sum(p[mask] * np.log(p[mask]))
        item_entropy[item_id] = float(h / log_K)  # normalized to [0, 1]

    # Passo 2: tutte le recensioni ordinate per entropia decrescente.
    boundary_items = sorted(item_entropy.items(), key=lambda x: x[1], reverse=True)

     # Passo 3: per ogni cluster, media delle entropie delle sue recensioni.
    # Un cluster con media alta è internamente confuso → candidato split.
    cluster_entropy: dict[int, float] = {}
    for cluster in state.clusters:
        if cluster.item_ids:
            cluster_entropy[cluster.id] = float(
                np.mean([item_entropy[i] for i in cluster.item_ids])
            )
    split_candidates = sorted(cluster_entropy.items(), key=lambda x: x[1], reverse=True)

    # Passo 4: centroide di ogni cluster come media dei vettori soft_probs.
    # Due cluster con centroidi vicini si assomigliano → candidati merge.
    cluster_centroids: dict[int, np.ndarray] = {}
    for cluster in state.clusters:
        assert len(cluster.item_ids) > 0, (
            f"f_uncertainty: cluster {cluster.id} has no items — "
            "empty clusters must be removed before f_uncertainty is called"
        )
        vecs = np.array([state.soft_probs[i] for i in cluster.item_ids])
        cluster_centroids[cluster.id] = vecs.mean(axis=0)

    # Calcola la distanza tra ogni coppia di centroidi e ordina crescente.
    merge_candidates = []
    cluster_ids = [c.id for c in state.clusters]
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            a, b = cluster_ids[i], cluster_ids[j]
            dist = float(np.linalg.norm(cluster_centroids[a] - cluster_centroids[b]))
            merge_candidates.append((a, b, dist))
    merge_candidates.sort(key=lambda x: x[2])  # coppia più simile per prima

    return UncertaintyReport(
        boundary_items=boundary_items,
        split_candidates=split_candidates,
        merge_candidates=merge_candidates,
    )
