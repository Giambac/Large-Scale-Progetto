"""
agent_functions.py — Le funzioni che trasformano lo stato in risposta al feedback.

Questo è il cuore del sistema: le funzioni che applicano concretamente le richieste dell'oracle al clustering.

Quattro funzioni principali:
    - f_output : controlla che lo stato sia completo e lo restituisce
    - f_next_best_step : delega alla strategia la scelta dell'azione
    - f_next_state : applica tutti i feedback dell'oracle e produce il nuovo stato
    - (f_uncertainty è in uncertainty.py)

Tre operazioni interne usate da f_next_state:
    - _apply_split : divide un cluster in due usando KMeans
    - _apply_merge : unisce due cluster in uno
    - _apply_move_item : sposta una singola recensione in un altro cluster

Queste funzioni non fanno I/O, non scrivono file, non chiamano API direttamente (tranne il namer per i nomi dei cluster). 
Ricevono uno stato, lo trasformano, restituiscono il nuovo stato.
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans

from src.state import Cluster, ClusteringState
from src.feedback import (
    FeedbackDelta, SplitFeedback, MergeFeedback,
    MoveItemFeedback, GlobalFeedback, InstructionalFeedback,
    ORACLE_MOVE_CONFIDENCE, UNIFORM_FALLBACK_THRESHOLD,
)
from src.uncertainty import UncertaintyReport

if TYPE_CHECKING:
    from src.embedding_store import EmbeddingStore
    from src.cluster_naming import ClusterNamer
    from src.strategy import StrategyProtocol, Action
    from src.hierarchy import HierarchyStore

"""
def f_output( )
    Controlla che lo stato sia completo e lo restituisce invariato.

    Tre controlli: che ci siano recensioni, che tutte abbiano le probabilità, che esista almeno un cluster. 
    Se uno di questi fallisce il programma crasha — uno stato incompleto non deve mai uscire dal sistema.
 """
def f_output(state: ClusteringState) -> ClusteringState:
    N = len(state.assignments)
    assert N > 0, "f_output: state has no items"
    assert len(state.soft_probs) == N, (
        f"f_output: soft_probs has {len(state.soft_probs)} items, expected {N}"
    )
    assert len(state.clusters) > 0, "f_output: state has no clusters"
    return state  # current state IS the complete assignment

"""
def f_next_best_step( )
    Sceglie la prossima azione delegando interamente alla strategia.

    Non fa niente da sola — chiama strategy.select() e restituisce il risultato.
"""
def f_next_best_step(
    state: ClusteringState,
    strategy: "StrategyProtocol",
    uncertainty_report: UncertaintyReport,
) -> "Action":
    return strategy.select(state, uncertainty_report)

# ── Funzioni di supporto interne ─────────────────────────────────────────────

"""
def _cluster_id_to_index( )
    Costruisce un dizionario cluster_id -> posizione nella lista clusters.
    Serve per trovare velocemente in quale posizione sta un cluster dato il suo ID.
    Viene ricostruito da zero ad ogni chiamata.
"""
def _cluster_id_to_index(state: ClusteringState) -> dict[int, int]:
    return {c.id: i for i, c in enumerate(state.clusters)}

"""
def _next_cluster_id( )
    Restituisce il prossimo ID disponibile per un nuovo cluster.
    È sempre il massimo ID esistente + 1, così un ID ritirato non viene mai riusato.
"""
def _next_cluster_id(state: ClusteringState) -> int:
    assert len(state.clusters) > 0, "_next_cluster_id called on empty state"
    return max(c.id for c in state.clusters) + 1

"""
def _apply_move_item( )
    Sposta una singola recensione da un cluster a un altro.

    Cosa succede alle probabilità:
        - La probabilità della recensione per il cluster di destinazione viene fissata a 0.95 (ORACLE_MOVE_CONFIDENCE) — il sistema è quasi certo.
        - Il restante 0.05 viene ridistribuito proporzionalmente tra gli altri cluster.
        - Caso limite: se tutti gli altri cluster avevano probabilità 0, la distribuzione diventa uniforme grazie a UNIFORM_FALLBACK_THRESHOLD.

    No-op guard: se l'oracle suggerisce di spostare una recensione nel cluster in cui è già, lo stato viene restituito invariato senza fare nulla.

    Dopo lo spostamento, il namer viene chiamato per aggiornare il nome del cluster di partenza e del cluster di destinazione.
"""
def _apply_move_item(
    feedback: MoveItemFeedback,
    state: ClusteringState,
    namer: "ClusterNamer",
    id_to_text: dict[int, str],
    global_instructions: list[str],
) -> ClusteringState:
    assert feedback.item_id in state.assignments, (
        f"_apply_move_item: item_id {feedback.item_id} not in assignments"
    )
    id_to_idx = _cluster_id_to_index(state)
    assert feedback.target_cluster_id in id_to_idx, (
        f"_apply_move_item: target_cluster_id {feedback.target_cluster_id} not in clusters"
    )

    source_cluster_id = state.assignments[feedback.item_id]

    # No-op: la recensione è già nel cluster di destinazione — niente da fare
    if source_cluster_id == feedback.target_cluster_id:
        return state

    target_idx = id_to_idx[feedback.target_cluster_id]

    # Aggiorna le probabilità della recensione spostata
    probs = list(state.soft_probs[feedback.item_id])
    original_non_target_sum = sum(p for i, p in enumerate(probs) if i != target_idx)
    assert original_non_target_sum >= 0, "negative sum is impossible"

    residual = 1.0 - ORACLE_MOVE_CONFIDENCE  # = 0.05
    probs[target_idx] = ORACLE_MOVE_CONFIDENCE

    if original_non_target_sum < UNIFORM_FALLBACK_THRESHOLD:
        # Caso limite: tutti gli altri erano a zero → distribuzione uniforme
        non_target_count = len(probs) - 1
        for idx in range(len(probs)):
            if idx != target_idx:
                probs[idx] = residual / non_target_count
    else:
        # Redistribuzione proporzionale del residuo tra gli altri cluster
        for idx in range(len(probs)):
            if idx != target_idx:
                probs[idx] = probs[idx] * (residual / original_non_target_sum)

    assert abs(sum(probs) - 1.0) < 1e-5, (
        f"_apply_move_item: soft_probs row for item {feedback.item_id} sums to {sum(probs)}"
    )

    # Copia tutti i soft_probs e sostituisce solo la riga della recensione spostata
    new_soft_probs = {item_id: list(row) for item_id, row in state.soft_probs.items()}
    new_soft_probs[feedback.item_id] = probs

    # Aggiorna le assegnazioni
    new_assignments = dict(state.assignments)
    new_assignments[feedback.item_id] = feedback.target_cluster_id

    # Aggiorna i cluster: il cluster di partenza perde la recensione, il cluster di destinazione la guadagna. Entrambi vengono rinominati.
    new_clusters = []
    for c in state.clusters:
        if c.id == source_cluster_id:
            new_item_ids = [i for i in c.item_ids if i != feedback.item_id]
            sample_texts = [id_to_text[i] for i in new_item_ids[:5] if i in id_to_text]
            if sample_texts:
                naming = namer.name_cluster(sample_texts, c.id)
                new_clusters.append(Cluster(
                    id=c.id,
                    name=naming["name"],
                    description=naming["description"],
                    item_ids=new_item_ids,
                ))
            else:
                new_clusters.append(Cluster(
                    id=c.id,
                    name=c.name,
                    description=c.description,
                    item_ids=new_item_ids,
                ))
        elif c.id == feedback.target_cluster_id:
            new_item_ids = list(c.item_ids) + [feedback.item_id]
            sample_texts = [id_to_text[i] for i in new_item_ids[:5] if i in id_to_text]
            if sample_texts:
                naming = namer.name_cluster(sample_texts, c.id)
                new_clusters.append(Cluster(
                    id=c.id,
                    name=naming["name"],
                    description=naming["description"],
                    item_ids=new_item_ids,
                ))
            else:
                new_clusters.append(Cluster(
                    id=c.id,
                    name=c.name,
                    description=c.description,
                    item_ids=new_item_ids,
                ))
        else:
            new_clusters.append(c)

    return ClusteringState(
        turn_index=state.turn_index,
        timestamp=state.timestamp,
        clusters=new_clusters,
        assignments=new_assignments,
        soft_probs=new_soft_probs,
    )

"""
def _apply_split( )
    Divide un cluster in due nuovi cluster usando KMeans con K=2.

    Come funziona:
        1. Raccoglie gli embeddings di tutte le recensioni del cluster da dividere.
        2. Esegue KMeans K=2 su quegli embeddings.
            - Se l'oracle ha indicato recensioni rappresentative dei due gruppi, vengono usate come centroidi iniziali.
            - Altrimenti KMeans sceglie da solo con k-means++.
        3. Le recensioni vengono divise in due gruppi in base all'etichetta KMeans.
        4. I due nuovi cluster ricevono ID nuovi (monotonicamente crescenti).
        5. Il cluster originale viene rimosso dallo stato e dalla gerarchia.
        6. Le probabilità vengono redistribuite: ogni recensione del cluster originale porta tutta la sua probabilità nel sottocluster a cui è stata assegnata.
        7. Tutte le righe di probabilità vengono rinormalizzate.
        8. I due nuovi cluster vengono rinominati dall'LLM.
"""
def _apply_split(
    delta: SplitFeedback,
    state: ClusteringState,
    store: "EmbeddingStore",
    namer: "ClusterNamer",
    hierarchy: "HierarchyStore",
    id_to_text: dict[int, str],
    global_instructions: list[str],
) -> ClusteringState:
    id_to_idx = _cluster_id_to_index(state)
    assert delta.cluster_id in id_to_idx, (
        f"_apply_split: cluster_id {delta.cluster_id} not found in state"
    )
    target = state.clusters[id_to_idx[delta.cluster_id]]
    assert len(target.item_ids) >= 2, (
        f"Cannot split cluster {delta.cluster_id}: has only {len(target.item_ids)} item(s)"
    )

    # Embeddings del cluster da dividere
    sub_embeddings = np.array([store.get(item_id) for item_id in target.item_ids])

    # KMeans con centroidi iniziali dall'oracle, o k-means++ se non ci sono seed
    if delta.seed_item_ids:
        target_item_set = set(target.item_ids)
        for sid in delta.seed_item_ids[:2]:
            assert sid in target_item_set, (
                f"_apply_split: seed_item_id {sid} is not in cluster {delta.cluster_id} "
                f"(items: {target.item_ids})"
            )

    if delta.seed_item_ids and len(delta.seed_item_ids) >= 2:
        seed_embeddings = np.array([store.get(sid) for sid in delta.seed_item_ids[:2]])
        centroids = seed_embeddings[:2]
        km = KMeans(n_clusters=2, init=centroids, n_init=1, random_state=0)
    else:
        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)

    km.fit(sub_embeddings)
    labels = km.labels_  # 0 o 1 per ogni recensione del cluster

    # Nuovi ID monotonicamente crescenti per i due sottocluster
    new_id_a = _next_cluster_id(state)
    new_id_b = new_id_a + 1

    retired_idx = id_to_idx[delta.cluster_id]
    old_clusters_kept = [c for c in state.clusters if c.id != delta.cluster_id]
    new_k = len(old_clusters_kept) + 2
    new_id_a_idx = len(old_clusters_kept)
    new_id_b_idx = len(old_clusters_kept) + 1

    # Mappa: vecchio cluster ID -> nuova posizione nel vettore di probabilità
    old_id_to_new_idx: dict[int, int] = {}
    for new_pos, c in enumerate(old_clusters_kept):
        old_id_to_new_idx[c.id] = new_pos

    # Indice locale di ogni recensione dentro il cluster da dividere.
    item_local_idx = {item_id: i for i, item_id in enumerate(target.item_ids)}

    new_soft_probs: dict[int, list[float]] = {}
    new_assignments: dict[int, int] = {}

    for item_id, old_probs in state.soft_probs.items():
        new_probs = [0.0] * new_k
        # Copia le probabilità dei cluster non toccati nelle nuove posizioni
        for old_c in old_clusters_kept:
            old_c_idx = id_to_idx[old_c.id]
            new_c_idx = old_id_to_new_idx[old_c.id]
            new_probs[new_c_idx] = old_probs[old_c_idx]

        if item_id in item_local_idx:
            # Questa recensione era nel cluster diviso. Tutta la sua probabilità va al sottocluster a cui è stata assegnata.
            local_idx = item_local_idx[item_id]
            retired_prob = old_probs[retired_idx]
            if labels[local_idx] == 0:
                new_probs[new_id_a_idx] = retired_prob
                new_probs[new_id_b_idx] = 0.0
            else:
                new_probs[new_id_a_idx] = 0.0
                new_probs[new_id_b_idx] = retired_prob
            # Assign to the appropriate sub-cluster
            new_assignments[item_id] = new_id_a if labels[local_idx] == 0 else new_id_b
        else:
            # Recensione in un altro cluster — i nuovi slot restano a 0
            new_probs[new_id_a_idx] = 0.0
            new_probs[new_id_b_idx] = 0.0
            new_assignments[item_id] = state.assignments[item_id]

        # Rinormalizza la riga
        p = np.array(new_probs, dtype=np.float64)
        row_sum = p.sum()
        if row_sum > 0:
            p = p / row_sum
        else:
            p = np.ones(new_k, dtype=np.float64) / new_k
        new_soft_probs[item_id] = p.tolist()

    # Costruisce i due nuovi cluster con nomi generati dall'LLM
    items_a = [item_id for item_id, li in zip(target.item_ids, labels) if li == 0]
    items_b = [item_id for item_id, li in zip(target.item_ids, labels) if li == 1]

    # Name sub-cluster A
    sample_texts_a = [id_to_text[i] for i in items_a[:5] if i in id_to_text]
    if sample_texts_a:
        naming_a = namer.name_cluster(sample_texts_a, new_id_a)
    else:
        naming_a = {"name": f"Cluster {new_id_a}", "description": "Split sub-cluster A."}

    # Name sub-cluster B
    sample_texts_b = [id_to_text[i] for i in items_b[:5] if i in id_to_text]
    if sample_texts_b:
        naming_b = namer.name_cluster(sample_texts_b, new_id_b)
    else:
        naming_b = {"name": f"Cluster {new_id_b}", "description": "Split sub-cluster B."}

    new_cluster_a = Cluster(id=new_id_a, name=naming_a["name"], description=naming_a["description"], item_ids=items_a)
    new_cluster_b = Cluster(id=new_id_b, name=naming_b["name"], description=naming_b["description"], item_ids=items_b)

    new_clusters = old_clusters_kept + [new_cluster_a, new_cluster_b]

    # Registra lo split nella gerarchia
    hierarchy.record_split(target.id, new_id_a, new_id_b)

    # Verifica che il numero di recensioni e la somma delle probabilità siano intatti
    assert len(new_soft_probs) == len(state.soft_probs), (
        f"_apply_split: soft_probs item count changed ({len(new_soft_probs)} != {len(state.soft_probs)})"
    )
    for item_id, probs in new_soft_probs.items():
        assert len(probs) == new_k, (
            f"_apply_split: item {item_id} soft_probs length {len(probs)} != {new_k}"
        )
        assert abs(sum(probs) - 1.0) < 1e-5, (
            f"_apply_split: item {item_id} soft_probs sum={sum(probs)}"
        )

    return ClusteringState(
        turn_index=state.turn_index,
        timestamp=state.timestamp,
        clusters=new_clusters,
        assignments=new_assignments,
        soft_probs=new_soft_probs,
    )

"""
def _apply_merge( )
    Unisce due cluster in uno solo.

    Come funziona:
        1. I due cluster originali vengono eliminati dallo stato.
        2. Viene creato un nuovo cluster con un ID nuovo.
        3. Le probabilità vengono aggiornate con il "column pooling": per ogni recensione, la nuova probabilità per il cluster unito 
           è la somma delle probabilità dei due cluster originali.
           Es: se item 5 aveva [0.6, 0.3, 0.1] e i cluster 0 e 1 vengono uniti, la nuova probabilità per il cluster unito è 0.6 + 0.3 = 0.9.
        4. Tutte le righe di probabilità vengono rinormalizzate.
        5. Il nuovo cluster viene rinominato dall'LLM.
        6. Il merge viene registrato nella gerarchia.
"""
def _apply_merge(
    delta: MergeFeedback,
    state: ClusteringState,
    namer: "ClusterNamer",
    hierarchy: "HierarchyStore",
    id_to_text: dict[int, str],
    global_instructions: list[str],
) -> ClusteringState:
    id_to_idx = _cluster_id_to_index(state)
    assert delta.cluster_a_id in id_to_idx, (
        f"_apply_merge: cluster_a_id {delta.cluster_a_id} not found in state"
    )
    assert delta.cluster_b_id in id_to_idx, (
        f"_apply_merge: cluster_b_id {delta.cluster_b_id} not found in state"
    )
    assert delta.cluster_a_id != delta.cluster_b_id, (
        f"_apply_merge: cluster_a_id and cluster_b_id are identical ({delta.cluster_a_id}) "
        "— cannot merge a cluster with itself"
    )

    a_idx = id_to_idx[delta.cluster_a_id]
    b_idx = id_to_idx[delta.cluster_b_id]

    # New cluster ID from monotonic counter (based on current max — after removing a and b, new_id > both)
    new_id = _next_cluster_id(state)

    # Cluster che rimangono invariati (tutti tranne i due da unire)
    old_clusters_kept = [c for c in state.clusters if c.id not in (delta.cluster_a_id, delta.cluster_b_id)]
    new_k = len(old_clusters_kept) + 1  # kept + merged
    new_merged_idx = len(old_clusters_kept)

    # Build old cluster id to new position mapping for non-retired clusters
    old_id_to_new_idx: dict[int, int] = {}
    for new_pos, c in enumerate(old_clusters_kept):
        old_id_to_new_idx[c.id] = new_pos

    # Build new soft_probs and assignments via column pooling
    new_soft_probs: dict[int, list[float]] = {}
    new_assignments: dict[int, int] = {}

    for item_id, old_probs in state.soft_probs.items():
        new_probs = [0.0] * new_k
        # Copia i cluster non toccati nelle nuove posizioni
        for old_c in old_clusters_kept:
            old_c_idx = id_to_idx[old_c.id]
            new_c_idx = old_id_to_new_idx[old_c.id]
            new_probs[new_c_idx] = old_probs[old_c_idx]

        # Column pooling: la prob del cluster unito è la somma delle due originali
        new_probs[new_merged_idx] = old_probs[a_idx] + old_probs[b_idx]

        # Rinormalizza
        p = np.array(new_probs, dtype=np.float64)
        row_sum = p.sum()
        if row_sum > 0:
            p = p / row_sum
        else:
            p = np.ones(new_k, dtype=np.float64) / new_k
        new_soft_probs[item_id] = p.tolist()

        # Aggiorna l'assegnazione: le recensioni di A o B ora appartengono al nuovo cluster
        old_assignment = state.assignments[item_id]
        if old_assignment in (delta.cluster_a_id, delta.cluster_b_id):
            new_assignments[item_id] = new_id
        else:
            new_assignments[item_id] = old_assignment

    # Costruisce il cluster unito con nome generato dall'LLM
    cluster_a = state.clusters[a_idx]
    cluster_b = state.clusters[b_idx]
    merged_item_ids = list(cluster_a.item_ids) + list(cluster_b.item_ids)

    sample_texts = [id_to_text[i] for i in merged_item_ids[:5] if i in id_to_text]
    if sample_texts:
        naming = namer.name_cluster(sample_texts, new_id)
    else:
        naming = {"name": f"Cluster {new_id}", "description": "Merged cluster."}

    merged_cluster = Cluster(
        id=new_id,
        name=naming["name"],
        description=naming["description"],
        item_ids=merged_item_ids,
    )

    new_clusters = old_clusters_kept + [merged_cluster]

    # Record merge in hierarchy
    hierarchy.record_merge(delta.cluster_a_id, delta.cluster_b_id, new_id)

    # Verify completeness
    assert len(new_soft_probs) == len(state.soft_probs), (
        f"_apply_merge: soft_probs item count changed"
    )
    for item_id, probs in new_soft_probs.items():
        assert abs(sum(probs) - 1.0) < 1e-5, (
            f"_apply_merge: item {item_id} soft_probs sum={sum(probs)}"
        )

    return ClusteringState(
        turn_index=state.turn_index,
        timestamp=state.timestamp,
        clusters=new_clusters,
        assignments=new_assignments,
        soft_probs=new_soft_probs,
    )

"""
def f_next_state( )
    Applica tutti i feedback dell'oracle e produce il nuovo stato.

    È la funzione più importante di questo file. Riceve lo stato corrente e una lista di feedback, e li applica tutti in ordine di priorità fisso:
        1. GlobalFeedback — istruzioni generali (non cambiano la struttura, vengono accumulate)
        2. SplitFeedback e MergeFeedback — operazioni strutturali sui cluster
        3. MoveItemFeedback — spostamenti di singole recensioni
        4. InstructionalFeedback — suggerimenti soft (vengono accumulati come i global)

    L'ordine è fisso indipendentemente da come l'oracle ha scritto i feedback.

    Nota su global_instructions: questa lista vive fuori dallo stato (lo schema di ClusteringState è congelato e non si può modificare). GlobalFeedback e
    InstructionalFeedback appendono il loro testo a questa lista in-place. 
    La lista persiste per tutta la sessione e viene usata nella Fase 3 per arricchire i prompt dell'oracle con le preferenze accumulate.

    Alla fine: controlla che il numero di recensioni non sia cambiato, che tutte abbiano le probabilità, e che esista ancora almeno un cluster. 
    Poi incrementa il turn_index e aggiorna il timestamp.
"""
def f_next_state(
    state: ClusteringState,
    deltas: list[FeedbackDelta],
    store: "EmbeddingStore",
    namer: "ClusterNamer",
    hierarchy: "HierarchyStore | None" = None,
    id_to_text: dict[int, str] | None = None,
    global_instructions: list[str] | None = None,
) -> ClusteringState:
    if global_instructions is None:
        global_instructions = []
    if id_to_text is None:
        id_to_text = {}
    if hierarchy is None:
        from src.hierarchy import HierarchyStore
        hierarchy = HierarchyStore()
        # Register all existing clusters so record_split/record_merge can find them
        for cluster in state.clusters:
            hierarchy.register(cluster.id)

    # Ordina i feedback per priorità: global=0, split/merge=1, move=2, instructional=3
    PRIORITY = {GlobalFeedback: 0, SplitFeedback: 1, MergeFeedback: 1,
                MoveItemFeedback: 2, InstructionalFeedback: 3}
    sorted_deltas = sorted(deltas, key=lambda d: PRIORITY[type(d)])

    current_state = state
    for delta in sorted_deltas:
        if isinstance(delta, GlobalFeedback):
            # Accumula l'istruzione — non cambia la struttura dei cluster
            global_instructions.append(delta.instruction_text)
        elif isinstance(delta, SplitFeedback):
            current_state = _apply_split(delta, current_state, store, namer, hierarchy, id_to_text, global_instructions)
        elif isinstance(delta, MergeFeedback):
            current_state = _apply_merge(delta, current_state, namer, hierarchy, id_to_text, global_instructions)
        elif isinstance(delta, MoveItemFeedback):
            current_state = _apply_move_item(delta, current_state, namer, id_to_text, global_instructions)
        elif isinstance(delta, InstructionalFeedback):
            # Accumula il suggerimento soft — non cambia la struttura dei cluster
            global_instructions.append(delta.instruction_text)
        else:
            assert False, f"Unknown FeedbackDelta type: {type(delta)}"

    # Verifica che gli invarianti fondamentali siano rispettati
    N = len(current_state.assignments)
    assert N == len(state.assignments), "f_next_state: item count changed"
    assert len(current_state.soft_probs) == N, "f_next_state: soft_probs incomplete"
    assert len(current_state.clusters) > 0, "f_next_state: no clusters remain"

    # Produce il nuovo stato con turn_index incrementato e timestamp aggiornato
    return ClusteringState(
        turn_index=current_state.turn_index + 1,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        clusters=current_state.clusters,
        assignments=current_state.assignments,
        soft_probs=current_state.soft_probs,
    )
