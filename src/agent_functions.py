"""
agent_functions.py — The four pure functions of the Clustering Agent (D-02).

f_output, f_uncertainty (in uncertainty.py), f_next_best_step, f_next_state.
Pure functions: no I/O, no global mutation. The loop (conversation_loop.py) owns all I/O.

GlobalFeedback accumulator (FB-01):
    f_next_state receives a `global_instructions: list[str]` parameter owned by the caller
    (conversation_loop.py). GlobalFeedback.instruction_text is appended to this list in-place
    so that the loop can pass accumulated instructions to the ClusterNamer in future phases.
    ClusteringState schema is FROZEN — this accumulator lives outside the state object.
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
    RenameFeedback, DeleteFeedback,
    ORACLE_MOVE_CONFIDENCE, UNIFORM_FALLBACK_THRESHOLD,
)
from src.uncertainty import UncertaintyReport
from src.logging_setup import deviation

if TYPE_CHECKING:
    from src.embedding_store import EmbeddingStore
    from src.cluster_naming import ClusterNamer
    from src.strategy import StrategyProtocol, Action
    from src.hierarchy import HierarchyStore


def f_output(state: ClusteringState) -> ClusteringState:
    """
    Anytime function: returns the complete current clustering assignment.
    Never partial. Called to publish the current best known assignment.
    """
    N = len(state.assignments)
    assert N > 0, "f_output: state has no items"
    assert len(state.soft_probs) == N, (
        f"f_output: soft_probs has {len(state.soft_probs)} items, expected {N}"
    )
    assert len(state.clusters) > 0, "f_output: state has no clusters"
    return state  # current state IS the complete assignment


def f_next_best_step(
    state: ClusteringState,
    strategy: "StrategyProtocol",
    uncertainty_report: UncertaintyReport,
) -> "Action":
    """
    Selects the next action via the injected strategy. Pure function — no I/O.
    RandomStrategy: uniform random over valid action types (Phase 2).
    Phase 5 adds UncertaintyDrivenStrategy and BoundaryDrivenStrategy.
    """
    return strategy.select(state, uncertainty_report)


def _cluster_id_to_index(state: ClusteringState) -> dict[int, int]:
    """Build cluster_id -> position-in-clusters-list mapping. Rebuilt fresh each call."""
    return {c.id: i for i, c in enumerate(state.clusters)}


def _next_cluster_id(state: ClusteringState) -> int:
    """Monotonic counter: max existing cluster ID + 1. D-11: never reuse retired IDs."""
    assert len(state.clusters) > 0, "_next_cluster_id called on empty state"
    return max(c.id for c in state.clusters) + 1


def _apply_move_item(
    feedback: MoveItemFeedback,
    state: ClusteringState,
    namer: "ClusterNamer",
    id_to_text: dict[int, str],
    global_instructions: list[str],
) -> ClusteringState:
    """
    Apply a point-move: move item_id to target_cluster_id.

    Implements D-10: soft_probs[item_id][target_idx] = ORACLE_MOVE_CONFIDENCE (0.95).
    Remaining 0.05 redistributed proportionally among non-target clusters.
    ClusterNamer called for source and target clusters (D-12).
    """
    assert feedback.item_id in state.assignments, (
        f"_apply_move_item: item_id {feedback.item_id} not in assignments"
    )
    id_to_idx = _cluster_id_to_index(state)
    assert feedback.target_cluster_id in id_to_idx, (
        f"_apply_move_item: target_cluster_id {feedback.target_cluster_id} not in clusters"
    )

    source_cluster_id = state.assignments[feedback.item_id]

    # No-op guard: LLM may suggest moving an item to its current cluster — return state unchanged.
    if source_cluster_id == feedback.target_cluster_id:
        return state  # already in target cluster — nothing to do

    target_idx = id_to_idx[feedback.target_cluster_id]

    # Copy probs for this item
    probs = list(state.soft_probs[feedback.item_id])
    original_non_target_sum = sum(p for i, p in enumerate(probs) if i != target_idx)
    assert original_non_target_sum >= 0, "negative sum is impossible"

    residual = 1.0 - ORACLE_MOVE_CONFIDENCE  # = 0.05
    probs[target_idx] = ORACLE_MOVE_CONFIDENCE

    if original_non_target_sum < UNIFORM_FALLBACK_THRESHOLD:
        # Edge case: uniform distribution across non-target clusters
        non_target_count = len(probs) - 1
        for idx in range(len(probs)):
            if idx != target_idx:
                probs[idx] = residual / non_target_count
    else:
        # Proportional redistribution (D-10)
        for idx in range(len(probs)):
            if idx != target_idx:
                probs[idx] = probs[idx] * (residual / original_non_target_sum)

    assert abs(sum(probs) - 1.0) < 1e-5, (
        f"_apply_move_item: soft_probs row for item {feedback.item_id} sums to {sum(probs)}"
    )

    # Build new soft_probs: copy all rows, replace row for feedback.item_id
    new_soft_probs = {item_id: list(row) for item_id, row in state.soft_probs.items()}
    new_soft_probs[feedback.item_id] = probs

    # Build new assignments
    new_assignments = dict(state.assignments)
    new_assignments[feedback.item_id] = feedback.target_cluster_id

    # Build updated cluster lists: source loses item, target gains item
    new_clusters = []
    for c in state.clusters:
        if c.id == source_cluster_id:
            new_item_ids = [i for i in c.item_ids if i != feedback.item_id]
            # Name the source cluster (D-12) — global_instructions available for Phase 3 enrichment
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
            # Name the target cluster (D-12) — global_instructions available for Phase 3 enrichment
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


def _apply_split(
    delta: SplitFeedback,
    state: ClusteringState,
    store: "EmbeddingStore",
    namer: "ClusterNamer",
    hierarchy: "HierarchyStore",
    id_to_text: dict[int, str],
    global_instructions: list[str],
) -> ClusteringState:
    """
    Apply a cluster split: divide target cluster into two sub-clusters using K-means.

    Implements D-08: oracle-seeded K-means (or k-means++ fallback).
    New cluster IDs assigned from monotonic counter (D-11).
    Retired cluster ID removed from new state.
    ClusterNamer called for both new sub-clusters (D-12).
    """
    id_to_idx = _cluster_id_to_index(state)
    assert delta.cluster_id in id_to_idx, (
        f"_apply_split: cluster_id {delta.cluster_id} not found in state"
    )
    target = state.clusters[id_to_idx[delta.cluster_id]]
    assert len(target.item_ids) >= 2, (
        f"Cannot split cluster {delta.cluster_id}: has only {len(target.item_ids)} item(s)"
    )

    # Gather embeddings for items in the target cluster
    sub_embeddings = np.array([store.get(item_id) for item_id in target.item_ids])

    # K-means with optional oracle seed centroids (WR-02: validate seed membership first)
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
        # 0 or 1 seeds — fall back to k-means++ (avoids identical centroid degenerate case)
        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)

    km.fit(sub_embeddings)
    labels = km.labels_  # shape (M,), values 0 or 1

    # Assign new IDs monotonically (D-11)
    new_id_a = _next_cluster_id(state)
    new_id_b = new_id_a + 1

    # Build soft_probs for new state
    # retired cluster position in old state
    retired_idx = id_to_idx[delta.cluster_id]

    # Number of new clusters = (old count - 1) + 2 = old count + 1
    # New cluster order: all old clusters except retired, then new_id_a, new_id_b
    # We build a mapping from old cluster positions to new positions
    old_clusters_kept = [c for c in state.clusters if c.id != delta.cluster_id]
    # new positions: 0..len(old_kept)-1 for kept; len(old_kept) for new_id_a; len(old_kept)+1 for new_id_b
    new_k = len(old_clusters_kept) + 2
    new_id_a_idx = len(old_clusters_kept)
    new_id_b_idx = len(old_clusters_kept) + 1

    # Build old cluster id to new position mapping for non-retired clusters
    old_id_to_new_idx: dict[int, int] = {}
    for new_pos, c in enumerate(old_clusters_kept):
        old_id_to_new_idx[c.id] = new_pos

    # Build item_id -> local index within target cluster
    item_local_idx = {item_id: i for i, item_id in enumerate(target.item_ids)}

    new_soft_probs: dict[int, list[float]] = {}
    new_assignments: dict[int, int] = {}

    for item_id, old_probs in state.soft_probs.items():
        new_probs = [0.0] * new_k
        # Copy non-retired columns to their new positions
        for old_c in old_clusters_kept:
            old_c_idx = id_to_idx[old_c.id]
            new_c_idx = old_id_to_new_idx[old_c.id]
            new_probs[new_c_idx] = old_probs[old_c_idx]

        if item_id in item_local_idx:
            # This item is in the split cluster — distribute retired prob to sub-clusters
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
            # Item not in split cluster — new sub-cluster columns stay at 0
            new_probs[new_id_a_idx] = 0.0
            new_probs[new_id_b_idx] = 0.0
            new_assignments[item_id] = state.assignments[item_id]

        # Re-normalize row
        p = np.array(new_probs, dtype=np.float64)
        row_sum = p.sum()
        if row_sum > 0:
            p = p / row_sum
        else:
            p = np.ones(new_k, dtype=np.float64) / new_k
        new_soft_probs[item_id] = p.tolist()

    # Build two new Cluster objects with new IDs (D-12: name both sub-clusters)
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

    # Record split in hierarchy (D-11)
    hierarchy.record_split(target.id, new_id_a, new_id_b)

    # Verify completeness
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


def _apply_merge(
    delta: MergeFeedback,
    state: ClusteringState,
    namer: "ClusterNamer",
    hierarchy: "HierarchyStore",
    id_to_text: dict[int, str],
    global_instructions: list[str],
) -> ClusteringState:
    """
    Apply a cluster merge: combine cluster_a and cluster_b into one new cluster.

    Implements D-09: column pooling on soft_probs.
    New cluster ID assigned from monotonic counter (D-11).
    Both retired IDs removed from new state.
    ClusterNamer called for merged cluster (D-12).
    """
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

    # Identify kept clusters (all except a and b)
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
        # Copy non-retired columns to their new positions
        for old_c in old_clusters_kept:
            old_c_idx = id_to_idx[old_c.id]
            new_c_idx = old_id_to_new_idx[old_c.id]
            new_probs[new_c_idx] = old_probs[old_c_idx]

        # Pool: merged prob = a_prob + b_prob (D-09)
        new_probs[new_merged_idx] = old_probs[a_idx] + old_probs[b_idx]

        # Re-normalize row
        p = np.array(new_probs, dtype=np.float64)
        row_sum = p.sum()
        if row_sum > 0:
            p = p / row_sum
        else:
            p = np.ones(new_k, dtype=np.float64) / new_k
        new_soft_probs[item_id] = p.tolist()

        # Update assignment: items from a or b now belong to new_id
        old_assignment = state.assignments[item_id]
        if old_assignment in (delta.cluster_a_id, delta.cluster_b_id):
            new_assignments[item_id] = new_id
        else:
            new_assignments[item_id] = old_assignment

    # Build merged Cluster (D-12: name the merged cluster)
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


def f_next_state(
    state: ClusteringState,
    deltas: list[FeedbackDelta],
    store: "EmbeddingStore",
    namer: "ClusterNamer",
    hierarchy: "HierarchyStore | None" = None,
    id_to_text: dict[int, str] | None = None,
    global_instructions: list[str] | None = None,
) -> ClusteringState:
    """
    Applies a list of FeedbackDelta objects to state in type-priority order (D-07):
    global → cluster-level (split/merge) → point-level (move_item) → instructional.

    GlobalFeedback (FB-01): instruction_text is appended to the global_instructions list
    (mutable, owned by the caller — conversation_loop.py). ClusteringState schema is FROZEN;
    this accumulator lives outside the state object. The list is passed to naming helpers
    so that Phase 3 can enrich LLM prompts with accumulated oracle preferences.

    Pure function with respect to ClusteringState. Does mutate global_instructions in-place.
    """
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

    # Sort deltas by type priority (D-07): global first, then cluster, then point, then instructional
    PRIORITY = {GlobalFeedback: 0, SplitFeedback: 1, MergeFeedback: 1,
                RenameFeedback: 1, DeleteFeedback: 1,
                MoveItemFeedback: 2, InstructionalFeedback: 3}
    sorted_deltas = sorted(deltas, key=lambda d: PRIORITY[type(d)])

    current_state = state
    for delta in sorted_deltas:
        # Validate cluster-referencing deltas against the LIVE state before applying.
        # Deltas originate from parse_feedback (an LLM call) — an untrusted boundary that
        # can hallucinate cluster ids, and earlier deltas in this batch may retire a
        # cluster a later delta names. deviation() skips the delta (warn in production,
        # raise under STRICT_MODE) rather than letting _apply_* assert and kill the caller.
        live_ids = {c.id for c in current_state.clusters}
        if isinstance(delta, GlobalFeedback):
            global_instructions.append(delta.instruction_text)
        elif isinstance(delta, RenameFeedback):
            _new_clusters = []
            for _c in current_state.clusters:
                if _c.id == delta.cluster_id:
                    _new_clusters.append(Cluster(id=_c.id, name=delta.new_name, description=_c.description, item_ids=_c.item_ids))
                else:
                    _new_clusters.append(_c)
            current_state = ClusteringState(
                turn_index=current_state.turn_index,
                timestamp=current_state.timestamp,
                clusters=_new_clusters,
                assignments=current_state.assignments,
                soft_probs=current_state.soft_probs,
            )
        elif isinstance(delta, DeleteFeedback):
            _id_to_idx = _cluster_id_to_index(current_state)
            if delta.cluster_id in _id_to_idx:
                _kept = [c for c in current_state.clusters if c.id != delta.cluster_id]
                _kept_ids = [c.id for c in _kept]
                _new_k = len(_kept_ids)
                _new_assignments = dict(current_state.assignments)
                _cluster_extra: dict[int, list[int]] = {cid: [] for cid in _kept_ids}
                _target = current_state.clusters[_id_to_idx[delta.cluster_id]]
                for _item_id in _target.item_ids:
                    _old_probs = current_state.soft_probs[_item_id]
                    _best_cid = max(_kept_ids, key=lambda cid: _old_probs[_id_to_idx[cid]])
                    _new_assignments[_item_id] = _best_cid
                    _cluster_extra[_best_cid].append(_item_id)
                _new_soft_probs = {}
                for _item_id, _old_probs in current_state.soft_probs.items():
                    _new_probs = [_old_probs[_id_to_idx[cid]] for cid in _kept_ids]
                    _p = np.array(_new_probs, dtype=np.float64)
                    _s = _p.sum()
                    _p = _p / _s if _s > 0 else np.ones(_new_k, dtype=np.float64) / _new_k
                    _new_soft_probs[_item_id] = _p.tolist()
                _new_clusters = []
                for _c in _kept:
                    _new_item_ids = list(_c.item_ids) + _cluster_extra[_c.id]
                    _new_clusters.append(Cluster(id=_c.id, name=_c.name, description=_c.description, item_ids=_new_item_ids))
                current_state = ClusteringState(
                    turn_index=current_state.turn_index,
                    timestamp=current_state.timestamp,
                    clusters=_new_clusters,
                    assignments=_new_assignments,
                    soft_probs=_new_soft_probs,
                )
        elif isinstance(delta, SplitFeedback):
            if delta.cluster_id not in live_ids:
                deviation("f_next_state: SplitFeedback references unknown cluster — skipping",
                          cluster_id=delta.cluster_id, live_ids=sorted(live_ids))
                continue
            current_state = _apply_split(delta, current_state, store, namer, hierarchy, id_to_text, global_instructions)
        elif isinstance(delta, MergeFeedback):
            if delta.cluster_a_id not in live_ids or delta.cluster_b_id not in live_ids:
                deviation("f_next_state: MergeFeedback references unknown cluster — skipping",
                          cluster_a_id=delta.cluster_a_id, cluster_b_id=delta.cluster_b_id,
                          live_ids=sorted(live_ids))
                continue
            if delta.cluster_a_id == delta.cluster_b_id:
                deviation("f_next_state: MergeFeedback names the same cluster twice — skipping",
                          cluster_id=delta.cluster_a_id)
                continue
            current_state = _apply_merge(delta, current_state, namer, hierarchy, id_to_text, global_instructions)
        elif isinstance(delta, MoveItemFeedback):
            if delta.target_cluster_id not in live_ids:
                deviation("f_next_state: MoveItemFeedback references unknown target cluster — skipping",
                          target_cluster_id=delta.target_cluster_id, item_id=delta.item_id,
                          live_ids=sorted(live_ids))
                continue
            current_state = _apply_move_item(delta, current_state, namer, id_to_text, global_instructions)
        elif isinstance(delta, InstructionalFeedback):
            global_instructions.append(delta.instruction_text)
        else:
            assert False, f"Unknown FeedbackDelta type: {type(delta)}"

    # Completeness invariants — fail loudly if violated
    N = len(current_state.assignments)
    assert N == len(state.assignments), "f_next_state: item count changed"
    assert len(current_state.soft_probs) == N, "f_next_state: soft_probs incomplete"
    assert len(current_state.clusters) > 0, "f_next_state: no clusters remain"

    # Bump turn_index and timestamp
    return ClusteringState(
        turn_index=current_state.turn_index + 1,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        clusters=current_state.clusters,
        assignments=current_state.assignments,
        soft_probs=current_state.soft_probs,
    )