"""
src/judge.py — Judge Agent: per-turn metrics, pairwise validation, no-dialogue baseline.

All functions are pure (no SQL, no I/O). SQL goes through src/db/ exclusively (recipe §6).
This module builds Pydantic TurnCreate / OracleFeedbackCreate models and hands them
to src/db/turns.py and src/db/oracle_feedback.py for persistence.

Key design decisions (CONTEXT.md):
  D-17: Ground truth = accumulated oracle feedback (no external labels).
  D-18: Pair extraction from structural FeedbackDelta types only (Move/Split/Merge).
  D-19: Sample min(50, bag_size) pairs per turn; seed=turn_index for deterministic replay.
  D-20: Contradiction handling — overwrite pairs for affected items.
  D-21: PairBag in memory; rebuilt from JSONL on resume via rebuild_pair_bag().
  D-22: Baseline = initial clustering + one oracle turn; strategy_id = "no_dialogue".
  D-26: No SQL here — pure functions + Pydantic models.
"""
from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING

from src.feedback import (
    FeedbackDelta,
    GlobalFeedback,
    InstructionalFeedback,
    MergeFeedback,
    MoveItemFeedback,
    SplitFeedback,
)
from src.logging_setup import deviation

if TYPE_CHECKING:
    from src.state import ClusteringState
    from src.oracle_protocol import OracleReply
    from src.stopping import StopReason
    import sqlite3


# ── PairBag ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _Pair:
    """A single oracle-asserted same/different-cluster pair."""
    item_x: int
    item_y: int
    expected_same: bool


class PairBag:
    """
    Accumulates (item_x, item_y, expected_same) pairs from oracle feedback.

    Pair sources (D-18):
      MoveItemFeedback(x, C)   → (x, y, True) for every y in cluster C at time of delta
      SplitFeedback([a, b])    → (a, b, False)
      MergeFeedback(A, B)      → (x, y, True) for every x in A, y in B
      Global/Instructional     → no pairs (too semantic)

    Contradiction handling (D-20): when a new delta contradicts prior state,
    drop all pairs involving the affected items, then add fresh pairs from the new delta.
    This mirrors CLUS-04 "latest intent wins".

    Deterministic sampling (D-19): random.Random(turn_index).sample — no global RNG.
    """

    def __init__(self) -> None:
        self._pairs: list[_Pair] = []

    def __len__(self) -> int:
        return len(self._pairs)

    def update(
        self,
        deltas: list[FeedbackDelta],
        state: "ClusteringState",
        is_contradiction: bool = False,
    ) -> None:
        """
        Apply a list of FeedbackDelta objects to the bag.

        For contradicting feedback (is_contradiction=True), affected-item pairs are
        dropped before adding new ones (D-20: overwrite on contradiction).

        Args:
            deltas:          FeedbackDelta list from parse_feedback() for this turn.
            state:           The NEW ClusteringState after f_next_state — cluster membership
                             is used to resolve "every y currently in C" for MoveItemFeedback.
            is_contradiction: True if reply.contradiction_detected — triggers overwrite for
                             all items mentioned in deltas.
        """
        for delta in deltas:
            if isinstance(delta, (GlobalFeedback, InstructionalFeedback)):
                continue  # no structural pairs from semantic feedback (D-18)

            # Gather affected item ids for potential overwrite
            affected_items: set[int] = set()
            if isinstance(delta, MoveItemFeedback):
                affected_items.add(delta.item_id)
            elif isinstance(delta, SplitFeedback):
                affected_items.update(delta.seed_item_ids)
            elif isinstance(delta, MergeFeedback):
                for cluster in state.clusters:
                    if cluster.id in (delta.cluster_a_id, delta.cluster_b_id):
                        affected_items.update(cluster.item_ids)

            # D-20: drop pairs for affected items on contradiction
            if is_contradiction and affected_items:
                self._pairs = [
                    p for p in self._pairs
                    if p.item_x not in affected_items and p.item_y not in affected_items
                ]

            # Add new pairs
            new_pairs = _extract_pairs(delta, state)
            self._pairs.extend(new_pairs)

    def sample(self, turn_index: int, n: int = 50) -> list[_Pair]:
        """
        Return up to n pairs sampled deterministically by turn_index (D-19).

        Uses random.Random(turn_index) — no global RNG mutation, deterministic replay.
        """
        k = min(n, len(self._pairs))
        if k == 0:
            return []
        rng = random.Random(turn_index)
        return rng.sample(self._pairs, k)


def _extract_pairs(delta: FeedbackDelta, state: "ClusteringState") -> list[_Pair]:
    """
    Extract (item_x, item_y, expected_same) pairs from a single structural delta (D-18).

    Global/Instructional → always returns [].
    """
    pairs: list[_Pair] = []

    if isinstance(delta, MoveItemFeedback):
        # x → same cluster as every y currently in target cluster
        target = next((c for c in state.clusters if c.id == delta.target_cluster_id), None)
        if target is None:
            deviation(
                "MoveItemFeedback target_cluster_id not found in state",
                target_cluster_id=delta.target_cluster_id,
                item_id=delta.item_id,
            )
            return []
        for y in target.item_ids:
            if y != delta.item_id:
                pairs.append(_Pair(item_x=delta.item_id, item_y=y, expected_same=True))

    elif isinstance(delta, SplitFeedback):
        seeds = delta.seed_item_ids
        if len(seeds) >= 2:
            # First two seeds should be in different clusters (D-18)
            pairs.append(_Pair(item_x=seeds[0], item_y=seeds[1], expected_same=False))
        else:
            deviation(
                "SplitFeedback has fewer than 2 seed_item_ids — no pair extracted",
                cluster_id=delta.cluster_id,
                seed_count=len(seeds),
            )

    elif isinstance(delta, MergeFeedback):
        # Every item in A × every item in B → expected same
        cluster_a = next((c for c in state.clusters if c.id == delta.cluster_a_id), None)
        cluster_b = next((c for c in state.clusters if c.id == delta.cluster_b_id), None)
        if cluster_a is None or cluster_b is None:
            deviation(
                "MergeFeedback cluster_id not found in state",
                cluster_a_id=delta.cluster_a_id,
                cluster_b_id=delta.cluster_b_id,
            )
            return []
        for x in cluster_a.item_ids:
            for y in cluster_b.item_ids:
                pairs.append(_Pair(item_x=x, item_y=y, expected_same=True))

    return pairs


# ── Accuracy computation ──────────────────────────────────────────────────────

def compute_pairwise_accuracy(
    state: "ClusteringState",
    bag: PairBag,
    turn_index: int,
    sample_size: int = 50,
) -> float:
    """
    Compute pairwise validation accuracy for current state vs. oracle preferences (D-19).

    Samples min(sample_size, bag_size) pairs deterministically using turn_index as seed.
    A pair (x, y, True) matches if state.assignments[x] == state.assignments[y].
    A pair (x, y, False) matches if state.assignments[x] != state.assignments[y].

    Returns 0.0 if the bag is empty (no oracle feedback yet — not an error).
    """
    pairs = bag.sample(turn_index, n=sample_size)
    if not pairs:
        return 0.0

    matches = 0
    for pair in pairs:
        same_in_state = state.assignments.get(pair.item_x) == state.assignments.get(pair.item_y)
        if same_in_state == pair.expected_same:
            matches += 1
    return matches / len(pairs)


# ── Turn metric assembly ──────────────────────────────────────────────────────

def assemble_turn_metrics(
    state: "ClusteringState",
    reply: "OracleReply",
    pair_acc: float,
    stop_reason: "StopReason | None",
    action_type: str,
    experiment_id: int,
    cumulative_contradiction_count: int,
    sample_size: int = 50,
) -> "TurnCreate":
    """
    Build a TurnCreate Pydantic model from per-turn signals (D-26, JUDG-02).

    Args:
        state:                           New ClusteringState after f_next_state (turn_index already incremented).
        reply:                           OracleReply from oracle.reply() — carries contradiction_detected + cognitive_load.
        pair_acc:                        Pairwise accuracy float from compute_pairwise_accuracy() this turn.
        stop_reason:                     StopReason if loop is stopping this turn, else None.
        action_type:                     The action.action_type string from f_next_best_step().
        experiment_id:                   FK to the experiments row for this run.
        cumulative_contradiction_count:  Running total of contradictions so far in this run (NOT a per-turn binary
                                         flag). Caller tracks the counter; this function just stores it (D-02 DB-02).
        sample_size:                     Number of pairs sampled (stored in details JSON for reproducibility).

    Returns:
        TurnCreate ready to pass to src.db.turns.create().
    """
    from src.db.turns import TurnCreate

    convergence_signal = stop_reason.value if stop_reason is not None else None

    return TurnCreate(
        experiment_id=experiment_id,
        turn_index=state.turn_index,
        action_type=action_type,
        cognitive_load_score=reply.turn_cognitive_load,
        cumulative_contradiction_count=cumulative_contradiction_count,
        convergence_signal=convergence_signal,
        details={
            "pairwise_accuracy": pair_acc,
            "pairwise_sample_size": sample_size,
        },
    )


def assemble_feedback_rows(
    turn_id: int,
    deltas: list[FeedbackDelta],
    reply: "OracleReply",
) -> list["OracleFeedbackCreate"]:
    """
    Build OracleFeedbackCreate rows — one per FeedbackDelta (DB-03).

    Compound oracle messages produce multiple rows (DB-03 requirement).
    parsed_delta stores type + fields as a dict for easy JSON serialization.

    Args:
        turn_id:   FK to the turns row just created.
        deltas:    FeedbackDelta list from parse_feedback().
        reply:     OracleReply — raw_text + contradiction_detected carried to each row.

    Returns:
        List of OracleFeedbackCreate, one per delta. Empty list if deltas=[].
    """
    from src.db.oracle_feedback import OracleFeedbackCreate

    rows = []
    for delta in deltas:
        type_name = type(delta).__name__
        # Serialize delta fields to dict for parsed_delta JSON column
        parsed = dataclasses.asdict(delta) if dataclasses.is_dataclass(delta) else {}
        rows.append(OracleFeedbackCreate(
            turn_id=turn_id,
            feedback_type=type_name,
            raw_text=reply.raw_text,
            parsed_delta=parsed,
            is_contradiction=reply.contradiction_detected,
        ))
    return rows


# ── No-dialogue baseline ──────────────────────────────────────────────────────

def run_baseline(
    records: list[dict],
    persona_id: str,
    seed: int,
    db: "sqlite3.Connection",
    oracle: "OracleProtocol | None" = None,
    llm_client: object | None = None,
    dataset_name: str = "default",
    criteria: "StoppingCriteria | None" = None,
) -> "ExperimentRead":
    """
    Run the no-dialogue baseline: build initial clustering, show oracle once, compute metrics (JUDG-03).

    This is the reusable function imported by examples/run_baseline.py (D-25) and Phase 5 harness.
    strategy_id is always "no_dialogue" (D-23).

    Write order per D-03: JSONL not applicable for baseline (no session dir);
    DB commit is the persistence point.

    Args:
        records:       List of {"item_id": int, "text": str} dicts.
        persona_id:    Oracle persona identifier (OracleSpec persona) for experiments.persona_id.
        seed:          Random seed for clustering backend + pairwise sampling.
        db:            Open sqlite3.Connection (caller owns lifecycle per D-04).
        oracle:        OracleProtocol instance. If None, uses a single-turn MockOracle with satisfied=True.
        dataset_name:  Human-readable dataset identifier written to experiments.dataset.
        criteria:      StoppingCriteria. None → defaults from StoppingCriteria().

    Returns:
        ExperimentRead with id, slug, and headline metrics in details JSON.
    """
    from datetime import datetime, timezone

    # NOTE: AnthropicClusterNamer intentionally NOT imported here — run_baseline uses
    # _NullNamer (generic labels) so it never calls an LLM for naming (D-22, D-26).
    from src.clustering import HDBSCANBackend, build_initial_clustering_state
    from src.conversation_loop import _format_message
    from src.db import experiments as exp_db
    from src.db import oracle_feedback as fb_db
    from src.db import turns as turn_db
    from src.db.connection import init_schema
    from src.db.experiments import ExperimentCreate, ExperimentUpdate
    from src.oracle_protocol import MockOracle, OracleReply
    from src.stopping import StoppingCriteria, compute_magnitude, FeedbackMagnitudeWeights
    from src.strategy import RandomStrategy
    from src.uncertainty import f_uncertainty
    from src.agent_functions import f_next_best_step

    if criteria is None:
        criteria = StoppingCriteria()

    init_schema(db)

    start_ts = datetime.now(timezone.utc).isoformat()

    # Create experiment row (strategy_id="no_dialogue" per D-23)
    exp_name = f"no_dialogue-{persona_id}-seed{seed}"
    exp_create = ExperimentCreate(
        name=exp_name,
        strategy_id="no_dialogue",
        persona_id=persona_id,
        seed=seed,
        dataset=dataset_name,
        start_timestamp=start_ts,
        details={},
    )
    experiment = exp_db.create(db, exp_create)

    # Build initial clustering (FOUND-02: HDBSCAN default)
    texts = [r["text"] for r in records]

    # Import EmbeddingStore for embeddings
    from src.embedding_store import EmbeddingStore

    # Use in-memory embedding computation — no file persistence for baseline
    # EmbeddingStore.compute_and_save requires a file path; use a temp path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        store = EmbeddingStore.compute_and_save(texts, tmp_path)
    finally:
        # Cleanup temp file after store is loaded (numpy keeps data in memory)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    backend = HDBSCANBackend()

    # Null namer returns generic cluster labels — no LLM call, no AnthropicClusterNamer.
    # IMPORTANT (V-4-01): name_cluster() MUST return dict[str, str] with "name" and
    # "description" keys. build_initial_clustering_state() accesses naming_result["name"]
    # and naming_result["description"]. A plain string return raises TypeError at runtime.
    class _NullNamer:
        def name_cluster(self, items, cluster_id=None):
            return {"name": f"Cluster {cluster_id}", "description": ""}

        def describe_cluster(self, items, name, cluster_id=None):
            return ""

    initial_state = build_initial_clustering_state(
        store.get_all(), records, _NullNamer(), backend=backend
    )

    # Get one oracle reply (D-22: oracle runs ONCE on initial clustering)
    if oracle is None:
        oracle = MockOracle(script=[OracleReply(raw_text="", satisfied=True, turn_cognitive_load=0.0)])

    uncertainty_report = f_uncertainty(initial_state)
    action = f_next_best_step(initial_state, RandomStrategy(seed=seed), uncertainty_report)
    message = _format_message(action, initial_state)
    reply = oracle.reply(initial_state, message, global_instructions=[], cognitive_load=0.0)

    # Parse feedback from reply (D-22: oracle runs once; parse_feedback extracts FeedbackDeltas)
    # Fast path: empty raw_text → [] (no LLM call needed)
    from src.feedback_parser import parse_feedback
    deltas = parse_feedback(reply.raw_text, initial_state, llm_client) if llm_client and reply.raw_text else []

    # Compute pairwise accuracy against the one-turn bag
    bag = PairBag()
    bag.update(deltas, initial_state, is_contradiction=False)
    pair_acc = compute_pairwise_accuracy(initial_state, bag, turn_index=0)

    # Assemble turn row (one turn, turn_index=0)
    from src.stopping import check_stopping
    stop_reason = check_stopping(
        turn_index=0,
        oracle_satisfied=reply.satisfied,
        recent_magnitudes=[],
        criteria=criteria,
    )
    turn_create = assemble_turn_metrics(
        state=initial_state,
        reply=reply,
        pair_acc=pair_acc,
        stop_reason=stop_reason,
        action_type=action.action_type,
        experiment_id=experiment.id,
        cumulative_contradiction_count=0,  # baseline = 1 turn, no prior contradictions
    )
    turn_row = turn_db.create(db, turn_create)

    # Insert oracle_feedback rows (one per delta; empty for no-dialogue baseline)
    fb_rows = assemble_feedback_rows(turn_row.id, deltas, reply)
    for fb in fb_rows:
        fb_db.create(db, fb)

    # Seal experiment row with summary metrics
    weights = FeedbackMagnitudeWeights()
    magnitude = compute_magnitude(deltas, weights)

    end_ts = datetime.now(timezone.utc).isoformat()
    experiment = exp_db.update(db, experiment.id, ExperimentUpdate(
        total_turns=1,
        convergence_reason=stop_reason.value if stop_reason else None,
        end_timestamp=end_ts,
        details={
            "mean_cognitive_load": reply.turn_cognitive_load,
            "mean_pairwise_accuracy": pair_acc,
            "contradiction_count": 0,
            "turns_to_convergence": 1,
            "oracle_satisfied": reply.satisfied,
        },
    ))

    return experiment
