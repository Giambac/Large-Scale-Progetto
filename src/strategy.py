"""
strategy.py — StrategyProtocol, Action, RandomStrategy.

Phase 5 adds UncertaintyDrivenStrategy and BoundaryDrivenStrategy without changing this file.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from src.state import ClusteringState


@dataclass
class Action:
    """
    Represents one possible next action for the agent to take.

    action_type: One of the four valid action types for Phase 2.
    payload:     Optional metadata (cluster_id, item_ids, question_text, etc.).
                 Phase 2 uses empty dict; Phase 5 enriches payloads.
    """
    action_type: Literal["show_full", "show_subset", "ask_question", "stop"]
    payload: dict = field(default_factory=dict)


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Interface for action-selection strategies.

    Phase 2: RandomStrategy (uniform random over valid actions).
    Phase 5: UncertaintyDrivenStrategy, BoundaryDrivenStrategy.

    Structural subtyping: implementations do NOT need to inherit from this class.
    """
    def select(self, state: ClusteringState, uncertainty_report: "object") -> Action:
        ...


def _enumerate_valid_actions(
    state: ClusteringState,
    uncertainty_report: object,
) -> list[Action]:
    """
    Build a non-empty list of valid Action objects for the current state.

    Phase 2 payload is always {} — Phase 5 enriches payloads with specific cluster_ids
    and item_ids drawn from the uncertainty_report.

    Rules:
      - "show_full"    always valid
      - "ask_question" always valid if there are clusters (any question makes sense)
      - "show_subset"  valid if there are >= 2 clusters (subset requires at least 2 to show)
      - "stop"         always valid (loop checks oracle satisfaction separately)
    """
    actions: list[Action] = []
    actions.append(Action(action_type="show_full"))
    if len(state.clusters) > 0:
        actions.append(Action(action_type="ask_question"))
    if len(state.clusters) >= 2:
        actions.append(Action(action_type="show_subset"))
    actions.append(Action(action_type="stop"))
    assert len(actions) > 0, (
        "BUG: _enumerate_valid_actions returned empty list — state has no clusters at all"
    )
    return actions


class RandomStrategy:
    """
    Phase 2 implementation: uniform random selection over valid actions.

    Uses a seeded random.Random instance (NOT the global random module) to ensure
    determinism when a seed is provided. Tests always pass a seed (T-02-07).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, state: ClusteringState, uncertainty_report: object) -> Action:
        """Select a uniformly random valid action for the current state."""
        valid_actions = _enumerate_valid_actions(state, uncertainty_report)
        assert len(valid_actions) > 0, "BUG: no valid actions to select from"
        return self._rng.choice(valid_actions)


class UncertaintyDrivenStrategy:
    """
    Phase 5 strategy (ALAB-01, D-01, D-02).

    Targets the cluster with the highest mean per-item entropy
    (split_candidates[0]) and emits a focused ask_question. Plan 02's
    enriched _format_message renders this as a cluster-targeted prompt.

    Degenerate fallback (D-02): when split_candidates is empty (K=0, all
    clusters empty, or all items equally confident), emit ask_question
    with an empty payload and fire deviation(). Hands initiative to the
    oracle ("what should we focus on?"). NEVER crashes; NEVER random-
    fallbacks (that would defeat the ablation signal).

    Determinism: tie-breaking among equally-ranked split_candidates is
    handled by random.Random(seed) — RandomStrategy pattern (Phase 2 D-19,
    Phase 4 D-19 carry-forward). When the underlying ranking has a clear
    winner, the seed has no effect.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, state: ClusteringState, uncertainty_report: object) -> Action:
        from src.uncertainty import UncertaintyReport
        from src.logging_setup import deviation
        assert isinstance(uncertainty_report, UncertaintyReport), (
            f"UncertaintyDrivenStrategy.select expected UncertaintyReport, "
            f"got {type(uncertainty_report).__name__}"
        )

        # Defensive: _enumerate_valid_actions guarantees a non-empty action list,
        # but UncertaintyDriven does not pick from it — it constructs its own Action.
        # Still call it to keep the invariant alive (mirrors RandomStrategy contract).
        _ = _enumerate_valid_actions(state, uncertainty_report)

        split = uncertainty_report.split_candidates
        if not split:
            deviation(
                "UncertaintyDrivenStrategy: no split_candidates — falling back to generic ask_question",
                n_clusters=len(state.clusters),
                turn_index=state.turn_index,
            )
            return Action(action_type="ask_question", payload={})

        # Tie-break among top candidates with identical mean entropy using seeded RNG.
        top_entropy = split[0][1]
        top_tied = [cid for cid, h in split if h == top_entropy]
        cluster_id = self._rng.choice(top_tied)
        return Action(action_type="ask_question", payload={"cluster_id": int(cluster_id)})


class BoundaryDrivenStrategy:
    """
    Phase 5 strategy (ALAB-01, D-03, D-04).

    Targets the most-confused cluster pair (merge_candidates[0]) and
    collects the ambiguous-zone items — items whose soft_probs put both
    P(A) and P(B) above a threshold (default 0.3) such that {A, B} are
    the dominant pair. Emits show_subset so Plan 02's enriched
    _format_message renders the item texts alongside the two cluster
    names. The oracle sees a group of ambiguous items as a coherent
    unit and can give merge/split/rule feedback on the pattern.

    Algorithm (D-03):
      1. Pair (A, B) = merge_candidates[0][:2]  (closest pair).
      2. For each item i in state.soft_probs:
         - p_a = soft_probs[i][pos[A]],  p_b = soft_probs[i][pos[B]]
         - if p_a >= AMBIGUOUS_P_MIN AND p_b >= AMBIGUOUS_P_MIN AND
           {A, B} are the top-2 dominant clusters for i:
               -> add i to ambiguous_items
      3. Cap the subset at MAX_SUBSET_SIZE for cognitive-load reasons (D-05);
         deterministic sample via random.Random(state.turn_index)
         (mirrors PairBag.sample pattern — Phase 4 D-19, CONTEXT.md specifics line 174).
      4. Emit show_subset with full triple payload.

    Degenerate fallback (D-04): if merge_candidates is empty OR the
    ambiguous zone is empty, fall back to ask_question with empty
    payload + deviation(). NEVER crash; NEVER random-fallback.
    """

    # Class-level constants (planner discretion per CONTEXT.md Claude's Discretion list).
    AMBIGUOUS_P_MIN: float = 0.3   # threshold from CONTEXT.md "P(A) > 0.3 AND P(B) > 0.3"
    MAX_SUBSET_SIZE: int = 8       # cognitive-load cap; bounds oracle prompt size

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, state: ClusteringState, uncertainty_report: object) -> Action:
        from src.uncertainty import UncertaintyReport
        from src.logging_setup import deviation
        assert isinstance(uncertainty_report, UncertaintyReport), (
            f"BoundaryDrivenStrategy.select expected UncertaintyReport, "
            f"got {type(uncertainty_report).__name__}"
        )

        _ = _enumerate_valid_actions(state, uncertainty_report)

        merge = uncertainty_report.merge_candidates
        if not merge:
            deviation(
                "BoundaryDrivenStrategy: no merge_candidates — falling back to generic ask_question",
                n_clusters=len(state.clusters),
                turn_index=state.turn_index,
            )
            return Action(action_type="ask_question", payload={})

        cluster_a, cluster_b, _dist = merge[0]

        # Build cluster_id -> positional index in soft_probs vector.
        # CRITICAL: soft_probs[item][j] corresponds to state.clusters[j].id (positional).
        pos = {c.id: j for j, c in enumerate(state.clusters)}
        if cluster_a not in pos or cluster_b not in pos:
            deviation(
                "BoundaryDrivenStrategy: merge_candidates references unknown cluster id",
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                known_ids=list(pos.keys()),
            )
            return Action(action_type="ask_question", payload={})

        ia, ib = pos[cluster_a], pos[cluster_b]
        ambiguous_items: list[int] = []
        for item_id, probs in state.soft_probs.items():
            p_a = probs[ia]
            p_b = probs[ib]
            if p_a < self.AMBIGUOUS_P_MIN or p_b < self.AMBIGUOUS_P_MIN:
                continue
            # {A, B} must be the top-2 dominant clusters for this item.
            top_two_indices = sorted(
                range(len(probs)), key=lambda j: probs[j], reverse=True
            )[:2]
            if set(top_two_indices) == {ia, ib}:
                ambiguous_items.append(int(item_id))

        if not ambiguous_items:
            deviation(
                "BoundaryDrivenStrategy: ambiguous zone empty for top merge pair — falling back",
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                n_items=len(state.soft_probs),
                turn_index=state.turn_index,
            )
            return Action(action_type="ask_question", payload={})

        # Cap subset deterministically by state.turn_index (mirrors PairBag.sample, D-19).
        ambiguous_items.sort()  # stable input to sampler
        if len(ambiguous_items) > self.MAX_SUBSET_SIZE:
            turn_rng = random.Random(state.turn_index)
            ambiguous_items = sorted(
                turn_rng.sample(ambiguous_items, self.MAX_SUBSET_SIZE)
            )

        return Action(
            action_type="show_subset",
            payload={
                "cluster_a": int(cluster_a),
                "cluster_b": int(cluster_b),
                "item_ids": ambiguous_items,
            },
        )
