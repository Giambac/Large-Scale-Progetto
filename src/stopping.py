"""
stopping.py — Stopping criteria specification (PRE-02).

Three OR-combined conditions. Any one firing stops the conversation loop.
Priority order: oracle_satisfied > turn_budget > diminishing_returns.

D-08: Oracle satisfaction — primary: OracleReply.satisfied=True token.
      Secondary fallback: magnitude drops below epsilon for N turns (Phase 4).
D-09: Turn budget — hard cap of 15 turns. Fires when turn_index >= 15.
D-10: Diminishing returns — weighted feedback magnitude near zero for N turns.
      Weighting scheme: global > cluster-level > point-level > instructional.
      Exact weights and threshold are Phase 4 decisions; Phase 1 specifies structure only.

Phase 4 implemented:
  1. FeedbackMagnitudeWeights fields set to real float values (D-13).
  2. StoppingCriteria.magnitude_threshold_epsilon = 0.05 (D-14).
  3. StoppingCriteria.magnitude_fallback_turns = 3 (D-15).
  4. Diminishing-returns branch implemented in check_stopping() (D-10).
  5. compute_magnitude() added for use by conversation_loop.py (Plan 04).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StopReason(Enum):
    """The reason the conversation loop stopped. Exactly three values (D-08, D-09, D-10)."""
    ORACLE_SATISFIED = "oracle_satisfied"
    TURN_BUDGET = "turn_budget"
    DIMINISHING_RETURNS = "diminishing_returns"


@dataclass(frozen=True)
class StoppingCriteria:
    """
    Configuration for all three stopping conditions.

    turn_budget: Hard cap — loop stops unconditionally at turn_index >= turn_budget.
                 Default 15 (D-09). Do not change this value without updating tests.

    magnitude_threshold_epsilon: Threshold for diminishing-returns condition (D-10).
                                 Set to float("nan") until Phase 4 fills in the value.

    magnitude_fallback_turns: Number of consecutive turns below epsilon to trigger
                              diminishing returns (D-08 secondary fallback).
                              Set to -1 until Phase 4 fills in the value.
    """
    turn_budget: int = 15
    magnitude_threshold_epsilon: float = 0.05   # D-14
    magnitude_fallback_turns: int = 3            # D-15


@dataclass
class FeedbackMagnitudeWeights:
    """
    Weighting scheme for feedback magnitude computation (D-10).

    Ordering: global_feedback > cluster_level > point_level > instructional.
    Exact float values are Phase 4 decisions; all fields are nan until then.

    Phase 4 must replace nan with real positive floats.
    The weighted magnitude formula:
        magnitude = (
            w.global_feedback * count_global
            + w.cluster_level * count_cluster
            + w.point_level * count_point
            + w.instructional * count_instructional
        )
    """
    global_feedback: float = 1.0   # D-13 — highest weight
    cluster_level: float = 0.5     # D-13
    point_level: float = 0.2       # D-13
    instructional: float = 0.1     # D-13 — lowest weight


def check_stopping(
    turn_index: int,
    oracle_satisfied: bool,
    recent_magnitudes: list[float],
    criteria: StoppingCriteria,
) -> Optional[StopReason]:
    """
    Evaluate all stopping conditions. Returns the first triggered condition, or None.

    Args:
        turn_index: 0-based index of the current turn.
        oracle_satisfied: True if the oracle emitted an explicit satisfaction token.
        recent_magnitudes: Sequence of feedback magnitude values for recent turns.
                           Used by the diminishing-returns condition (stub in Phase 1).
        criteria: StoppingCriteria configuration instance.

    Returns:
        StopReason if a condition fires, else None (loop should continue).

    Condition priority (OR-combined, first wins):
        1. oracle_satisfied (D-08 primary)
        2. turn_index >= turn_budget (D-09)
        3. diminishing returns (D-10) — stub; Phase 4 implements threshold check
    """
    # D-08: primary oracle satisfaction token
    if oracle_satisfied:
        return StopReason.ORACLE_SATISFIED

    # D-09: hard turn budget — unconditional
    if turn_index >= criteria.turn_budget:
        return StopReason.TURN_BUDGET

    # D-10: diminishing returns — check last N turns for epsilon silence
    if criteria.magnitude_fallback_turns > 0 and len(recent_magnitudes) >= criteria.magnitude_fallback_turns:
        last_n = recent_magnitudes[-criteria.magnitude_fallback_turns:]
        if all(m < criteria.magnitude_threshold_epsilon for m in last_n):
            return StopReason.DIMINISHING_RETURNS

    return None


def compute_magnitude(deltas: list, weights: FeedbackMagnitudeWeights) -> float:
    """
    Compute weighted feedback magnitude from a list of FeedbackDelta objects.

    Used by conversation_loop.py to populate recent_magnitudes for check_stopping().
    Replaces the Phase 2 stub of float(len(deltas)) (conversation_loop.py line 247).

    Args:
        deltas: List of FeedbackDelta objects from parse_feedback().
        weights: FeedbackMagnitudeWeights with non-nan float values.

    Returns:
        Scalar magnitude >= 0.0.
    """
    from src.feedback import GlobalFeedback, SplitFeedback, MergeFeedback, MoveItemFeedback, InstructionalFeedback
    assert not math.isnan(weights.global_feedback), "FeedbackMagnitudeWeights must be filled in before calling compute_magnitude"
    total = 0.0
    for d in deltas:
        if isinstance(d, GlobalFeedback):
            total += weights.global_feedback
        elif isinstance(d, (SplitFeedback, MergeFeedback)):
            total += weights.cluster_level
        elif isinstance(d, MoveItemFeedback):
            total += weights.point_level
        elif isinstance(d, InstructionalFeedback):
            total += weights.instructional
    return total
