"""
tests/test_judge.py — Unit tests for src/judge.py (JUDG-01, JUDG-02).

Tests use minimal in-memory ClusteringState instances — no LLM calls, no file I/O.
"""
import os

import pytest

from src.feedback import (
    GlobalFeedback,
    InstructionalFeedback,
    MergeFeedback,
    MoveItemFeedback,
    SplitFeedback,
)
from src.judge import PairBag, compute_pairwise_accuracy, assemble_turn_metrics, assemble_feedback_rows
from src.logging_setup import UnexpectedDeviation
from src.oracle_protocol import OracleReply
from src.stopping import check_stopping, StoppingCriteria, StopReason, compute_magnitude, FeedbackMagnitudeWeights


# ── Stopping conditions (JUDG-01) ────────────────────────────────────────────

def test_oracle_satisfied_stops():
    c = StoppingCriteria()
    assert check_stopping(0, True, [], c) == StopReason.ORACLE_SATISFIED


def test_turn_budget_stops():
    c = StoppingCriteria(turn_budget=5)
    assert check_stopping(5, False, [], c) == StopReason.TURN_BUDGET


def test_diminishing_returns_stops():
    c = StoppingCriteria(magnitude_threshold_epsilon=0.05, magnitude_fallback_turns=3)
    # Three turns all below epsilon
    assert check_stopping(0, False, [0.01, 0.01, 0.01], c) == StopReason.DIMINISHING_RETURNS


def test_diminishing_returns_requires_n_turns():
    c = StoppingCriteria(magnitude_threshold_epsilon=0.05, magnitude_fallback_turns=3)
    # Only 2 turns below epsilon — not enough
    assert check_stopping(0, False, [0.01, 0.01], c) is None


def test_no_stop_condition():
    c = StoppingCriteria()
    assert check_stopping(0, False, [1.0, 0.5], c) is None


# ── Magnitude computation ─────────────────────────────────────────────────────

def test_compute_magnitude_weights():
    w = FeedbackMagnitudeWeights()
    deltas = [GlobalFeedback("recluster"), MoveItemFeedback(1, 2)]
    mag = compute_magnitude(deltas, w)
    assert abs(mag - (1.0 + 0.2)) < 1e-9


def test_compute_magnitude_empty():
    w = FeedbackMagnitudeWeights()
    assert compute_magnitude([], w) == 0.0


# ── PairBag pair extraction ───────────────────────────────────────────────────

def _make_state(assignments: dict, clusters: list):
    """Build a minimal ClusteringState-like object for testing."""
    from dataclasses import dataclass

    @dataclass
    class _Cluster:
        id: int
        item_ids: list
        name: str = "test"
        description: str = ""

    @dataclass
    class _State:
        assignments: dict
        clusters: list
        turn_index: int = 0
        soft_probs: dict = None
        timestamp: str = "2026-01-01T00:00:00+00:00"

        def __post_init__(self):
            if self.soft_probs is None:
                self.soft_probs = {}

    return _State(
        assignments=assignments,
        clusters=[_Cluster(id=cid, item_ids=items) for cid, items in clusters],
    )


def test_pair_bag_move_feedback():
    state = _make_state(
        assignments={0: 1, 1: 1, 2: 2},
        clusters=[(1, [0, 1]), (2, [2])],
    )
    bag = PairBag()
    bag.update([MoveItemFeedback(item_id=2, target_cluster_id=1)], state)
    assert len(bag) == 2  # (2,0,True) + (2,1,True)


def test_pair_bag_split_feedback():
    state = _make_state(assignments={0: 1, 1: 1}, clusters=[(1, [0, 1])])
    bag = PairBag()
    bag.update([SplitFeedback(cluster_id=1, seed_item_ids=[0, 1])], state)
    assert len(bag) == 1  # (0, 1, False)


def test_pair_bag_merge_feedback():
    state = _make_state(
        assignments={0: 1, 1: 2},
        clusters=[(1, [0]), (2, [1])],
    )
    bag = PairBag()
    bag.update([MergeFeedback(cluster_a_id=1, cluster_b_id=2)], state)
    assert len(bag) == 1  # (0, 1, True)


def test_pair_bag_global_feedback_no_pairs():
    state = _make_state(assignments={0: 1}, clusters=[(1, [0])])
    bag = PairBag()
    bag.update([GlobalFeedback("recluster")], state)
    assert len(bag) == 0


def test_pair_bag_instructional_no_pairs():
    state = _make_state(assignments={0: 1}, clusters=[(1, [0])])
    bag = PairBag()
    bag.update([InstructionalFeedback("hint")], state)
    assert len(bag) == 0


def test_pair_bag_contradiction_overwrites():
    """D-20: contradicting delta drops old pairs for affected items."""
    state = _make_state(
        assignments={0: 1, 1: 2},
        clusters=[(1, [0]), (2, [1])],
    )
    bag = PairBag()
    # First: merge → (0, 1, True)
    bag.update([MergeFeedback(cluster_a_id=1, cluster_b_id=2)], state)
    assert len(bag) == 1

    # Now contradiction: split → should overwrite
    state2 = _make_state(
        assignments={0: 3, 1: 3},
        clusters=[(3, [0, 1])],
    )
    bag.update([SplitFeedback(cluster_id=3, seed_item_ids=[0, 1])], state2, is_contradiction=True)
    # Old pair (0,1,True) dropped; new pair (0,1,False) added
    assert len(bag) == 1
    pairs = bag.sample(0, n=50)
    assert pairs[0].expected_same is False


# ── Pairwise accuracy ────────────────────────────────────────────────────────

def test_pairwise_accuracy_empty_bag():
    state = _make_state(assignments={0: 1}, clusters=[(1, [0])])
    bag = PairBag()
    assert compute_pairwise_accuracy(state, bag, 0) == 0.0


def test_pairwise_accuracy_perfect():
    """All pairs match oracle expectation → 1.0."""
    state = _make_state(
        assignments={0: 1, 1: 1},
        clusters=[(1, [0, 1])],
    )
    bag = PairBag()
    # Oracle says 0 and 1 should be same-cluster → state has them in cluster 1 → perfect
    bag.update([MoveItemFeedback(item_id=0, target_cluster_id=1)], state)
    acc = compute_pairwise_accuracy(state, bag, turn_index=0)
    assert acc == 1.0


# ── assemble_turn_metrics ─────────────────────────────────────────────────────

def test_assemble_turn_metrics_builds_turn_create():
    state = _make_state(assignments={0: 1}, clusters=[(1, [0])])
    state.turn_index = 3
    reply = OracleReply(raw_text="ok", satisfied=False, turn_cognitive_load=0.4)
    tc = assemble_turn_metrics(
        state=state,
        reply=reply,
        pair_acc=0.75,
        stop_reason=None,
        action_type="ask_question",
        experiment_id=1,
        cumulative_contradiction_count=0,
    )
    assert tc.turn_index == 3
    assert tc.cognitive_load_score == 0.4
    assert tc.convergence_signal is None
    assert tc.details["pairwise_accuracy"] == 0.75


def test_assemble_turn_metrics_with_stop_reason():
    state = _make_state(assignments={0: 1}, clusters=[(1, [0])])
    state.turn_index = 15
    reply = OracleReply(raw_text="done", satisfied=True, turn_cognitive_load=0.1)
    tc = assemble_turn_metrics(
        state=state,
        reply=reply,
        pair_acc=0.9,
        stop_reason=StopReason.ORACLE_SATISFIED,
        action_type="stop",
        experiment_id=1,
        cumulative_contradiction_count=2,
    )
    assert tc.convergence_signal == "oracle_satisfied"
    assert tc.cumulative_contradiction_count == 2


# ── assemble_feedback_rows ────────────────────────────────────────────────────

def test_assemble_feedback_rows_compound():
    """Compound oracle message = multiple rows (DB-03)."""
    deltas = [SplitFeedback(1, [0, 1]), MergeFeedback(2, 3)]
    reply = OracleReply(raw_text="split and merge", satisfied=False)
    rows = assemble_feedback_rows(turn_id=42, deltas=deltas, reply=reply)
    assert len(rows) == 2
    assert rows[0].feedback_type == "SplitFeedback"
    assert rows[1].feedback_type == "MergeFeedback"
    assert rows[0].turn_id == 42


def test_assemble_feedback_rows_empty():
    reply = OracleReply(raw_text="", satisfied=False)
    rows = assemble_feedback_rows(turn_id=1, deltas=[], reply=reply)
    assert rows == []


# ── deviation() / STRICT_MODE ─────────────────────────────────────────────────

def test_deviation_strict_mode():
    """deviation() raises UnexpectedDeviation when STRICT_MODE=1 (D-29)."""
    original = os.environ.get("STRICT_MODE", "0")
    try:
        os.environ["STRICT_MODE"] = "1"
        from src.logging_setup import deviation
        with pytest.raises(UnexpectedDeviation):
            deviation("test unexpected branch", key="val")
    finally:
        os.environ["STRICT_MODE"] = original


def test_deviation_normal_mode():
    """deviation() does NOT raise when STRICT_MODE is not set."""
    original = os.environ.get("STRICT_MODE", "0")
    try:
        os.environ["STRICT_MODE"] = "0"
        from src.logging_setup import deviation
        deviation("this should just log a warning")  # must not raise
    finally:
        os.environ["STRICT_MODE"] = original
