"""Tests for stopping.py — PRE-02: stopping criteria spec."""
import pytest

from src.stopping import StoppingCriteria, StopReason, check_stopping, FeedbackMagnitudeWeights


def test_stopping_criteria_turn_budget_default():
    """Default StoppingCriteria.turn_budget is 15 (per D-09)."""
    criteria = StoppingCriteria()
    assert criteria.turn_budget == 15


def test_stop_reason_enum_values():
    """StopReason has exactly three members matching D-08, D-09, D-10."""
    reasons = {r.value for r in StopReason}
    assert "oracle_satisfied" in reasons
    assert "turn_budget" in reasons
    assert "diminishing_returns" in reasons


def test_check_stopping_oracle_satisfied():
    """oracle_satisfied=True returns StopReason.ORACLE_SATISFIED regardless of turn index."""
    criteria = StoppingCriteria()
    result = check_stopping(
        turn_index=0,
        oracle_satisfied=True,
        recent_magnitudes=[],
        criteria=criteria,
    )
    assert result == StopReason.ORACLE_SATISFIED


def test_check_stopping_turn_budget_fires_at_15():
    """turn_index=15 returns StopReason.TURN_BUDGET (per D-09: fires when turn_index >= 15)."""
    criteria = StoppingCriteria()
    result = check_stopping(
        turn_index=15,
        oracle_satisfied=False,
        recent_magnitudes=[],
        criteria=criteria,
    )
    assert result == StopReason.TURN_BUDGET


def test_check_stopping_turn_budget_fires_past_15():
    """turn_index > 15 also fires TURN_BUDGET."""
    criteria = StoppingCriteria()
    result = check_stopping(
        turn_index=20,
        oracle_satisfied=False,
        recent_magnitudes=[],
        criteria=criteria,
    )
    assert result == StopReason.TURN_BUDGET


def test_check_stopping_no_trigger_mid_conversation():
    """Mid-conversation (turn_index=7, not satisfied) returns None."""
    criteria = StoppingCriteria()
    result = check_stopping(
        turn_index=7,
        oracle_satisfied=False,
        recent_magnitudes=[1.0, 0.8, 0.6],
        criteria=criteria,
    )
    assert result is None


def test_oracle_satisfied_beats_turn_budget():
    """Oracle satisfaction takes priority over turn budget (checked first)."""
    criteria = StoppingCriteria()
    result = check_stopping(
        turn_index=15,
        oracle_satisfied=True,
        recent_magnitudes=[],
        criteria=criteria,
    )
    assert result == StopReason.ORACLE_SATISFIED


def test_feedback_magnitude_weights_has_four_fields():
    """FeedbackMagnitudeWeights has fields for all four feedback types (D-10)."""
    w = FeedbackMagnitudeWeights()
    assert hasattr(w, "global_feedback")
    assert hasattr(w, "cluster_level")
    assert hasattr(w, "point_level")
    assert hasattr(w, "instructional")
