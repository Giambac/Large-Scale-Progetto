"""Tests for feedback.py — FB-01, FB-02, FB-03: FeedbackDelta union type and dataclasses."""
import pytest


def test_split_feedback_construction():
    """SplitFeedback stores cluster_id and seed_item_ids (may be empty)."""
    from src.feedback import SplitFeedback
    fb = SplitFeedback(cluster_id=2, seed_item_ids=[10, 20])
    assert fb.cluster_id == 2
    assert fb.seed_item_ids == [10, 20]


def test_split_feedback_empty_seeds():
    """SplitFeedback with empty seed_item_ids is valid (k-means++ fallback per D-08)."""
    from src.feedback import SplitFeedback
    fb = SplitFeedback(cluster_id=3, seed_item_ids=[])
    assert fb.seed_item_ids == []


def test_merge_feedback_construction():
    """MergeFeedback stores cluster_a_id and cluster_b_id."""
    from src.feedback import MergeFeedback
    fb = MergeFeedback(cluster_a_id=0, cluster_b_id=1)
    assert fb.cluster_a_id == 0
    assert fb.cluster_b_id == 1


def test_move_item_feedback_construction():
    """MoveItemFeedback stores item_id and target_cluster_id."""
    from src.feedback import MoveItemFeedback
    fb = MoveItemFeedback(item_id=5, target_cluster_id=2)
    assert fb.item_id == 5
    assert fb.target_cluster_id == 2


def test_global_feedback_construction():
    """GlobalFeedback stores instruction_text string."""
    from src.feedback import GlobalFeedback
    fb = GlobalFeedback(instruction_text="Focus on sentiment.")
    assert fb.instruction_text == "Focus on sentiment."


def test_instructional_feedback_construction():
    """InstructionalFeedback stores instruction_text string."""
    from src.feedback import InstructionalFeedback
    fb = InstructionalFeedback(instruction_text="Treat X and Y as synonyms.")
    assert fb.instruction_text == "Treat X and Y as synonyms."


def test_feedback_dataclasses_are_frozen():
    """All FeedbackDelta subtypes are frozen — mutation must raise FrozenInstanceError."""
    from src.feedback import SplitFeedback
    fb = SplitFeedback(cluster_id=0, seed_item_ids=[])
    with pytest.raises(Exception):  # FrozenInstanceError is a subclass of AttributeError
        fb.cluster_id = 99


def test_oracle_move_confidence_is_0_95():
    """ORACLE_MOVE_CONFIDENCE module constant must equal 0.95 (per CONTEXT.md Specifics)."""
    from src.feedback import ORACLE_MOVE_CONFIDENCE
    assert ORACLE_MOVE_CONFIDENCE == 0.95


def test_uniform_fallback_threshold_is_positive():
    """UNIFORM_FALLBACK_THRESHOLD must be a small positive float (zero-sum edge case guard)."""
    from src.feedback import UNIFORM_FALLBACK_THRESHOLD
    assert UNIFORM_FALLBACK_THRESHOLD > 0


def test_split_feedback_is_feedbackdelta():
    """SplitFeedback is a valid FeedbackDelta (isinstance check via Union type)."""
    from src.feedback import (
        SplitFeedback, MergeFeedback, MoveItemFeedback,
        GlobalFeedback, InstructionalFeedback,
    )
    fb = SplitFeedback(cluster_id=1, seed_item_ids=[])
    assert isinstance(fb, (SplitFeedback, MergeFeedback, MoveItemFeedback,
                            GlobalFeedback, InstructionalFeedback))
