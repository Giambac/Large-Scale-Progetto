"""Tests for feedback_parser.py — parse_feedback() with MockLLMClient."""
import json
import pytest
from unittest.mock import MagicMock


def _make_mock_client(response_payload):
    """Build a mock Anthropic client returning response_payload as JSON text."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(response_payload))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_parse_feedback_returns_list(tiny_state_3cluster):
    """parse_feedback always returns a list (may be empty)."""
    from src.feedback_parser import parse_feedback
    client = _make_mock_client([])
    result = parse_feedback("No feedback.", tiny_state_3cluster, client)
    assert isinstance(result, list)


def test_parse_split_feedback(tiny_state_3cluster):
    """parse_feedback extracts SplitFeedback from oracle utterance."""
    from src.feedback_parser import parse_feedback
    from src.feedback import SplitFeedback
    payload = [{"type": "split", "cluster_id": 0, "seed_item_ids": [0, 1]}]
    client = _make_mock_client(payload)
    deltas = parse_feedback("Please split cluster 0.", tiny_state_3cluster, client)
    assert len(deltas) == 1
    assert isinstance(deltas[0], SplitFeedback)
    assert deltas[0].cluster_id == 0


def test_parse_merge_feedback(tiny_state_3cluster):
    """parse_feedback extracts MergeFeedback from oracle utterance."""
    from src.feedback_parser import parse_feedback
    from src.feedback import MergeFeedback
    payload = [{"type": "merge", "cluster_a_id": 0, "cluster_b_id": 1}]
    client = _make_mock_client(payload)
    deltas = parse_feedback("Merge clusters 0 and 1.", tiny_state_3cluster, client)
    assert len(deltas) == 1
    assert isinstance(deltas[0], MergeFeedback)


def test_parse_move_item_feedback(tiny_state_3cluster):
    """parse_feedback extracts MoveItemFeedback from oracle utterance."""
    from src.feedback_parser import parse_feedback
    from src.feedback import MoveItemFeedback
    payload = [{"type": "move_item", "item_id": 2, "target_cluster_id": 0}]
    client = _make_mock_client(payload)
    deltas = parse_feedback("Move item 2 to cluster 0.", tiny_state_3cluster, client)
    assert len(deltas) == 1
    assert isinstance(deltas[0], MoveItemFeedback)


def test_parse_compound_feedback(tiny_state_3cluster):
    """Compound oracle messages produce multiple FeedbackDelta objects."""
    from src.feedback_parser import parse_feedback
    payload = [
        {"type": "split", "cluster_id": 1, "seed_item_ids": []},
        {"type": "move_item", "item_id": 0, "target_cluster_id": 2},
    ]
    client = _make_mock_client(payload)
    deltas = parse_feedback("Split cluster 1 and move item 0 to cluster 2.", tiny_state_3cluster, client)
    assert len(deltas) == 2


def test_parse_empty_oracle_message(tiny_state_3cluster):
    """Empty oracle message returns empty list (no crash)."""
    from src.feedback_parser import parse_feedback
    client = _make_mock_client([])
    deltas = parse_feedback("", tiny_state_3cluster, client)
    assert deltas == []


def test_parse_feedback_crashes_on_invalid_cluster_id(tiny_state_3cluster):
    """parse_feedback raises AssertionError if LLM returns cluster_id not in current state."""
    from src.feedback_parser import parse_feedback
    payload = [{"type": "split", "cluster_id": 999, "seed_item_ids": []}]
    client = _make_mock_client(payload)
    with pytest.raises(AssertionError):
        parse_feedback("Split cluster 999.", tiny_state_3cluster, client)


def test_parse_feedback_crashes_on_unknown_type(tiny_state_3cluster):
    """parse_feedback raises AssertionError if LLM returns unknown feedback type."""
    from src.feedback_parser import parse_feedback
    payload = [{"type": "teleport", "cluster_id": 0}]
    client = _make_mock_client(payload)
    with pytest.raises(AssertionError):
        parse_feedback("Teleport cluster 0.", tiny_state_3cluster, client)


@pytest.mark.llm
def test_parse_feedback_real_llm(tiny_state_3cluster):
    """Integration test: parse_feedback with real Anthropic client (skipped in CI)."""
    import anthropic
    import os
    from src.feedback_parser import parse_feedback
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)
    deltas = parse_feedback("Split cluster 0 into two groups.", tiny_state_3cluster, client)
    assert isinstance(deltas, list)
