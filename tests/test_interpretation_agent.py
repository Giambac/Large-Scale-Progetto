"""
tests/test_interpretation_agent.py — Unit tests for interpret_feedback().

Uses a mock LLM client so no real API key is needed.
Run with: pytest tests/test_interpretation_agent.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.state import ClusteringState, Cluster
from src.interpretation_agent import interpret_feedback, _build_cluster_list


def _make_state(clusters: list[tuple[int, str, int]]) -> ClusteringState:
    """Build a minimal ClusteringState with given (id, name, n_items) tuples."""
    cluster_objs = [
        Cluster(id=cid, name=name, description="", item_ids=list(range(n)))
        for cid, name, n in clusters
    ]
    return ClusteringState(
        turn_index=0,
        timestamp="2026-01-01T00:00:00",
        clusters=cluster_objs,
        assignments={},
        soft_probs={},
    )


def _mock_client(response: str) -> object:
    """Return a mock (provider, client) tuple that returns `response` from chat()."""
    mock = MagicMock()
    mock.messages.create.return_value.content = [MagicMock(text=response)]
    return ("anthropic", mock)


# ── _build_cluster_list ────────────────────────────────────────────────────

def test_build_cluster_list_format():
    state = _make_state([(3, "Billing Issues", 142), (7, "Login Problems", 89)])
    result = _build_cluster_list(state)
    assert "3" in result and "Billing Issues" in result and "142" in result
    assert "7" in result and "Login Problems" in result and "89" in result


# ── interpret_feedback: normal cases ──────────────────────────────────────

def test_returns_llm_output_when_successful():
    state = _make_state([(3, "Billing", 100), (7, "Login", 50)])
    client = _mock_client("merge cluster 3 and cluster 7")
    result = interpret_feedback("merge the last two", state, client)
    assert result == "merge cluster 3 and cluster 7"


def test_empty_text_returns_empty():
    state = _make_state([(3, "Billing", 100)])
    client = _mock_client("anything")
    assert interpret_feedback("", state, client) == ""
    assert interpret_feedback("   ", state, client) == "   "


def test_fallback_on_llm_exception():
    """If the LLM call raises, interpret_feedback returns raw_text unchanged."""
    state = _make_state([(3, "Billing", 100)])
    broken_client = ("anthropic", MagicMock())
    broken_client[1].messages.create.side_effect = RuntimeError("API down")
    result = interpret_feedback("split the biggest", state, broken_client)
    assert result == "split the biggest"


def test_fallback_on_suspiciously_long_output():
    """If LLM returns text >3x longer than input, fall back to raw_text."""
    state = _make_state([(3, "Billing", 100)])
    long_response = "x" * 500
    client = _mock_client(long_response)
    result = interpret_feedback("hi", state, client)
    assert result == "hi"


def test_turn_history_passed_through():
    """interpret_feedback accepts turn_history without raising."""
    state = _make_state([(3, "Billing", 100), (7, "Login", 50)])
    client = _mock_client("merge cluster 3 and cluster 7")
    history = ["Turn 0 - user: keep Billing and Login separate"]
    result = interpret_feedback("merge them", state, client, turn_history=history)
    assert result == "merge cluster 3 and cluster 7"