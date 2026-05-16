"""Tests for Phase 5 _format_message enrichment (D-05, D-16) and W-02 (empty payload is not a deviation)."""
import os

import pytest

from src.conversation_loop import _format_message
from src.logging_setup import UnexpectedDeviation
from src.state import Cluster, ClusteringState
from src.strategy import Action


@pytest.fixture
def state():
    clusters = [
        Cluster(id=0, name="Electronics", description="", item_ids=[10, 11]),
        Cluster(id=1, name="Gadgets",     description="", item_ids=[20, 21]),
        Cluster(id=2, name="Peripherals", description="", item_ids=[30, 31]),
    ]
    return ClusteringState(
        turn_index=3,
        timestamp="2026-05-17T00:00:00+00:00",
        clusters=clusters,
        assignments={10: 0, 11: 0, 20: 1, 21: 1, 30: 2, 31: 2},
        soft_probs={i: [0.3, 0.3, 0.4] for i in [10, 11, 20, 21, 30, 31]},
    )


@pytest.fixture
def id_to_text():
    return {
        10: "iphone 14 case",
        11: "android charger",
        20: "laptop bag",
        21: "usb hub",
        30: "wireless mouse",
        31: "mechanical keyboard",
    }


def test_show_subset_enriched_with_items_and_cluster_names(state, id_to_text):
    action = Action(
        action_type="show_subset",
        payload={"cluster_a": 0, "cluster_b": 1, "item_ids": [10, 20]},
    )
    msg = _format_message(action, state, id_to_text)
    assert "iphone 14 case" in msg
    assert "laptop bag" in msg
    assert "Electronics" in msg
    assert "Gadgets" in msg


def test_ask_question_enriched_with_cluster_name(state, id_to_text):
    action = Action(action_type="ask_question", payload={"cluster_id": 2})
    msg = _format_message(action, state, id_to_text)
    assert "Peripherals" in msg


def test_show_subset_empty_payload_falls_back_to_placeholder(state):
    action = Action(action_type="show_subset")
    msg = _format_message(action, state)
    assert "subset" in msg.lower()
    assert "turn 3" in msg


def test_ask_question_empty_payload_falls_back_to_placeholder(state):
    action = Action(action_type="ask_question")
    msg = _format_message(action, state)
    assert "split" in msg.lower() or "merged" in msg.lower()


def test_show_full_unchanged(state):
    action = Action(action_type="show_full")
    msg = _format_message(action, state)
    assert "Current clustering" in msg
    assert "Electronics" in msg


def test_stop_unchanged(state):
    action = Action(action_type="stop")
    msg = _format_message(action, state)
    assert "satisfactory" in msg


def test_backward_compat_no_id_to_text_kwarg(state):
    # Existing callers (src/judge.py:run_baseline line 408) call _format_message(action, state).
    # This MUST keep working — id_to_text defaults to None.
    action_show_full = Action(action_type="show_full")
    assert "Electronics" in _format_message(action_show_full, state)


# ── W-02 (checker revision): empty payload is NOT a deviation ──────────────
# RandomStrategy emits empty payloads by design — under STRICT_MODE=1, these
# MUST NOT raise UnexpectedDeviation, otherwise the entire `random` column of
# the Phase 5 ablation matrix would abort.

def test_strict_mode_show_subset_empty_payload_does_NOT_raise(state, monkeypatch):
    monkeypatch.setenv("STRICT_MODE", "1")
    action = Action(action_type="show_subset")
    # No exception — empty payload is the legitimate Phase 2 / RandomStrategy contract.
    msg = _format_message(action, state)
    assert "subset" in msg.lower()


def test_strict_mode_ask_question_empty_payload_does_NOT_raise(state, monkeypatch):
    monkeypatch.setenv("STRICT_MODE", "1")
    action = Action(action_type="ask_question")
    msg = _format_message(action, state)
    assert msg  # non-empty string returned


# ── Genuine deviation cases: payload references state that doesn't exist ──

def test_strict_mode_unknown_cluster_id_in_ask_question_DOES_raise(state, id_to_text, monkeypatch):
    monkeypatch.setenv("STRICT_MODE", "1")
    action = Action(action_type="ask_question", payload={"cluster_id": 999})
    with pytest.raises(UnexpectedDeviation):
        _format_message(action, state, id_to_text)


def test_strict_mode_unknown_cluster_in_show_subset_DOES_raise(state, id_to_text, monkeypatch):
    monkeypatch.setenv("STRICT_MODE", "1")
    action = Action(
        action_type="show_subset",
        payload={"cluster_a": 999, "cluster_b": 1, "item_ids": [10]},
    )
    with pytest.raises(UnexpectedDeviation):
        _format_message(action, state, id_to_text)


def test_strict_mode_missing_item_id_in_show_subset_DOES_raise(state, monkeypatch):
    monkeypatch.setenv("STRICT_MODE", "1")
    partial_map = {10: "iphone 14 case"}  # item 20 missing
    action = Action(
        action_type="show_subset",
        payload={"cluster_a": 0, "cluster_b": 1, "item_ids": [10, 20]},
    )
    with pytest.raises(UnexpectedDeviation):
        _format_message(action, state, partial_map)
