"""
Tests for conversation_loop.py — 30-turn MockOracle loop with state integrity checks.
Covers CLUS-03 integration test.
"""
import pytest


def _build_30_turn_script():
    """
    Build a 30-turn scripted OracleReply sequence.
    Turn 29 sets satisfied=True to terminate the loop cleanly.
    """
    from src.oracle_protocol import OracleReply
    replies = []
    for i in range(29):
        replies.append(OracleReply(
            raw_text="",  # empty → parse_feedback returns [] → no state change
            satisfied=False,
            turn_cognitive_load=0.0,
        ))
    replies.append(OracleReply(raw_text="", satisfied=True, turn_cognitive_load=0.0))
    return replies


def test_30_turn_loop_completes(tiny_state_3cluster, mock_embeddings_3cluster, tmp_path):
    """CLUS-03: 30-turn MockOracle loop runs to completion without exception."""
    from src.oracle_protocol import MockOracle
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    oracle = MockOracle(script=_build_30_turn_script())
    criteria = StoppingCriteria(turn_budget=50)  # generous budget; satisfied=True stops it
    log_path = str(tmp_path / "audit.jsonl")
    final_state = run_conversation(
        initial_state=tiny_state_3cluster,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,    # None → use RandomStrategy(seed=0) as default
        log_path=log_path,
        criteria=criteria,
        socketio=None,    # None → skip emit calls
    )
    assert final_state is not None


def test_30_turn_loop_state_integrity_at_turn_10(tiny_state_3cluster, mock_embeddings_3cluster, tmp_path):
    """State at turn 10 has all items assigned (completeness invariant)."""
    from src.oracle_protocol import MockOracle
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore
    from src.serialization import load_audit_log
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    oracle = MockOracle(script=_build_30_turn_script())
    log_path = str(tmp_path / "audit.jsonl")
    run_conversation(
        initial_state=tiny_state_3cluster,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=StoppingCriteria(turn_budget=50),
        socketio=None,
    )
    states = load_audit_log(log_path)
    state_at_10 = states[min(10, len(states) - 1)]
    assert len(state_at_10.assignments) == 6


def test_audit_log_written_each_turn(tiny_state_3cluster, mock_embeddings_3cluster, tmp_path):
    """AuditLog has one entry per turn (D-04: loop owns the JSONL write)."""
    from src.oracle_protocol import MockOracle
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore
    from src.serialization import load_audit_log
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    oracle = MockOracle(script=_build_30_turn_script())
    log_path = str(tmp_path / "audit.jsonl")
    run_conversation(
        initial_state=tiny_state_3cluster,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=StoppingCriteria(turn_budget=50),
        socketio=None,
    )
    states = load_audit_log(log_path)
    # Loop runs until oracle says satisfied=True at turn 29 (0-indexed)
    assert len(states) >= 29
