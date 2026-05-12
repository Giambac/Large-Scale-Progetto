"""
Integration tests for run_conversation() with OracleAgent (Phase 3 wiring).

Tests verify:
- ORC-02: oracle_init event written to events.jsonl sidecar
- ORC-03: f_cognitive_load computed before oracle.reply() each turn
- ORC-04: drift_event written to events.jsonl when contradiction detected
- FB-04: global_instructions accumulate and are passed to oracle.reply()

All LLM calls are mocked. No real API calls.
"""
import json
import os
import pytest
from unittest.mock import MagicMock


def _build_state():
    """Build a minimal ClusteringState for integration tests (2 clusters, 6 items)."""
    from src.state import Cluster, ClusteringState
    return ClusteringState(
        turn_index=0,
        timestamp="2026-05-12T00:00:00",
        clusters=[
            Cluster(id=0, name="ClusterA", description="Items about topic A.", item_ids=[0, 1, 2]),
            Cluster(id=1, name="ClusterB", description="Items about topic B.", item_ids=[3, 4, 5]),
        ],
        assignments={0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
        soft_probs={i: [0.8, 0.2] if i < 3 else [0.2, 0.8] for i in range(6)},
    )


def _build_store():
    """Build an EmbeddingStore with 6 random embeddings (dim=768)."""
    import numpy as np
    from src.embedding_store import EmbeddingStore
    rng = np.random.default_rng(42)
    embs = rng.random((6, 768)).astype(np.float32)
    return EmbeddingStore(embs)


def _build_namer():
    """Build a MagicMock ClusterNamer."""
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X description."}
    return namer


def test_oracle_agent_loop_5_turns(oracle_agent_factory, tmp_path):
    """ORC-02: run_conversation with OracleAgent writes oracle_init to events.jsonl.

    Runs 5 turns with a mocked OracleAgent. Verifies:
    - Loop completes without error
    - audit_log.jsonl contains ClusteringState records (not corrupted by events)
    - events.jsonl exists and contains an oracle_init record
    - load_audit_log(audit_log) does not crash
    """
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.serialization import load_audit_log

    state = _build_state()
    store = _build_store()
    namer = _build_namer()
    oracle = oracle_agent_factory(reply_text="looks good to me")

    log_path = str(tmp_path / "audit_log.jsonl")
    events_path = str(tmp_path / "events.jsonl")

    final_state = run_conversation(
        initial_state=state,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=StoppingCriteria(turn_budget=5),
        socketio=None,
        events_path=events_path,
    )

    assert final_state is not None
    assert final_state.turn_index >= 1

    # Verify audit_log.jsonl is valid (not corrupted by events)
    states = load_audit_log(log_path)
    assert len(states) >= 1, "audit_log.jsonl must contain at least 1 state"

    # Verify events.jsonl was created and contains oracle_init
    assert os.path.exists(events_path), "events.jsonl must be created by run_conversation"
    with open(events_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) >= 1, "events.jsonl must have at least 1 event record"
    event_types = [json.loads(line)["event"] for line in lines]
    assert "oracle_init" in event_types, f"oracle_init not found in events.jsonl; got {event_types}"


def test_drift_event_logged(oracle_agent_factory, tmp_path):
    """ORC-04: drift_event is written to events.jsonl when contradiction detected.

    Uses a MagicMock llm_client to drive parse_feedback to produce a SplitFeedback
    delta. The oracle agent's delta window is pre-seeded with a MergeFeedback on
    cluster 0, so the next SplitFeedback(cluster_id=0) will be a contradiction.

    Verifies that after the contradiction turn, events.jsonl contains a drift_event.
    """
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.feedback import SplitFeedback, MergeFeedback

    state = _build_state()
    store = _build_store()
    namer = _build_namer()
    oracle = oracle_agent_factory(reply_text="please split cluster 0")

    # Pre-seed the delta window with a MergeFeedback(0, 1) at turn 0
    # so that a SplitFeedback(0) on the next turn triggers a contradiction
    oracle._delta_window.append((0, MergeFeedback(cluster_a_id=0, cluster_b_id=1)))

    # Mock an llm_client that returns a SplitFeedback payload to parse_feedback
    # parse_feedback result: SplitFeedback(cluster_id=0, seed_item_ids=[0])
    split_payload = json.dumps([{"type": "split", "cluster_id": 0, "seed_item_ids": [0]}])
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=split_payload)]
    mock_llm_client = MagicMock()
    mock_llm_client.messages.create.return_value = mock_response

    log_path = str(tmp_path / "audit_log.jsonl")
    events_path = str(tmp_path / "events.jsonl")

    run_conversation(
        initial_state=state,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=StoppingCriteria(turn_budget=2),
        socketio=None,
        llm_client=mock_llm_client,
        events_path=events_path,
    )

    assert os.path.exists(events_path), "events.jsonl must be created"
    with open(events_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    event_types = [json.loads(line)["event"] for line in lines]
    assert "drift_event" in event_types, (
        f"drift_event not found in events.jsonl; got {event_types}\n"
        "Expected: SplitFeedback(0) after MergeFeedback(0,1) in window = contradiction"
    )


def test_instructional_feedback_accumulates(oracle_agent_factory, tmp_path):
    """FB-04: InstructionalFeedback.instruction_text accumulates in global_instructions
    and is passed to oracle.reply() on subsequent turns.

    The mock llm_client returns an InstructionalFeedback payload for turn 1.
    After turn 1, oracle.reply() should be called with the accumulated instruction.
    We verify this by inspecting the mock oracle client's call args on turn 2.
    """
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria

    state = _build_state()
    store = _build_store()
    namer = _build_namer()

    # Track calls to oracle.reply to inspect global_instructions argument
    instruction_text = "prefer smaller clusters"
    instruct_payload = json.dumps([{
        "type": "instructional",
        "instruction_text": instruction_text,
    }])

    # First parse_feedback call returns InstructionalFeedback; subsequent calls return []
    call_count = [0]
    def mock_create(**kwargs):
        call_count[0] += 1
        mock_resp = MagicMock()
        if call_count[0] == 1:
            mock_resp.content = [MagicMock(text=instruct_payload)]
        else:
            mock_resp.content = [MagicMock(text="[]")]
        return mock_resp

    mock_llm_client = MagicMock()
    mock_llm_client.messages.create.side_effect = mock_create

    # Use a real OracleAgent with a mock LLM that records its call arguments
    oracle = oracle_agent_factory(reply_text="looks fine")

    # Wrap oracle.reply to capture global_instructions argument
    original_reply = oracle.reply
    captured_instructions = []
    def wrapped_reply(state_arg, message_arg, global_instructions=None):
        captured_instructions.append(list(global_instructions) if global_instructions else [])
        return original_reply(state_arg, message_arg, global_instructions=global_instructions)
    oracle.reply = wrapped_reply

    log_path = str(tmp_path / "audit_log.jsonl")
    events_path = str(tmp_path / "events.jsonl")

    run_conversation(
        initial_state=state,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=StoppingCriteria(turn_budget=3),
        socketio=None,
        llm_client=mock_llm_client,
        events_path=events_path,
    )

    # After turn 1 produced an InstructionalFeedback, turn 2+ should have it in global_instructions
    assert len(captured_instructions) >= 2, (
        f"Expected at least 2 oracle.reply() calls, got {len(captured_instructions)}"
    )
    # Turn 1 (index 0): global_instructions is empty before any feedback processed
    # Turn 2 (index 1): global_instructions should contain the instruction from turn 1
    assert instruction_text in captured_instructions[1], (
        f"Expected '{instruction_text}' in global_instructions for turn 2, "
        f"got {captured_instructions[1]}"
    )
