"""Integration tests for OracleAgent wired into run_conversation()."""
import os
import pytest
from unittest.mock import MagicMock, patch


def test_oracle_agent_loop_5_turns(tiny_state_3cluster, mock_embeddings_3cluster,
                                    oracle_agent_factory, tmp_path):
    """OracleAgent substituted for MockOracle runs 5 turns without crashing."""
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore

    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    oracle = oracle_agent_factory(reply_text="looks good to me")
    criteria = StoppingCriteria(turn_budget=5)
    log_path = str(tmp_path / "audit.jsonl")
    final_state = run_conversation(
        initial_state=tiny_state_3cluster,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=criteria,
        socketio=None,
        llm_client=None,
    )
    assert final_state is not None
    assert final_state.turn_index >= 1


def test_drift_event_logged(tiny_state_3cluster, mock_embeddings_3cluster,
                             oracle_agent_factory, tmp_path):
    """
    Drift event is written to events.jsonl when contradiction_detected=True.

    Patches parse_feedback to return:
      turn 1: [SplitFeedback(cluster_id=0, seed_item_ids=[])]
      turn 2: [MergeFeedback(cluster_a_id=0, cluster_b_id=1)]

    Wave 3 (Plan 04) must wire update_delta_window and drift logging into run_conversation.
    This test stays RED until Plan 04 is executed.
    """
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore
    from src.feedback import SplitFeedback, MergeFeedback

    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    oracle = oracle_agent_factory(reply_text="merge clusters 0 and 1")
    events_path = str(tmp_path / "events.jsonl")
    log_path = str(tmp_path / "audit.jsonl")
    criteria = StoppingCriteria(turn_budget=2)

    # Mock parse_feedback to return contradiction-triggering sequence:
    # turn 1 → SplitFeedback(cluster_id=0), turn 2 → MergeFeedback(cluster_a_id=0, cluster_b_id=1)
    with patch("src.conversation_loop.parse_feedback") as mock_parse:
        mock_parse.side_effect = [
            [SplitFeedback(cluster_id=0, seed_item_ids=[])],   # turn 1 return
            [MergeFeedback(cluster_a_id=0, cluster_b_id=1)],   # turn 2 return
        ]
        # Pass a non-None llm_client so parse_feedback is actually called
        run_conversation(
            initial_state=tiny_state_3cluster,
            oracle=oracle,
            store=store,
            namer=namer,
            strategy=None,
            log_path=log_path,
            criteria=criteria,
            socketio=None,
            llm_client=MagicMock(),
            events_path=events_path,
        )

    assert os.path.exists(events_path), \
        "events.jsonl not created — Wave 3 (Plan 04) must wire drift_event logging"
    with open(events_path, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert any('"drift_event"' in line for line in lines), \
        f"No drift_event found in events.jsonl — Plan 04 Task 2 must write drift_event when contradiction_detected=True"


def test_instructional_feedback_accumulates(tiny_state_3cluster, mock_embeddings_3cluster,
                                             tmp_path):
    """
    InstructionalFeedback accumulates into global_instructions and appears in oracle prompt.

    Creates OracleAgent manually (not via fixture) so mock_client is accessible for assertion.
    Mocks parse_feedback to return InstructionalFeedback on every call.
    After 2 turns, asserts that 'test_instruction' appears in the most recent LLM call args.
    """
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.serialization import load_audit_log
    from src.feedback import InstructionalFeedback

    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    log_path = str(tmp_path / "audit.jsonl")
    criteria = StoppingCriteria(turn_budget=2)

    # Build OracleAgent manually so mock_client is available for assertion
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="looks good to me")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"],
                      persona_description="A neutral test analyst.")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    oracle = OracleAgent(spec=spec, noise_params=noise, client=mock_client)

    with patch("src.conversation_loop.parse_feedback") as mock_parse:
        # Every turn returns InstructionalFeedback
        mock_parse.return_value = [InstructionalFeedback(instruction_text="test_instruction")]
        run_conversation(
            initial_state=tiny_state_3cluster,
            oracle=oracle,
            store=store,
            namer=namer,
            strategy=None,
            log_path=log_path,
            criteria=criteria,
            socketio=None,
            llm_client=MagicMock(),  # non-None so parse_feedback is called
        )

    # After 2 turns, audit log should have >= 2 entries
    states = load_audit_log(log_path)
    assert len(states) >= 2, f"Expected >= 2 audit entries, got {len(states)}"

    # The accumulated instruction should appear in the oracle's system prompt on turn 2
    assert mock_client.messages.create.call_args is not None, \
        "mock_client.messages.create was never called"
    call_args_str = str(mock_client.messages.create.call_args)
    assert "test_instruction" in call_args_str, \
        f"'test_instruction' not found in oracle LLM call args on turn 2: {call_args_str[:500]}"
