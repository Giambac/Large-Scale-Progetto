"""Unit tests for OracleAgent, OracleSpec, NoiseParams — all LLM calls mocked."""
import pytest
from unittest.mock import MagicMock


def _make_oracle_client(reply_text: str = "looks good to me"):
    """MagicMock client returning reply_text for every messages.create() call."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=reply_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_oracle_agent_satisfies_protocol(tiny_state_3cluster):
    """OracleAgent is an instance of OracleProtocol (structural subtyping)."""
    from src.oracle_protocol import OracleProtocol
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    assert isinstance(agent, OracleProtocol)


def test_reply_returns_oracle_reply(tiny_state_3cluster):
    """agent.reply(state, msg) returns OracleReply with non-empty raw_text."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.oracle_protocol import OracleReply
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise,
                        client=_make_oracle_client("some non-empty reply"))
    result = agent.reply(tiny_state_3cluster, "show me the clusters")
    assert isinstance(result, OracleReply)
    assert len(result.raw_text) > 0, "OracleReply.raw_text must be non-empty"


def test_oracle_spec_fields(tiny_state_3cluster):
    """OracleSpec fields are accessible from the constructed OracleAgent."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic", "tone"],
                      persona_description="A neutral test analyst.")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    assert agent.spec.preferred_k == 3
    assert len(agent.spec.semantic_axes) > 0
    assert isinstance(agent.spec.persona_description, str)


def test_oracle_init_logged(tmp_path):
    """OracleAgent.__init__ writes an oracle_init event to events_path."""
    from pathlib import Path
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    events_path = tmp_path / "events.jsonl"
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="tester")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client(),
                events_path=events_path)
    assert events_path.exists(), "events_path file not created by OracleAgent.__init__"
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert any('"oracle_init"' in line for line in lines), \
        f"No oracle_init event in {events_path}: {lines}"


def test_noise_params_in_prompt(tiny_state_3cluster):
    """_build_system_prompt includes str(int(noise.consistency_rate * 100)) in output."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.3, [])
    # consistency_rate=0.8 → str(int(0.8 * 100)) == "80"
    assert str(int(noise.consistency_rate * 100)) in prompt, \
        f"consistency_rate percentage not found in prompt: {prompt[:200]}"


def test_overload_prompt_injected(tiny_state_3cluster):
    """_build_system_prompt with load=0.8 contains 'OVERLOAD' warning."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.8, [])
    assert "OVERLOAD" in prompt, \
        f"Expected 'OVERLOAD' in prompt for load=0.8, but not found: {prompt[:300]}"


def test_no_overload_below_threshold(tiny_state_3cluster):
    """_build_system_prompt with load=0.5 does NOT contain 'OVERLOAD'."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.5, [])
    assert "OVERLOAD" not in prompt, \
        f"'OVERLOAD' should NOT appear in prompt for load=0.5, but found: {prompt[:300]}"


def test_contradiction_merge_after_split(tiny_state_3cluster):
    """MergeFeedback(A,B) after SplitFeedback(A) is detected as a contradiction."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import SplitFeedback, MergeFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    # Append a SplitFeedback on cluster_id=0 at turn 1
    prior_split = SplitFeedback(cluster_id=0, seed_item_ids=[])
    agent._delta_window.append((1, prior_split))
    # Now check contradiction: MergeFeedback(cluster_a_id=0, cluster_b_id=1) at turn 2
    merge = MergeFeedback(cluster_a_id=0, cluster_b_id=1)
    contradicted, prior_turn = agent._check_contradiction(merge, current_turn=2)
    assert contradicted is True, "MergeFeedback(0,1) after SplitFeedback(0) should be a contradiction"
    assert prior_turn is not None, "prior_turn should be set when contradiction is detected"


def test_contradiction_move_item(tiny_state_3cluster):
    """MoveItemFeedback(item=0, target=2) after MoveItemFeedback(item=0, target=1) is contradiction."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import MoveItemFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    # Append first MoveItemFeedback(item_id=0, target_cluster_id=1) at turn 1
    prior_move = MoveItemFeedback(item_id=0, target_cluster_id=1)
    agent._delta_window.append((1, prior_move))
    # Now check contradiction: same item moved to different cluster at turn 3
    new_move = MoveItemFeedback(item_id=0, target_cluster_id=2)
    contradicted, prior_turn = agent._check_contradiction(new_move, current_turn=3)
    assert contradicted is True, "Moving item 0 to cluster 2 after moving it to cluster 1 should be contradiction"
    assert prior_turn == 1, f"prior_turn should be 1, got {prior_turn}"


def test_no_contradiction_empty_window(tiny_state_3cluster):
    """Fresh agent with empty delta window returns (False, None) for any delta."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import MergeFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    assert len(agent._delta_window) == 0, "Window should be empty for fresh agent"
    delta = MergeFeedback(cluster_a_id=0, cluster_b_id=1)
    contradicted, prior_turn = agent._check_contradiction(delta, current_turn=1)
    assert contradicted is False, "Empty window should never produce contradiction"
    assert prior_turn is None, "prior_turn should be None when no contradiction"


def test_no_false_positive_contradiction(tiny_state_3cluster):
    """SplitFeedback(A) then SplitFeedback(A) is NOT a contradiction (same operation)."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import SplitFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    prior_split = SplitFeedback(cluster_id=0, seed_item_ids=[])
    agent._delta_window.append((1, prior_split))
    # Same operation on same cluster — NOT a contradiction
    new_split = SplitFeedback(cluster_id=0, seed_item_ids=[1])
    contradicted, prior_turn = agent._check_contradiction(new_split, current_turn=2)
    assert contradicted is False, \
        "SplitFeedback(0) after SplitFeedback(0) should NOT be a contradiction"
    assert prior_turn is None


def test_global_instructions_in_prompt(tiny_state_3cluster):
    """_build_system_prompt includes global_instructions content in the output."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client())
    global_instructions = ["keep clusters small"]
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.3, global_instructions)
    assert "keep clusters small" in prompt, \
        f"global_instructions not found in prompt: {prompt[:300]}"


def test_oracle_agent_crashes_on_invalid_noise_params():
    """NoiseParams with consistency_rate > 1.0 causes AssertionError on OracleAgent construction."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    bad_noise = NoiseParams(consistency_rate=1.5, drift_probability=0.1,
                            sycophancy_resistance=0.9)
    with pytest.raises(AssertionError):
        OracleAgent(spec=spec, noise_params=bad_noise, client=MagicMock())
