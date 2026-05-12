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
    """ORC-01: OracleAgent is a structural subtype of OracleProtocol."""
    from src.oracle_protocol import OracleProtocol
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    assert isinstance(agent, OracleProtocol)


def test_reply_returns_oracle_reply(tiny_state_3cluster):
    """ORC-01: reply() returns OracleReply with non-empty raw_text when mock client returns text."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.oracle_protocol import OracleReply
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    client = _make_oracle_client("These clusters look reasonable.")
    agent = OracleAgent(spec=spec, noise_params=noise, client=client)
    result = agent.reply(tiny_state_3cluster, "How does the clustering look?")
    assert isinstance(result, OracleReply)
    assert result.raw_text == "These clusters look reasonable."


def test_oracle_spec_fields(tiny_state_3cluster):
    """ORC-01: OracleSpec fields are accessible via agent.spec property."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic", "sentiment"], persona_description="expert")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    assert agent.spec.preferred_k == 3
    assert len(agent.spec.semantic_axes) > 0
    assert isinstance(agent.spec.persona_description, str)


def test_oracle_init_logged(tmp_path):
    """ORC-02: OracleAgent.__init__ writes oracle_init event when events_path provided."""
    from pathlib import Path
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    events_path = tmp_path / "events.jsonl"
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="tester")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    OracleAgent(spec=spec, noise_params=noise, client=_make_oracle_client(),
                events_path=events_path)
    assert events_path.exists(), "events_path file not created by OracleAgent.__init__"
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert any('"oracle_init"' in line for line in lines), \
        f"No oracle_init event in {events_path}: {lines}"


def test_noise_params_in_prompt(tiny_state_3cluster):
    """ORC-02: NoiseParams appear in assembled system prompt as percentages."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.3, [])
    assert str(int(noise.consistency_rate * 100)) in prompt


def test_overload_prompt_injected(tiny_state_3cluster):
    """ORC-03: OVERLOAD instruction appears in system prompt when load > 0.7."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.8, [])
    assert "OVERLOAD" in prompt


def test_no_overload_below_threshold(tiny_state_3cluster):
    """ORC-03: OVERLOAD instruction is absent when load <= 0.7."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.5, [])
    assert "OVERLOAD" not in prompt


def test_contradiction_merge_after_split(tiny_state_3cluster):
    """ORC-04: MergeFeedback(A,B) contradicts prior SplitFeedback(cluster_id=A)."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import SplitFeedback, MergeFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    # Append a prior SplitFeedback(cluster_id=0) at turn 1
    agent._delta_window.append((1, SplitFeedback(cluster_id=0, seed_item_ids=[])))
    # MergeFeedback(0, 1) at turn 2 should contradict the prior SplitFeedback(0)
    contradicted, prior_turn = agent._check_contradiction(
        MergeFeedback(cluster_a_id=0, cluster_b_id=1), current_turn=2
    )
    assert contradicted is True
    assert prior_turn == 1


def test_contradiction_move_item(tiny_state_3cluster):
    """ORC-04: MoveItemFeedback(item, B) contradicts prior MoveItemFeedback(item, C) where C != B."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import MoveItemFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    agent._delta_window.append((1, MoveItemFeedback(item_id=0, target_cluster_id=1)))
    contradicted, prior_turn = agent._check_contradiction(
        MoveItemFeedback(item_id=0, target_cluster_id=2), current_turn=3
    )
    assert contradicted is True
    assert prior_turn == 1


def test_no_contradiction_empty_window(tiny_state_3cluster):
    """ORC-04: fresh agent with empty window returns (False, None)."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import MergeFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    contradicted, prior_turn = agent._check_contradiction(
        MergeFeedback(cluster_a_id=0, cluster_b_id=1), current_turn=1
    )
    assert contradicted is False
    assert prior_turn is None


def test_no_false_positive_contradiction(tiny_state_3cluster):
    """ORC-04: Two SplitFeedback on same cluster is NOT a contradiction."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    from src.feedback import SplitFeedback
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    agent._delta_window.append((1, SplitFeedback(cluster_id=0, seed_item_ids=[])))
    contradicted, prior_turn = agent._check_contradiction(
        SplitFeedback(cluster_id=0, seed_item_ids=[]), current_turn=2
    )
    assert contradicted is False
    assert prior_turn is None


def test_global_instructions_in_prompt(tiny_state_3cluster):
    """FB-04: global_instructions content appears in oracle system prompt."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="analyst")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1, sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    prompt = agent._build_system_prompt(tiny_state_3cluster, 0.3, ["keep clusters small"])
    assert "keep clusters small" in prompt


def test_oracle_agent_crashes_on_invalid_noise_params():
    """ORC-02: NoiseParams values outside [0,1] raise AssertionError at construction."""
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    bad_noise = NoiseParams(consistency_rate=1.5, drift_probability=0.1, sycophancy_resistance=0.9)
    with pytest.raises(AssertionError):
        OracleAgent(spec=spec, noise_params=bad_noise, client=MagicMock())
