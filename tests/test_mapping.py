"""
tests/test_mapping.py — Unit tests for src/mapping.py (GEN-01, 06-04).

Covers:
  - OracleRuleSet field construction and validation
  - MAPPING_REGISTRY key set
  - MappingProtocol isinstance checks (structural subtyping)
  - extract_oracle_rules with empty / no-instructional-delta audit log
  - CentroidMappingStrategy correct cluster assignment (monkeypatched encode)
  - CentroidMappingStrategy ValueError on empty cluster.item_ids
  - LLMMappingStrategy ValueError on unknown cluster ID from LLM

No test makes a live Anthropic API call — all LLM surfaces are monkeypatched.
"""
import json

import numpy as np
import pytest

from src.embedding_store import EmbeddingStore
from src.mapping import (
    CentroidMappingStrategy,
    LLMMappingStrategy,
    MAPPING_REGISTRY,
    MappingProtocol,
    OracleRuleSet,
    extract_oracle_rules,
)
from src.state import Cluster, ClusteringState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_state() -> ClusteringState:
    """Minimal 4-item, 2-cluster ClusteringState for test use."""
    clusters = [
        Cluster(id=0, name="A", description="desc A", item_ids=[0, 1]),
        Cluster(id=1, name="B", description="desc B", item_ids=[2, 3]),
    ]
    assignments = {0: 0, 1: 0, 2: 1, 3: 1}
    soft_probs = {
        0: [0.9, 0.1],
        1: [0.8, 0.2],
        2: [0.1, 0.9],
        3: [0.2, 0.8],
    }
    return ClusteringState(
        turn_index=1,
        timestamp="2026-01-01T00:00:00+00:00",
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )


def _make_test_store() -> EmbeddingStore:
    """
    4×384 float32 EmbeddingStore with structured embeddings:
      - Items 0, 1 (cluster 0) have high values on dim 0
      - Items 2, 3 (cluster 1) have high values on dim 1
    """
    emb = np.zeros((4, 384), dtype=np.float32)
    emb[0, 0] = 1.0
    emb[1, 0] = 0.9   # cluster 0
    emb[2, 1] = 1.0
    emb[3, 1] = 0.9   # cluster 1
    return EmbeddingStore(emb)


def _make_rule_set() -> OracleRuleSet:
    """Test OracleRuleSet with representative non-empty fields."""
    return OracleRuleSet(
        synonyms=[["error", "fail"]],
        focus_areas=["billing"],
        exclusions=[],
        cluster_rules=["technical issues go in A"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_oracle_rule_set_fields():
    """OracleRuleSet stores all four fields correctly."""
    r = OracleRuleSet(
        synonyms=[["a", "b"]],
        focus_areas=["x"],
        exclusions=[],
        cluster_rules=[],
    )
    assert r.synonyms == [["a", "b"]]
    assert r.focus_areas == ["x"]
    assert r.exclusions == []
    assert r.cluster_rules == []


def test_mapping_registry_keys():
    """MAPPING_REGISTRY exposes exactly the 'llm' and 'centroid' keys."""
    assert set(MAPPING_REGISTRY) == {"llm", "centroid"}


def test_mapping_protocol_isinstance():
    """Both strategies satisfy MappingProtocol (structural/runtime_checkable)."""
    assert isinstance(LLMMappingStrategy(), MappingProtocol)
    assert isinstance(CentroidMappingStrategy(), MappingProtocol)


def test_extract_oracle_rules_empty_audit_log(tmp_path, monkeypatch):
    """
    When audit_log.jsonl contains no instructional deltas, extract_oracle_rules
    calls deviation("no_fb04_entries", ...) and returns an empty OracleRuleSet.
    """
    audit_path = str(tmp_path / "audit_log.jsonl")
    # Write a valid ClusteringState JSONL line with only a global delta (type != "instructional").
    # deserialize_state() requires: turn_index, timestamp, clusters, assignments, soft_probs.
    # Extra keys (e.g. "deltas") are ignored by the deserializer.
    line = json.dumps({
        "turn_index": 0,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "clusters": [
            {"id": 0, "name": "A", "description": "desc A", "item_ids": [0, 1]},
        ],
        "assignments": {"0": 0, "1": 0},
        "soft_probs": {"0": [1.0], "1": [1.0]},
        "deltas": [{"type": "global", "instruction_text": "too many clusters"}],
    })
    (tmp_path / "audit_log.jsonl").write_text(line + "\n", encoding="utf-8")

    # Capture deviation() calls
    deviations: list[str] = []
    monkeypatch.setattr("src.mapping.deviation", lambda msg, **kw: deviations.append(msg))

    result = extract_oracle_rules(audit_path)

    assert result.synonyms == []
    assert result.focus_areas == []
    assert result.exclusions == []
    assert result.cluster_rules == []
    assert len(deviations) == 1
    assert "no_fb04_entries" in deviations[0]


def test_centroid_strategy_assigns_correctly(monkeypatch):
    """
    CentroidMappingStrategy assigns to cluster 0 when the new-item embedding
    is closest to the cluster-0 centroid (high dim-0 value).
    """
    state = _make_test_state()
    store = _make_test_store()
    rule_set = _make_rule_set()
    strategy = CentroidMappingStrategy()

    # New item vector is close to cluster-0 centroid (high dim-0, zero elsewhere)
    test_vec = np.array([[1.0] + [0.0] * 383], dtype=np.float32)
    monkeypatch.setattr(strategy._model, "encode", lambda texts, **kw: test_vec)

    result = strategy.assign("some item text", state, rule_set, store)
    assert result == "0"  # cluster 0 centroid has high dim-0 value


def test_centroid_strategy_empty_cluster_raises():
    """
    CentroidMappingStrategy raises ValueError when a cluster has no hard-assigned
    items (cluster.item_ids is empty) — fail loudly per CLAUDE.md.
    """
    # Build state where cluster 1 has no items
    clusters = [
        Cluster(id=0, name="A", description="d", item_ids=[0]),
        Cluster(id=1, name="B", description="d", item_ids=[]),  # EMPTY
    ]
    assignments = {0: 0}
    soft_probs = {0: [1.0, 0.0]}
    state = ClusteringState(
        turn_index=0,
        timestamp="2026-01-01T00:00:00+00:00",
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )
    emb = np.ones((1, 384), dtype=np.float32)
    store = EmbeddingStore(emb)
    strategy = CentroidMappingStrategy()
    rule_set = _make_rule_set()

    with pytest.raises(ValueError, match="has no items"):
        strategy.assign("text", state, rule_set, store)


def test_llm_strategy_validates_cluster_id(monkeypatch):
    """
    LLMMappingStrategy raises ValueError when the LLM returns a cluster ID
    not present in state.clusters (threat T-06-01-01 mitigation).
    """
    state = _make_test_state()
    store = _make_test_store()
    rule_set = _make_rule_set()
    strategy = LLMMappingStrategy()

    # Mock the LLM surface in src.mapping: assign() calls
    # build_client(*resolve_llm_key()) then chat(...). Patch all three so no live
    # provider key or network call is needed, and chat returns an invalid id "99".
    monkeypatch.setattr("src.mapping.resolve_llm_key", lambda: ("anthropic", "test-key"))
    monkeypatch.setattr("src.mapping.build_client", lambda *a, **kw: object())
    monkeypatch.setattr("src.mapping.chat", lambda client, **kw: "99")

    with pytest.raises(ValueError, match="unknown cluster id"):
        strategy.assign("item text", state, rule_set, store)
