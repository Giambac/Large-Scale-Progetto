"""Tests for Phase 5 strategies (ALAB-01, D-01..D-04)."""
import pytest

from src.logging_setup import UnexpectedDeviation
from src.state import Cluster, ClusteringState
from src.strategy import (
    Action,
    BoundaryDrivenStrategy,
    RandomStrategy,
    StrategyProtocol,
    UncertaintyDrivenStrategy,
)
from src.uncertainty import UncertaintyReport, f_uncertainty


def _state_two_clusters_one_ambiguous():
    """Two clusters with one clearly ambiguous item (high P on both)."""
    clusters = [
        Cluster(id=0, name="A", description="", item_ids=[10, 11]),
        Cluster(id=1, name="B", description="", item_ids=[20, 21]),
    ]
    soft = {
        10: [0.9, 0.1],  # firmly in A
        11: [0.85, 0.15],
        20: [0.1, 0.9],  # firmly in B
        21: [0.15, 0.85],
        # Add genuinely-ambiguous items split between A and B so the
        # ambiguous zone is non-empty AND items belong to those clusters.
        # NOTE: assignments include these items; they must appear in some cluster.
    }
    # Add two ambiguous items to cluster A (membership) but with split probs
    soft[12] = [0.5, 0.5]
    soft[13] = [0.45, 0.55]
    clusters[0].item_ids.extend([12])
    clusters[1].item_ids.extend([13])
    assignments = {10: 0, 11: 0, 12: 0, 20: 1, 21: 1, 13: 1}
    return ClusteringState(
        turn_index=0,
        timestamp="2026-05-17T00:00:00+00:00",
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft,
    )


def _state_three_clusters_clear_winner():
    """Three clusters; cluster 2 has highest mean entropy (most uncertain)."""
    clusters = [
        Cluster(id=0, name="Sharp",  description="", item_ids=[100]),
        Cluster(id=1, name="Sharp2", description="", item_ids=[101]),
        Cluster(id=2, name="Fuzzy",  description="", item_ids=[200, 201]),
    ]
    soft = {
        100: [0.95, 0.025, 0.025],
        101: [0.025, 0.95, 0.025],
        200: [0.34, 0.33, 0.33],   # very uncertain
        201: [0.33, 0.34, 0.33],
    }
    assignments = {100: 0, 101: 1, 200: 2, 201: 2}
    return ClusteringState(
        turn_index=5,
        timestamp="2026-05-17T00:00:00+00:00",
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft,
    )


# -- UncertaintyDrivenStrategy -----------------------------------------------

def test_uncertainty_driven_targets_highest_entropy_cluster():
    state = _state_three_clusters_clear_winner()
    report = f_uncertainty(state)
    # Cluster 2 has the highest mean entropy -> top of split_candidates.
    assert report.split_candidates[0][0] == 2

    strat = UncertaintyDrivenStrategy(seed=0)
    action = strat.select(state, report)
    assert action.action_type == "ask_question"
    assert action.payload == {"cluster_id": 2}


def test_uncertainty_driven_fallback_when_no_split_candidates():
    # Empty report -> degenerate fallback
    state = _state_three_clusters_clear_winner()
    empty_report = UncertaintyReport(boundary_items=[], split_candidates=[], merge_candidates=[])
    strat = UncertaintyDrivenStrategy(seed=0)
    action = strat.select(state, empty_report)
    assert action.action_type == "ask_question"
    assert action.payload == {}


def test_uncertainty_driven_fallback_fires_deviation_under_strict_mode(monkeypatch):
    monkeypatch.setenv("STRICT_MODE", "1")
    state = _state_three_clusters_clear_winner()
    empty_report = UncertaintyReport(boundary_items=[], split_candidates=[], merge_candidates=[])
    strat = UncertaintyDrivenStrategy(seed=0)
    with pytest.raises(UnexpectedDeviation):
        strat.select(state, empty_report)


def test_uncertainty_driven_deterministic_same_seed():
    state = _state_three_clusters_clear_winner()
    report = f_uncertainty(state)
    a1 = UncertaintyDrivenStrategy(seed=42).select(state, report)
    a2 = UncertaintyDrivenStrategy(seed=42).select(state, report)
    assert a1.payload == a2.payload


# -- BoundaryDrivenStrategy --------------------------------------------------

def test_boundary_driven_targets_closest_pair_with_item_ids():
    state = _state_two_clusters_one_ambiguous()
    report = f_uncertainty(state)
    # With only two clusters, merge_candidates has exactly one entry.
    assert len(report.merge_candidates) == 1

    strat = BoundaryDrivenStrategy(seed=0)
    action = strat.select(state, report)
    assert action.action_type == "show_subset"
    assert {"cluster_a", "cluster_b", "item_ids"}.issubset(action.payload.keys())
    # Items 12 and 13 are the genuinely ambiguous ones (approximately 0.5/0.5 probs).
    assert 12 in action.payload["item_ids"]
    assert 13 in action.payload["item_ids"]
    # Sharp items must NOT be in the ambiguous zone.
    assert 10 not in action.payload["item_ids"]
    assert 20 not in action.payload["item_ids"]


def test_boundary_driven_fallback_when_no_merge_candidates():
    state = _state_two_clusters_one_ambiguous()
    empty_report = UncertaintyReport(boundary_items=[], split_candidates=[], merge_candidates=[])
    strat = BoundaryDrivenStrategy(seed=0)
    action = strat.select(state, empty_report)
    assert action.action_type == "ask_question"
    assert action.payload == {}


def test_boundary_driven_fallback_when_ambiguous_zone_empty():
    """Only sharp items exist -> ambiguous zone is empty -> fallback."""
    clusters = [
        Cluster(id=0, name="A", description="", item_ids=[10]),
        Cluster(id=1, name="B", description="", item_ids=[20]),
    ]
    sharp = {10: [0.99, 0.01], 20: [0.01, 0.99]}
    state = ClusteringState(
        turn_index=2,
        timestamp="2026-05-17T00:00:00+00:00",
        clusters=clusters,
        assignments={10: 0, 20: 1},
        soft_probs=sharp,
    )
    report = f_uncertainty(state)
    strat = BoundaryDrivenStrategy(seed=0)
    action = strat.select(state, report)
    assert action.action_type == "ask_question"
    assert action.payload == {}


def test_boundary_driven_fallback_fires_deviation_under_strict_mode(monkeypatch):
    monkeypatch.setenv("STRICT_MODE", "1")
    state = _state_two_clusters_one_ambiguous()
    empty_report = UncertaintyReport(boundary_items=[], split_candidates=[], merge_candidates=[])
    strat = BoundaryDrivenStrategy(seed=0)
    with pytest.raises(UnexpectedDeviation):
        strat.select(state, empty_report)


def test_boundary_driven_deterministic_same_seed_same_turn_index():
    state = _state_two_clusters_one_ambiguous()
    report = f_uncertainty(state)
    a1 = BoundaryDrivenStrategy(seed=7).select(state, report)
    a2 = BoundaryDrivenStrategy(seed=7).select(state, report)
    assert a1.payload == a2.payload


# -- Structural Protocol conformance -----------------------------------------

@pytest.mark.parametrize("cls", [RandomStrategy, UncertaintyDrivenStrategy, BoundaryDrivenStrategy])
def test_strategy_satisfies_protocol_structurally(cls):
    instance = cls(seed=0)
    assert isinstance(instance, StrategyProtocol), (
        f"{cls.__name__} does not satisfy StrategyProtocol structurally"
    )
