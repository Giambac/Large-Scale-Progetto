"""Tests for uncertainty.py — CLUS-02: f_uncertainty ranked boundary/split/merge lists."""
import pytest


def test_f_uncertainty_returns_report(tiny_state_3cluster):
    """f_uncertainty returns an UncertaintyReport instance."""
    from src.uncertainty import f_uncertainty, UncertaintyReport
    report = f_uncertainty(tiny_state_3cluster)
    assert isinstance(report, UncertaintyReport)


def test_boundary_items_covers_all_items(tiny_state_3cluster):
    """boundary_items list contains all N items (6 items in tiny_state_3cluster)."""
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    item_ids = [item_id for item_id, _ in report.boundary_items]
    assert len(item_ids) == 6
    assert set(item_ids) == {0, 1, 2, 3, 4, 5}


def test_boundary_items_sorted_descending(tiny_state_3cluster):
    """boundary_items is sorted by entropy descending (highest uncertainty first)."""
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    entropies = [e for _, e in report.boundary_items]
    assert entropies == sorted(entropies, reverse=True)


def test_entropy_values_in_zero_one(tiny_state_3cluster):
    """Normalized entropy values are in [0.0, 1.0]."""
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    for _, entropy in report.boundary_items:
        assert 0.0 <= entropy <= 1.0, f"Entropy {entropy} outside [0, 1]"


def test_split_candidates_covers_all_clusters(tiny_state_3cluster):
    """split_candidates contains one entry per cluster."""
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    cluster_ids = [cid for cid, _ in report.split_candidates]
    assert set(cluster_ids) == {0, 1, 2}


def test_split_candidates_sorted_descending(tiny_state_3cluster):
    """split_candidates sorted by mean entropy descending."""
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    mean_entropies = [e for _, e in report.split_candidates]
    assert mean_entropies == sorted(mean_entropies, reverse=True)


def test_merge_candidates_sorted_ascending(tiny_state_3cluster):
    """merge_candidates sorted by distance ascending (closest pair first)."""
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    distances = [d for _, _, d in report.merge_candidates]
    assert distances == sorted(distances)


def test_merge_candidates_count(tiny_state_3cluster):
    """3 clusters produce C(3,2)=3 merge candidate pairs."""
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    assert len(report.merge_candidates) == 3


def test_f_uncertainty_pure_no_side_effects(tiny_state_3cluster):
    """Calling f_uncertainty twice with same state returns equal results (pure function)."""
    from src.uncertainty import f_uncertainty
    r1 = f_uncertainty(tiny_state_3cluster)
    r2 = f_uncertainty(tiny_state_3cluster)
    assert r1.boundary_items == r2.boundary_items
    assert r1.split_candidates == r2.split_candidates


def test_f_uncertainty_crashes_on_empty_state():
    """f_uncertainty raises AssertionError on state with no clusters (fail loudly)."""
    from src.uncertainty import f_uncertainty
    from src.state import ClusteringState
    empty = ClusteringState(
        turn_index=0,
        timestamp="2026-05-07T00:00:00",
        clusters=[],
        assignments={},
        soft_probs={},
    )
    with pytest.raises(AssertionError):
        f_uncertainty(empty)
