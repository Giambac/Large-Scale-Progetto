"""Tests for src/analysis.py (ALAB-03, D-13) and examples/compute_ci.py (D-14)."""
import json
import os
import subprocess
import sys

import pytest

from src.analysis import compute_bootstrap_ci


# ── compute_bootstrap_ci ────────────────────────────────────────────────

def test_ci_brackets_mean_for_simple_input():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = compute_bootstrap_ci(values, n_bootstrap=5000, seed=0)
    assert lo < hi
    # Mean = 3.0; for n=5, 95% CI should comfortably contain it.
    assert lo <= 3.0 <= hi


def test_ci_seed_determinism():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = compute_bootstrap_ci(values, n_bootstrap=2000, seed=42)
    b = compute_bootstrap_ci(values, n_bootstrap=2000, seed=42)
    assert a == b


def test_ci_different_seeds_give_different_intervals():
    # Use 10 values so bootstrap means have enough resolution for seeds to diverge.
    # With n=5 and integer values [1..5], all means are multiples of 0.2 and the
    # 2.5/97.5 percentiles collapse to the same discrete values regardless of seed.
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    a = compute_bootstrap_ci(values, n_bootstrap=2000, seed=1)
    b = compute_bootstrap_ci(values, n_bootstrap=2000, seed=2)
    assert a != b  # almost surely


def test_ci_empty_input_raises():
    with pytest.raises(AssertionError, match="empty values"):
        compute_bootstrap_ci([], n_bootstrap=1000)


def test_ci_invalid_ci_raises():
    with pytest.raises(AssertionError, match="ci must be in"):
        compute_bootstrap_ci([1.0, 2.0], ci=0.0)
    with pytest.raises(AssertionError, match="ci must be in"):
        compute_bootstrap_ci([1.0, 2.0], ci=1.0)


def test_ci_single_value_is_degenerate():
    lo, hi = compute_bootstrap_ci([5.0], n_bootstrap=1000)
    assert lo == 5.0 == hi


# ── examples/compute_ci.py CLI ──────────────────────────────────────────

def test_compute_ci_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "examples.compute_ci", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--oracle-type" in result.stdout
    assert "--json" in result.stdout
    assert "--n-bootstrap" in result.stdout


def test_group_by_strategy_filters_by_oracle_type():
    from examples.compute_ci import _group_by_strategy
    from src.db.experiments import ExperimentRead

    def _make_row(strategy, oracle_type, total_turns, _id):
        return ExperimentRead(
            id=_id, name=f"r{_id}", slug=f"r{_id}",
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
            deleted_at=None,
            strategy_id=strategy, persona_id="p", seed=1,
            dataset="d", oracle_type=oracle_type,
            total_turns=total_turns, convergence_reason=None,
            start_timestamp="2026-05-17T00:00:00+00:00",
            end_timestamp="2026-05-17T00:00:01+00:00",
            details={},
        )

    rows = [
        _make_row("random", "llm", 10, 1),
        _make_row("random", "human", 20, 2),
        _make_row("uncertainty_driven", "llm", 8, 3),
        _make_row("random", "llm", None, 4),  # unsealed — must be skipped
    ]

    all_groups = _group_by_strategy(rows)
    assert sorted(all_groups.keys()) == ["random", "uncertainty_driven"]
    assert sorted(all_groups["random"]) == [10.0, 20.0]
    assert all_groups["uncertainty_driven"] == [8.0]

    llm_only = _group_by_strategy(rows, oracle_type="llm")
    assert llm_only["random"] == [10.0]
    assert llm_only["uncertainty_driven"] == [8.0]


def test_format_table_includes_strategy_and_ci_bracket():
    from examples.compute_ci import _format_table
    entries = [
        {"strategy": "random", "mean": 14.2, "ci_lo": 12.1, "ci_hi": 16.3, "n": 9},
        {"strategy": "uncertainty_driven", "mean": 10.8, "ci_lo": 9.1, "ci_hi": 12.5, "n": 9},
    ]
    out = _format_table(entries)
    assert "random" in out
    assert "uncertainty_driven" in out
    assert "[12.1" in out
    assert "16.3]" in out
    assert "9" in out


def test_format_table_empty():
    from examples.compute_ci import _format_table
    assert _format_table([]) == "No experiments found."
