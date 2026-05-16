"""
examples/compute_ci.py — Bootstrap 95% CI table CLI (ALAB-03, D-14).

Usage:
    python -m examples.compute_ci
    python -m examples.compute_ci --dataset dataset/train.jsonl
    python -m examples.compute_ci --oracle-type llm --json
    python -m examples.compute_ci --n-bootstrap 5000

Prints the comparison table per CONTEXT.md specifics:

    Strategy              | Mean turns | 95% CI         | N runs
    ----------------------|------------|----------------|-------
    random                |       14.2 | [12.1 – 16.3]  |     9
    uncertainty_driven    |       10.8 | [9.1 – 12.5]   |     9
    boundary_driven       |       11.4 | [9.8 – 13.0]   |     9
    no_dialogue (baseline)|        1.0 | [1.0 – 1.0]    |     9

With --json, emits the same data as a JSON array of
{strategy, mean, ci_lo, ci_hi, n} entries.
"""
from __future__ import annotations

import argparse
import json
import statistics


def _group_by_strategy(rows, oracle_type=None):
    """Group ExperimentRead rows by strategy_id, optionally filtered by oracle_type."""
    groups: dict[str, list[float]] = {}
    for r in rows:
        if r.total_turns is None:
            continue  # skip unsealed rows
        if oracle_type is not None and r.oracle_type != oracle_type:
            continue
        groups.setdefault(r.strategy_id, []).append(float(r.total_turns))
    return groups


def _format_table(entries):
    """entries: list of dict(strategy, mean, ci_lo, ci_hi, n) — returns table string."""
    if not entries:
        return "No experiments found."
    header = f"{'Strategy':<22}| {'Mean turns':>10} | {'95% CI':<14} | {'N runs':>6}"
    sep = "-" * 22 + "|" + "-" * 12 + "|" + "-" * 16 + "|" + "-" * 7
    lines = [header, sep]
    for e in entries:
        ci_str = f"[{e['ci_lo']:.1f} – {e['ci_hi']:.1f}]"
        lines.append(
            f"{e['strategy']:<22}| {e['mean']:>10.1f} | {ci_str:<14} | {e['n']:>6}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CI on turns-to-convergence per strategy (ALAB-03)"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Filter experiments by dataset path (default: all datasets)"
    )
    parser.add_argument(
        "--oracle-type", default=None, choices=("llm", "human"),
        help="Filter experiments by oracle_type (default: all)"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Bootstrap resamples (default: 10000)"
    )
    parser.add_argument(
        "--ci", type=float, default=0.95,
        help="Confidence level in (0, 1) (default: 0.95)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of the printed table"
    )
    args = parser.parse_args()

    # Lazy imports — keep --help fast (D-25 pattern)
    from src.db import experiments as exp_db
    from src.db.connection import connect, init_schema
    from src.analysis import compute_bootstrap_ci

    db = connect()
    init_schema(db)
    try:
        rows = exp_db.query(db, dataset=args.dataset)
    finally:
        db.close()

    groups = _group_by_strategy(rows, oracle_type=args.oracle_type)

    entries = []
    for strategy_id, values in sorted(groups.items()):
        mean_val = statistics.fmean(values)
        lo, hi = compute_bootstrap_ci(
            values, n_bootstrap=args.n_bootstrap, ci=args.ci, seed=0
        )
        entries.append({
            "strategy": strategy_id,
            "mean": mean_val,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(values),
        })

    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        print(_format_table(entries))


if __name__ == "__main__":
    main()
