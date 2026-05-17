"""
examples/compare_oracle_types.py — LLM-vs-human oracle comparison CLI (EXP-V2-01).

Usage:
    python -m examples.compare_oracle_types
    python -m examples.compare_oracle_types --dataset dataset/train.jsonl
    python -m examples.compare_oracle_types --json
    python -m examples.compare_oracle_types --n-bootstrap 5000 --ci 0.90

Prints two tables:
  1. turns-to-convergence split by oracle_type AND strategy_id (with 95% CI)
  2. oracle satisfaction rates split by oracle_type (with 95% CI)

Comparison table format:
  oracle_type | strategy             | mean turns |    95% CI      | N runs
  ------------|----------------------|------------|----------------|-------
  llm         | random               |       14.2 | [12.1–16.3]    |      9
  human       | human_study          |       18.5 | [15.2–21.8]    |     10

No new packages. No DB writes. No LLM calls.
All DB access through src/db/ only (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-vs-human oracle comparison: bootstrap CI on turns-to-convergence "
            "split by oracle_type and strategy_id (EXP-V2-01)"
        )
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Filter experiments by dataset path (default: all datasets)"
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
        help="Emit JSON instead of the printed tables"
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

    # --- Table 1: turns-to-convergence split by (oracle_type, strategy_id) ---
    groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.total_turns is None:
            continue  # skip unsealed rows
        key = (r.oracle_type, r.strategy_id)
        groups.setdefault(key, []).append(float(r.total_turns))

    entries = []
    for (oracle_type, strategy_id) in sorted(groups.keys()):
        values = groups[(oracle_type, strategy_id)]
        mean_val = statistics.fmean(values)
        lo, hi = compute_bootstrap_ci(
            values, n_bootstrap=args.n_bootstrap, ci=args.ci, seed=0
        )
        entries.append({
            "oracle_type": oracle_type,
            "strategy": strategy_id,
            "mean": mean_val,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(values),
        })

    # --- Table 2: oracle satisfaction rates split by oracle_type ---
    sat_by_type: dict[str, list[float]] = {}
    for r in rows:
        sat_by_type.setdefault(r.oracle_type, [])
        if r.convergence_reason is not None:
            sat_by_type[r.oracle_type].append(
                1.0 if r.convergence_reason == "oracle_satisfied" else 0.0
            )

    # Total counts for satisfaction rate denominator (all rows, not just sealed ones)
    total_by_type: dict[str, int] = {}
    for r in rows:
        total_by_type[r.oracle_type] = total_by_type.get(r.oracle_type, 0) + 1

    satisfaction_rates = []
    for oracle_type in sorted(sat_by_type.keys()):
        sat_values = sat_by_type[oracle_type]
        total = total_by_type.get(oracle_type, 0)
        rate = statistics.fmean(sat_values) if sat_values else 0.0
        if len(sat_values) >= 5:
            lo, hi = compute_bootstrap_ci(
                sat_values, n_bootstrap=args.n_bootstrap, ci=args.ci, seed=0
            )
        else:
            lo, hi = rate, rate
        satisfaction_rates.append({
            "oracle_type": oracle_type,
            "metric": "satisfaction_rate",
            "rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": total,
        })

    if args.json:
        output = {
            "turns_by_oracle_and_strategy": entries,
            "satisfaction_rates": satisfaction_rates,
        }
        print(json.dumps(output, indent=2))
    else:
        _print_turns_table(entries, ci=args.ci)
        _print_satisfaction_table(satisfaction_rates)


def _print_turns_table(entries: list[dict], ci: float = 0.95) -> None:
    """Print the turns-to-convergence comparison table."""
    if not entries:
        print("No experiments found.")
        return

    ci_label = f"{int(ci * 100)}% CI"
    header = (
        f"{'oracle_type':<12}| {'strategy':<20}| {'mean turns':>10} | "
        f"{ci_label:<14} | {'N runs':>6}"
    )
    sep = "-" * 12 + "|" + "-" * 21 + "|" + "-" * 12 + "|" + "-" * 16 + "|" + "-" * 7
    print(header)
    print(sep)
    for e in entries:
        ci_str = f"[{e['ci_lo']:.1f}–{e['ci_hi']:.1f}]"
        print(
            f"{e['oracle_type']:<12}| {e['strategy']:<20}| {e['mean']:>10.1f} | "
            f"{ci_str:<14} | {e['n']:>6}"
        )


def _print_satisfaction_table(satisfaction_rates: list[dict]) -> None:
    """Print the oracle satisfaction rates table."""
    print("\n--- Oracle Satisfaction Rates ---")
    if not satisfaction_rates:
        print("No experiment data with convergence_reason.")
        return

    for s in satisfaction_rates:
        ci_str = f"[{s['ci_lo']:.3f}–{s['ci_hi']:.3f}]"
        print(
            f"{s['oracle_type']:<12}| {'satisfaction_rate':<20}| {s['rate']:>10.3f} | "
            f"{ci_str:<14} | {s['n']:>6}"
        )


if __name__ == "__main__":
    main()
