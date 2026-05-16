"""
examples/run_harness.py — Phase 5 ablation harness CLI (ALAB-02, D-10, D-25).

Usage:
    python -m examples.run_harness
    python -m examples.run_harness --config experiments/configs/harness.yaml

Env flags:
    HARNESS_DRY_RUN=1 — swap real OracleAgent for MockOracle (CI smoke-test only).
                        Default (unset): uses real anthropic.Anthropic client via
                        resolve_llm_key().
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 5 N×M×K ablation harness (ALAB-02)"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/harness.yaml",
        help="Path to harness YAML config (default: experiments/configs/harness.yaml)",
    )
    args = parser.parse_args()

    # Lazy import so --help is fast (Phase 4 D-25 pattern).
    from src.harness import run_harness

    results = run_harness(args.config)
    print(json.dumps([r.model_dump() for r in results], indent=2, default=str))


if __name__ == "__main__":
    main()
