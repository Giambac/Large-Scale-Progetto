"""
examples/run_baseline.py — No-dialogue baseline CLI (D-25, JUDG-03).

Runs the baseline: builds initial clustering, shows oracle once, writes DB rows,
prints the ExperimentRead as JSON.

Usage:
    python -m examples.run_baseline --dataset dataset/train.jsonl --persona "curious_researcher" --seed 42
    python -m examples.run_baseline --dataset dataset/train.jsonl --persona p1 --seed 0 --config experiments/configs/default.yaml
    python -m examples.run_baseline --dataset dataset/train.jsonl --persona p1 --seed 0 --name my-run-slug

For paired comparison:
    # Interactive run (via web UI) uses strategy_id="random" or other strategy
    # Baseline run uses strategy_id="no_dialogue" automatically
    # Join by: SELECT * FROM experiments WHERE dataset=? AND persona_id=? AND seed=?
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-dialogue baseline (JUDG-03)")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--persona", required=True, help="Oracle persona identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--config", default=None, help="Path to YAML config file (default: experiments/configs/default.yaml)")
    parser.add_argument("--name", default=None, help="Optional slug override for experiment name")
    args = parser.parse_args()

    # Load dataset
    records = []
    with open(args.dataset, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rec = json.loads(line)
                if "text" not in rec:
                    assert "reviewText" in rec or "body" in rec or "content" in rec, (
                        f"Record {i} has no 'text' field: {list(rec.keys())}"
                    )
                    text_key = next(k for k in ("reviewText", "body", "content") if k in rec)
                    rec = {"item_id": i, "text": rec[text_key]}
                else:
                    rec["item_id"] = i
                records.append(rec)

    assert records, f"Dataset {args.dataset} is empty"

    # Load config
    import yaml
    config_path = args.config or "experiments/configs/default.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from src.stopping import StoppingCriteria, FeedbackMagnitudeWeights

    stopping_cfg = config.get("stopping", {})
    criteria = StoppingCriteria(
        magnitude_threshold_epsilon=stopping_cfg.get("epsilon", 0.05),
        magnitude_fallback_turns=stopping_cfg.get("n_fallback", 3),
    )

    # Open DB and run baseline
    from src.db.connection import connect, init_schema
    from src.judge import run_baseline

    db = connect()
    init_schema(db)

    try:
        experiment = run_baseline(
            records=records,
            persona_id=args.persona,
            seed=args.seed,
            db=db,
            oracle=None,  # defaults to MockOracle satisfied=True; Phase 5 passes real OracleAgent
            dataset_name=args.dataset,
            criteria=criteria,
        )
    finally:
        db.close()

    print(experiment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
