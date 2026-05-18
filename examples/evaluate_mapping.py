"""
examples/evaluate_mapping.py — Held-out evaluation for mapping strategies (GEN-02).

Runs both mapping strategies on a frozen sample from dataset/held_out.jsonl,
uses a direct LLM call as ground-truth labeler, computes accuracy + bootstrap
95% CI, and prints a comparison table.

Usage:
    python -m examples.evaluate_mapping --session-dir sessions/<ts>/
    python -m examples.evaluate_mapping --session-dir sessions/<ts>/ --n-items 25
    python -m examples.evaluate_mapping --session-dir sessions/<ts>/ --json
    python -m examples.evaluate_mapping --session-dir sessions/<ts>/ --held-out dataset/held_out.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out evaluation for mapping strategies (GEN-02)"
    )
    parser.add_argument(
        "--session-dir", required=True,
        help="Path to sessions/<ts>/ directory (must contain state.json and audit_log.jsonl)"
    )
    parser.add_argument(
        "--n-items", type=int, default=25,
        help="Number of held-out items to evaluate (default: 25)"
    )
    parser.add_argument(
        "--held-out", default="dataset/held_out.jsonl",
        help="Path to held-out JSONL file (default: dataset/held_out.jsonl)"
    )
    parser.add_argument(
        "--embeddings", default="embeddings/embeddings.npy",
        help="Path to embeddings .npy file (default: embeddings/embeddings.npy)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for held-out sample selection (default: 42)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON array instead of table"
    )
    args = parser.parse_args()

    # Lazy imports — keep --help fast (D-25 pattern)
    from src.analysis import compute_bootstrap_ci
    from src.embedding_store import EmbeddingStore
    from src.llm_call import build_client, chat
    from src.llm_key import resolve_llm_key
    from src.logging_setup import deviation
    from src.mapping import MAPPING_REGISTRY, extract_oracle_rules
    from src.serialization import deserialize_state

    # --- 1. Assert session_dir exists and contains required files ---
    assert os.path.isdir(args.session_dir), (
        f"session-dir not found: {args.session_dir}"
    )
    state_path = os.path.join(args.session_dir, "state.json")
    audit_log_path = os.path.join(args.session_dir, "audit_log.jsonl")
    assert os.path.exists(state_path), (
        f"state.json not found in session-dir: {state_path}"
    )
    assert os.path.exists(audit_log_path), (
        f"audit_log.jsonl not found in session-dir: {audit_log_path}"
    )

    # --- 2. Load state from state.json ---
    with open(state_path, encoding="utf-8") as f:
        state_line = f.read().strip()
    state = deserialize_state(state_line)

    # --- 3. Load embedding store ---
    embedding_store = EmbeddingStore.load(args.embeddings)

    # --- 4. Load held-out records (read-only — never write) ---
    assert os.path.exists(args.held_out), (
        f"held-out file not found: {args.held_out}"
    )
    records = []
    with open(args.held_out, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" not in rec:
                assert "reviewText" in rec or "body" in rec or "content" in rec, (
                    f"Record {i} has no 'text' field: {list(rec.keys())}"
                )
                text_key = next(k for k in ("reviewText", "body", "content") if k in rec)
                rec = {"item_id": rec.get("item_id", i), "text": rec[text_key]}
            records.append(rec)

    assert records, f"held-out file is empty: {args.held_out}"

    # --- 5. Sample n_items records deterministically ---
    if len(records) < args.n_items:
        deviation(
            "held_out_smaller_than_n_items",
            held_out_size=len(records),
            n_items=args.n_items,
        )
        sampled = records
    else:
        sampled = random.Random(args.seed).sample(records, args.n_items)

    # --- 6. Build OracleRuleSet from audit log ---
    rule_set = extract_oracle_rules(audit_log_path)

    # --- 7. Get ground-truth labels via direct LLM calls (D-06) ---
    cluster_list_str = ", ".join(
        f"{c.id}: {c.name}" for c in state.clusters
    )
    valid_cluster_ids = {c.id for c in state.clusters}

    client = build_client(*resolve_llm_key())
    gt_labels: list[int | None] = []
    for item in sampled:
        item_text = item["text"]
        prompt = (
            "You are a cluster labeler. Assign this item to exactly one cluster.\n"
            f"Clusters: {cluster_list_str}\n"
            f"Item text: {item_text}\n"
            "Reply with ONLY the cluster ID integer."
        )
        raw = chat(client, system=None, user=prompt, max_tokens=8).strip()
        try:
            label = int(raw)
        except ValueError:
            deviation(
                "oracle_labeling_invalid_cluster_id",
                raw=raw,
                item_text=item_text[:80],
            )
            gt_labels.append(None)
            continue
        if label not in valid_cluster_ids:
            deviation(
                "oracle_labeling_invalid_cluster_id",
                label=label,
                valid=sorted(valid_cluster_ids),
                item_text=item_text[:80],
            )
            gt_labels.append(None)
            continue
        gt_labels.append(label)

    # --- 8. Evaluate each strategy ---
    entries = []
    for name in sorted(MAPPING_REGISTRY.keys()):
        strategy_cls = MAPPING_REGISTRY[name]
        strategy = strategy_cls()
        agreement_values: list[float] = []

        for item, oracle_label in zip(sampled, gt_labels):
            if oracle_label is None:
                continue  # skip items with invalid GT labels
            item_text = item["text"]
            try:
                prediction = strategy.assign(item_text, state, rule_set, embedding_store)
            except anthropic.APIError:
                raise
            try:
                pred_int = int(prediction)
            except ValueError:
                deviation(
                    "mapping_strategy_invalid_prediction",
                    prediction=prediction,
                    strategy=name,
                )
                continue  # skip item rather than crash
            agreement = 1.0 if pred_int == oracle_label else 0.0
            agreement_values.append(agreement)

        n = len(agreement_values)
        accuracy = statistics.fmean(agreement_values) if agreement_values else 0.0

        if n >= 5:
            lo, hi = compute_bootstrap_ci(agreement_values, seed=args.seed)
        else:
            lo, hi = accuracy, accuracy

        entries.append({
            "strategy": name,
            "accuracy": accuracy,
            "n": n,
            "ci_lo": lo,
            "ci_hi": hi,
        })

    # --- 9. Output ---
    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        header = f"{'Mapping strategy':<20}| {'Accuracy':>8} | {'N items':>7} | {'95% CI'}"
        sep = "-" * 20 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 8
        print(header)
        print(sep)
        for e in entries:
            ci_str = f"[{e['ci_lo']:.2f}–{e['ci_hi']:.2f}]"
            print(f"{e['strategy']:<20}| {e['accuracy']:>8.2f} | {e['n']:>7} | {ci_str}")


if __name__ == "__main__":
    main()
