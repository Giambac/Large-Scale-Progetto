---
phase: "06"
plan: "02"
subsystem: "evaluation"
tags: [generalization, held-out-eval, mapping-evaluation, bootstrap-ci, llm-oracle, gen-02]
dependency_graph:
  requires:
    - src/mapping.py           # MAPPING_REGISTRY, extract_oracle_rules, LLMMappingStrategy, CentroidMappingStrategy
    - src/analysis.py          # compute_bootstrap_ci
    - src/serialization.py     # deserialize_state
    - src/embedding_store.py   # EmbeddingStore.load
    - src/logging_setup.py     # deviation()
    - dataset/held_out.jsonl   # frozen held-out split (read-only)
  provides:
    - examples/evaluate_mapping.py  # GEN-02 evaluation CLI
  affects: []
tech_stack:
  added: []
  patterns:
    - examples/ CLI pattern (D-25): argparse at top, lazy imports inside main(), table or JSON output
    - Direct anthropic.Anthropic() client call for GT labeling (D-06: avoids OracleAgent noise params)
    - deviation() for unexpected-but-possible branches (invalid GT label, smaller held-out file)
    - try/except anthropic.APIError only at LLM boundary; all other exceptions propagate
    - random.Random(seed).sample() for deterministic reproducible sampling
key_files:
  created:
    - examples/evaluate_mapping.py
  modified: []
decisions:
  - "GT labeling uses direct anthropic.Anthropic() client call (not OracleAgent) — avoids noise params interfering with labeling accuracy (D-06, planner discretion)"
  - "Items with invalid GT labels (parse fails or ID not in cluster IDs) are skipped; deviation() called; remaining items still evaluated"
  - "Both strategies share the same GT labels for fair comparison — GT labels computed once, reused across MAPPING_REGISTRY loop"
  - "state.json contains a single line (last ClusteringState snapshot); loaded via deserialize_state()"
  - "Script asserts os.path.exists(args.held_out) before opening; never opens with write mode (T-06-02-01 mitigation)"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-17T12:15:00Z"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 0
---

# Phase 6 Plan 02: Held-out Evaluation — examples/evaluate_mapping.py (GEN-02)

**One-liner:** CLI evaluation script that samples 25 held-out items deterministically, labels them with a direct LLM oracle call, runs both mapping strategies, and prints an accuracy + bootstrap 95% CI comparison table.

## What Was Built

`examples/evaluate_mapping.py` — the GEN-02 evaluation script:

1. **CLI argument parsing** — `--session-dir` (required), `--n-items` (default 25), `--held-out` (default dataset/held_out.jsonl), `--embeddings` (default embeddings/embeddings.npy), `--seed` (default 42), `--json` flag.

2. **Session loading** — asserts `state.json` and `audit_log.jsonl` exist in `--session-dir`; reads `state.json` as a single JSON line; calls `deserialize_state()` to produce `ClusteringState`.

3. **Deterministic sampling** — loads all records from `held_out.jsonl` (read-only, never written); calls `random.Random(args.seed).sample(records, args.n_items)` for reproducible selection; calls `deviation("held_out_smaller_than_n_items")` when the file is smaller than requested.

4. **OracleRuleSet extraction** — calls `extract_oracle_rules(audit_log_path)` which reads the audit log for FB-04 instructional deltas; returns an empty `OracleRuleSet` (with deviation logged) for sessions without such entries.

5. **Ground-truth labeling** — one `anthropic.Anthropic()` client call per sampled item with cluster list + item text prompt; parses response as int; validates against cluster IDs; calls `deviation("oracle_labeling_invalid_cluster_id")` and skips invalid responses.

6. **Strategy evaluation loop** — iterates `sorted(MAPPING_REGISTRY.keys())` ("centroid", "llm"); instantiates each strategy; calls `strategy.assign(item_text, state, rule_set, embedding_store)` for each item; computes agreement rate; calls `compute_bootstrap_ci(agreement_values, seed=args.seed)` when N >= 5.

7. **Output** — comparison table in specified format (or JSON array with `--json`):
   ```
   Mapping strategy    | Accuracy | N items | 95% CI
   --------------------|----------|---------|--------
   centroid            |     0.74 |      25 | [0.57–0.87]
   llm                 |     0.87 |      25 | [0.72–0.96]
   ```

## Verification Results

All acceptance criteria pass:

1. `python -c 'import examples.evaluate_mapping'` exits 0 — PASSED
2. `python -m examples.evaluate_mapping --help` shows all flags (--session-dir, --n-items, --held-out, --embeddings, --seed, --json) — PASSED
3. Script does NOT import sqlite3 and does NOT reference src.db — PASSED
4. `assert os.path.exists(args.held_out)` visible in source before opening — PASSED
5. `random.Random(args.seed).sample(records, args.n_items)` present for deterministic sampling — PASSED
6. `extract_oracle_rules(audit_log_path)` called to build OracleRuleSet — PASSED
7. Script loops over `MAPPING_REGISTRY` keys for both strategies — PASSED
8. `compute_bootstrap_ci` called when `len(agreement_values) >= 5` — PASSED
9. `dataset/held_out.jsonl` never opened with write mode — PASSED

## Deviations from Plan

### Auto-fixed Issues

None.

### Architectural Observations

**Line count vs. plan estimate:** The plan estimated ~60-80 lines; the script is 211 lines. The plan's estimate was based on `run_baseline.py`'s structure but the actual logic requires: two nested loops (GT labeling loop + strategy evaluation loop), per-item deviation handling, cluster list construction, and the complete table formatting. The analogous `compute_ci.py` is 117 lines. The extra length adds clarity not complexity — no code should be removed.

**GT labels shared across strategies:** The plan described getting GT labels inside the strategy loop (`for each sampled item`) but this would make N redundant API calls for each additional strategy. The implementation computes GT labels once and passes the shared list to each strategy evaluation — this is strictly more correct (both strategies compared on the same oracle judgment) and was the intended design per D-07 ("Both mapping strategies evaluated on the same 20-30 item sample").

## Threat Surface Scan

No new network endpoints, auth paths, or DB writes. The two new LLM call sites (GT labeling in the evaluation script, and any calls inside `LLMMappingStrategy.assign`) follow the `try/except anthropic.APIError: raise` pattern. `held_out.jsonl` is opened with `open(..., encoding="utf-8")` (read mode only — T-06-02-01 satisfied).

## Known Stubs

None — the script is fully wired. It requires a valid `--session-dir` with `state.json` and `audit_log.jsonl` to produce output; without a real session, it will assert-fail at the session directory check (correct behavior).

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| T-06-02-01 | Implement examples/evaluate_mapping.py | ee84865 |

## Self-Check

- [x] `examples/evaluate_mapping.py` exists
- [x] Commit `ee84865` exists in git log
- [x] All 9 acceptance criteria pass
- [x] No stubs (no TODO/FIXME/placeholder text)
- [x] No sqlite3 or src.db references
- [x] held_out.jsonl opened read-only only

## Self-Check: PASSED
