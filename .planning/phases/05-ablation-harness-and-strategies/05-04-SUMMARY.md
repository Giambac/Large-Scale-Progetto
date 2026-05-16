---
phase: 05-ablation-harness-and-strategies
plan: "04"
subsystem: harness
tags: [harness, parallel, threadpool, yaml, config, sqlite, wal, oracle, baseline, embeddings]
dependency_graph:
  requires: [05-01, 05-02, 05-03]
  provides: [ALAB-02, EXP-V2-01-partial]
  affects: [src/harness.py, examples/run_harness.py, experiments/configs/harness.yaml, tests/phase5/test_harness.py]
tech_stack:
  added: [ThreadPoolExecutor, as_completed, yaml.safe_load]
  patterns: [per-thread sqlite3 connection, lazy imports in workers, shared embedding store, baseline dedup]
key_files:
  created:
    - src/harness.py
    - examples/run_harness.py
    - experiments/configs/harness.yaml
    - tests/phase5/test_harness.py
    - tests/phase5/__init__.py
  modified: []
decisions:
  - "B-02: real OracleAgent + anthropic.Anthropic in production; MockOracle only under HARNESS_DRY_RUN=1"
  - "W-01: run_baseline deduped once per (persona, seed) across all strategies"
  - "W-04: EmbeddingStore.compute_and_save called once in orchestrator, shared read-only"
  - "Fail-loudly: no try/except around fut.result() in as_completed() loop"
  - "Per-thread DB connections via connect() in each worker (D-04 + D-08 pattern)"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-17"
  tasks_completed: 3
  files_created: 5
  tests_added: 11
---

# Phase 05 Plan 04: N×M×K Ablation Harness Summary

## One-liner

ThreadPoolExecutor harness running 3×3×3 strategy/persona/seed combos via real OracleAgent + anthropic.Anthropic client, with deduped baseline rows and single-pass embedding caching.

## What Was Done

### Task 1: experiments/configs/harness.yaml

Created the YAML config file defining the full D-09 shape:
- 3 strategies: `[random, uncertainty_driven, boundary_driven]`
- 3 personas: `[curious, skeptical, drifty]` with OracleSpec + NoiseParams for each
- 3 seeds: `[1, 2, 3]`
- max_workers: 4
- Carries forward `stopping:` and `judge:` sections from `default.yaml`

Persona noise parameters chosen to produce meaningfully different convergence behaviour:
- curious: high consistency (0.9), low drift (0.05), medium sycophancy resistance (0.7)
- skeptical: medium consistency (0.75), low drift (0.05), high sycophancy resistance (0.9)
- drifty: medium consistency (0.7), high drift (0.25), low sycophancy resistance (0.5)

### Task 2: src/harness.py

Implemented the full orchestrator module (~300 lines) with:

- `STRATEGY_REGISTRY`: module-level dict mapping strategy strings to classes
- `_Combo`: dataclass for (strategy_id, persona_id, seed) tuples
- `_is_dry_run()`: reads `HARNESS_DRY_RUN` env var
- `_build_llm_client()`: returns real `anthropic.Anthropic` or `None` in dry-run (B-02 fix)
- `_build_oracle()`: returns real `OracleAgent` or `MockOracle` (B-02 fix); fires `deviation()` for MockOracle path
- `_parse_personas()`: converts YAML personas section to `{name: (OracleSpec, NoiseParams)}`
- `_build_combos()`: cross-product builder with exclude filtering; asserts unknown strategies fail loudly
- `_load_dataset()`: same JSONL record loading pattern as `examples/run_baseline.py`
- `_build_embedding_store()`: W-04 fix - computes embedding store once per orchestrator invocation
- `_run_one()`: per-combo worker; opens its own DB connection; writes `oracle_type='llm'`; uses pre-computed embedding store; calls `run_conversation()` with real llm_client
- `_run_baseline_for_combo()`: W-01 fix - per-(persona,seed) baseline runner with its own DB connection
- `run_harness()`: orchestrator; builds embedding_store ONCE; builds llm_client ONCE; submits all combos + deduped baseline jobs to ThreadPoolExecutor; uses `as_completed()` with NO try/except around `fut.result()`

### Task 3: examples/run_harness.py + tests/phase5/test_harness.py

`examples/run_harness.py` (30 lines): thin argparse wrapper with lazy import of `run_harness` for fast `--help`.

`tests/phase5/test_harness.py` (11 tests, all passing):
1. `test_strategy_registry_keys` - verifies all 3 strategy keys present
2. `test_build_combos_full_cross_product` - 3×3×3=27 combos, deterministic ordering
3. `test_build_combos_exclude_drops_specific_combo` - exclude list works correctly
4. `test_build_combos_rejects_unknown_strategy` - AssertionError with strategy name
5. `test_parse_personas_returns_dataclasses` - OracleSpec + NoiseParams correctly instantiated
6. `test_threadpool_propagates_worker_exception` - fail-loudly contract verified
7. `test_dry_run_flag_swaps_in_mock_oracle` - HARNESS_DRY_RUN=1 yields MockOracle
8. `test_real_oracle_constructed_when_client_provided` - B-02: real OracleAgent
9. `test_run_harness_invokes_baseline_once_per_persona_seed` - W-01: 12 interactive + 4 baseline = 16 rows
10. `test_embedding_store_built_once_in_orchestrator` - W-04: exactly 1 _build_embedding_store call
11. `test_run_harness_writes_oracle_type_llm` - end-to-end with mocked workers

## Commits

| Hash | Message |
|------|---------|
| c4c2efc | chore(05-04): add experiments/configs/harness.yaml for N×M×K ablation harness |
| d1e38e1 | feat(05-04): N×M×K ablation harness with real OracleAgent and baseline dedup |
| 779f459 | feat(05-04): add run_harness CLI wrapper and test_harness.py test suite |

## Test Results

```
tests/phase5/test_harness.py: 11 passed in 0.55s
tests/test_db.py: 20 passed
```

Note: `tests/test_baseline.py::test_run_baseline_returns_experiment_read` has a pre-existing failure (HDBSCAN `k <= n_training_points` constraint on 3-record test dataset). Confirmed pre-existing via git stash check — not caused by Plan 04 changes.

## Deviations from Plan

None - plan executed exactly as written. The plan code was followed precisely.

## Known Stubs

None. All harness functions are wired; the dry-run path (`HARNESS_DRY_RUN=1`) is explicitly documented as CI-only with a `deviation()` call.

## Threat Flags

No new threat surface introduced beyond what the plan's threat model covers (T-05-07 through T-05-13).

## Self-Check: PASSED

- experiments/configs/harness.yaml: EXISTS
- src/harness.py: EXISTS
- examples/run_harness.py: EXISTS
- tests/phase5/test_harness.py: EXISTS
- Commits c4c2efc, d1e38e1, 779f459: VERIFIED in git log
- 11/11 tests passing
- 27 combos from 3×3×3: VERIFIED (`python -c "... print(len(_build_combos(...)))"` outputs 27)
- W-04: `compute_and_save` NOT in `_run_one` source: VERIFIED
