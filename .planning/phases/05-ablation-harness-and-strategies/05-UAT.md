---
status: complete
phase: 05-ablation-harness-and-strategies
source:
  - 05-01-SUMMARY.md
  - 05-02-SUMMARY.md
  - 05-03-SUMMARY.md
  - 05-04-SUMMARY.md
  - 05-05-SUMMARY.md
started: 2026-05-17T00:00:00Z
updated: 2026-05-17T00:00:00Z
---

## Current Test

number: 8
name: Jupyter notebook — valid nbformat-4 structure
expected: COMPLETE
awaiting: none

## Tests

### 1. Strategy Protocol — all 3 strategies importable and satisfy StrategyProtocol
expected: |
  Run:
    python -c "from src.strategy import RandomStrategy, UncertaintyDrivenStrategy, BoundaryDrivenStrategy, StrategyProtocol; print([isinstance(s(seed=0), StrategyProtocol) for s in [RandomStrategy, UncertaintyDrivenStrategy, BoundaryDrivenStrategy]])"
  Expected output: [True, True, True]
  All 3 strategy classes (Random, UncertaintyDriven, BoundaryDriven) exist and satisfy the
  StrategyProtocol structurally. ALAB-01 complete.
result: PASS

### 2. oracle_type column — experiments table has oracle_type with default 'llm'
expected: |
  Run:
    python -c "import sqlite3; from src.db.connection import init_schema; c=sqlite3.connect(':memory:'); init_schema(c); cols=[r[1] for r in c.execute('PRAGMA table_info(experiments)').fetchall()]; print('oracle_type in cols:', 'oracle_type' in cols)"
  Expected output: oracle_type in cols: True
  The experiments table has the new oracle_type column (NOT NULL DEFAULT 'llm') enabling
  Phase 6 LLM-vs-human run separation.
result: PASS

### 3. _format_message enrichment — targeted oracle messages contain item text
expected: |
  When UncertaintyDrivenStrategy or BoundaryDrivenStrategy produces a targeted payload,
  the oracle message shows the actual item text and cluster names — not generic placeholders.
  Example: "These items are ambiguous between cluster 'Electronics' and cluster 'Gadgets':
  'iphone 14 case', 'laptop bag'. How would you distinguish them?"
  (Already verified by automated check in the conversation above — confirm if this looks right.)
result: PASS

### 4. Harness cross-product — 27 combos from 3×3×3 config
expected: |
  Run:
    python -c "import yaml; from src.harness import _build_combos; cfg=yaml.safe_load(open('experiments/configs/harness.yaml', encoding='utf-8')); print(len(_build_combos(cfg['harness'])), 'combos')"
  Expected output: 27 combos
  The harness.yaml config (3 strategies × 3 personas × 3 seeds) produces exactly 27 experiment
  combinations. ALAB-02 cross-product builder functional.
result: PASS

### 5. Harness CLI smoke test (dry-run, no API key needed)
expected: |
  Run:
    python -m examples.run_harness --help
  Expected output: shows --config CONFIG option, exits 0.
  HARNESS_DRY_RUN=1 swaps in MockOracle (1-turn convergence, no API cost) for smoke-testing.
  Real ablation runs require ANTHROPIC_API_KEY and produce statistically valid rows.
result: PASS

### 6. Bootstrap CI CLI — --oracle-type and --json flags present
expected: |
  Run:
    python -m examples.compute_ci --help
  Expected output: shows --oracle-type {llm,human}, --json, --n-bootstrap flags.
  With no experiments.db (or empty DB), prints "No experiments found." and exits 0.
result: PASS
note: Stale experiments.db (pre-Plan-01 schema) caused OperationalError on first attempt.
  Deleted stale DB — fresh DB auto-created with oracle_type column. Now working correctly.
  Per .continue-here.md D-11: always delete stale experiments.db before running harness.

### 7. Bootstrap CI pure function — deterministic and correct
expected: |
  Run:
    python -c "from src.analysis import compute_bootstrap_ci; lo, hi = compute_bootstrap_ci([1.0,2.0,3.0,4.0,5.0], n_bootstrap=5000, seed=0); print(f'CI: [{lo:.2f}, {hi:.2f}]'); assert lo < hi and lo <= 3.0 <= hi; print('PASS')"
  Expected output: CI: [1.80, 4.20] (approximately), then PASS
  The function is pure (no I/O/SQL/global state), deterministic per seed, and fails loudly on
  empty input. ALAB-03 requirement met.
result: PASS

### 8. Jupyter notebook — valid structure
expected: |
  python -c "import json; nb=json.load(open('notebooks/analysis.ipynb', encoding='utf-8')); print(nb['nbformat'], len(nb['cells']))"
  Expected output: 4 5
result: PASS

## Summary

total: 8
passed: 8
issues: 0
pending: 0
skipped: 0

## Gaps

### GAP-01: Stale experiments.db breaks compute_ci and harness on first run
severity: low
description: |
  If a pre-Phase-5 experiments.db exists at repo root, init_schema() silently
  skips the CREATE TABLE (IF NOT EXISTS) and the oracle_type column is missing.
  Both `python -m examples.compute_ci` and `run_harness` then fail with
  sqlite3.OperationalError: no such column: oracle_type.
mitigation: |
  Delete experiments.db before running any Phase 5 code on a machine that
  previously ran the Phase 4 harness. The DB is gitignored; no data is lost.
  Already documented in .continue-here.md (D-11) but surfaced during UAT.
status: documented — no code fix required (D-11 decision: no ALTER TABLE migration)
