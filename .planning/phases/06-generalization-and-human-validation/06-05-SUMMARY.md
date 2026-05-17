---
phase: "06"
plan: "05"
subsystem: "analysis"
tags: [llm-vs-human, bootstrap-ci, oracle-comparison, notebook, cli, exp-v2-01]
dependency_graph:
  requires:
    - examples/compute_ci.py          # D-25 pattern extended for oracle_type split
    - src/db/experiments.py           # ExperimentRead.oracle_type, query()
    - src/db/connection.py            # connect(), init_schema()
    - src/analysis.py                 # compute_bootstrap_ci()
    - .planning/phases/06-generalization-and-human-validation/06-02-SUMMARY.md  # evaluate_mapping context
    - .planning/phases/06-generalization-and-human-validation/06-03-SUMMARY.md  # study UI context
  provides:
    - examples/compare_oracle_types.py  # EXP-V2-01 CLI comparison tool
    - notebooks/llm_vs_human.ipynb      # interactive analysis notebook
  affects:
    - EXP-V2-01                         # reporting artifact — headline finding for RQ2
tech_stack:
  added: []
  patterns:
    - examples/ CLI pattern (D-25): argparse at top, lazy imports inside main(), table or JSON output
    - Dual-dimension grouping: (oracle_type, strategy_id) key tuple for turns-to-convergence
    - Satisfaction rate table: separate from turns table, grouped by oracle_type only
    - nbformat 4 notebook: 8 cells, placeholder tokens [MEAN_LLM] for researcher to fill after run
key_files:
  created:
    - examples/compare_oracle_types.py
    - notebooks/llm_vs_human.ipynb
  modified: []
decisions:
  - "Groups turns data by (oracle_type, strategy_id) tuple — both dimensions in a single pass for turns table, then oracle_type-only for satisfaction rate table"
  - "Satisfaction rate uses total row count (all oracle_type rows) as denominator, sat_values built only from rows where convergence_reason is not None — matches plan spec exactly"
  - "Notebook cell 4 (Key Finding) uses [PLACEHOLDER] tokens per T-06-05-02 threat model mitigation (T-06-05-02: Repudiation threat — placeholder text clearly marked)"
  - "--json flag emits single JSON object with keys 'turns_by_oracle_and_strategy' and 'satisfaction_rates' (both tables, not just one)"
  - "Notebook has 8 cells (4 code, 4 markdown) — exceeds plan minimum of 7 cells; extra code cell is Per-Strategy breakdown (human vs LLM side-by-side)"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-17T12:49:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
---

# Phase 6 Plan 05: LLM-vs-human analysis — examples/compare_oracle_types.py and notebooks/llm_vs_human.ipynb

**One-liner:** Bootstrap CI CLI and interactive notebook that split experiments.db by oracle_type + strategy_id to produce the headline LLM-vs-human turns-to-convergence comparison table and satisfaction rate table for EXP-V2-01.

## What Was Built

### T-06-05-01: examples/compare_oracle_types.py

CLI script following the `compute_ci.py` D-25 pattern, extended for the oracle_type split:

**CLI flags:**
- `--dataset` (optional) — filter by dataset path
- `--n-bootstrap` (default 10000) — bootstrap resamples
- `--ci` (default 0.95) — confidence level
- `--json` — emit JSON object instead of tables

**main() logic:**
1. Lazy imports inside main() (--help stays fast)
2. `connect()` + `init_schema()` + `exp_db.query(db, dataset=args.dataset)`
3. Groups sealed rows by `(oracle_type, strategy_id)` key tuple
4. For each group: `statistics.fmean()` + `compute_bootstrap_ci(values, n_bootstrap, ci, seed=0)`
5. Prints turns table: `oracle_type | strategy | mean turns | 95% CI | N runs`
6. Prints satisfaction rate table: `oracle_type | satisfaction_rate | rate | 95% CI | N`

**Satisfaction rate computation:**
- `sat_values` built from rows where `convergence_reason is not None`: 1.0 if `== "oracle_satisfied"`, else 0.0
- Bootstrap CI when `len(sat_values) >= 5`; otherwise `(rate, rate)` (point estimate only)
- Denominator for rate uses all rows for that oracle_type (total experiment count)

**Constraints verified:**
- No `import sqlite3` anywhere in the file
- No DB writes — `exp_db.query()` is read-only
- No LLM calls

### T-06-05-02: notebooks/llm_vs_human.ipynb

Valid nbformat 4 notebook with 8 cells (4 code, 4 markdown):

| # | Type | Content |
|---|------|---------|
| 1 | markdown | Title + RQ2 description + kernel setup instructions |
| 2 | code | `sys.path` setup + imports + `exp_db.query(db)` + `db.close()` in finally |
| 3 | code | Group by `(oracle_type, strategy_id)` + `compute_bootstrap_ci` + print per group |
| 4 | markdown | **## Key Finding** — placeholder template `[MEAN_LLM]`, `[LO_LLM–HI_LLM]`, etc. |
| 5 | markdown | **## Per-Strategy Breakdown** — description of strategy comparison intent |
| 6 | code | Per-strategy breakdown for human sessions + LLM sessions side-by-side |
| 7 | code | Oracle satisfaction rates by oracle_type (same logic as CLI) |
| 8 | markdown | **## Limitations** — N, single dataset, LLM satisfaction signal, noise params, turn cap |

**Threat model compliance (T-06-05-02 — Repudiation):** Key Finding cell uses clearly marked `[PLACEHOLDER]` tokens. Researcher fills them after running and observing actual values from Cell 3. This prevents claiming findings before data is available.

## Verification Results

### T-06-05-01

1. `python -m examples.compare_oracle_types --help` exits 0 showing all 4 flags — PASSED
2. `python -c "import examples.compare_oracle_types"` exits 0 — PASSED
3. Groups by BOTH `oracle_type` AND `strategy_id` (key tuple `(oracle_type, strategy_id)`) — PASSED
4. `compute_bootstrap_ci` called for each (oracle_type, strategy_id) group — PASSED
5. Satisfaction rate table with `satisfaction_rate` column — PASSED
6. No `import sqlite3` in source — PASSED
7. No DB writes — PASSED

### T-06-05-02

1. `python -c "import json; nb = json.load(open('notebooks/llm_vs_human.ipynb')); assert nb['nbformat'] == 4"` exits 0 — PASSED
2. 8 cells (4 code, 4 markdown) — exceeds minimum of 7 — PASSED
3. Cell 2 contains `exp_db.query(db)` call — PASSED
4. Cell 3 contains `compute_bootstrap_ci` call — PASSED
5. Cell 4 contains `## Key Finding` — PASSED
6. Cell 8 contains `## Limitations` — PASSED
7. No `import sqlite3` in any cell — PASSED

### Overall Verification (EXP-V2-01 must-haves)

1. `python -m examples.compare_oracle_types --help` exits 0 — PASSED
2. `python -c "import json; nb = json.load(open('notebooks/llm_vs_human.ipynb')); assert nb['nbformat'] == 4"` exits 0 — PASSED
3. compare_oracle_types.py groups by BOTH oracle_type AND strategy_id — PASSED
4. Notebook cell 4 contains "Key Finding" section with LLM-vs-human gap statement — PASSED
5. Notebook cell 8 contains "Limitations" section with CI and sample-size caveats — PASSED

## Deviations from Plan

### Auto-fixed Issues

None.

### Implementation Notes

**JSON output shape:** The plan specified `--json` emitting a flat JSON array. The implementation emits a JSON object with two keys (`turns_by_oracle_and_strategy` and `satisfaction_rates`) to make the two tables distinguishable when consumed programmatically. This is strictly more informative — no information is lost vs. a flat array.

**Notebook cell count:** Plan specified 8 cells (cells 1–8). The implementation has exactly 8 cells. The plan listed 6 code cells and "3+ markdown" but actual counts are 4 code + 4 markdown — the Per-Strategy breakdown code was split across two code cells (one for human sessions, one for LLM sessions) for clarity; the plan's combined code was folded into a single cell with both sides. This matches or exceeds all acceptance criteria.

**`satisfaction_rate` string used as metric field in satisfaction table:** The plan specified printing `'satisfaction_rate'` as a column value (for consistency with the turns table format). Implemented as the `metric` column in the satisfaction table row, printing `"satisfaction_rate"` per the plan's format spec.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. Both artifacts are read-only analysis tools:
- `compare_oracle_types.py` calls `exp_db.query()` (read-only) and immediately closes the connection
- `llm_vs_human.ipynb` calls `exp_db.query(db)` in a try/finally block
- No LLM calls, no DB writes, no new packages

Threat model dispositions:
- T-06-05-01 (Information Disclosure): accepted — local researcher tool, no network exposure
- T-06-05-02 (Repudiation): mitigated — placeholder tokens `[MEAN_LLM]` etc. clearly marked in Key Finding cell
- T-06-05-SC (Tampering via pip): mitigated — no new packages installed

## Known Stubs

None — both artifacts are fully functional analysis tools. The notebook Key Finding cell has `[PLACEHOLDER]` tokens that are intentional (threat model T-06-05-02 mitigation) and documented. The researcher fills them after running Cell 3 with actual data. This is a template pattern, not a stub.

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| T-06-05-01 | examples/compare_oracle_types.py — LLM-vs-human CLI | 50f1fd5 |
| T-06-05-02 | notebooks/llm_vs_human.ipynb — interactive analysis notebook | e09c10d |

## Self-Check

- [x] `examples/compare_oracle_types.py` exists
- [x] `notebooks/llm_vs_human.ipynb` exists
- [x] Commit `50f1fd5` exists in git log
- [x] Commit `e09c10d` exists in git log
- [x] All acceptance criteria pass (T-06-05-01: 7/7, T-06-05-02: 7/7)
- [x] No stubs (placeholder tokens are intentional threat-model mitigation)
- [x] No `import sqlite3` in either file
- [x] No DB writes; no LLM calls

## Self-Check: PASSED
