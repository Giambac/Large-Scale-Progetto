---
phase: 05-ablation-harness-and-strategies
plan: "01"
subsystem: db
tags: [sqlite, schema, ddl, pydantic, experiments, oracle_type]
dependency_graph:
  requires: []
  provides: [oracle_type column in experiments table, idx_experiments_oracle_type index, ExperimentCreate.oracle_type, ExperimentRead.oracle_type]
  affects: [src/db/experiments.py, src/db/connection.py, docs/MODEL.md]
tech_stack:
  added: []
  patterns: [D-31 docs-first discipline, Pydantic model extension, idempotent DDL with IF NOT EXISTS]
key_files:
  created:
    - tests/phase5/__init__.py
    - tests/phase5/test_db_oracle_type.py
  modified:
    - docs/MODEL.md
    - src/db/connection.py
    - src/db/experiments.py
decisions:
  - oracle_type defaults to 'llm' so all existing code paths continue to work without modification
  - Field placed after dataset in both DDL and Pydantic models to match D-31 column ordering discipline
metrics:
  duration: ~8 minutes
  completed: "2026-05-17"
  tasks_completed: 3
  tasks_total: 3
---

# Phase 5 Plan 01: Add oracle_type Column to Experiments Table Summary

**One-liner:** Added `oracle_type TEXT NOT NULL DEFAULT 'llm'` column to the experiments table with an index, enabling Phase 6 LLM-vs-human run filtering (EXP-V2-01).

---

## What Was Done

Executed three tasks per D-31 discipline (docs before DDL):

1. **docs/MODEL.md** — Added `oracle_type` row after `dataset` in the experiments table, and added `idx_experiments_oracle_type` entry in the Indexes section. Both reference EXP-V2-01.

2. **src/db/connection.py** — Added `oracle_type TEXT NOT NULL DEFAULT 'llm'` after the `dataset` column in the `CREATE TABLE IF NOT EXISTS experiments` block, and added `CREATE INDEX IF NOT EXISTS idx_experiments_oracle_type ON experiments(oracle_type)` after `idx_experiments_strategy`.

3. **src/db/experiments.py** — Added `oracle_type: str = "llm"` to `ExperimentCreate` (after `dataset`), `oracle_type: str` to `ExperimentRead` (after `dataset`), and updated the INSERT SQL from 10 to 11 columns/placeholders to include `oracle_type`. Created `tests/phase5/__init__.py` (empty) and `tests/phase5/test_db_oracle_type.py` with three round-trip tests.

---

## Files Modified

| File | Change |
|------|--------|
| `docs/MODEL.md` | Added oracle_type column row + idx_experiments_oracle_type index row |
| `src/db/connection.py` | Added oracle_type DDL column + index in executescript block |
| `src/db/experiments.py` | ExperimentCreate + ExperimentRead + INSERT updated |
| `tests/phase5/__init__.py` | Created (empty package init) |
| `tests/phase5/test_db_oracle_type.py` | Created with 3 round-trip tests |

---

## Test Results

- `pytest tests/phase5/test_db_oracle_type.py -x -q` — 3 passed (default llm, explicit human, index existence)
- `pytest tests/test_db.py -x -q` — 20 passed (no Phase 4 regression)
- `python -c "from src.db.experiments import ExperimentCreate; ..."` — prints "llm" (default confirmed)
- `python -c "from src.db.connection import init_schema; ..."` — prints "OK" (DDL idempotency confirmed)

---

## Commits

| Hash | Message |
|------|---------|
| `4abfd7e` | docs(05-01): update MODEL.md with oracle_type column + index (D-31) |
| `68c8ad5` | feat(05-01): add oracle_type column + index to experiments DDL |
| `33ef904` | feat(05-01): add oracle_type column to experiments table |

---

## Deviations from Plan

None — plan executed exactly as written.

---

## Self-Check: PASSED

All files found on disk. All commits verified in git log. All tests green.
