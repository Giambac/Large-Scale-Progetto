---
phase: 4
plan: "04-05"
subsystem: testing
tags: [test-suite, db-layer, judge, pairwise-accuracy, baseline, sqlite]
dependency_graph:
  requires: [04-03, 04-04]
  provides: [phase-4-test-suite]
  affects: [src/db/turns.py]
tech_stack:
  added: []
  patterns: [":memory: SQLite fixture (recipe §9)", "TDD unit tests", "STRICT_MODE env var save/restore"]
key_files:
  created:
    - tests/test_db.py
    - tests/test_judge.py
    - tests/test_baseline.py
  modified:
    - tests/conftest.py
    - src/db/turns.py
    - pyproject.toml
decisions:
  - "Fixed turns.query() to filter deleted_at IS NULL — required for cascade_delete soft-delete test to pass (Rule 1 bug fix)"
  - "Registered 'slow' marker in pyproject.toml to avoid pytest warnings on @pytest.mark.slow tests"
  - "Placed Phase 4 fixture imports at module level after Phase 1-3 fixtures — valid Python, conftest loaded once"
metrics:
  duration: "~20min"
  completed: "2026-05-16"
  tasks_completed: 2
  files_modified: 6
---

# Phase 4 Plan 05: Test Suite Summary

**One-liner:** DB layer CRUD unit tests, PairBag/pairwise-accuracy judge unit tests, and run_baseline() integration test using :memory: SQLite fixture — 43 tests covering all Phase 4 requirements.

## Tasks Completed

| Task | Name | Files |
|------|------|-------|
| 1 | conftest.py db fixture + test_db.py | tests/conftest.py, tests/test_db.py, src/db/turns.py |
| 2 | test_judge.py + test_baseline.py | tests/test_judge.py, tests/test_baseline.py, pyproject.toml |

## Test Coverage

### test_db.py (20 tests)
- Slugify basic and underscore conversion
- Slug uniqueness (-2 suffix on collision)
- Experiment CRUD: create, get, get_by_slug, update, soft-delete, query with filters
- Turn CRUD: create, get, query (with FK violation raising IntegrityError)
- OracleFeedback CRUD: compound multi-row insert, FK violation
- Cascade delete (soft and hard) including ON DELETE CASCADE FK chain

### test_judge.py (21 tests)
- All 3 stopping conditions independently triggered (oracle_satisfied, turn_budget, diminishing_returns)
- Diminishing returns requires exactly N turns below epsilon
- Magnitude computation for GlobalFeedback + MoveItemFeedback weights
- PairBag pair extraction: Move/Split/Merge produce correct pairs; Global/Instructional produce 0
- Contradiction overwrite: old pairs dropped, new pair replaces with opposite polarity
- Pairwise accuracy: empty bag = 0.0, perfect alignment = 1.0
- assemble_turn_metrics: TurnCreate with convergence_signal, cumulative_contradiction_count
- assemble_feedback_rows: compound deltas → multiple OracleFeedbackCreate rows; empty → []
- deviation() STRICT_MODE=1 raises UnexpectedDeviation; STRICT_MODE=0 logs silently

### test_baseline.py (2 tests, @pytest.mark.slow)
- run_baseline() returns ExperimentRead with strategy_id='no_dialogue', total_turns=1, mean_pairwise_accuracy in details
- run_baseline() writes exactly 1 turn row to turns table

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed turns.query() to filter soft-deleted rows**
- **Found during:** Task 1 (test_cascade_delete_soft)
- **Issue:** `turns.query()` had no `deleted_at IS NULL` filter. After `cascade_delete()` soft-deletes turns, `turn_query(db, exp.id)` would still return them, causing `assert turns == []` to fail.
- **Fix:** Added `AND deleted_at IS NULL` to the WHERE clause in `turns.query()`.
- **Files modified:** `src/db/turns.py`
- **Note:** `TurnRead` has no `deleted_at` field; Pydantic v2 ignores extra dict keys by default, so `SELECT *` rows continue to work correctly.

**2. [Rule 2 - Enhancement] Registered 'slow' pytest marker**
- **Found during:** Task 2 (test_baseline.py uses @pytest.mark.slow)
- **Issue:** pytest would emit PytestUnknownMarkWarning for unregistered markers.
- **Fix:** Added `"slow: ..."` to `markers` list in `pyproject.toml`.
- **Files modified:** `pyproject.toml`

## Key Technical Notes

- V-4-02 (known issue): `TurnCreate` field is `cumulative_contradiction_count`, NOT `contradiction_count`. All `_make_turn()` helpers use the correct field name.
- STRICT_MODE tests use `try/finally` to save/restore `os.environ["STRICT_MODE"]` — prevents test-ordering contamination (T-04-05-01).
- test_baseline.py tests marked `@pytest.mark.slow` and can be excluded via `-m "not slow"` if embedding computation is unavailable.
- The `:memory:` db fixture in conftest.py calls `init_schema()` (which calls `conn.commit()`), then tests add foreign_keys=ON PRAGMA manually in the fixture before init_schema so FKs are enforced.

## Self-Check

### Files created/exist:
- FOUND: tests/test_db.py
- FOUND: tests/test_judge.py
- FOUND: tests/test_baseline.py
- FOUND: tests/conftest.py (modified — db fixture appended)
- FOUND: src/db/turns.py (modified — query filter added)
- FOUND: pyproject.toml (modified — slow marker added)

### Threat Surface Scan
No new network endpoints, auth paths, or file access patterns introduced. Test files use :memory: SQLite only. No new threat flags.
