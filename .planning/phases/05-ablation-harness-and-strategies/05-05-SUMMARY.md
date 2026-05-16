---
phase: 05-ablation-harness-and-strategies
plan: 05
subsystem: analysis
tags: [bootstrap, ci, statistics, numpy, jupyter, notebook, cli]

requires:
  - phase: 05-01
    provides: oracle_type column on ExperimentRead and experiments table
  - phase: 05-04
    provides: N×M×K ablation harness that populates experiments rows with total_turns

provides:
  - compute_bootstrap_ci pure function (src/analysis.py) — percentile bootstrap 95% CI on mean
  - examples/compute_ci.py CLI — grouped comparison table + JSON output + --oracle-type filter
  - notebooks/analysis.ipynb — 5-cell Jupyter scaffold for interactive exploration
  - 10-test suite in tests/phase5/test_analysis.py

affects: [phase-6-generalization, llm-vs-human-comparison]

tech-stack:
  added: []
  patterns:
    - "Pure-function analysis module (no I/O, no SQL, no global state) per D-13"
    - "Seeded numpy.random.Generator for deterministic vectorised bootstrap resampling"
    - "Lazy imports in CLI main() to keep --help fast (D-25 pattern)"
    - "oracle_type Python-side filter (not SQL) to keep plan self-contained for Phase 6 extension"

key-files:
  created:
    - src/analysis.py
    - examples/compute_ci.py
    - notebooks/analysis.ipynb
    - tests/phase5/test_analysis.py
  modified: []

key-decisions:
  - "Bootstrap seed defaults to 0 and is explicit in CLI; rerun produces identical CI (T-05-12 mitigation)"
  - "oracle_type filtered in Python post-query (not SQL) to avoid adding param to exp_db.query() — Phase 6 concern"
  - "test_ci_different_seeds_give_different_intervals uses n=10 values: with n=5 integer values [1..5], all bootstrap means are multiples of 0.2 and percentiles collapse to identical values regardless of seed"

patterns-established:
  - "Analysis layer: pure function in src/analysis.py → CLI in examples/ → notebook in notebooks/"
  - "All SQL isolation enforced: compute_ci.py uses exp_db.query(), zero raw conn.execute calls"

requirements-completed: [ALAB-03]

duration: 20min
completed: 2026-05-17
---

# Phase 5 Plan 05: Bootstrap CI Analysis Layer Summary

**Percentile bootstrap 95% CI pure function, CLI comparison table with --oracle-type filter, and Jupyter notebook scaffold — operationalising ALAB-03 headline quantitative claim**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-05-17T00:00:00Z
- **Completed:** 2026-05-17T00:20:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- `compute_bootstrap_ci` pure function: vectorised numpy resampling, seeded RNG, deviation() on low n_bootstrap, fail-loud asserts on empty input and invalid ci
- `examples/compute_ci.py` CLI: prints CONTEXT.md table format per D-14; --json emits machine-readable array; --oracle-type enables Phase 6 LLM-vs-human comparison; zero raw SQL (CLAUDE.md enforced)
- `notebooks/analysis.ipynb`: valid nbformat-4 JSON, 5 cells, imports compute_bootstrap_ci + exp_db.query, documents kernel registration per D-15
- 10 tests all passing; no regressions in tests/phase5/ or tests/test_db.py

## Task Commits

1. **Task 1: src/analysis.py** — `5ec12b5` (feat)
2. **Task 2: examples/compute_ci.py + tests/phase5/test_analysis.py** — `19873a7` (feat)
3. **Task 3: notebooks/analysis.ipynb** — `f7e8ceb` (feat)

## Files Created/Modified

- `src/analysis.py` — compute_bootstrap_ci pure function (ALAB-03, D-13)
- `examples/compute_ci.py` — CLI: grouped table or JSON, --oracle-type filter (D-14)
- `notebooks/analysis.ipynb` — Jupyter exploration scaffold, 5 cells (D-15)
- `tests/phase5/test_analysis.py` — 10 tests: pure-function paths + CLI helpers

## Decisions Made

- Bootstrap seed defaults to 0 and is explicit; the same code reruns produce the same CI (T-05-12 mitigation — ALAB-03 reproducibility).
- oracle_type filtered in Python post-query to keep plan self-contained; adding it to exp_db.query() is a Phase 6 concern.
- Test `test_ci_different_seeds_give_different_intervals` upgraded from n=5 to n=10 values: with 5 integer values [1..5] and 2000 bootstrap samples, all bootstrap means are multiples of 0.2 and the 2.5/97.5 percentiles collapse to identical values regardless of seed. With n=10 the discrete resolution is sufficient for seeds to diverge.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_ci_different_seeds_give_different_intervals using n=10 values**
- **Found during:** Task 2 (test execution)
- **Issue:** Plan's test used values=[1.0..5.0] (n=5) with n_bootstrap=2000. With 5 integer values, bootstrap means can only be multiples of 0.2; the 2.5/97.5 percentiles collapse to (1.8, 4.2) regardless of seed 1 or 2 — the assertion `a != b` always fails.
- **Fix:** Changed values to 10-element list [1.0..10.0]; seeds 1 and 2 produce (3.7, 7.3) and (3.7, 7.4) respectively.
- **Files modified:** tests/phase5/test_analysis.py
- **Verification:** All 10 tests pass
- **Committed in:** 19873a7

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in plan's test)
**Impact on plan:** Minimal — test intent preserved; the fix makes the test actually exercise seed independence.

## Issues Encountered

None beyond the seed-determinism test fix above.

## User Setup Required

None — no external service configuration required. Run `python -m examples.compute_ci` to see the comparison table once experiments are populated by the Phase 5 harness.

## Next Phase Readiness

- ALAB-03 met: bootstrap 95% CI computable from experiments rows via CLI
- Phase 6 LLM-vs-human comparison ready: `python -m examples.compute_ci --oracle-type llm` vs `--oracle-type human`
- Phase 5 complete: all 5 plans (05-01 through 05-05) executed

## Self-Check

- [x] `src/analysis.py` exists
- [x] `examples/compute_ci.py` exists
- [x] `notebooks/analysis.ipynb` exists (valid JSON, nbformat=4, 5 cells)
- [x] `tests/phase5/test_analysis.py` exists (10 tests pass)
- [x] Commits 5ec12b5, 19873a7, f7e8ceb verified in git log
- [x] No raw SQL outside src/db/ (acceptance criterion verified)
- [x] No regressions in tests/phase5/ + tests/test_db.py

## Self-Check: PASSED

---
*Phase: 05-ablation-harness-and-strategies*
*Completed: 2026-05-17*
