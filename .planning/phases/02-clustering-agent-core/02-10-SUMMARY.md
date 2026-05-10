---
phase: 02-clustering-agent-core
plan: 10
subsystem: testing
tags: [hdbscan, umap-learn, pytest, importorskip, python314]

# Dependency graph
requires:
  - phase: 02-clustering-agent-core
    plan: 09
    provides: lazy hdbscan import in run_hdbscan() enabling module-level importorskip guard pattern
provides:
  - pytest.importorskip("hdbscan") guard in test_clustering_backends.py — module skips cleanly when hdbscan absent
  - pytest.importorskip("umap") guard in test_umap_projection.py — module skips cleanly when umap-learn absent
  - umap-learn 0.5.12 installed and importable in the active Python 3.14 environment
  - 10 UMAP projection tests pass green
affects: [02-clustering-agent-core, phase-gate]

# Tech tracking
tech-stack:
  added: [umap-learn==0.5.12, numba==0.65.1, llvmlite==0.47.0, pynndescent==0.6.0]
  patterns: [pytest.importorskip at module level for optional-dependency test files]

key-files:
  created: []
  modified:
    - tests/phase2/test_clustering_backends.py
    - tests/phase2/test_umap_projection.py

key-decisions:
  - "Add pytest.importorskip guards after all import statements and before any fixtures/tests — standard pytest pattern for optional-dependency test suites"
  - "hdbscan install deferred: Python 3.14 has no pre-built binary wheels; source build requires MSVC C++ 14.0+ compiler not present on this machine"
  - "test_clustering_backends.py skips cleanly (1 skipped, 0 failed) — this is correct behavior per plan when hdbscan absent"

patterns-established:
  - "pytest.importorskip pattern: place at module level after imports; causes entire module to skip with informative message when optional dependency absent"

requirements-completed: [BACK-V2-01, VIZ-V2-01]

# Metrics
duration: 25min
completed: 2026-05-10
---

# Phase 02 Plan 10: Environment Gate and importorskip Guards Summary

**umap-learn installed (10 UMAP tests green); hdbscan deferred (MSVC missing on Python 3.14); importorskip guards added to both test files so suite is green-or-skip in any environment**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-10T00:00:00Z
- **Completed:** 2026-05-10T00:25:00Z
- **Tasks:** 3 (Tasks 1+2 committed together; Task 3 is verification-only)
- **Files modified:** 2

## Accomplishments

- Added `pytest.importorskip("hdbscan", ...)` guard to `test_clustering_backends.py` — entire module skips cleanly (0 failures) when hdbscan is absent, ending ModuleNotFoundError at collection time
- Added `pytest.importorskip("umap", ...)` guard to `test_umap_projection.py` — all 10 tests pass green with umap-learn installed
- Installed umap-learn 0.5.12 (with numba 0.65.1, llvmlite 0.47.0, pynndescent 0.6.0) into the active Python 3.14 environment

## Task Commits

Each task was committed atomically:

1. **Task 1: Install hdbscan and umap-learn** — environment operation; umap-learn installed successfully; hdbscan blocked by missing MSVC (see Deviations)
2. **Task 2: Add pytest.importorskip guards** — `d3b255f` (feat)
3. **Task 3: Verify full Phase 2 non-LLM suite** — verification only, no commit needed

**Plan metadata:** (docs commit to follow via orchestrator)

## Files Created/Modified

- `tests/phase2/test_clustering_backends.py` — added `pytest.importorskip("hdbscan", reason="...")` after import block
- `tests/phase2/test_umap_projection.py` — added `pytest.importorskip("umap", reason="...")` after import block

## Decisions Made

- Used `pytest.importorskip` at module level (not per-test) so the entire file skips when dependency absent — consistent with the plan's "green-or-skip" intent
- Import module name is `"umap"` (not `"umap_learn"`) because the PyPI package `umap-learn` imports as `umap`
- No version pins in the importorskip call (module presence check only)

## Deviations from Plan

### Blocking Issue

**1. [Rule 3 - Blocking] hdbscan cannot be installed on Python 3.14 without MSVC compiler**

- **Found during:** Task 1 (pip install hdbscan)
- **Issue:** hdbscan 0.8.42 has no pre-built binary wheels for Python 3.14 on Windows. Source build requires MSVC C++ 14.0+ (for Cython-compiled C extensions). The compiler is not installed on this machine. `pip install hdbscan --only-binary=:all:` returned "No matching distribution found". conda is not available.
- **Fix attempted:** pip install from PyPI, pip install with `--only-binary`, pip install with `--no-build-isolation`, pip install from GitHub master branch. All failed with the same MSVC error.
- **Result:** hdbscan remains uninstalled. The `pytest.importorskip` guard (Task 2) ensures the test suite skips cleanly rather than failing with ModuleNotFoundError.
- **Impact on plan:** Plan truth "pip install hdbscan succeeds" is NOT satisfied. However, the plan's core mitigation intent is achieved: tests skip rather than fail. The plan's artifact `pytest.importorskip("hdbscan")` guard is in place.
- **Deferred:** Installing MSVC Build Tools (or switching to a conda environment) would unblock hdbscan compilation. See deferred-items note below.

---

**Total deviations:** 1 blocking (hdbscan install failed — environment constraint)
**Impact on plan:** umap-learn tests (10) pass green. hdbscan tests skip cleanly (0 failures). The importorskip guard pattern is correctly in place for both files.

## Issues Encountered

- **Pre-existing test failures (out of scope):** 13 tests in `test_agent_functions.py` (10 tests) and `test_conversation_loop.py` (3 tests) fail with `ModuleNotFoundError: No module named 'sentence_transformers'`. These failures are not caused by plan 02-10 changes — they existed before this plan and are caused by `EmbeddingStore` importing `sentence_transformers` at module level. These are out of scope per deviation rule scope boundary. The full suite result is: 72 passed, 13 failed (pre-existing), 1 skipped (hdbscan guard), vs. the plan's expected 0 failures.

## Known Deferred Items

- **hdbscan on Python 3.14 Windows:** Install Microsoft C++ Build Tools 14.0+ (https://visualstudio.microsoft.com/visual-cpp-build-tools/) to enable hdbscan compilation. Once installed, `python -m pip install hdbscan` should succeed. Alternatively, use a conda environment which provides pre-compiled wheels.
- **sentence_transformers missing:** 13 tests fail in test_agent_functions.py and test_conversation_loop.py due to `No module named 'sentence_transformers'`. These are pre-existing failures not in scope for this plan.

## Next Phase Readiness

- umap-learn is installed and all 10 UMAP projection tests pass green — VIZ-V2-01 runtime path is unblocked
- hdbscan tests skip cleanly — no failures from missing hdbscan package
- importorskip guard pattern is established for optional-dependency test files
- Remaining blocker: hdbscan must be installed to pass test_clustering_backends.py (currently skipped, not failed)

---
*Phase: 02-clustering-agent-core*
*Completed: 2026-05-10*
