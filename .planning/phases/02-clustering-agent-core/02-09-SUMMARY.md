---
phase: 02-clustering-agent-core
plan: "09"
subsystem: clustering-backend
tags: [bug-fix, lazy-import, type-annotation, hdbscan, feedback]
dependency_graph:
  requires: []
  provides: [importable-clustering-module, list-typed-seed-item-ids]
  affects: [src/clustering.py, src/feedback.py, src/feedback_parser.py, tests/phase2/test_umap_projection.py]
tech_stack:
  added: []
  patterns: [lazy-import, fail-loudly]
key_files:
  modified:
    - src/clustering.py
    - src/feedback.py
    - src/feedback_parser.py
    - tests/phase2/test_umap_projection.py
decisions:
  - "Lazy import of hdbscan inside run_hdbscan() body — no try/except per fail-loudly policy"
  - "Standardize SplitFeedback.seed_item_ids on list[int] to match JSON array type from parser"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-10T10:13:09Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 4
---

# Phase 2 Plan 09: Fix Lazy hdbscan Import and Standardize SplitFeedback list[int] Summary

**One-liner:** Move module-level `import hdbscan` to lazy load inside `run_hdbscan()` and standardize `SplitFeedback.seed_item_ids` from `tuple[int, ...]` to `list[int]` across all files.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Move hdbscan import to lazy load inside run_hdbscan() | 897213e | src/clustering.py |
| 2 | Standardize SplitFeedback.seed_item_ids to list[int] across feedback.py, feedback_parser.py, and tests | 93cad0d | src/feedback.py, src/feedback_parser.py, tests/phase2/test_umap_projection.py |

## What Was Done

### Task 1 — Lazy hdbscan Import

Removed the module-level `import hdbscan` from `src/clustering.py` line 18 and added `import hdbscan` as the first statement inside `run_hdbscan()`. This makes the entire `src.clustering` module importable even when the `hdbscan` package is not installed in the environment — only calling `run_hdbscan()` or `HDBSCANBackend.fit()` (which delegates to `run_hdbscan()`) triggers the import. No try/except was added; if hdbscan is absent at call time, the `ImportError` propagates with a clear traceback (fail-loudly policy from CLAUDE.md).

### Task 2 — list[int] Standardization for SplitFeedback

Three targeted edits across three files:

1. `src/feedback.py` — Changed `SplitFeedback.seed_item_ids` annotation from `tuple[int, ...]` to `list[int]` and updated the docstring accordingly.
2. `src/feedback_parser.py` — Changed `tuple(item["seed_item_ids"])` to `list(item["seed_item_ids"])` in the `split` branch of `_build_delta()`.
3. `tests/phase2/test_umap_projection.py` — Changed `seed_item_ids=()` (tuple literal) to `seed_item_ids=[]` (list literal) in `test_should_recompute_projection_split()`.

`tests/phase2/test_feedback.py` already used list literals and list equality assertions — no changes were needed there.

## Verification Results

- `from src.clustering import ClusteringBackend, KMeansBackend, HDBSCANBackend` exits 0 and prints "clustering module imports OK"
- All 10 tests in `tests/phase2/test_feedback.py` pass green
- All 6 KMeans-only tests in `tests/phase2/test_clustering_backends.py` pass green (hdbscan-specific tests correctly skipped when package absent)
- `test_should_recompute_projection_split` and all other non-UMAP-compute tests in `tests/phase2/test_umap_projection.py` pass green

## Deviations from Plan

None — plan executed exactly as written. Both tasks required only the edits specified in the plan.

## Known Stubs

None. No placeholder values or unconnected data sources introduced.

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- src/clustering.py exists and contains lazy import at line 249
- src/feedback.py exists with `seed_item_ids: list[int]`
- src/feedback_parser.py exists with `list(item["seed_item_ids"])`
- tests/phase2/test_umap_projection.py exists with `seed_item_ids=[]`
- Commit 897213e exists (Task 1)
- Commit 93cad0d exists (Task 2)
