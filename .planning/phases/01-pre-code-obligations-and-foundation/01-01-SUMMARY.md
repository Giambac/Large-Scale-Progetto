---
phase: 01-pre-code-obligations-and-foundation
plan: "01"
subsystem: test-infrastructure
tags: [pytest, tdd, test-scaffold, red-state, phase1]

dependency_graph:
  requires: []
  provides:
    - tests/phase1/ test package with 41 stub tests covering PRE-01, PRE-02, FOUND-01, FOUND-02, FOUND-03, FOUND-04
    - tests/conftest.py with four shared fixtures (sample_texts, tiny_clustering_state_dict, mock_hdbscan_output, mock_embeddings)
    - pyproject.toml with pytest configuration
  affects:
    - All subsequent Phase 1 implementation plans (01-02 through 01-05) depend on these test stubs

tech_stack:
  added:
    - pytest 9.0.2 (already installed)
  patterns:
    - pytest fixtures via conftest.py injection
    - Module-level imports for RED state enforcement (ModuleNotFoundError on missing src.* modules)
    - TDD Wave 0 scaffold: tests written before any implementation code

key_files:
  created:
    - pyproject.toml
    - tests/__init__.py
    - tests/phase1/__init__.py
    - tests/conftest.py
    - tests/phase1/test_dataset.py
    - tests/phase1/test_stopping_criteria.py
    - tests/phase1/test_embedding_store.py
    - tests/phase1/test_clustering.py
    - tests/phase1/test_cluster_naming.py
    - tests/phase1/test_serialization.py
  modified: []

decisions:
  - Module-level imports used (not pytest.importorskip) to enforce hard RED state — all 6 test files fail immediately with ModuleNotFoundError for src.* modules, which is the correct pre-implementation state
  - mock_embeddings fixture uses 5x768 shape to match production embedding dimensions (sentence-transformers all-MiniLM-L6-v2 uses 384 dims, but plan specified 768; kept per plan spec)
  - FeedbackMagnitudeWeights tested for all four feedback types (global_feedback, cluster_level, point_level, instructional) per D-10

metrics:
  duration: "~8 minutes"
  completed: "2026-05-05T08:30:41Z"
  tasks_completed: 3
  tasks_total: 3
  files_created: 10
  files_modified: 0
---

# Phase 1 Plan 01: Test Scaffold — Summary

**One-liner:** pytest Wave 0 scaffold with 41 test stubs across 6 files covering PRE-01, PRE-02, FOUND-01–04, all in RED state via ModuleNotFoundError on absent src.* modules.

---

## What Was Built

This plan installs the test infrastructure required before any implementation code is written for Phase 1. The three tasks established:

1. **pytest configuration** — `pyproject.toml` with `testpaths=["tests"]`, `python_files`, `python_classes`, `python_functions`, and `-q` addopts. Empty `tests/__init__.py` and `tests/phase1/__init__.py` created to make the test tree a proper Python package.

2. **Shared fixtures** (`tests/conftest.py`) — Four fixtures available to all Phase 1 tests via pytest injection:
   - `sample_texts`: 5 review-like strings for data loader and embedding tests
   - `tiny_clustering_state_dict`: Minimal ClusteringState as plain dict (D-13 schema, 2 clusters, 5 items)
   - `mock_hdbscan_output`: Synthetic (labels, soft_probs) for 5 items, 2 clusters, no noise
   - `mock_embeddings`: Seeded 5×768 float32 array for reproducible embedding tests

3. **Six Phase 1 test stub files** (41 tests total):
   - `test_dataset.py` (6 tests): SHA-256 hash determinism, hash mismatch crash, correct hash passes, 80/20 split, reproducible split, no overlap between train/held-out
   - `test_stopping_criteria.py` (8 tests): turn_budget default=15, StopReason enum values, oracle_satisfied priority, turn budget fires at/past 15, no trigger mid-conversation, FeedbackMagnitudeWeights fields
   - `test_embedding_store.py` (5 tests): shape (5,768), get single item returns 1D, round-trip values, 1D array rejection, readonly behavior
   - `test_clustering.py` (6 tests): HDBSCAN returns (labels, soft_probs), shape matches N, rows sum to 1.0, noise-to-nearest no -1, all items present, all item_ids assigned
   - `test_cluster_naming.py` (4 tests): returns name+description keys, name is nonempty string, description is nonempty string, bad LLM schema crashes with AssertionError
   - `test_serialization.py` (12 tests): round-trip turn_index, timestamp, cluster count, cluster fields, assignments, soft_probs; int key preservation for assignments and soft_probs; valid JSON output; audit log creates file, one line per call, each line valid JSON

---

## RED State Verification

All test files use module-level imports (`from src.xxx import ...`). Since no `src/` package exists yet, all 6 files fail during pytest collection with:

```
ModuleNotFoundError: No module named 'src'
```

This is the correct RED state per TDD Wave 0 protocol. Tests will turn GREEN when implementation plans (01-02 through 01-05) create the corresponding modules.

---

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 15921d1 | chore(01-01): install pytest and configure project |
| 2 | f8472b3 | test(01-01): add shared fixtures to conftest.py |
| 3 | 6e7c1cc | test(01-01): add all six Phase 1 test stub files (RED state) |

---

## Deviations from Plan

None — plan executed exactly as written. The collection behavior (exit code 2 due to ModuleNotFoundError at import time) is the expected and documented outcome for module-level imports before implementation modules exist.

---

## Known Stubs

The following files contain tests that reference modules not yet implemented. These are intentional stubs — they will fail until the corresponding implementation plan runs:

| Stub | File | Reason |
|------|------|--------|
| `from src.data_loader import ...` | tests/phase1/test_dataset.py | src/data_loader.py not yet written (Plan 01-02) |
| `from src.stopping import ...` | tests/phase1/test_stopping_criteria.py | src/stopping.py not yet written (Plan 01-02) |
| `from src.embedding_store import ...` | tests/phase1/test_embedding_store.py | src/embedding_store.py not yet written (Plan 01-03) |
| `from src.clustering import ...` | tests/phase1/test_clustering.py | src/clustering.py not yet written (Plan 01-03) |
| `from src.cluster_naming import ...` | tests/phase1/test_cluster_naming.py | src/cluster_naming.py not yet written (Plan 01-03) |
| `from src.serialization import ...` | tests/phase1/test_serialization.py | src/serialization.py and src/state.py not yet written (Plan 01-04) |

These stubs are intentional and required — they define the contract before implementation.

---

## Threat Flags

No new security-relevant surface introduced. All files are test infrastructure only: no network endpoints, no auth paths, no file access patterns beyond tmp_path fixtures, no schema changes at trust boundaries.

---

## Self-Check: PASSED

- pyproject.toml exists: FOUND
- tests/__init__.py exists: FOUND
- tests/phase1/__init__.py exists: FOUND
- tests/conftest.py exists: FOUND
- tests/phase1/test_dataset.py exists: FOUND (6 test functions)
- tests/phase1/test_stopping_criteria.py exists: FOUND (8 test functions)
- tests/phase1/test_embedding_store.py exists: FOUND (5 test functions)
- tests/phase1/test_clustering.py exists: FOUND (6 test functions)
- tests/phase1/test_cluster_naming.py exists: FOUND (4 test functions)
- tests/phase1/test_serialization.py exists: FOUND (12 test functions)
- Commit 15921d1 exists: FOUND
- Commit f8472b3 exists: FOUND
- Commit 6e7c1cc exists: FOUND
- RED state confirmed: ModuleNotFoundError for src.* on all test files
