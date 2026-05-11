---
phase: 02-clustering-agent-core
plan: "06"
subsystem: clustering-backends
tags: [clustering, kmeans, hdbscan, protocol, backend, bic, soft-probs, tdd]
dependency_graph:
  requires: [02-05]
  provides: [ClusteringBackend Protocol, HDBSCANBackend, KMeansBackend, KMEANS_SOFTMAX_TEMP, --backend CLI flag]
  affects: [src/clustering.py, web/app.py, tests/phase2/test_clustering_backends.py]
tech_stack:
  added: [sklearn.cluster.KMeans, sklearn.mixture.GaussianMixture]
  patterns: [Protocol (runtime_checkable), BIC K selection, softmax of negative centroid distances, argparse module-level parsing]
key_files:
  created: [tests/phase2/test_clustering_backends.py]
  modified: [src/clustering.py, web/app.py]
decisions:
  - "ClusteringBackend is a runtime_checkable Protocol (not ABC) — satisfies D-16 with duck typing; future backends need no inheritance"
  - "KMeansBackend lazy K selection: _k is None until first fit() call; can be overridden directly for tests (backend._k = N)"
  - "BIC loop uses covariance_type='diag' and max_iter=50 for speed on 768-dim embeddings (D-17 constraint)"
  - "HDBSCANBackend calls assign_noise_to_nearest internally so fit() always returns no -1 labels"
  - "backend_init event written as raw JSONL dict (not ClusteringState) — consistent with AuditLog patterns"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-09"
  tasks_completed: 3
  files_changed: 3
requirements_satisfied: [BACK-V2-01]
---

# Phase 2 Plan 06: Clustering Backends (BACK-V2-01) Summary

**One-liner:** ClusteringBackend Protocol with HDBSCANBackend and KMeansBackend (BIC K selection, softmax soft-probs) wired via --backend CLI flag into web/app.py.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write failing tests (RED) | 2fdaca6 | tests/phase2/test_clustering_backends.py |
| 2 | Implement Protocol + backends (GREEN) | 3265efe | src/clustering.py |
| 3 | Wire --backend CLI flag | 281409d | web/app.py |

## What Was Built

### src/clustering.py

Added:
- `KMEANS_SOFTMAX_TEMP = 1.0` — named constant (D-20)
- `ClusteringBackend` — `@runtime_checkable` Protocol with `fit(embeddings) -> (labels, soft_probs)` (D-16)
- `HDBSCANBackend` — wraps existing `run_hdbscan()` + `assign_noise_to_nearest()`; fit() always returns labels with no -1 values
- `KMeansBackend` — lazy BIC K selection (K=2..sqrt(N) via GaussianMixture, covariance_type='diag', D-17); fit() returns labels + softmax soft_probs; `_compute_soft_probs()` static method with numerically stable softmax (D-19)
- Updated `build_initial_clustering_state()` signature to accept `backend: ClusteringBackend | None = None`; defaults to HDBSCANBackend when None

Preserved: `run_hdbscan()`, `assign_noise_to_nearest()` — backward compatible.

### web/app.py

Added:
- `_parse_args()` — argparse with `--backend hdbscan|kmeans`, default hdbscan, asserts on unknown values (D-18)
- `_args = _parse_args()` and `_backend_name: str` — module-level, evaluated once at import
- Backend instantiation in `_run_conversation_background` (HDBSCANBackend or KMeansBackend per `_backend_name`)
- `backend=backend` passed to `build_initial_clustering_state`
- `backend_init` JSONL event appended to audit_log.jsonl after clustering (D-17); also prints chosen K to stdout

### tests/phase2/test_clustering_backends.py

8 tests covering:
1. `test_clustering_backend_protocol_importable` — import check
2. `test_hdbscan_backend_fit_returns_labels_and_soft_probs` — shape + row normalization
3. `test_kmeans_backend_bic_selects_k` — K within [2, sqrt(N)+1]
4. `test_kmeans_backend_fit_returns_correct_shape` — labels (N,), soft_probs (N, K), no -1
5. `test_kmeans_soft_probs_boundary_symmetry` — equidistant point gets ~[0.5, 0.5]
6. `test_kmeans_softmax_temp_constant` — KMEANS_SOFTMAX_TEMP == 1.0
7. `test_build_initial_clustering_state_with_kmeans_backend` — integration test
8. `test_build_initial_clustering_state_with_hdbscan_backend` — integration test

## Test Results

- 8 new backend tests: ALL PASS GREEN
- Prior Phase 2 tests: 60 passing, 13 pre-existing failures (unchanged from before this plan)
- `python -c "from src.clustering import ClusteringBackend, HDBSCANBackend, KMeansBackend, KMEANS_SOFTMAX_TEMP; print('imports ok')"` prints "imports ok"

## Deviations from Plan

None — plan executed exactly as written. The KMeansBackend uses lazy K initialization (set on first fit() call rather than __init__) which is consistent with the plan's test guidance (`backend._k = 3` override pattern).

## Threat Model Coverage

| Threat ID | Status |
|-----------|--------|
| T-02-19 | Mitigated: assert in _parse_args + argparse choices= |
| T-02-20 | Mitigated: assert np.allclose(soft_probs.sum...) in KMeansBackend.fit() |
| T-02-21 | Accepted: diag covariance + max_iter=50 caps BIC loop cost |
| T-02-22 | Mitigated: assert np.all(labels >= 0) in KMeansBackend.fit() |

## Self-Check: PASSED

- tests/phase2/test_clustering_backends.py exists: FOUND
- src/clustering.py modified with Protocol + backends: FOUND
- web/app.py modified with --backend flag: FOUND
- Commits 2fdaca6, 3265efe, 281409d: FOUND in git log
