---
phase: 02-clustering-agent-core
plan: "07"
subsystem: web-ui
tags: [umap, projection, visualization, socketio, canvas, tdd]
dependency_graph:
  requires:
    - 02-06  # ClusteringBackend Protocol + web/app.py --backend flag
  provides:
    - VIZ-V2-01  # UMAP 2D projection panel in web UI
  affects:
    - src/conversation_loop.py  # post_turn_callback parameter added
    - web/app.py                # projection helpers + emit logic
    - web/templates/index.html  # canvas panel added
    - web/static/main.js        # projection_update handler
    - web/static/style.css      # projection panel styles
tech_stack:
  added:
    - umap-learn 0.5.12 (server-side UMAP; anaconda env)
    - pynndescent 0.6.0 (UMAP dependency)
  patterns:
    - TDD RED/GREEN: 10 failing tests first, then implementation
    - SocketIO instance emit in background thread (safe pattern)
    - Canvas 2D API for client-side scatter plot rendering
    - Closure-based post_turn_callback hook for extensible per-turn side effects
key_files:
  created:
    - tests/phase2/test_umap_projection.py
  modified:
    - web/app.py
    - src/conversation_loop.py
    - web/templates/index.html
    - web/static/main.js
    - web/static/style.css
decisions:
  - "UMAP random_state=42 for reproducibility — same coords across runs on same dataset (D-21)"
  - "Projection recomputed on split/merge only — not every turn (D-22)"
  - "Full-width canvas panel above two-column layout (D-23)"
  - "Server-side UMAP, coords emitted via projection_update SocketIO event (D-24)"
  - "post_turn_callback pattern is backward-compatible: default None, existing callers unaffected"
  - "SplitFeedback.seed_item_ids uses tuple[int,...] (frozen dataclass) — test uses empty tuple ()"
metrics:
  duration: "14 minutes"
  completed: "2026-05-09"
  tasks_completed: 2
  files_modified: 5
  files_created: 1
---

# Phase 2 Plan 07: UMAP Projection Panel Summary

**One-liner:** Server-side UMAP 2D projection emitted via SocketIO `projection_update` event; vanilla JS canvas scatter plot with per-cluster colors and soft-probability opacity encoding.

## What Was Built

### Task 1 — TDD RED: tests/phase2/test_umap_projection.py

10 failing tests were written before any implementation:

1. `test_compute_and_emit_projection_importable` — ImportError gate
2. `test_compute_projection_coords_shape` — shape (N, 2) assertion
3. `test_compute_projection_reproducible` — identical coords with fixed random_state=42
4. `test_projection_payload_structure` — payload keys + lengths
5. `test_projection_cluster_ids_match_state` — ordered by item_id
6. `test_projection_max_probs_range` — all probs in [0.0, 1.0]
7. `test_should_recompute_projection_split` — True for SplitFeedback
8. `test_should_recompute_projection_merge` — True for MergeFeedback
9. `test_should_not_recompute_projection_move` — False for MoveItemFeedback
10. `test_should_not_recompute_projection_global` — False for GlobalFeedback

### Task 2 — TDD GREEN: implementation across 5 files

**web/app.py:**
- `_compute_projection(embeddings)` — UMAP with n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
- `_CLUSTER_COLORS` — 20-color palette, cycles for K > 20
- `_build_projection_payload(coords, state)` — builds `{coords, cluster_ids, max_probs, cluster_colors}` dict
- `_should_recompute_projection(deltas)` — True iff any delta is SplitFeedback or MergeFeedback
- `compute_and_emit_projection(store, state, sio)` — orchestrates UMAP + payload build + emit
- `_run_conversation_background`: calls `compute_and_emit_projection` after initial clustering; defines `_projection_post_turn` closure; passes it as `post_turn_callback` to `run_conversation`

**src/conversation_loop.py:**
- `post_turn_callback: Optional[Callable] = None` parameter added to `run_conversation`
- Called with `(new_state, deltas)` after each AuditLog write (Step 6b)
- Backward-compatible: existing callers pass no `post_turn_callback`, default is `None`

**web/templates/index.html:**
- `<section class="projection-panel">` with `<canvas id="projection-canvas">` inserted above `<main class="layout">`
- No new `<script src=` tags added (count remains 2)

**web/static/main.js:**
- `socket.on('projection_update', ...)` handler calls `drawProjection`
- `drawProjection(coords, clusterIds, maxProbs, clusterColors)` — normalizes coords to canvas bounding box, draws 3px arcs with per-cluster color and opacity from max_probs
- `hexToRgba(hex, alpha)` — converts `#rrggbb` to `rgba(r,g,b,alpha)`

**web/static/style.css:**
- `.projection-panel`, `.projection-title`, `#projection-canvas` rules appended

## TDD Gate Compliance

| Gate | Commit | Notes |
|------|--------|-------|
| RED  | ff504f7 | 10 tests, all ImportError/AttributeError failures |
| GREEN | 537761c | All 10 tests pass; existing test_app.py unbroken |
| REFACTOR | — | Code was clean on first pass; no refactor needed |

## Test Results

```
tests/phase2/test_umap_projection.py: 10 passed (34s)
tests/phase2/test_app.py: 5 passed
```

Pre-existing failures in `test_conversation_loop.py` and `test_agent_functions.py` are caused by `sentence_transformers` not being installed in the anaconda environment — confirmed pre-existing before this plan by reverting changes and running tests.

## Deviations from Plan

### Auto-fixed Issues

None.

### Notes

**TDD sequence interrupted by git stash:** After confirming the tests were RED (Task 1 commit ff504f7), a `git stash` was used to verify pre-existing test failures. The stash reverted the working tree source file edits (web/app.py, conversation_loop.py, etc.) because they had not been committed yet. All changes were reapplied to disk and committed cleanly as 537761c. This is documented for transparency; no work was lost and all tests pass.

**SplitFeedback tuple type:** The plan's test template used `SplitFeedback(cluster_id=0, seed_item_ids=[])` (list) but the actual frozen dataclass requires `seed_item_ids: tuple[int, ...]`. The test was written with `()` (empty tuple) to match the actual dataclass contract — a minor deviation from the plan template, applied automatically per Rule 1 (correctness).

## Known Stubs

None — `projection_update` event is fully wired. MockOracle never produces SplitFeedback or MergeFeedback in Phase 2, so `_should_recompute_projection` returns False on every turn at runtime. This is by design (D-22): the wiring is correct and will activate when Phase 3's Oracle Agent ships. This is documented in the code, not a stub.

## Threat Flags

No new trust boundaries beyond what the plan's threat model covers.

The `socketio.emit("projection_update", payload)` call uses the instance method `sio.emit()` (not the context-bound `from flask_socketio import emit`) — mitigates T-02-25 as specified.

The `assert coords.shape == (embeddings.shape[0], 2)` in `_compute_projection` mitigates T-02-24 (malformed UMAP output).

## Self-Check: PASSED

Files exist:
- `tests/phase2/test_umap_projection.py` — FOUND
- `web/app.py` (contains compute_and_emit_projection) — FOUND
- `web/templates/index.html` (contains projection-canvas) — FOUND
- `web/static/main.js` (contains drawProjection) — FOUND
- `web/static/style.css` (contains .projection-panel) — FOUND

Commits exist:
- ff504f7 (TDD RED) — FOUND
- 537761c (TDD GREEN) — FOUND
