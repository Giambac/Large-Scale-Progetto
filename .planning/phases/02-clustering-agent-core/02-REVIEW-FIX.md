---
phase: 02-clustering-agent-core
fixed_at: 2026-05-07T00:00:00Z
review_path: .planning/phases/02-clustering-agent-core/02-REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-05-07T00:00:00Z
**Source review:** `.planning/phases/02-clustering-agent-core/02-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 8 (CR-01 through CR-04, WR-01 through WR-04)
- Fixed: 8
- Skipped: 0

All 65 non-LLM tests pass after all fixes (`python -m pytest tests/phase2/ -m "not llm" --tb=no`).

## Fixed Issues

### CR-01: `_apply_move_item` silently corrupts state when source and target cluster are the same

**Files modified:** `src/agent_functions.py`
**Commit:** 3d078df
**Applied fix:** Added an early-return guard immediately after `source_cluster_id` is resolved. When `source_cluster_id == feedback.target_cluster_id` the function returns the original state unchanged, preventing the loop from removing the item from its cluster without re-inserting it. Used early return (not assert) per the fix guidance because the LLM may legitimately emit no-op moves.

---

### CR-02: `_apply_merge` does not guard against merging a cluster with itself

**Files modified:** `src/agent_functions.py`, `src/feedback_parser.py`
**Commit:** 3d078df (agent_functions.py), 70b1d45 (feedback_parser.py)
**Applied fix:** Added `assert delta.cluster_a_id != delta.cluster_b_id` in `_apply_merge` immediately after the two `id_to_idx` membership assertions. Also added the same guard in `_build_delta` in `feedback_parser.py` for the `"merge"` case, providing an earlier catch with a clear error message at the parser layer.

---

### CR-03: `_apply_split` produces degenerate K-Means when exactly one seed item is provided

**Files modified:** `src/agent_functions.py`
**Commit:** 3d078df
**Applied fix:** Replaced the single-seed centroid duplication (`np.vstack([seed_embeddings, seed_embeddings])`) with a conditional: if `len(delta.seed_item_ids) >= 2` use oracle-seeded KMeans with `n_init=1`; otherwise fall back to `KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)`. This eliminates the identical-centroid degenerate case that caused all items to land in label 0 and produced an empty sub-cluster.

---

### CR-04: `f_uncertainty` produces `nan` centroids for empty clusters

**Files modified:** `src/uncertainty.py`
**Commit:** 977f5aa
**Applied fix:** Added `assert len(cluster.item_ids) > 0` at the top of the centroid-computation loop in `f_uncertainty`. This fails loudly at the point of the bad state rather than propagating `nan` through centroid arithmetic and silently corrupting `merge_candidates` sort order.

---

### WR-01: `SplitFeedback.seed_item_ids` mutable list in frozen dataclass

**Files modified:** `src/feedback.py`, `src/feedback_parser.py`
**Commit:** 70b1d45
**Applied fix:** Changed `SplitFeedback.seed_item_ids` type annotation from `list[int]` to `tuple[int, ...]` in `feedback.py`. Updated `_build_delta` in `feedback_parser.py` to wrap `item["seed_item_ids"]` in `tuple()` when constructing `SplitFeedback`, ensuring all parser-produced instances hold an immutable tuple. Updated the docstring to reflect the change.

---

### WR-02: `_apply_split` does not validate that `seed_item_ids` elements belong to the target cluster

**Files modified:** `src/agent_functions.py`
**Commit:** 3d078df
**Applied fix:** Added a membership assertion loop before building the KMeans centroids in `_apply_split`. For each seed in `delta.seed_item_ids[:2]`, asserts it is in `set(target.item_ids)`, crashing immediately with a clear message if the LLM hallucinates seed items from a different cluster.

---

### WR-03: Concurrent `/upload` requests corrupt shared session state without locking

**Files modified:** `web/app.py`
**Commit:** 1cf49d3
**Applied fix:** Added `import threading` and `_session_lock = threading.Lock()` as a module-level lock. Wrapped the session reset (`_session["state"] = None`, `_session["task"] = None`) and `socketio.start_background_task(...)` call inside `with _session_lock:` in `upload_dataset`. This serialises concurrent POST `/upload` requests and prevents two background tasks from racing on `_session["state"]`, the embeddings `.npy` file, and `audit_log.jsonl`.

---

### WR-04: Redundant `try/except` with immediate re-raise in `parse_feedback`

**Files modified:** `src/feedback_parser.py`
**Commit:** 70b1d45
**Applied fix:** Removed the `try/except json.JSONDecodeError: raise` block entirely. Left the bare `raw_items = json.loads(cleaned_text)` call, which propagates `json.JSONDecodeError` naturally. This eliminates misleading exception-handling machinery that did nothing and could have misled future maintainers.

---

_Fixed: 2026-05-07T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
