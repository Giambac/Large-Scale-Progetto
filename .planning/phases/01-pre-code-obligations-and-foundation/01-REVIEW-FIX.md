---
phase: 01-pre-code-obligations-and-foundation
fixed_at: 2026-05-05T00:00:00Z
review_path: .planning/phases/01-pre-code-obligations-and-foundation/01-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-05-05T00:00:00Z
**Source review:** .planning/phases/01-pre-code-obligations-and-foundation/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (CR-01, CR-02, CR-03, WR-01, WR-02, WR-03, WR-04)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: item_id / embedding row index conflation

**Files modified:** `src/data_loader.py`
**Commit:** a3ec851
**Applied fix:** In `download_and_save_dataset`, after `split_dataset()` returns the train
list, rewrite it via `[{"item_id": i, "text": r["text"]} for i, r in enumerate(train)]`.
This makes `train[i]["item_id"] == i` unconditionally, so the EmbeddingStore row index
matches item_id throughout. The held-out list is left unchanged because it is never used
with EmbeddingStore. Option A from the review was applied.

---

### CR-02: Unguarded json.loads in name_cluster Anthropic path

**Files modified:** `src/cluster_naming.py`
**Commit:** edfcaff
**Applied fix:** Added markdown code-fence stripping to the free function `name_cluster`
(the Anthropic path) immediately before `json.loads`, matching the guard already present in
`GoogleClusterNamer.name_cluster`. The three-line strip pattern is identical: check for
leading triple-backticks, split on them, remove leading "json" tag, then strip whitespace.

---

### CR-03: Dead branch with false assertion in test_embedding_store_get_all_is_readonly

**Files modified:** `tests/phase1/test_embedding_store.py`
**Commit:** 563f7a5
**Applied fix:** Replaced the if/else branching test (which contained a permanently-dead
writeable branch) with a single, explicit test that matches the actual implementation
contract: assert `arr.flags.writeable` is False, then assert mutation raises `ValueError`.
The dead branch and its misleading assertion were removed entirely.

---

### WR-01: setup_phase1.py subset mode inherits CR-01

**Files modified:** none (resolved by CR-01)
**Commit:** a3ec851 (shared with CR-01)
**Applied fix:** WR-01 is fully resolved by the CR-01 re-indexing fix. After CR-01,
`train.jsonl` always has contiguous item_ids 0..N_train-1. Slicing the first N records in
`setup_phase1.py` (`records[:args.n_items]`) therefore yields item_ids 0..N-1, which is
exactly what `EmbeddingStore.get(item_id)` expects. No change to `scripts/setup_phase1.py`
was required.

---

### WR-02: split_dataset builds unnecessary set

**Files modified:** `src/data_loader.py`
**Commit:** a3ec851 (shared with CR-01)
**Applied fix:** Replaced the set-based filter (`train_indices = set(indices[:n_train])`
followed by two full-list scans) with direct index slicing:
`train = [records[i] for i in indices[:n_train]]` and
`held_out = [records[i] for i in indices[n_train:]]`.
Simpler, preserves shuffled order, and easier to audit.

---

### WR-03: GoogleClusterNamer.name_cluster signature diverges from Protocol

**Files modified:** `src/cluster_naming.py`
**Commit:** edfcaff (shared with CR-02)
**Applied fix:** Removed the `max_samples: int = 5` parameter from
`GoogleClusterNamer.name_cluster` and replaced `sample_texts[:max_samples]` with the
literal `sample_texts[:5]`. The signature now matches the `ClusterNamer` Protocol exactly.
`name_all_clusters` already slices to 5 items before calling `name_cluster`, making the
parameter redundant.

---

### WR-04: conftest.py rng.random() dtype kwarg

**Files modified:** `tests/conftest.py`
**Commit:** 5a67f4c
**Applied fix:** Changed `rng.random((5, 768), dtype=np.float32)` to
`rng.random((5, 768)).astype(np.float32)`. The `.astype()` form is idiomatic and works
across all NumPy versions; passing `dtype` as a kwarg to `Generator.random()` was not
formally stabilised before NumPy 1.17 and would surface as a confusing fixture-level
`TypeError` rather than a test failure.

---

## Skipped Issues

None — all findings were fixed.

---

_Fixed: 2026-05-05T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
