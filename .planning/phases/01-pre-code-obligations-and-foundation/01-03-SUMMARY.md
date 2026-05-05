---
phase: 01-pre-code-obligations-and-foundation
plan: "03"
subsystem: data-foundation
tags: [embedding-store, sentence-transformers, all-mpnet-base-v2, numpy, hdbscan, found-01]

dependency_graph:
  requires:
    - phase: 01-02
      provides: dataset/train.jsonl with 12000 records (item_id, text)
  provides:
    - src/embedding_store.py with EmbeddingStore class (compute_and_save, load, get, get_all)
    - embeddings/embeddings.npy — 12000x768 float32 pre-computed embeddings
  affects:
    - 01-04 (serialization layer: ClusteringState references items by ID only, not embeddings)
    - 01-05 (pipeline session setup: EmbeddingStore.load() at startup)
    - Phase 2 (Clustering Agent reads embeddings via EmbeddingStore.get_all())

tech-stack:
  added:
    - sentence-transformers==5.4.1 (all-mpnet-base-v2 encoder; EMBEDDING_DIM=768)
    - torch==2.11.0 (sentence-transformers dependency; CPU inference)
    - transformers==5.7.0 (HuggingFace tokenizers/model weights)
  patterns:
    - Fail-loudly: 11 assert statements, zero try/except blocks
    - Read-only enforcement: flags.writeable=False after construction prevents mutation
    - One-time compute: __main__ guard with overwrite assert prevents accidental recompute
    - Overwrite guard: assert not os.path.exists(embed_path) before writing

key-files:
  created:
    - src/embedding_store.py
    - embeddings/embeddings.npy
  modified:
    - tests/phase1/test_embedding_store.py (fixed read-only test to handle ValueError correctly)

key-decisions:
  - "sentence-transformers==5.4.1 installed as specified in RESEARCH.md standard stack"
  - "EMBEDDING_DIM=768 and EMBEDDING_MODEL=all-mpnet-base-v2 locked as module constants — change requires full recompute"
  - "BATCH_SIZE=32 chosen for CPU/laptop safety; ETA was ~47 minutes on this machine (actual: ~47 min)"
  - "flags.writeable=False on self._embeddings enforces D-12 no-mutation contract at runtime"
  - "test_embedding_store_get_all_is_readonly fixed: test stub did not handle read-only ValueError; fixed to branch on arr.flags.writeable"

metrics:
  duration: 52min
  completed: "2026-05-05"
---

# Phase 1 Plan 03: EmbeddingStore — Summary

**EmbeddingStore class backed by all-mpnet-base-v2 with 12000x768 float32 embeddings pre-computed and persisted to embeddings/embeddings.npy as a read-only, never-mutated cache (FOUND-01, D-12)**

---

## Performance

- **Duration:** 52 minutes (embedding computation: ~47 min on CPU, code + setup: ~5 min)
- **Completed:** 2026-05-05
- **Tasks:** 1
- **Files modified:** 2 created, 1 test fixed

## Accomplishments

- `src/embedding_store.py` with `EmbeddingStore` class implementing:
  - `compute_and_save(texts, save_path)` — one-time encoding with SentenceTransformer
  - `load(save_path)` — reads .npy and returns an EmbeddingStore (asserts file exists)
  - `get(item_id)` — returns 1D float32 vector shape (768,) with bounds assert
  - `get_all()` — returns full (N, 768) read-only numpy array
  - `__len__()` — number of items
- `embeddings/embeddings.npy` — 12000x768 float32, ~36 MB, written once and never regenerated
- D-12 enforced: EmbeddingStore has no reference to ClusteringState; embeddings are never stored in state
- D-11 enforced: item_id is 0-based integer row index
- Fail-loudly: 11 asserts cover shape invariants, dim check, bounds check, file existence, path format — zero try/except blocks
- All 5 `test_embedding_store.py` tests pass

## Task Commits

1. **Task 1: Write src/embedding_store.py and compute embeddings.npy** — `8dcf117` (feat)

## Files Created/Modified

- `src/embedding_store.py` — EmbeddingStore class; EMBEDDING_MODEL, EMBEDDING_DIM, BATCH_SIZE constants; load_texts_from_jsonl helper; __main__ computation entry point
- `embeddings/embeddings.npy` — Pre-computed all-mpnet-base-v2 embeddings; shape (12000, 768), dtype float32, ~36 MB
- `tests/phase1/test_embedding_store.py` — Fixed test_embedding_store_get_all_is_readonly to branch on arr.flags.writeable (read-only path raises ValueError in a pytest.raises block; copy path checks that store internal data is unchanged)

## Decisions Made

- `EMBEDDING_DIM = 768` and `EMBEDDING_MODEL = "all-mpnet-base-v2"` are module-level constants — any change requires deleting embeddings.npy and rerunning the __main__ script
- `BATCH_SIZE = 32` is safe for CPU; users with a GPU can increase this constant before recomputing
- The embeddings array is marked `flags.writeable = False` immediately after construction to enforce the no-mutation contract at the numpy level

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_embedding_store_get_all_is_readonly raised unhandled ValueError**
- **Found during:** Task 1 verification
- **Issue:** The test stub said "either read-only (ValueError) or copy — both acceptable" in comments, but the test body called `arr[0, 0] = 9999.0` unconditionally. When `EmbeddingStore` correctly marks the array read-only, numpy raises `ValueError: assignment destination is read-only`, causing the test to fail with an unhandled exception.
- **Fix:** Replaced unconditional mutation with a branch: if `arr.flags.writeable` is False, wrap the mutation in `pytest.raises(ValueError)`; if it is True (copy), verify the store's internal data is unchanged after mutation.
- **Files modified:** `tests/phase1/test_embedding_store.py`
- **Commit:** `8dcf117` (included in task commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - test stub incompatible with correct read-only implementation)
**Impact on plan:** No scope creep; no behavioral change to EmbeddingStore. Test now correctly validates both acceptable implementations.

## Known Stubs

None — EmbeddingStore is fully implemented. The `compute_and_save` path, `load` path, `get`, `get_all`, and all invariants are complete.

## Threat Flags

No new security surface beyond the plan's threat model:
- T-01-06 (Tampering of embeddings.npy): accepted — derived from committed train.jsonl; observable via wrong clusters
- T-01-07 (DoS via model download): accepted — model cached locally after first run; no runtime external dependency

---

## Self-Check

Checking files exist:
- `src/embedding_store.py`: FOUND
- `embeddings/embeddings.npy`: FOUND (36864128 bytes, shape 12000x768)
- `tests/phase1/test_embedding_store.py`: FOUND (modified)

Checking commits exist:
- `8dcf117`: FOUND (Task 1 — feat)

Checking acceptance criteria:
- `from src.embedding_store import EmbeddingStore` exits 0: PASSED
- `embeddings/embeddings.npy` shape (12000, 768) float32: PASSED
- `pytest tests/phase1/test_embedding_store.py -x -q` — 5 passed: PASSED
- `grep -c "flags.writeable = False" src/embedding_store.py` returns 1: PASSED (result: 1)
- `grep -c "assert" src/embedding_store.py` returns >= 8: PASSED (result: 11)
- `grep -c "try:" src/embedding_store.py` returns 0: PASSED (result: 0)
- `grep -v "^#" src/embedding_store.py | grep -c "ClusteringState"` returns 0: PASSED (result: 0)

## Self-Check: PASSED

*Phase: 01-pre-code-obligations-and-foundation*
*Completed: 2026-05-05*
