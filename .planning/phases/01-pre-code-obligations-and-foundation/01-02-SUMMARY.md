---
phase: 01-pre-code-obligations-and-foundation
plan: "02"
subsystem: data-foundation
tags: [amazon-reviews, hdbscan, sha256, stopping-criteria, datasets, jsonl, split]

dependency_graph:
  requires:
    - phase: 01-01
      provides: pytest test scaffold with 41 stubs for PRE-01 and PRE-02
  provides:
    - src/data_loader.py with compute_sha256, verify_held_out_hash, split_dataset, download_and_save_dataset
    - src/stopping.py with StopReason, StoppingCriteria, FeedbackMagnitudeWeights, check_stopping
    - dataset/held_out.jsonl — 3000-record frozen held-out split (SHA-256 sealed, never regenerate)
    - dataset/held_out.sha256 — ea53dfa14bfe72f4... (committed, never change)
    - dataset/train.jsonl — 12000-record training split
    - dataset/arts_crafts_15k.jsonl — raw 15K Amazon Reviews 2023 (Arts, Crafts & Sewing)
  affects:
    - 01-03 (FOUND-01 embedding store reads from train.jsonl)
    - 01-04 (serialization layer uses ClusteringState schema from D-13)
    - 01-05 (pipeline uses verify_held_out_hash at session start)
    - Phase 4 (Judge Agent fills in FeedbackMagnitudeWeights values and diminishing-returns branch)

tech-stack:
  added:
    - datasets==2.21.0 (HuggingFace streaming download; 4.8.5 incompatible with Amazon Reviews loading script)
  patterns:
    - Fail-loudly: all invariants enforced via assert, zero try/except outside CLI/API boundary
    - Overwrite guard: download_and_save_dataset asserts held_out.jsonl does not exist before writing
    - Chunked SHA-256: compute_sha256 reads in 64KB chunks to handle large files
    - Seeded reproducible split: random.Random(seed).shuffle for deterministic 80/20 split
    - Enum-based stopping reasons with dataclass configuration and typed stub

key-files:
  created:
    - src/__init__.py
    - src/data_loader.py
    - src/stopping.py
    - dataset/arts_crafts_15k.jsonl
    - dataset/train.jsonl
    - dataset/held_out.jsonl
    - dataset/held_out.sha256
  modified: []

key-decisions:
  - "datasets downgraded to 2.21.0: datasets==4.8.5 dropped trust_remote_code support, which Amazon Reviews 2023 requires for its Python loading script"
  - "HF_TEXT_FIELD='text': confirmed per RESEARCH.md correction that actual HuggingFace field is 'text', not 'review_text'"
  - "Held-out SHA-256: ea53dfa14bfe72f4200091592a0657ce93a028e466d5a3c37587c13a08ca69ff — committed and immutable"
  - "Diminishing-returns branch in check_stopping is an explicit stub — Phase 4 fills in threshold and implementation"
  - "FeedbackMagnitudeWeights fields initialized to float('nan') as Phase 4 placeholders per D-10"

patterns-established:
  - "Pattern: Overwrite guard — assert not os.path.exists() before writing irreversible files"
  - "Pattern: SHA-256 lock — compute hash immediately after writing, verify immediately after computing"
  - "Pattern: Typed stub — Phase 1 specifies structure with nan/sentinel placeholders; later phases fill values"

requirements-completed:
  - PRE-01
  - PRE-02

duration: 5min
completed: "2026-05-05"
---

# Phase 1 Plan 02: Data Split and Stopping Criteria — Summary

**Amazon Reviews 2023 Arts/Crafts/Sewing 15K downloaded, 80/20 split sealed with SHA-256 ea53dfa1, and three OR-combined stopping criteria typed as Python spec with turn_budget=15 and Phase 4 magnitude stubs**

---

## Performance

- **Duration:** 5 minutes
- **Started:** 2026-05-05T08:34:09Z
- **Completed:** 2026-05-05T08:39:41Z
- **Tasks:** 3 (1a + 1b + 2)
- **Files modified:** 7 created, 0 modified

## Accomplishments

- Frozen held-out split: `dataset/held_out.jsonl` (3000 records) sealed with SHA-256 hash committed to git — PRE-01 complete
- `verify_held_out_hash()` uses `assert` so any held-out contamination crashes immediately with "hash mismatch" in the message
- Stopping criteria typed as Python with `StopReason` enum (3 values), `StoppingCriteria` (turn_budget=15), `FeedbackMagnitudeWeights` (all nan, Phase 4 placeholders), and `check_stopping()` with oracle_satisfied > turn_budget > diminishing_returns (stub) — PRE-02 complete
- All 14 tests across `test_dataset.py` (6) and `test_stopping_criteria.py` (8) pass

## Task Commits

1. **Task 1a: Write src/__init__.py and src/data_loader.py** — `6302c45` (feat)
2. **Task 1b: Run live dataset download** — `470d788` (feat)
3. **Task 2: Write src/stopping.py** — `952e7a0` (feat)

## Files Created/Modified

- `src/__init__.py` — empty package init
- `src/data_loader.py` — compute_sha256, verify_held_out_hash (assert crash on mismatch), split_dataset (seeded 80/20), download_and_save_dataset (idempotent with overwrite guard)
- `src/stopping.py` — StopReason enum, StoppingCriteria dataclass, FeedbackMagnitudeWeights dataclass, check_stopping function (diminishing-returns as documented Phase 4 stub)
- `dataset/arts_crafts_15k.jsonl` — 15000 raw records (item_id, text)
- `dataset/train.jsonl` — 12000 train records (seed=42)
- `dataset/held_out.jsonl` — 3000 held-out records — NEVER REGENERATE
- `dataset/held_out.sha256` — `ea53dfa14bfe72f4200091592a0657ce93a028e466d5a3c37587c13a08ca69ff` — NEVER CHANGE

## Decisions Made

- Downgraded `datasets` from 4.8.5 to 2.21.0: `datasets==4.8.5` removed `trust_remote_code` support, which the Amazon Reviews 2023 dataset requires for its Python loading script. 2.21.0 works correctly.
- Confirmed `HF_TEXT_FIELD = "text"` (not "review_text") per RESEARCH.md: the HuggingFace field is `text`; `review_text` is a project-level naming convention in CONTEXT.md D-02.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] datasets==4.8.5 incompatible with Amazon Reviews 2023 trust_remote_code**
- **Found during:** Task 1b (live dataset download)
- **Issue:** `datasets==4.8.5` dropped `trust_remote_code` parameter support. The Amazon Reviews 2023 dataset uses a Python loading script, which requires this parameter. Error: `RuntimeError: Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py`
- **Fix:** Downgraded to `datasets==2.21.0` which supports `trust_remote_code=True` and successfully loads the streaming dataset. Restored `trust_remote_code=True` in `download_and_save_dataset()`.
- **Files modified:** `src/data_loader.py` (parameter removed then restored)
- **Verification:** Full 15000 records downloaded, 12000/3000 split confirmed, hash verification passed
- **Committed in:** `470d788` (Task 1b commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - library API compatibility bug)
**Impact on plan:** Essential fix — without it the dataset cannot be downloaded. No scope creep.

## Issues Encountered

- HuggingFace hub cache contained a stale snapshot with the Amazon Reviews 2023 Python loading script. Cache cleared before retry succeeded.

## User Setup Required

None — no external service configuration required beyond internet access for the one-time dataset download (already completed and committed).

## Next Phase Readiness

- `dataset/train.jsonl` is ready for Phase 1 Plan 03 (embedding store: load texts, encode with SentenceTransformer, persist as .npy)
- `dataset/held_out.jsonl` is sealed — do not pass to any training or clustering code
- `src/stopping.py` provides the typed contract for Phase 4 (Judge Agent) to implement the diminishing-returns branch
- 14 tests green; remaining 27 test stubs (test_embedding_store.py, test_clustering.py, test_cluster_naming.py, test_serialization.py) are RED pending Plans 01-03 and 01-04

---

## Known Stubs

| Stub | File | Reason |
|------|------|--------|
| `diminishing_returns` branch in `check_stopping()` | src/stopping.py | Phase 4 decision: exact epsilon threshold and N_fallback turns not yet specified (D-10) |
| `FeedbackMagnitudeWeights` fields all `float("nan")` | src/stopping.py | Phase 4 decision: exact weight values not yet specified (D-10) |
| `StoppingCriteria.magnitude_threshold_epsilon = float("nan")` | src/stopping.py | Phase 4 decision placeholder |
| `StoppingCriteria.magnitude_fallback_turns = -1` | src/stopping.py | Phase 4 decision placeholder |

These stubs are intentional per the plan — Phase 1 specifies structure only; Phase 4 fills values. The stubs do not prevent this plan's goals (PRE-01 and PRE-02) from being achieved.

---

## Threat Flags

No new security-relevant surface beyond what was specified in the plan threat model (T-01-02 through T-01-05). The `verify_held_out_hash()` function implements T-01-02 mitigation: any held-out file tampering crashes immediately with `AssertionError` containing "hash mismatch".

---

## Self-Check

Checking files exist:
- `src/__init__.py`: FOUND
- `src/data_loader.py`: FOUND
- `src/stopping.py`: FOUND
- `dataset/held_out.jsonl`: FOUND (3000 lines)
- `dataset/held_out.sha256`: FOUND (ea53dfa14bfe72f4...)
- `dataset/train.jsonl`: FOUND (12000 lines)
- `dataset/arts_crafts_15k.jsonl`: FOUND (15000 lines)

Checking commits exist:
- `6302c45`: FOUND (Task 1a)
- `470d788`: FOUND (Task 1b)
- `952e7a0`: FOUND (Task 2)

All 14 tests pass: `pytest tests/phase1/test_dataset.py tests/phase1/test_stopping_criteria.py` — 14 passed in 0.12s

## Self-Check: PASSED

*Phase: 01-pre-code-obligations-and-foundation*
*Completed: 2026-05-05*
