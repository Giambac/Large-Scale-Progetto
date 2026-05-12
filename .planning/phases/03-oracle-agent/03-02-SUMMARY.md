---
phase: "03"
plan: "02"
subsystem: oracle-agent
tags: [cognitive-load, pure-function, tdd, orc-03]
dependency_graph:
  requires:
    - "03-00"  # OracleProtocol and OracleReply types already present
  provides:
    - "src/cognitive_load.py — f_cognitive_load, MAX_K, MAX_MSG_LEN, TOP_K_ITEMS_PER_CLUSTER, COG_LOAD_THRESHOLD"
  affects:
    - "src/oracle_agent.py — imports f_cognitive_load, COG_LOAD_THRESHOLD (Plan 03-01)"
    - "src/conversation_loop.py — calls f_cognitive_load before oracle.reply() (Plan 03-03)"
tech_stack:
  added: []
  patterns:
    - "Pure function with named constants (same structure as src/uncertainty.py)"
    - "TDD RED/GREEN cycle — test file committed before implementation"
    - "Fail-loudly asserts at function entry — no try/except"
key_files:
  created:
    - src/cognitive_load.py
    - tests/phase3/__init__.py
    - tests/phase3/test_cognitive_load.py
  modified: []
decisions:
  - "Equal weights w=1/3 hard-wired (D-07); configurable weights deferred to Phase 5 ablations"
  - "COG_LOAD_THRESHOLD=0.7 as named constant (D-08); per-persona threshold deferred to NoiseParams extension"
  - "TYPE_CHECKING import guard used for ClusteringState to avoid circular import"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-12"
  tasks_completed: 1
  files_created: 3
  files_modified: 0
---

# Phase 03 Plan 02: Cognitive Load Pure Function Summary

**One-liner:** Pure `f_cognitive_load` function with named constants computing normalized [0,1] load score from cluster count, items shown, and message length — equal-weighted (w=1/3), fail-loudly asserts on empty state.

---

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Add failing tests for f_cognitive_load | 108e0a2 | tests/phase3/__init__.py, tests/phase3/test_cognitive_load.py |
| GREEN | Implement f_cognitive_load pure function | 53d888b | src/cognitive_load.py |

---

## What Was Built

### `src/cognitive_load.py`

A standalone pure-function module implementing ORC-03. Exports:

- `MAX_K: int = 20` — cap for cluster-count normalization
- `MAX_MSG_LEN: int = 500` — cap for message-length normalization
- `TOP_K_ITEMS_PER_CLUSTER: int = 5` — matches `_format_message` default (top-5 items per cluster)
- `COG_LOAD_THRESHOLD: float = 0.7` — D-08 threshold; above this the oracle system prompt injects OVERLOAD instruction
- `f_cognitive_load(state, message) -> float` — returns score in `[0.0, 1.0]`

**Formula (D-07):**
```
w = 1/3
load = min(len(clusters) / MAX_K, 1.0) * w
     + min((len(clusters) * TOP_K_ITEMS_PER_CLUSTER) / total_items, 1.0) * w
     + min(len(message) / MAX_MSG_LEN, 1.0) * w
```

**Preconditions (fail-loudly):**
- `assert len(state.clusters) > 0, "f_cognitive_load: no clusters in state"`
- `assert total_items > 0, "f_cognitive_load: no items in state"`

No `try/except`. No I/O. No global state mutation. No LLM calls.

### `tests/phase3/test_cognitive_load.py`

5 unit tests covering all success criteria:
- `test_load_in_range` — return value always in [0.0, 1.0]
- `test_load_above_threshold` — MAX_K clusters + MAX_MSG_LEN message exceeds COG_LOAD_THRESHOLD
- `test_load_zero_for_minimal_state` — single cluster + empty message, exact value verified
- `test_f_cognitive_load_crashes_on_empty_state` — AssertionError on no-clusters and no-items states
- `test_f_cognitive_load_pure` — same inputs → same output (deterministic)

---

## Verification Results

```
pytest tests/phase3/test_cognitive_load.py -v -m "not llm"
5 passed in 0.05s

python -c "from src.cognitive_load import f_cognitive_load, COG_LOAD_THRESHOLD, MAX_K, MAX_MSG_LEN, TOP_K_ITEMS_PER_CLUSTER; ..."
load=0.353 OK
```

Pre-existing failures in `tests/phase1/` and `tests/phase2/` (missing `hdbscan` and `sentence_transformers` packages not installed in environment) are unchanged — these are environment-level failures unrelated to this plan. The 5 cognitive load tests all pass; no regressions introduced.

---

## Deviations from Plan

None — plan executed exactly as written. The single TDD task followed the RED/GREEN sequence specified in the plan: tests committed first (failing), then implementation committed (all passing).

---

## TDD Gate Compliance

- RED gate: commit `108e0a2` — `test(03-02): add failing tests for f_cognitive_load pure function (ORC-03)` — 5 tests, all failing with `ModuleNotFoundError`
- GREEN gate: commit `53d888b` — `feat(03-02): implement f_cognitive_load pure function (ORC-03)` — 5 tests passing
- REFACTOR gate: not needed — implementation is already clean and minimal

---

## Threat Surface Scan

No new network endpoints, auth paths, or file access patterns introduced. `src/cognitive_load.py` is a pure in-memory computation with no external calls. Threat model mitigations from the plan were fully applied:

| T-03-02-01 | assert total_items > 0 — zero denominator crashes loudly | Mitigated |
| T-03-02-02 | Module constants, no external mutation path | Accepted |

---

## Known Stubs

None — the function is fully implemented with its real formula and all constants wired.

---

## Self-Check: PASSED

- `src/cognitive_load.py` exists: FOUND
- `tests/phase3/test_cognitive_load.py` exists: FOUND
- RED commit `108e0a2` exists: FOUND
- GREEN commit `53d888b` exists: FOUND
- All 5 tests pass: CONFIRMED
