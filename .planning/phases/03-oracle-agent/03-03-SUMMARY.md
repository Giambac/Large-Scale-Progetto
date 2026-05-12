---
phase: "03"
plan: "03"
subsystem: oracle-agent
tags: [drift-detection, contradiction, deque, orc-04, structural-comparison]
dependency_graph:
  requires:
    - "03-01"  # OracleAgent with _check_contradiction and _delta_window
    - "03-02"  # cognitive_load.py (imported by oracle_agent.py)
  provides:
    - "src/oracle_agent.py — update_delta_window() method, structural drift detection complete"
  affects:
    - "src/conversation_loop.py — Wave 3 (Plan 04) will call oracle.update_delta_window() after parse_feedback()"
tech_stack:
  added: []
  patterns:
    - "Check-then-append deque pattern: contradictions checked BEFORE appending so same-turn deltas don't conflict"
    - "Fail-loudly: no try/except in _contradicts, _check_contradiction, or update_delta_window"
    - "Structural isinstance dispatch over frozen dataclasses (D-09)"
key_files:
  created: []
  modified:
    - src/oracle_agent.py
decisions:
  - "update_delta_window() is the public API for the loop; _check_contradiction() remains a private helper (single-delta check)"
  - "GlobalFeedback and InstructionalFeedback are skipped for both checking AND appending — structurally opaque per D-09"
  - "Same-turn deltas cannot contradict each other: check runs before append (all checks see only prior turns)"
  - "Wave 2 comment placeholder in reply() replaced with explanatory comment about Wave 3 loop wiring"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-12"
  tasks_completed: 1
  files_created: 0
  files_modified: 1
---

# Phase 03 Plan 03: Structural Drift Detection (update_delta_window) Summary

**One-liner:** Added `update_delta_window(deltas, turn_index)` to OracleAgent implementing the check-then-append deque pattern for structural contradiction detection (ORC-04 / D-09) — all 4 contradiction unit tests pass, no regressions.

---

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add update_delta_window() and finalize structural drift detection | 8a8b03e | src/oracle_agent.py |

---

## What Was Built

### `src/oracle_agent.py` — changes from Wave 1 baseline

**New method: `update_delta_window(self, deltas, turn_index)`**

The public API for the conversation loop (Wave 3 / Plan 04). Called by `run_conversation()` after `parse_feedback()` returns deltas.

Behavior:
1. Iterates `deltas`, skipping `GlobalFeedback` and `InstructionalFeedback` (D-09)
2. For each structural delta, calls `_check_contradiction(delta, turn_index)` against the existing rolling deque
3. Tracks the first contradiction found (earliest detection per turn)
4. After all checks, appends all structural deltas to `_delta_window`
5. Returns `(first_contradiction, first_contradicted_turn)` — `(False, None)` if none

**Critical invariant:** Check runs BEFORE append — deltas within the same turn do not contradict each other. This matches D-09: "compare each delta against a rolling window of PRIOR deltas".

**Pre-existing methods confirmed working:**
- `_contradicts(new_delta, prior_delta)` — module-level function implementing all D-09 rules
- `_check_contradiction(self, delta, current_turn)` — iterates deque, no mutation

**Comment update in reply():**
Replaced the Wave 2 TODO comment with an explanatory note describing that `contradiction_detected=False` is always returned by `reply()` itself; the loop overwrites it after calling `update_delta_window()` (Wave 3 wiring).

---

## Contradiction Rules (D-09) — Verified

| Rule | New delta | Prior delta | Condition |
|------|-----------|-------------|-----------|
| Merge vs Split | MergeFeedback(A,B) | SplitFeedback(X) | X == A or X == B |
| Split vs Merge | SplitFeedback(X) | MergeFeedback(A,B) | X == A or X == B (retired IDs) |
| Move vs Move | MoveItemFeedback(item,B) | MoveItemFeedback(item,C) | same item_id AND B != C |
| Global/Instructional | any | any | always False |

**Merged cluster ID handling (Pitfall 3 / RESEARCH.md):** Verified via `src/agent_functions.py _apply_merge()` — merged cluster receives a brand-new ID via `_next_cluster_id()`, NOT reusing `cluster_a_id` or `cluster_b_id`. Therefore `SplitFeedback(X)` is contradictory when `X == cluster_a_id` or `X == cluster_b_id` (the retired IDs), not when `X == new_merged_id`.

---

## Verification Results

```
pytest tests/phase3/test_oracle_agent.py -q -m "not llm" -k "contradiction"
4 passed in 0.28s

pytest tests/phase3/test_oracle_agent.py -m "not llm"
13 passed, 1 warning in 0.84s

pytest tests/phase3/test_cognitive_load.py tests/phase3/test_oracle_agent.py -m "not llm"
18 passed, 1 warning in 1.84s

python -c "from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams; from unittest.mock import MagicMock; agent = OracleAgent(OracleSpec(3, ['t'], 'x'), NoiseParams(0.8, 0.1, 0.9), MagicMock()); assert hasattr(agent, '_check_contradiction'); assert hasattr(agent, 'update_delta_window'); print('OK')"
OK
```

Pre-existing failures in `tests/phase3/test_oracle_loop_integration.py` (3 tests) are due to `sentence_transformers` not installed in this environment — same as documented in 03-02 SUMMARY. They are out of scope for this plan (Wave 3 / Plan 04 work). Pre-existing failures in `tests/phase1/` have the same root cause.

---

## Deviations from Plan

**Deviation: Wave 1 files not present in worktree at start**

The worktree was initialized from a pre-Wave-1 branch (da36c91). The `git reset --hard e94e210ae062303c8cc0992dc72d8f3e44aa0f64` step from the `<worktree_branch_check>` brought it to the correct base (Wave 1 merge commit), making `src/oracle_agent.py`, `src/cognitive_load.py`, and `tests/phase3/` available.

**Deviation: _contradicts and _check_contradiction already present**

Wave 1 (Plan 03-01) had already implemented `_contradicts()` (module-level) and `_check_contradiction()` (method) in oracle_agent.py. Only `update_delta_window()` was missing. The plan's Task 1 action items were partially pre-done; only the method addition and comment cleanup were needed.

No Rule 4 (architectural) deviations. Plan executed within scope.

---

## Known Stubs

None — `update_delta_window()` is fully implemented. The `contradiction_detected=False` in `reply()` is intentionally a stub that Wave 3 (Plan 04) will fill in by calling `update_delta_window()` in the loop and overwriting the field on the reply object. This is documented explicitly in the reply() method's inline comment.

---

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. `update_delta_window()` is a pure in-memory deque operation. All FeedbackDelta types are frozen dataclasses — no mutation risk.

Threat model mitigations from the plan:

| Threat ID | Mitigation | Status |
|-----------|-----------|--------|
| T-03-03-01 | deque(maxlen=10) auto-evicts; frozen dataclasses cannot be mutated | Applied (deque was already bounded in Wave 1) |
| T-03-03-02 | Window holds only structural delta types, no raw text or PII | Applied |
| T-03-03-03 | deque(maxlen=window_size) prevents unbounded growth | Applied |

---

## Self-Check: PASSED

- `src/oracle_agent.py` exists: FOUND
- `def update_delta_window` in oracle_agent.py: FOUND
- `def _contradicts` in oracle_agent.py: FOUND
- `def _check_contradiction` in oracle_agent.py: FOUND
- Task commit `8a8b03e` exists: CONFIRMED
- `test_contradiction_merge_after_split` passes: CONFIRMED
- `test_contradiction_move_item` passes: CONFIRMED
- `test_no_contradiction_empty_window` passes: CONFIRMED
- `test_no_false_positive_contradiction` passes: CONFIRMED
- All 13 tests in test_oracle_agent.py pass: CONFIRMED
