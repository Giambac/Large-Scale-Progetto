---
plan: 02-04
phase: 02-clustering-agent-core
status: complete
completed: 2026-05-07
subsystem: agent-functions-conversation-loop
tags: [agent-functions, conversation-loop, f_output, f_next_state, f_next_best_step, split, merge, move-item, global-feedback, audit-log, fail-loudly]
dependency_graph:
  requires: [02-01, 02-02, 02-03]
  provides: [src/agent_functions.py, src/conversation_loop.py]
  affects: [web/app.py]
tech_stack:
  added: []
  patterns: [type-priority-dispatch, global-instructions-accumulator, column-pooling-merge, kmeans-seeded-split, monotonic-cluster-ids, anytime-f_output, oracle-protocol-boundary, fail-loudly-assert]
key_files:
  created:
    - src/agent_functions.py
    - src/conversation_loop.py
  modified:
    - tests/phase2/test_agent_functions.py
decisions:
  - "_apply_split and _apply_merge implemented in Task 1 commit alongside f_output/f_next_best_step — no stubs needed since all test cases required both helpers from the start"
  - "f_next_state accepts hierarchy=None; creates a HierarchyStore and registers existing clusters when called without one — enables test convenience without requiring callers to always build hierarchy"
  - "global_instructions defaults to [] when None in f_next_state — safe for callers that do not need FB-01 accumulation"
  - "_apply_move_item skips namer.name_cluster when sample_texts is empty (id_to_text dict has no entries for affected cluster) — keeps clusters with original name rather than crashing on empty texts"
metrics:
  duration: "~6 minutes"
  completed_date: "2026-05-07"
  tasks_completed: 3
  tasks_total: 3
  files_created: 2
  files_modified: 1
commits:
  - 44e372f
  - dec8b5c
requirements:
  - CLUS-01
  - CLUS-03
  - CLUS-04
  - FB-01
  - FB-02
  - FB-03
  - HIER-01
  - HIER-02
---

# Phase 02 Plan 04: Agent Functions and Conversation Loop Summary

f_output, f_next_best_step, f_next_state (split/merge/move/global dispatch) pure functions plus the run_conversation() while-loop orchestrator that owns all I/O and maintains the GlobalFeedback accumulator across turns.

## What Was Built

**Task 1 + 2 — src/agent_functions.py** (commit `44e372f`):

- `f_output(state)` — anytime function: asserts `len(assignments) == N > 0`, `len(soft_probs) == N`, `len(clusters) > 0`; returns state unchanged (CLUS-01 anytime behavior)
- `f_next_best_step(state, strategy, uncertainty_report)` — pure delegation to `strategy.select()`; no I/O (CLUS-03)
- `_cluster_id_to_index(state)` — helper: builds `{cluster_id: position}` mapping fresh on each call
- `_next_cluster_id(state)` — monotonic counter: `max(c.id for c in state.clusters) + 1` (D-11)
- `_apply_move_item(delta, state, namer, id_to_text, global_instructions)` — D-10: sets `soft_probs[item_id][target_idx] = 0.95`; redistributes residual 0.05 proportionally among non-target clusters; UNIFORM_FALLBACK_THRESHOLD guard for zero-sum edge case (T-02-10 mitigated); asserts row sums to 1.0; names source and target clusters (D-12)
- `_apply_split(delta, state, store, namer, hierarchy, id_to_text, global_instructions)` — D-08: oracle-seeded K-means (or k-means++ fallback); distributes retired cluster prob proportionally into two new sub-clusters; re-normalizes all rows; asserts soft_probs length == new K for every item (T-02-08 mitigated); asserts len(target.item_ids) >= 2 (T-02-11 mitigated); names both sub-clusters (D-12); calls `hierarchy.record_split()`
- `_apply_merge(delta, state, namer, hierarchy, id_to_text, global_instructions)` — D-09: column pooling: `new_prob = probs[a_idx] + probs[b_idx]`; drops retired columns; re-normalizes; names merged cluster (D-12); calls `hierarchy.record_merge()`
- `f_next_state(state, deltas, store, namer, hierarchy, id_to_text, global_instructions)` — type-priority dispatch (D-07): GlobalFeedback (0) → SplitFeedback/MergeFeedback (1) → MoveItemFeedback (2) → InstructionalFeedback (3); sorts by priority before applying; appends GlobalFeedback.instruction_text to `global_instructions` in-place (FB-01); completeness invariants asserted at return site; bumps turn_index and sets UTC timestamp
- No try/except anywhere in the file
- 15 tests GREEN in `test_agent_functions.py` (including `test_global_feedback_accumulates`)

**Task 3 — src/conversation_loop.py** (commit `dec8b5c`):

- `_format_message(action, state)` — converts Action to human-readable oracle message; asserts isinstance(action, Action)
- `run_conversation(initial_state, oracle, store, namer, strategy, log_path, criteria, socketio, id_to_text, llm_client)` — plain while loop (D-01):
  - Strategy=None default: uses `RandomStrategy(seed=0)`
  - Criteria=None default: uses `StoppingCriteria()` (turn_budget=15)
  - HierarchyStore instantiated at session start; all initial clusters registered
  - `global_instructions: list[str] = []` initialized at session start; passed to `f_next_state` every turn (FB-01)
  - Loop body: f_uncertainty → f_next_best_step → oracle.reply → parse_feedback (llm_client required) → f_next_state → append_to_audit_log → socketio.emit (if not None) → check_stopping → advance or break
  - SocketIO emits skipped when `socketio is None` (unit test mode)
  - `parse_feedback` only called when `llm_client is not None` and `reply.raw_text.strip()` non-empty
  - No try/except in loop body (parse_feedback handles its own LLM boundary)
- 3 tests GREEN in `test_conversation_loop.py`

## Verification Results

```
pytest tests/phase2/test_agent_functions.py tests/phase2/test_conversation_loop.py -q
..................
18 passed in 2.1s
```

```
pytest tests/phase2/ -m "not llm" --ignore=tests/phase2/test_app.py
60 passed, 1 deselected, 1 warning in 2.09s
```

```
pytest tests/phase2/test_agent_functions.py::test_global_feedback_accumulates -v
1 passed in 2.10s
```

## Deviations from Plan

### Implementation Decision

**[Plan structure] Tasks 1 and 2 implemented in a single commit**

- **Found during:** Task 1 implementation
- **Issue:** The plan called for Task 1 to have `_apply_split` and `_apply_merge` as `raise NotImplementedError` stubs, with Task 2 replacing them. However, the test suite in `test_agent_functions.py` includes split/merge tests in the same file, and these tests would fail (with `NotImplementedError`) if the stubs were left in place during Task 1's GREEN phase. Since TDD requires tests to pass GREEN after each task, the full implementations were included in Task 1.
- **Fix:** Implemented `_apply_split` and `_apply_merge` fully in Task 1; Task 2 had no additional source changes (same commit `44e372f` covers both). All Task 2 acceptance criteria pass.
- **Files modified:** `src/agent_functions.py`
- **Commit:** `44e372f`

### Auto-fixed: test_global_feedback_accumulates stub

**[Rule 1 - Bug] Implemented `raise NotImplementedError` test stub**

- **Found during:** Task 1
- **Issue:** `test_agent_functions.py::test_global_feedback_accumulates` had `raise NotImplementedError` as its body — a scaffold stub from plan 02-01
- **Fix:** Implemented the test body per the plan's `<behavior>` spec: creates two `GlobalFeedback` deltas, calls `f_next_state` with a `global_instructions` list, asserts `len(global_instructions) == 2` after the call
- **Files modified:** `tests/phase2/test_agent_functions.py`
- **Commit:** `44e372f`

## Known Stubs

- `global_instructions` is accumulated and passed through to `_apply_split`, `_apply_merge`, `_apply_move_item`, but NOT passed to `namer.name_cluster` calls in Phase 2 — Phase 3 will enrich naming prompts with accumulated oracle instructions. The accumulation itself is the Phase 2 observable behavior (tested).
- `Action.payload` is always `{}` in Phase 2 (from strategy.py) — Phase 5 enriches payloads.

## Threat Surface Scan

All threats in the plan's `<threat_model>` are addressed:

| Threat | Mitigation Status |
|--------|-------------------|
| T-02-08: soft_probs vector length mismatch after split/merge | Mitigated — `assert len(probs) == new_k` for every item in `_apply_split`; `assert abs(sum(probs) - 1.0) < 1e-5` after every operation |
| T-02-09: retired cluster ID reused | Mitigated — `_next_cluster_id` uses `max(c.id)+1`; new IDs always > any existing cluster ID |
| T-02-10: zero-sum proportional redistribution edge case | Mitigated — `UNIFORM_FALLBACK_THRESHOLD` guard in `_apply_move_item`; distributes residual uniformly when `original_non_target_sum < 1e-9` |
| T-02-11: KMeans on cluster with < 2 items | Mitigated — `assert len(target.item_ids) >= 2` with descriptive message before `km.fit()` |
| T-02-12: AuditLog written each turn | Accepted — JSONL contains clustering state (no PII); file path controlled by caller; loop writes every turn per D-04 |

No new threat surface introduced beyond what the plan enumerated.

## Self-Check: PASSED

- [x] `src/agent_functions.py` exists with `f_output`, `f_next_best_step`, `f_next_state`, `_apply_split`, `_apply_merge`, `_apply_move_item`
- [x] `src/conversation_loop.py` exists with `run_conversation`
- [x] commit `44e372f` exists: Tasks 1+2 combined
- [x] commit `dec8b5c` exists: Task 3
- [x] 18 tests GREEN (15 from test_agent_functions.py + 3 from test_conversation_loop.py)
- [x] 60 total Phase 2 non-LLM non-app tests GREEN
- [x] `test_global_feedback_accumulates` passes
- [x] No try/except in agent_functions.py
- [x] No try/except in conversation_loop.py
- [x] `global_instructions.append` present in f_next_state (FB-01)
- [x] `hierarchy.record_split` and `hierarchy.record_merge` called (HIER-01, HIER-02)
- [x] `ORACLE_MOVE_CONFIDENCE` used in `_apply_move_item`
- [x] `UNIFORM_FALLBACK_THRESHOLD` guard in `_apply_move_item`
- [x] `NotImplementedError` count in agent_functions.py = 0
