---
phase: "03-oracle-agent"
plan: "04"
subsystem: "conversation-loop"
tags: [oracle-agent, conversation-loop, drift-detection, cognitive-load, events-sidecar, fb-04]
dependency_graph:
  requires:
    - "03-01"  # OracleAgent, OracleSpec, NoiseParams, OracleReply Phase 3 fields
    - "03-02"  # f_cognitive_load pure function
    - "03-03"  # update_delta_window (added as deviation — plan 03 agent did not complete this)
  provides:
    - "run_conversation() with full Phase 3 wiring (ORC-02, ORC-03, ORC-04, FB-04)"
    - "events.jsonl sidecar with oracle_init + drift_event records"
    - "_write_event() helper function"
  affects:
    - "src/conversation_loop.py"
    - "src/agent_functions.py"
    - "src/oracle_agent.py"
tech_stack:
  added: []
  patterns:
    - "JSONL append mode for events.jsonl sidecar (mirrors serialization.py append_to_audit_log pattern)"
    - "Local import of OracleAgent inside run_conversation() body to avoid circular imports"
    - "isinstance(oracle, _OracleAgent) guard for OracleAgent-specific code paths"
key_files:
  created:
    - "tests/phase3/test_oracle_loop_integration.py"
  modified:
    - "src/conversation_loop.py"
    - "src/agent_functions.py"
    - "src/oracle_agent.py"
decisions:
  - "events_path derived from log_path if not provided (os.path.dirname(log_path) / events.jsonl)"
  - "Local import of OracleAgent inside run_conversation() to avoid circular imports at module level"
  - "update_delta_window called with new_state.turn_index (AFTER f_next_state) to avoid off-by-one"
  - "InstructionalFeedback fixed to append to global_instructions (was a pass stub — Rule 1 bug fix)"
  - "oracle_init written in run_conversation loop start in addition to OracleAgent.__init__ (plan spec)"
metrics:
  duration: "~25 minutes"
  completed: "2026-05-12"
  tasks_completed: 2
  files_modified: 3
  files_created: 1
---

# Phase 03 Plan 04: Oracle Loop Integration Summary

Wire all Phase 3 oracle agent components into `run_conversation()`: cognitive load computation before oracle.reply(), OracleAgent-aware reply call with global_instructions injection (FB-04), oracle_init and drift_event logging to events.jsonl sidecar, and delta window contradiction detection using new_state.turn_index.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Add failing oracle loop integration tests | 8653935 | tests/phase3/test_oracle_loop_integration.py |
| GREEN (Task 1+2) | Wire OracleAgent into run_conversation (ORC-02, ORC-03, ORC-04, FB-04) | 3939bf4 | src/conversation_loop.py, src/agent_functions.py, tests/phase3/test_oracle_loop_integration.py |

## What Was Built

### `src/conversation_loop.py` modifications

**`_write_event(record, events_path)` helper:**
- Appends one JSON record to the events.jsonl sidecar file
- Creates parent directories if needed
- No try/except (fail loudly per CLAUDE.md)
- MUST NOT write to audit_log.jsonl (Pitfall 5 — load_audit_log() crashes on non-state records)

**`run_conversation()` signature extension:**
- Added `events_path: str | None = None` parameter
- If None, derived from `os.path.dirname(log_path) / "events.jsonl"`

**Three surgical insertions in the while loop body:**

1. `f_cognitive_load` import at function body level (before while loop) — ORC-03
2. `from src.oracle_agent import OracleAgent as _OracleAgent` at function body level — avoid circular
3. oracle_init event written at session start (before while loop) if oracle is OracleAgent — ORC-02
4. Modification A: compute `cognitive_load = f_cognitive_load(state, message)` before oracle.reply()
5. Modification A: pass `global_instructions=global_instructions` to OracleAgent.reply() — FB-04
6. Modification B: call `oracle.update_delta_window(deltas, new_state.turn_index)` after f_next_state, set `reply.contradiction_detected` and `reply.contradicted_turn`
7. Modification C: write drift_event to events.jsonl if `reply.contradiction_detected` is True — ORC-04

### `src/oracle_agent.py` (deviation fix)

Added `update_delta_window(deltas, turn_index)` method — plan 03 agent did not implement this method (it was specified in 03-03-PLAN.md but the agent worktree only had `_check_contradiction`). See Deviations section.

### `src/agent_functions.py` (bug fix)

Fixed `InstructionalFeedback` handling in `f_next_state()`:
- Was: `pass` (no-op stub with "structural storage in Phase 3" comment)
- Now: `global_instructions.append(delta.instruction_text)` — same pattern as GlobalFeedback
- This was necessary for `test_instructional_feedback_accumulates` to pass (FB-04)

## Verification

```
pytest tests/phase3/ -m "not llm"
21 passed, 1 warning

pytest tests/ -m "not llm"
3 failed (pre-existing hdbscan not installed), 144 passed, 1 skipped, 1 deselected
```

The 3 pre-existing failures are `ModuleNotFoundError: No module named 'hdbscan'` in phase1 tests — unrelated to this plan's changes.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] `update_delta_window` absent from OracleAgent**

- **Found during:** Pre-task analysis before writing tests
- **Issue:** Plan 03's agent (worktree `worktree-agent-acefc71069c7cdf11`) implemented `_contradicts()` and `_check_contradiction()` but did NOT implement `update_delta_window()`. This method is the public API that `run_conversation()` needs to call to check and update the delta deque.
- **Fix:** Added `update_delta_window(deltas, turn_index) -> tuple[bool, int | None]` to OracleAgent per the design in 03-03-PLAN.md §behavior. Checks contradictions BEFORE appending deltas (so same-turn deltas don't contradict each other).
- **Files modified:** `src/oracle_agent.py`
- **Commit:** a03e89c

**2. [Rule 1 - Bug] `InstructionalFeedback` accumulated as no-op in `f_next_state`**

- **Found during:** Task 2 test failure (test_instructional_feedback_accumulates)
- **Issue:** `f_next_state()` in `src/agent_functions.py` had `elif isinstance(delta, InstructionalFeedback): pass` with comment "structural storage in Phase 3". This meant `InstructionalFeedback.instruction_text` was never appended to `global_instructions`, making FB-04 a no-op.
- **Fix:** Changed to `global_instructions.append(delta.instruction_text)` — same pattern as `GlobalFeedback`.
- **Files modified:** `src/agent_functions.py`
- **Commit:** 3939bf4

**3. [Rule 3 - Blocking Issue] Wave 1+2 agent work needed merging**

- **Found during:** Execution start — tests/phase3/ directory and oracle_agent.py did not exist on this worktree
- **Fix:** Merged `worktree-agent-acefc71069c7cdf11` (plan 01: oracle_agent.py, oracle_protocol.py, tests/phase3/__init__.py, test_oracle_agent.py, conftest.py) and `worktree-agent-a17da87bd14921317` (plan 02: cognitive_load.py, test_cognitive_load.py). Resolved trivial docstring conflict in cognitive_load.py.
- **Commits:** fe0f841 (merge plan 01), bf2309e (merge plan 02)

## Known Stubs

None — all FB-04, ORC-02, ORC-03, ORC-04 wiring is fully implemented and tested.

## Threat Flags

None beyond those already in the plan's threat model (T-03-04-01 through T-03-04-04). The `import dataclasses` inside the while loop's socketio block (line 228) is a pre-existing Phase 2 pattern not introduced by this plan.

## Self-Check: PASSED

Files exist:
- src/conversation_loop.py: contains _write_event, events_path param, oracle_init block, drift_event block
- src/oracle_agent.py: contains update_delta_window
- src/agent_functions.py: InstructionalFeedback appends to global_instructions
- tests/phase3/test_oracle_loop_integration.py: 3 integration tests

Commits exist (git log confirms):
- 3939bf4: feat(03-04): wire OracleAgent into run_conversation
- 8653935: test(03-04): add failing tests for oracle loop integration
- a03e89c: fix(03-04): add update_delta_window to OracleAgent
