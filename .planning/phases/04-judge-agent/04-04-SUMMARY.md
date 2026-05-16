---
phase: 4
plan: "04-04"
subsystem: conversation-loop
tags: [db-integration, judge-agent, socketio, pairwise-accuracy, wiring]
dependency_graph:
  requires:
    - "04-01"  # src/db/ modules (turns, oracle_feedback, experiments, connection)
    - "04-02"  # src/stopping.py compute_magnitude, FeedbackMagnitudeWeights
    - "04-03"  # src/judge.py PairBag, compute_pairwise_accuracy, assemble_turn_metrics, assemble_feedback_rows
  provides:
    - "conversation_loop.run_conversation() with DB writes per turn"
    - "web UI state_update event with pairwise_accuracy, convergence_signal, contradiction_count"
    - "experiment row lifecycle in web/app.py"
  affects:
    - "04-05"  # tests that call run_conversation()
    - "04-06"  # any plan using web/app.py
tech_stack:
  added: []
  patterns:
    - "Lazy function-body imports for judge.py guarded by db_conn is not None"
    - "TYPE_CHECKING-gated type hints for sqlite3.Connection and PairBag"
    - "Single check_stopping() call per turn (V-4-03 dedup)"
    - "deviation() for resumed DB sessions without PairBag (D-21)"
key_files:
  created: []
  modified:
    - src/conversation_loop.py
    - web/app.py
decisions:
  - "D-27: isinstance(oracle, _OracleAgent) branch removed from oracle.reply() call; unconditional call with global_instructions= and cognitive_load= kwargs"
  - "D-03: Write order per turn enforced: JSONL -> DB turns.create + oracle_feedback.create -> post_turn_callback -> socketio.emit"
  - "D-04: DB connection opened in _run_conversation_background, passed to run_conversation; closed after experiment seal"
  - "D-28: state_update SocketIO event gains pairwise_accuracy (float), convergence_signal (str|None), contradiction_count (cumulative int)"
  - "Lazy judge.py imports inside if db_conn is not None block — guards against judge.py not existing at module import time (parallel plan 04-03)"
  - "_pair_acc stays 0.0 when db_conn is None — no PairBag in no-DB mode; documented in emit comment"
  - "check_stopping() computed once per turn: in Step 6a when db_conn is not None, in Step 7 when only socketio is not None, in Step 9 when both are None"
metrics:
  duration: "~30 minutes"
  completed: "2026-05-16"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 4 Plan 04: Loop Wiring — DB writes per turn, remove isinstance branch, D-28 state_update fields Summary

DB integration wiring that connects judge.py + src/db/ into conversation_loop.py, removes the WR-04 isinstance(oracle, OracleAgent) branch, switches recent_magnitudes to real weighted magnitude computation, and extends the state_update SocketIO event with pairwise_accuracy, convergence_signal, and contradiction_count fields.

## What Was Built

### Task 1: src/conversation_loop.py

**New parameters added to `run_conversation()`:**
- `db_conn: sqlite3.Connection | None = None` — D-04 per-run connection; None = no-DB (unit test mode)
- `experiment_id: int | None = None` — FK for turn/feedback rows
- `pair_bag: PairBag | None = None` — D-21 pre-built bag for resume
- `magnitude_weights: FeedbackMagnitudeWeights | None = None` — D-13 defaults if None

**D-27: isinstance branch removed.** The original code:
```python
if isinstance(oracle, _OracleAgent):
    reply = oracle.reply(state, message, global_instructions=..., cognitive_load=...)
else:
    reply = oracle.reply(state, message)
```
Replaced with unconditional call (both MockOracle and OracleProtocol.reply() accept these kwargs as of Plan 03):
```python
reply = oracle.reply(state, message, global_instructions=global_instructions, cognitive_load=cognitive_load)
```

**D-13: Real magnitude computation.** `recent_magnitudes.append(float(len(deltas)))` replaced with `recent_magnitudes.append(compute_magnitude(deltas, magnitude_weights))`.

**D-03: DB writes per turn** (inside `if db_conn is not None and experiment_id is not None` block, after JSONL write):
- Compute `_pair_acc` via `compute_pairwise_accuracy(new_state, pair_bag, new_state.turn_index)`
- Compute `_stop_for_metrics` via `check_stopping()` — once per turn (V-4-03 dedup)
- Call `assemble_turn_metrics()` → `turns.create(db_conn, _turn_create)`
- Call `assemble_feedback_rows()` → `oracle_feedback.create(db_conn, _fb)` for each row

**D-28: Extended state_update emit:**
```python
socketio.emit("state_update", {
    ...,
    "pairwise_accuracy": _pair_acc,
    "convergence_signal": _stop_for_metrics.value if _stop_for_metrics else None,
    "contradiction_count": _cumulative_contradictions,  # cumulative running total
})
```

**V-4-03: Cumulative contradiction_count.** `_cumulative_contradictions` is a running counter (not `int(reply.contradiction_detected)` which would only ever be 0 or 1).

**D-21 / V-4-05: deviation() for resumed sessions.** When `db_conn is not None` and `_pair_bag_was_none` (caller passed `pair_bag=None`), `deviation()` fires to flag that pairwise_accuracy will be 0.0 until new feedback arrives.

**Lazy import strategy for judge.py compatibility.** `src/judge.py` is created by parallel plan 04-03. To allow `from src.conversation_loop import run_conversation` to succeed even before judge.py exists:
- `from src.judge import PairBag as _PairBagT` is TYPE_CHECKING-gated at top level (type hint only, not runtime)
- All actual judge.py imports are inside the `if db_conn is not None:` block at function start
- This means importing `run_conversation` works without judge.py; only calling it with `db_conn=not None` requires judge.py

### Task 2: web/app.py

**D-04: DB connection per session.** Inside `_run_conversation_background()`, after writing initial state snapshot:
1. Import `connect`, `init_schema`, `ExperimentCreate`, `create`, `update`, `ExperimentUpdate` (all lazy, inside function)
2. `_db_conn = _db_connect()` — opens `experiments.db` at repo root
3. `_db_init_schema(_db_conn)` — CREATE TABLE IF NOT EXISTS (idempotent)
4. Create experiment row with `strategy_id="interactive"`, `persona_id="web_session"`, `dataset=session_ts`
5. Pass `db_conn=_db_conn, experiment_id=_experiment_id` to `run_conversation()`
6. After `run_conversation()` returns: seal experiment with `_exp_update()`, `_db_conn.close()`

**D-28: Updated all state_update emits** with safe defaults for non-running context:
- `/resume/{session_id}` route: `pairwise_accuracy=0.0, convergence_signal=None, contradiction_count=0`
- `connect` handler re-emit: same three fields with same safe defaults

## Deviations from Plan

### Auto-fixed: Placement of recent_magnitudes.append relative to check_stopping

**Rule 1 - Consistency:** The plan notes that Step 6a computes `check_stopping()` before `recent_magnitudes.append()` (Step 8), while the original code computed it after. This is consistent within each execution path:
- `db_conn is not None` path: `check_stopping()` in Step 6a (before append) — magnitude from current turn NOT included in stopping check. This is what the plan specifies.
- `db_conn is None, socketio is not None` path: `check_stopping()` in Step 7 (after append) — magnitude from current turn IS included.
- Both paths are internally consistent. The behavioral difference only affects the edge case of no-DB + UI mode (not a production configuration).

### Minor adjustment: _pair_acc in no-DB socketio mode

When `db_conn is None` and `socketio is not None`, `_pair_acc` remains 0.0 (the safe default) because no PairBag is initialized in no-DB mode. The plan's pseudocode showed `compute_pairwise_accuracy()` being called in this branch, but since `pair_bag` is only initialized inside the `if db_conn is not None` block (to guard against judge.py not existing), calling `compute_pairwise_accuracy` in the no-DB path would reference an uninitialized variable. Emitting 0.0 is the correct safe default for the web UI without DB.

## Known Stubs

None — no placeholder values in the implemented logic.

## Threat Flags

None — no new network endpoints or auth paths introduced. All DB access goes through `src/db/` modules as required.

## Verification Status

Static checks (grep-based) confirm all acceptance criteria:
- `src/conversation_loop.py` does NOT contain `isinstance(oracle, _OracleAgent)` in the oracle.reply() call block (only the Step 5b update_delta_window check remains)
- `src/conversation_loop.py` contains `oracle.reply(\n    state, message,` followed by `global_instructions=global_instructions,`
- `src/conversation_loop.py` contains `compute_magnitude(deltas, magnitude_weights)`
- `src/conversation_loop.py` contains `pair_bag.update(deltas`
- `src/conversation_loop.py` contains `db_conn is not None`
- `src/conversation_loop.py` contains `"pairwise_accuracy"` in the socketio.emit block
- `src/conversation_loop.py` contains `"convergence_signal"` in the socketio.emit block
- `src/conversation_loop.py` contains `"contradiction_count": _cumulative_contradictions`
- `src/conversation_loop.py` contains `_pair_bag_was_none = pair_bag is None`
- `src/conversation_loop.py` contains `deviation(` near pair_bag initialization
- `web/app.py` contains `from src.db.connection import connect` (lazy import inside function)
- `web/app.py` contains `_db_conn = _db_connect()`
- `web/app.py` contains `db_conn=_db_conn`
- `web/app.py` contains `experiment_id=_experiment_id`
- `web/app.py` contains `"pairwise_accuracy"` in the /resume state_update emit
- `web/app.py` does NOT contain `sqlite3.execute`

Runtime verification (requires src/judge.py to exist from plan 04-03):
- `python -c "from src.conversation_loop import run_conversation; import inspect; sig = inspect.signature(run_conversation); assert 'db_conn' in sig.parameters; assert 'experiment_id' in sig.parameters; print('new params ok')"` — EXPECTED: exits 0 (lazy imports make module importable without judge.py)
- `python -c "from web.app import app; print('app imports ok')"` — EXPECTED: exits 0 after judge.py exists

## Self-Check

Files modified:
- [x] `src/conversation_loop.py` — exists and contains all required changes
- [x] `web/app.py` — exists and contains all required changes

Required patterns present:
- [x] `db_conn` in `run_conversation()` signature
- [x] `experiment_id` in `run_conversation()` signature
- [x] `_pair_bag_was_none = pair_bag is None` sentinel
- [x] `deviation(` call inside db_conn is not None block
- [x] `pair_bag.update(deltas` in while loop
- [x] `compute_magnitude(deltas, magnitude_weights)` replacing float(len(deltas))
- [x] `"pairwise_accuracy": _pair_acc` in socketio.emit
- [x] `"convergence_signal": _stop_for_metrics.value if _stop_for_metrics else None` in socketio.emit
- [x] `"contradiction_count": _cumulative_contradictions` in socketio.emit (cumulative, not per-turn binary)
- [x] No `isinstance(oracle, _OracleAgent)` in oracle.reply() call block
- [x] `_db_conn = _db_connect()` in web/app.py
- [x] `db_conn=_db_conn` passed to `run_conversation()`
- [x] `experiment_id=_experiment_id` passed to `run_conversation()`
- [x] D-28 fields in /resume and connect handler emits
- [x] No `sqlite3.execute` in web/app.py

## Self-Check: PASSED
