---
phase: 06-generalization-and-human-validation
fixed_at: 2026-05-17T00:00:00Z
review_path: .planning/phases/06-generalization-and-human-validation/06-REVIEW.md
iteration: 1
findings_in_scope: 12
fixed: 12
skipped: 0
status: all_fixed
---

# Phase 06: Code Review Fix Report

**Fixed at:** 2026-05-17T00:00:00Z
**Source review:** .planning/phases/06-generalization-and-human-validation/06-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 12
- Fixed: 12
- Skipped: 0

## Fixed Issues

### CR-01: Guard `feedback_queue.pop(0)` at both pop sites

**Files modified:** `web/app.py`
**Commit:** 05075a6
**Applied fix:** Added `if not sess["feedback_queue"]: return` guard immediately after each `feedback_event.wait()` + `clear()` block — both the primary pop (human_text) and the confirmation pop (confirm_text). This prevents `IndexError` when `_end_study_session` signals the event without appending to the queue.

---

### CR-02: Re-raise `APIError` in `_detect_satisfaction`

**Files modified:** `web/app.py`
**Commit:** cadcab2
**Applied fix:** Replaced the `deviation()` call + `return False` in the `except _anthropic.APIError` block with `raise`. Updated the docstring accordingly. API errors now propagate loudly per CLAUDE.md, so callers can handle study session failures cleanly.

---

### CR-03: Guard `int(prediction)` with try/except `ValueError` in `evaluate_mapping.py`

**Files modified:** `examples/evaluate_mapping.py`
**Commit:** 406d572
**Applied fix:** Wrapped `int(prediction)` in a `try/except ValueError` block that calls `deviation("mapping_strategy_invalid_prediction", ...)` and `continue`s rather than crashing the entire evaluation loop. The comparison now uses `pred_int` (the safe integer result).

---

### CR-04: Fix `_make_session_timestamp()` to use UTC and random suffix

**Files modified:** `web/app.py`
**Commit:** 3d8b8c5
**Applied fix:** Changed `datetime.datetime.now()` to `datetime.datetime.now(datetime.timezone.utc)` per CLAUDE.md mandate. Added `secrets.token_hex(4)` suffix to prevent same-second session ID collisions that would silently overwrite session state.

---

### WR-01: Remove bare `except Exception` from `_generate_session_name`

**Files modified:** `web/app.py`
**Commit:** fecf034
**Applied fix:** Removed the entire `try/except Exception as e: print(...)` wrapping the function body. The function now lets all exceptions propagate as required by CLAUDE.md "fail loudly" philosophy. The unused `import re, time` was also removed as it was only present in the old try block.

---

### WR-02: Move `norm_item` computation outside the centroid loop

**Files modified:** `src/mapping.py`
**Commit:** 71fdc01
**Applied fix:** Moved `norm_item = np.linalg.norm(item_vec)` to before the `for cluster_id_str, centroid in centroids:` loop with a comment `# computed once before the loop`. This eliminates redundant recomputation on every loop iteration since `item_vec` is immutable within the loop.

---

### WR-03: Fix `test_study_max_turns_env_override` to use `importlib.reload`

**Files modified:** `tests/test_study_ui.py`
**Commit:** 69cd961
**Applied fix:** Replaced the test body (which was testing the Python standard library inline expression rather than the module constant) with `importlib.reload(app_module)` after setting the env var, then asserting `app_module.STUDY_MAX_TURNS == 5`. The test now actually verifies the module constant reflects the environment variable.

---

### WR-04: Increment `turn_index` in satisfaction-rejected branch

**Files modified:** `web/app.py`
**Commit:** d0c4433
**Applied fix:** Added `turn_index += 1` and `sess["turn_index"] = turn_index` in the `else` branch (participant said "no, continue" after satisfaction detection) before the `continue` statement. This ensures oracle round-trips (including satisfaction check interactions) are counted against the turn budget, preventing `turn_index` from diverging from the actual number of interactions consumed.

---

### WR-05: Remove `init_schema` call from read-only `compare_oracle_types` CLI

**Files modified:** `examples/compare_oracle_types.py`
**Commit:** 6409fd8
**Applied fix:** Removed `init_schema(db)` call and its import from `src.db.connection`. This is a read-only analysis CLI — running DDL on every invocation was an unnecessary write side-effect. The schema is expected to already exist when analysis runs.

---

### IN-01: Replace `print()` statements with `log.debug()` / `log.info()`

**Files modified:** `web/app.py`
**Commit:** 80cee57
**Applied fix:** Added `import logging` and `log = logging.getLogger(__name__)` at module level. Replaced all 9 `print()` calls with appropriate `log.debug()` (timing/diagnostic) or `log.info()` (startup/parse information) calls using %-style formatting. This enables proper log level filtering and structured output consistent with the rest of the project.

---

### IN-02: Remove unused `rule_set = _make_rule_set()` in `test_mapping_protocol_isinstance`

**Files modified:** `tests/test_mapping.py`
**Commit:** 5f9103c
**Applied fix:** Removed the `rule_set = _make_rule_set()` line from `test_mapping_protocol_isinstance`. The variable was constructed but never used in the test body — it was a copy-paste artifact.

---

### IN-03: Add SRI integrity hash to socket.io CDN script in `study.html`

**Files modified:** `web/templates/study.html`
**Commit:** 7b153a2
**Applied fix:** Added `integrity="sha384-mZLF4UVrpi/QTWPA7BjNPEnkIfRFn4ZEO3Qt/HFklTJBj/gBOV8G3HcKn4NfQbl"` and `crossorigin="anonymous"` attributes to the socket.io CDN `<script>` tag, as specified in the review guidance.

---

**Test results after all fixes:**
```
14 passed in 20.14s (tests/test_mapping.py + tests/test_study_ui.py)
```

---

_Fixed: 2026-05-17T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
