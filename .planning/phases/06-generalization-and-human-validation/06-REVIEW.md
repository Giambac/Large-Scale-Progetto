---
phase: 06-generalization-and-human-validation
reviewed: 2026-05-17T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/mapping.py
  - examples/evaluate_mapping.py
  - examples/compare_oracle_types.py
  - web/app.py
  - web/templates/study.html
  - web/static/study.js
  - tests/test_mapping.py
  - tests/test_study_ui.py
findings:
  critical: 4
  warning: 5
  info: 3
  total: 12
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-05-17T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed all eight files introduced in Phase 06 (mapping layer, evaluation scripts,
compare oracle types CLI, study web routes, study HTML/JS, and their test suites).

The mapping layer itself (`src/mapping.py`) is largely well-structured and follows
project conventions. The critical issues are concentrated in two areas: (1) the study
session background worker in `web/app.py` has a race condition and a turn-budget bypass
bug; (2) `_detect_satisfaction` silently swallows an `APIError` via `deviation()`,
violating the "fail loudly" rule; (3) the `_make_session_timestamp()` helper uses a
naive (non-UTC) `datetime.now()`, producing non-deterministic, timezone-skewed names on
any machine not running in UTC. Smaller but still blocking issues exist in
`evaluate_mapping.py` (type mismatch in accuracy comparison) and the test suite (one test
tests its own re-implementation rather than the module constant).

---

## Critical Issues

### CR-01: `feedback_queue.pop(0)` called without checking queue length — crash if event fires before message is appended

**File:** `web/app.py:934` and `web/app.py:952`

**Issue:** `sess["feedback_event"].wait()` is signalled from two places: `study_feedback` SocketIO handler (line 1073) sets the event *after* appending to the queue, and `_end_study_session` (line 1053) sets the event without appending anything. The background worker always does `sess["feedback_queue"].pop(0)` immediately after `.wait()` + `.clear()`. If `_end_study_session` fires concurrently (e.g. called from a separate HTTP request while the worker is in the confirmation sub-loop at line 946–959), the queue may be empty when `pop(0)` executes, producing an unhandled `IndexError` that kills the daemon thread silently. This is a real race, not a theoretical one: the `/study/sessions` endpoint, HTTP disconnects, or any future admin endpoint could call `_end_study_session` at any time.

**Fix:**
```python
# After sess["feedback_event"].wait() / .clear() always guard pop:
if sess["ended"]:
    return
if not sess["feedback_queue"]:
    # Spurious wake (ended flag raced) — treat as session end
    return
human_text = sess["feedback_queue"].pop(0)
```
The same guard must be applied at line 952 (confirmation pop).

---

### CR-02: `_detect_satisfaction` swallows `anthropic.APIError` — silent wrong answer, violates "fail loudly"

**File:** `web/app.py:1024-1027`

**Issue:** The docstring says "Only catches anthropic.APIError — all other exceptions propagate (fail loudly)." But catching `APIError` and calling `deviation()` (which only logs a warning unless `STRICT_MODE=1`) means a transient or permanent API failure causes the function to silently return `False`. In the study loop this means the participant's satisfaction message is treated as ordinary feedback, the session never ends via the satisfaction path, and the participant is stuck until the turn budget is exhausted. This is a silent wrong answer — exactly what CLAUDE.md says is "the actual failure mode to avoid." Per project philosophy, API call errors are the *one permitted* place to handle errors at call boundaries — but here the correct handling is to re-raise or to propagate the failure upward so the study session worker can emit an error event and terminate cleanly.

**Fix:**
```python
except _anthropic.APIError:
    raise   # re-raise per CLAUDE.md — API errors at the call boundary propagate loudly
```
If graceful degradation is truly required for the study UX, it must be handled by the *caller* (`_run_study_background`), not swallowed inside the helper.

---

### CR-03: `evaluate_mapping.py` line 178 — `int(prediction)` raises `ValueError` on valid non-integer cluster IDs

**File:** `examples/evaluate_mapping.py:178`

**Issue:** `strategy.assign()` returns a cluster ID as a string (e.g. `"0"`, `"1"`) per `MappingProtocol`. The ground-truth `oracle_label` is stored as `int` (line 161: `label = int(raw)`). The agreement check is:
```python
agreement = 1.0 if int(prediction) == oracle_label else 0.0
```
`int(prediction)` will raise `ValueError` if the LLM strategy ever returns a string that cannot be parsed as an integer (e.g. `"cluster_0"`, `"A"`, `"None"`). While `LLMMappingStrategy` validates against `valid_cluster_ids` (which are stringified integers for typical datasets), the comparison is fragile and will break entirely the moment cluster IDs are non-integer strings. Worse, the guard `if oracle_label is None: continue` is checked before calling `strategy.assign()` but there is no corresponding guard for a bad `prediction`. An unhandled `ValueError` here would crash the entire evaluation loop after spending API budget on all prior items.

Also, `valid_cluster_ids` on line 121 is built as `{c.id for c in state.clusters}` — a set of `int` — while `LLMMappingStrategy` builds its own `valid_cluster_ids` as `{str(c.id) for c in state.clusters}`. If `LLMMappingStrategy` validates against string IDs and returns a string, `int(prediction)` converts back, so it works for integer cluster IDs. But the conversion should be explicit and protected:

**Fix:**
```python
try:
    pred_int = int(prediction)
except ValueError:
    deviation(
        "mapping_strategy_invalid_prediction",
        prediction=prediction,
        strategy=name,
    )
    continue  # skip item rather than crash
agreement = 1.0 if pred_int == oracle_label else 0.0
```

---

### CR-04: `_make_session_timestamp()` uses naive local time — sessions get wrong timestamps on non-UTC machines

**File:** `web/app.py:41`

**Issue:** `datetime.datetime.now()` returns local time without timezone info. CLAUDE.md explicitly forbids `datetime.utcnow()` and mandates `datetime.now(timezone.utc)`. While `_make_session_timestamp` is used for filesystem names (not DB timestamps), using local time means:
1. Session directory names embed local wall-clock time, making sessions from different timezones non-comparable.
2. Two concurrent sessions started within the same second on any machine will produce identical `session_id` values, causing `_study_sessions[session_id]` to overwrite the first session's state dict — a silent data loss bug. (The same-second collision risk applies to both study and regular sessions.)

The timestamp format has no sub-second precision, making collisions realistic under any load.

**Fix:**
```python
def _make_session_timestamp() -> str:
    import secrets
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    # Append random suffix to prevent same-second collisions
    return f"{ts}-{secrets.token_hex(4)}"
```

---

## Warnings

### WR-01: `_generate_session_name` swallows all exceptions — violates "fail loudly"

**File:** `web/app.py:437-439`

**Issue:** `_generate_session_name` contains a bare `except Exception as e: print(...)` and returns `""`. This is a pattern explicitly listed as forbidden in CLAUDE.md: "No swallowed exceptions (`except: pass`, `except Exception as e: log and continue`)." This particular helper is non-critical (session naming), but the pattern normalises the anti-pattern and hides real failures (e.g. wrong namer object type). If naming fails, the correct behaviour per project philosophy is to let it crash.

**Fix:** Remove the try/except entirely, or at minimum only catch the specific `anthropic.APIError` for the Anthropic path (the one external call).

---

### WR-02: `CentroidMappingStrategy` recomputes `norm_item` inside the centroid loop — wasteful and misleading

**File:** `src/mapping.py:364`

**Issue:** `norm_item = np.linalg.norm(item_vec)` is computed once per loop iteration (once per cluster), but `item_vec` never changes inside the loop. This is not a correctness bug but it is a logic error in code structure: a reader must carefully check that `item_vec` is not mutated to conclude the result is correct. The sentinel `best_sim: float = -2.0` combined with computing norms inside the loop suggests the loop body was partially copy-pasted from another context. The norm should be computed once before the loop.

**Fix:**
```python
norm_item = np.linalg.norm(item_vec)  # computed ONCE, before the loop
for cluster_id_str, centroid in centroids:
    norm_centroid = np.linalg.norm(centroid)
    cosine_sim = float(np.dot(item_vec, centroid) / (norm_item * norm_centroid + 1e-10))
    ...
```

---

### WR-03: `test_study_max_turns_env_override` does not test the module constant — tests its own inline expression

**File:** `tests/test_study_ui.py:136-143`

**Issue:** The test is supposed to verify that `STUDY_MAX_TURNS` (the module-level constant in `web/app.py`, set at import time) respects the environment variable. Instead it evaluates a standalone expression `int(os.environ.get("STUDY_MAX_TURNS", "30"))` inline and asserts on that. Because `web.app` is already imported (by the `study_client` fixture and module-level imports), the `STUDY_MAX_TURNS` constant was frozen at import time, before this test set the env var. The test always passes but never proves the module constant is overrideable — it tests the Python standard library.

**Fix:**
```python
def test_study_max_turns_env_override(monkeypatch):
    """STUDY_MAX_TURNS reflects the env var when web.app is re-imported."""
    monkeypatch.setenv("STUDY_MAX_TURNS", "5")
    import importlib
    import web.app as app_module
    importlib.reload(app_module)
    assert app_module.STUDY_MAX_TURNS == 5
```
Note: `reload` has ordering caveats with pytest fixtures, but it is the only way to test a module-level constant that is set at import time.

---

### WR-04: Study session worker loop does not increment `turn_index` when satisfaction is detected and rejected — turn budget erodes invisibly

**File:** `web/app.py:923-959`

**Issue:** When the satisfaction detection branch fires (line 939) and the participant says "no, continue" (line 956-959), the `while` loop `continue`s without incrementing `turn_index`. This means one full oracle LLM call (satisfaction detection) plus one human round-trip was consumed without advancing the turn counter. Over many false positive detections the budget silently under-counts turns consumed, potentially running many more actual turns than `STUDY_MAX_TURNS`. This is not a correctness crash but it is incorrect state: `sess["turn_index"]` diverges from the real number of oracle interactions.

**Fix:** Increment `turn_index` after the satisfaction-rejected branch before `continue`, or count all oracle round-trips (including satisfaction checks) uniformly.

---

### WR-05: `compare_oracle_types.py` closes the DB before querying — query result used after connection closed

**File:** `examples/compare_oracle_types.py:60-65`

**Issue:**
```python
db = connect()
init_schema(db)
try:
    rows = exp_db.query(db, dataset=args.dataset)
finally:
    db.close()
```
`rows` is a list of `ExperimentRead` Pydantic objects, so it is materialised in memory before `db.close()`. This is safe as written. However, if `exp_db.query` were to return a lazy cursor or generator in a future refactor, all subsequent iteration would fail silently. More concretely, `init_schema` is called on every CLI invocation of a read-only analysis tool, which runs DDL (`CREATE TABLE IF NOT EXISTS`) against the DB unnecessarily. This is a correctness-adjacent quality issue.

**Fix:** For a read-only CLI, omit `init_schema` (the schema should already exist). If schema initialisation must run, document that this CLI has a write side effect.

---

## Info

### IN-01: `web/app.py` uses `print()` for timing and diagnostic output rather than `logging`

**File:** `web/app.py:80, 85, 96, 438, 494, 522, 533, 546, 556, 670`

**Issue:** Ten `print()` statements are used for diagnostic output. The project uses Python's `logging` module everywhere else (via `src/logging_setup.py`). `print()` bypasses log level filtering, cannot be redirected, and is not structured. This is consistent with the "debug UI" nature of the server, but several of these are in code paths (`_compute_projection`, `_run_conversation_background`) that will also run during study sessions with human participants.

**Fix:** Replace with `log = logging.getLogger(__name__)` and `log.debug(...)` / `log.info(...)` calls.

---

### IN-02: `test_mapping.py` — `test_mapping_protocol_isinstance` instantiates `CentroidMappingStrategy()` twice (once for isinstance, unnecessarily)

**File:** `tests/test_mapping.py:107-109`

**Issue:**
```python
rule_set = _make_rule_set()   # created but never used
assert isinstance(LLMMappingStrategy(), MappingProtocol)
assert isinstance(CentroidMappingStrategy(), MappingProtocol)
```
`rule_set` is constructed but never referenced in the test body. `CentroidMappingStrategy.__init__` loads a `SentenceTransformer` model, making this test slow. The unused `rule_set` suggests a copy-paste from another test.

**Fix:** Remove the `rule_set = _make_rule_set()` line. If `CentroidMappingStrategy` model loading is too slow for the isinstance check, mock the `SentenceTransformer` import.

---

### IN-03: `study.html` loads socket.io from a CDN without Subresource Integrity (SRI)

**File:** `web/templates/study.html:192`

**Issue:**
```html
<script src="https://cdn.socket.io/4.6.2/socket.io.min.js"></script>
```
There is no `integrity="sha384-..."` attribute. For a research tool used in human subject studies, CDN script substitution (or CDN unavailability) would break all sessions silently. This is low severity in a research context but worth noting.

**Fix:** Add an integrity hash:
```html
<script
  src="https://cdn.socket.io/4.6.2/socket.io.min.js"
  integrity="sha384-mZLF4UVrpi/QTWPA7BjNPEnkIfRFn4ZEO3Qt/HFklTJBj/gBOV8G3HcKn4NfQbl"
  crossorigin="anonymous"
></script>
```
Or serve socket.io locally from `/static/`.

---

_Reviewed: 2026-05-17T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
