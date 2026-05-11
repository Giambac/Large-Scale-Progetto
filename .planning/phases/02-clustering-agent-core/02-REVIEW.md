---
phase: 02-clustering-agent-core
reviewed: 2026-05-10T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - src/clustering.py
  - web/app.py
  - src/conversation_loop.py
  - web/templates/index.html
  - web/static/main.js
  - web/static/style.css
  - tests/phase2/test_clustering_backends.py
  - tests/phase2/test_umap_projection.py
  - tests/phase2/test_sessions.py
findings:
  critical: 5
  warning: 6
  info: 3
  total: 14
status: issues_found
---

# Phase 02 (Trio): Code Review Report

**Reviewed:** 2026-05-10T00:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Nine files were reviewed covering three Trio additions to Phase 2: `ClusteringBackend` Protocol + `KMeansBackend` (BACK-V2-01), UMAP projection panel (VIZ-V2-01), and session persistence (UI-V2-01).

The core clustering math, Protocol design, and soft-prob normalization are solid. The most serious issues are: a path-traversal vulnerability in the `/resume/<session_id>` route, `_parse_args()` executing at module import time which breaks test isolation, a soft-probs column-index mismatch bug when HDBSCAN returns non-contiguous cluster labels, a race condition where the background thread writes to `_session` without holding the lock, and blurry/misaligned UMAP rendering due to a canvas logical-vs-CSS size mismatch.

---

## Critical Issues

### CR-01: Path-traversal vulnerability in `/resume/<session_id>`

**File:** `web/app.py:259`
**Issue:** `session_id` comes directly from the URL and is joined to `SESSIONS_DIR` with `os.path.join` without any sanitisation. An attacker (or a buggy client) can send `../../etc/passwd` as `session_id`, causing the server to read arbitrary files. The `assert os.path.isdir(session_dir)` guard only checks existence — it does not prevent escape from `SESSIONS_DIR`.

**Fix:**
```python
import pathlib
sessions_root = pathlib.Path(SESSIONS_DIR).resolve()
session_dir = (sessions_root / session_id).resolve()
assert session_dir.parent == sessions_root, (
    f"Invalid session_id (path traversal attempt): {session_id!r}"
)
```

---

### CR-02: `_parse_args()` called at module import time — corrupts test isolation

**File:** `web/app.py:177`
**Issue:** `_args = _parse_args()` executes unconditionally when the module is imported. During pytest, `sys.argv` is full of pytest flags (`--tb=short`, `-x`, `-k filter`, etc.). Although `parse_known_args` silently ignores unknown args, the backend selection still reads from `sys.argv` non-deterministically. Any CI run with extra pytest flags produces a different `_backend_name` than a local run, making backend selection unreliable. Additionally, the module-level `_args = _parse_args()` triggers the `assert args.backend in (...)` assertion at import time even in tests that import `web.app` only to access `_compute_projection` or `_write_session_state`.

**Fix:** Defer argument parsing to `__main__` or use a factory function pattern:
```python
# At module level:
_backend_name: str = "hdbscan"  # default; overridden by main() if __name__ == "__main__"

def main() -> None:
    global _backend_name
    args = _parse_args()
    _backend_name = args.backend
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()
```

---

### CR-03: Soft-probs column-index mismatch when HDBSCAN returns non-contiguous cluster labels

**File:** `src/clustering.py:371, 398-401`
**Issue:** `build_initial_clustering_state` builds `assignments` by storing raw HDBSCAN integer labels directly:

```python
assignments: dict[int, int] = {i: int(labels[i]) for i in range(len(labels))}
```

HDBSCAN can return labels `[0, 1, 3]` (skipping 2 if cluster 2 was noise-only and collapsed). The `soft_probs_matrix` returned by `run_hdbscan` has columns indexed `0, 1, 2` — three consecutive positions, not `0, 1, 3`. The dict then stores:

```python
soft_probs: dict[int, list[float]] = {
    i: soft_probs_matrix[i].tolist()  # column 0,1,2 only
    for i in range(len(embeddings))
}
```

But `state.assignments[item_id]` may return `3`. Any downstream code that does `state.soft_probs[item_id][state.assignments[item_id]]` will either raise `IndexError` (list index out of range) or silently return the wrong probability column.

**Fix:** Re-map labels to contiguous 0-based indices before building the dicts:
```python
unique_ids = sorted(set(int(l) for l in labels))
remap = {old: new for new, old in enumerate(unique_ids)}
labels = np.array([remap[int(l)] for l in labels], dtype=np.intp)
assignments = {i: int(labels[i]) for i in range(len(labels))}
```

---

### CR-04: Race condition — background thread writes to `_session` without holding `_session_lock`

**File:** `web/app.py:363, 389, 449`
**Issue:** `_run_conversation_background` writes `_session["session_dir"]` (line 363), `_session["state"] = initial_state` (line 389), and `_session["state"] = final_state` (line 449) from the background thread. `_session_lock` is released at the end of the `with` block in `upload_dataset` (line 325) — immediately after `start_background_task` returns, before the background thread has done any work. A concurrent `/upload` request can then enter the `with` block, reset `_session["state"] = None` and `_session["session_dir"] = None`, while the first background thread is mid-flight writing to both. The result is state from session A being attributed to session B's directory, or `_write_session_state` writing to `None` (which raises `TypeError` rather than crashing cleanly).

**Fix:** Pass `session_dir` as an explicit argument to the background task so the thread does not need to write back to `_session`:
```python
# In upload_dataset (before start_background_task):
session_ts = _make_session_timestamp()
session_dir = os.path.join(SESSIONS_DIR, session_ts)
_session["task"] = socketio.start_background_task(
    _run_conversation_background,
    records,
    _session["log_path"],
    session_dir,   # pass explicitly — no shared write-back needed
)

# Remove _session["session_dir"] = session_dir from _run_conversation_background
```
Also protect the `/status` route's `_session["state"]` read with the lock, or switch to a thread-safe queue.

---

### CR-05: Canvas logical size vs. CSS size mismatch — UMAP dots drawn off-canvas or bunched

**File:** `web/templates/index.html:20` and `web/static/style.css:122-124`
**Issue:** The `<canvas>` element has hardcoded HTML attributes `width="1200" height="300"`. These are the bitmap dimensions used for all drawing math in `drawProjection()`. CSS sets `width: 100%` on the canvas, which scales the bitmap to fill the container — but the `W = canvas.width` in JavaScript still reads `1200`, not the rendered pixel width. On a viewport narrower than 1200px the computed `toCanvas()` coordinates map off the right edge of the visible canvas. On wider viewports dots are compressed into the left portion. On HiDPI displays the bitmap is also blurry because the devicePixelRatio is not applied.

**Fix:**
```javascript
// At the start of drawProjection(), before computing bounding box:
canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
canvas.height = canvas.clientHeight * (window.devicePixelRatio || 1);
const ctx = canvas.getContext('2d');
ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
const W = canvas.clientWidth;   // logical width
const H = canvas.clientHeight;  // logical height
// Remove ctx.clearRect(0, 0, W, H) from its current position and let canvas.width= handle clear
```
Remove the hardcoded `width="1200"` HTML attribute.

---

## Warnings

### WR-01: `list_sessions` opens files without a `with` statement — file handle leak

**File:** `web/app.py:236`
**Issue:** `open(state_path, encoding="utf-8").read().strip()` opens a file without a context manager. The handle is never explicitly closed. CPython's reference counting will typically close it promptly, but this is not guaranteed and masks resource-leak bugs at scale (many sessions = many leaked handles before GC).

**Fix:**
```python
with open(state_path, encoding="utf-8") as fh:
    line = fh.read().strip()
```

---

### WR-02: `resume_session` opens files without a `with` statement — file handle leak

**File:** `web/app.py:268`
**Issue:** Same pattern as WR-01. `open(state_path, encoding="utf-8").read().strip()` leaks the file handle.

**Fix:**
```python
with open(state_path, encoding="utf-8") as fh:
    line = fh.read().strip()
```

---

### WR-03: `_select_k_via_bic` upper bound produces K=2-only selection for N < 9

**File:** `src/clustering.py:171`
**Issue:** `k_max = max(2, int(math.sqrt(N)))` evaluates to `2` for any `N <= 8` (since `int(sqrt(8)) == 2`). The BIC loop `for k in range(2, 3)` then runs one iteration, unconditionally returning `best_k = 2` with no comparison between candidates. Unit tests with small synthetic datasets (N=10 yields `k_max=3` — barely one extra candidate) will silently select K=2 regardless of true cluster structure. There is no assertion or documentation warning callers of this degenerate behavior.

**Fix:** Add an assertion to fail loudly on small inputs, or document the minimum required N:
```python
assert N >= 9, (
    f"KMeansBackend._select_k_via_bic: N={N} is too small for BIC selection "
    f"(need N >= 9 to explore K=2 vs K=3). Override backend._k directly for small datasets."
)
```

---

### WR-04: `_session["state"]` read in `/status` without the lock — TOCTOU hazard

**File:** `web/app.py:209-215`
**Issue:** The `/status` route reads `_session["state"]` twice without holding `_session_lock`:
```python
if _session["state"] is None:       # read 1
    return jsonify({"status": "idle", ...})
state = _session["state"]           # read 2 — may differ from read 1
```
The background thread can write `_session["state"]` between these two reads, causing `state` on line 211 to be non-None even though the `None` check on line 210 passed. In CPython this is unlikely to cause a crash (dict reads are individually GIL-protected), but it is a logical TOCTOU: the response attributes could reflect a partially-initialized state object.

**Fix:** Take a local snapshot under the lock:
```python
with _session_lock:
    state = _session["state"]
if state is None:
    return jsonify({"status": "idle", "turn_index": None})
return jsonify({...})
```

---

### WR-05: `umap` imported lazily inside `_compute_projection` — `ImportError` surfaces only at runtime

**File:** `web/app.py:64`
**Issue:** `import umap as umap_lib` is inside `_compute_projection`. If `umap-learn` is not installed, the `ImportError` is raised only when the first projection is computed — potentially long after the server started and a dataset was uploaded. Per the fail-loudly philosophy, missing dependencies should crash immediately at startup.

**Fix:** Move to module-level:
```python
import umap as umap_lib  # top of web/app.py — fails fast if umap-learn not installed
```
Then remove the lazy import inside `_compute_projection`.

---

### WR-06: `resumeSession` status banner uses unescaped `sessionId` in `textContent`

**File:** `web/static/main.js:43`
**Issue:** `document.getElementById('status-banner').textContent = 'Status: Resuming session ' + sessionId + '...'` uses `textContent`, which is safe against HTML injection. However, `sessionId` also appears in `li.title` (line 33) inside `renderSessionsList`:
```javascript
li.title = 'Click to resume session ' + s.session_id;
```
`title` is an HTML attribute set via the DOM property, which also does not interpret HTML. Both usages are currently safe.

The actual risk is that no server-side validation constrains what characters a session ID may contain (the ID comes from a directory name). If the session directory creation is ever attackable (e.g., via the path-traversal bug in CR-01 allowing attacker-created directories), the session ID rendered in the UI could be an adversarially crafted string. Currently not exploitable but structurally fragile.

**Fix:** Apply `escapeHtml()` to `s.session_id` in `li.title` for defensive depth:
```javascript
li.title = 'Click to resume session ' + escapeHtml(s.session_id);
```
And add server-side validation in `list_sessions` rejecting directory names that contain characters outside `[A-Za-z0-9\-T]`.

---

## Info

### IN-01: `console.error` in `loadSessionsList` swallows fetch failure silently

**File:** `web/static/main.js:17`
**Issue:** A failed `/sessions` fetch is logged to the browser console but shows no user-visible feedback. The sessions list stays showing "No sessions yet." which is indistinguishable from an empty sessions directory. This contradicts fail-loudly intent at the UI layer.

**Fix:**
```javascript
.catch(function (err) {
    const list = document.getElementById('sessions-list');
    if (list) list.innerHTML =
        '<li class="placeholder">Failed to load sessions: ' + escapeHtml(String(err)) + '</li>';
});
```

---

### IN-02: Magic number `30` for `MockOracle` script length must stay in sync with `turn_budget`

**File:** `web/app.py:423-425`
**Issue:** `oracle = MockOracle(script=[neutral_reply] * 30)` and `criteria = StoppingCriteria(turn_budget=30)` both hardcode `30`. If either is changed independently the oracle will exhaust its script before the turn budget fires (or the budget fires before the oracle runs out), causing an `IndexError` from `MockOracle` or a silent under-run.

**Fix:**
```python
TURN_BUDGET = 30
criteria = StoppingCriteria(turn_budget=TURN_BUDGET)
oracle = MockOracle(script=[neutral_reply] * TURN_BUDGET)
```

---

### IN-03: Test `test_per_turn_write_via_callback` has backwards `turn_index` ordering

**File:** `tests/phase2/test_sessions.py:149-172`
**Issue:** `state_turn1 = simple_state` (which has `turn_index=3`) and `state_turn2` has `turn_index=2`. The "second" write has a lower `turn_index` than the "first". The test verifies that `loaded.turn_index == 2` (last write wins), which is correct for the overwrite behavior being tested, but the variable names and comment ("turn-1 baseline", "turn-2") are backwards relative to the `turn_index` values. A future maintainer reading this test will be confused about which state is newer.

**Fix:** Rename variables to `state_first_write` / `state_second_write`, or set `turn_index` values in ascending order (`0` and `1`) so they match the implied ordering.

---

_Reviewed: 2026-05-10T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
