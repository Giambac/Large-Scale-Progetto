# FastAPI Migration Plan

**Status:** Ready to execute
**Author:** ale-bena (planned with Opus 4.7)
**Executor:** Sonnet
**Branch:** `FastAPI` (already checked out)
**Estimated commits:** 7

---

## 0. Why this exists

The current debug UI runs on **Flask + Flask-SocketIO** with `async_mode='threading'`. We are migrating it to **FastAPI + python-socketio (ASGI mode)** in preparation for future async-first features (LLM streaming, multi-client sessions, parallel ablation runs). The migration is **transport-only** — no clustering / oracle / feedback logic changes.

---

## 1. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **WebSocket transport: python-socketio (ASGI)** | Keeps the existing Socket.IO v4 client (`web/static/main.js`, `web/templates/index.html`) and event-name protocol untouched. Smallest diff, identical protocol. |
| 2 | **Emit boundary: thin `Emitter` wrapper** | Preserves `src/conversation_loop.py` exactly as-is. The loop stays framework-agnostic and synchronous (correct for CPU-bound numpy/sklearn work). |
| 3 | **Server: `uvicorn` only — never eventlet/gevent** | uvicorn is native asyncio with zero monkey-patching. eventlet/gevent globally replace stdlib primitives and break numpy/sklearn/HDBSCAN/UMAP at runtime. See section 2. |
| 4 | **Plan shape: one atomic plan** | Tightly coupled — server, tests, and docs can't merge independently because tests import from the rewritten `web/app.py`. |

---

## 2. Monkey-patching guardrail — READ FIRST

**The threat:** Some sync→async adapters (eventlet, gevent) work by *monkey-patching* the stdlib at import time — replacing `threading`, `socket`, `time.sleep`, `os` with green-thread versions. This happens **process-globally**. C-extension libraries (numpy, sklearn, HDBSCAN, UMAP) use Python's threading primitives internally; monkey-patched primitives produce silent deadlocks, corrupted intermediate state, or hangs that look like "the algorithm is slow."

**Why FastAPI *removes* this risk:** FastAPI is ASGI-native. uvicorn runs a real asyncio event loop. No monkey-patching is ever required or recommended. python-socketio's `AsyncServer(async_mode='asgi')` is also fully ASGI-native.

**Rules (enforce forever):**

1. **Never** add `eventlet` or `gevent` to dependencies, even transitively. Run `pip show eventlet gevent` after install — both must error with "Package not found".
2. **CPU-bound work** (HDBSCAN, UMAP, sklearn, embedding compute) runs in a `threading.Thread` — *not* on the asyncio event loop. The conversation loop already runs in a background thread; we preserve that exact pattern.
3. **Cross-thread emit** uses `asyncio.run_coroutine_threadsafe(sio.emit(...), loop)` — the standard, supported asyncio pattern. No patching involved.
4. **Run command must be `uvicorn`** — not gunicorn with gevent workers, not anything else. Documented in `CLAUDE.md` (see Task 7).

---

## 3. Files touched

### Source code (modified)

| File | Change |
|---|---|
| `web/app.py` | Full rewrite: Flask → FastAPI; Flask-SocketIO → python-socketio AsyncServer + ASGIApp wrap. All route bodies preserved; only decorators / signatures / response helpers change. |
| `tests/phase2/test_app.py` | Replace `flask_client` fixture (`app.test_client()`) with `client = TestClient(app)` from `fastapi.testclient`. Drop `app.config["TESTING"]`. |
| `tests/phase2/test_sessions.py` | Same fixture migration. Drop `PROPAGATE_EXCEPTIONS=False` (Flask-only). FastAPI re-raises by default; use a context manager pattern instead. |

### Source code (UNCHANGED — this is the win)

- `src/conversation_loop.py` — the `socketio` parameter is duck-typed (`object | None`); the `Emitter` wrapper satisfies its `.emit(event, payload)` contract identically.
- `src/clustering.py`, `src/oracle_agent.py`, `src/feedback.py`, `src/cognitive_load.py`, `src/cluster_naming.py`, `src/embedding_store.py`, etc. — pure logic, no framework coupling.
- `tests/phase2/test_conversation_loop.py`, `tests/phase2/test_umap_projection.py`, `tests/phase3/test_oracle_loop_integration.py` — all pass `socketio=None`, no changes needed.

### Frontend (UNCHANGED)

- `web/templates/index.html` — Socket.IO v4 CDN client stays.
- `web/static/main.js` — `io()` and `socket.on(...)` calls unchanged.
- `web/static/style.css` — unchanged.

### Docs / planning (modified)

| File | Change |
|---|---|
| `CLAUDE.md` | Add a "Web stack" section with deps, run command, monkey-patching warning. |
| `.planning/STATE.md` | Add a note under "Accumulated Context" → "Architecture Constraints"; bump `last_updated`. |
| `.planning/ROADMAP.md` | Add footnote on Phase 2: "Web layer migrated Flask → FastAPI on YYYY-MM-DD; UI-01/UI-02 reqs unchanged." Update header timestamp. |
| `.planning/REQUIREMENTS.md` | Update the "Last updated" trailer only. Requirement wording stays ("web-based debug interface" is framework-agnostic). |

### New files

| File | Purpose |
|---|---|
| `.planning/fastAPI_plan.md` | This file. |
| `requirements.txt` *(if missing)* or `pyproject.toml` deps section | Add `fastapi`, `uvicorn[standard]`, `python-socketio`. Remove `flask`, `flask-socketio`. |

---

## 4. Execution tasks (in order — each = one commit)

### Task 1 — Dependencies

**Goal:** install FastAPI stack; remove Flask stack; verify no eventlet/gevent.

**Actions:**
1. Check whether the project uses `requirements.txt`, `pyproject.toml` `[project.dependencies]`, or just ad-hoc installs (currently `pyproject.toml` has only `[tool.pytest.ini_options]` — no deps section).
2. If no dependency manifest exists, **create `requirements.txt`** at repo root with at minimum:
   ```
   fastapi>=0.110
   uvicorn[standard]>=0.27
   python-socketio>=5.11
   ```
3. Run `pip install fastapi "uvicorn[standard]" python-socketio` in the active venv.
4. Run `pip uninstall -y flask flask-socketio` (ignore "not installed" warnings).
5. **Verify guardrail:** `pip show eventlet gevent` must report both packages missing. If either is installed, run `pip uninstall -y eventlet gevent` and document why it was present.

**Commit:** `chore(deps): switch web stack from Flask to FastAPI + uvicorn`

---

### Task 2 — Rewrite `web/app.py`

**Goal:** Replace Flask + Flask-SocketIO with FastAPI + python-socketio AsyncServer. Preserve all routes, event names, payload shapes, and threading semantics.

**Key structural changes (concrete patterns to follow):**

```python
# Imports
import asyncio
import threading
import socketio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Capture the running event loop at startup so worker threads can schedule emits onto it.
_loop: asyncio.AbstractEventLoop | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Wrap FastAPI with socketio; exported as the ASGI entrypoint.
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

# The Emitter shim — see Task 3 for the class body; instantiate it here:
emitter = SocketIOEmitter(sio, lambda: _loop)
```

**Route translation rules:**

| Flask pattern | FastAPI replacement |
|---|---|
| `@app.route("/")` | `@app.get("/")` returning `templates.TemplateResponse("index.html", {"request": request})` |
| `@app.route("/status")` | `@app.get("/status")` returning a plain dict (FastAPI auto-JSONifies) |
| `@app.route("/sessions")` | `@app.get("/sessions")` returning a list |
| `@app.route("/resume/<session_id>", methods=["POST"])` | `@app.post("/resume/{session_id}")` |
| `@app.route("/upload", methods=["POST"])` | `@app.post("/upload")` with `file: UploadFile = File(...)` parameter |
| `request.files["file"]` | the `UploadFile` parameter directly (`await file.read()`) |
| `jsonify({...}), 400` | `raise HTTPException(status_code=400, detail="...")` |
| `socketio.start_background_task(fn, *args)` | `threading.Thread(target=fn, args=args, daemon=True).start()` |
| `@socketio.on("connect")` | `@sio.event\nasync def connect(sid, environ): ...` (use `await sio.emit(..., to=sid)`) |

**Replace `socketio.emit(...)` in `_run_conversation_background`:** pass `socketio=emitter` (instance of `SocketIOEmitter`) into `run_conversation()` and `compute_and_emit_projection()`. The loop already calls `.emit()` on whatever it's given — the Emitter satisfies that interface.

**Routes to preserve (verbatim behavior):**
- `GET /` → render index.html
- `GET /status` → `{"status": "idle"|"running", "turn_index": ..., "cluster_count": ...}`
- `GET /sessions` → list of session metadata (newest-first)
- `POST /resume/{session_id}` → load state.json, emit state_update, return `{"status": "resumed", ...}`
- `POST /upload` → parse file, start background thread, return `{"status": "session_started", "records": N}`

**Module-level state to preserve verbatim:**
- `SESSIONS_DIR = "sessions"`
- `_CLUSTER_COLORS` palette
- `_session` dict
- `_session_lock` (still a `threading.Lock()` — concurrent POSTs go to the event loop, but the lock guards shared mutable state across the worker threads they spawn)
- `_make_session_timestamp`, `_write_session_state`, `_compute_projection`, `_build_projection_payload`, `_should_recompute_projection`, `compute_and_emit_projection`, `_parse_upload`, `_parse_args` — copy these unchanged.

**Entry point:**
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:asgi_app", host="0.0.0.0", port=5000, reload=False)
```

**Commit:** `refactor(web): rewrite app.py on FastAPI + python-socketio (ASGI)`

---

### Task 3 — Implement the `Emitter` shim

**Goal:** Provide a `.emit(event, payload)` interface that bridges sync worker threads to the asyncio event loop, with **zero changes to `src/conversation_loop.py`**.

**Where:** inside `web/app.py`, above the `app = FastAPI(...)` block.

**Class body (paste verbatim, no modifications):**

```python
class SocketIOEmitter:
    """
    Bridges a sync worker thread to an asyncio.AsyncServer.

    The conversation loop calls `.emit(event, payload)` synchronously from
    a worker thread (CPU-bound clustering work runs there to avoid blocking
    the event loop). This wrapper schedules the coroutine on the main loop
    via run_coroutine_threadsafe — the standard, monkey-patch-free pattern.

    Fire-and-forget by design: matches the original Flask-SocketIO threading
    semantic. If the loop is closed during shutdown, the future errors out
    but the worker keeps running — same behavior as the previous stack.
    """
    def __init__(self, sio, loop_getter):
        self._sio = sio
        self._loop_getter = loop_getter  # callable returning the asyncio loop (deferred so startup captures it)

    def emit(self, event: str, payload: dict) -> None:
        loop = self._loop_getter()
        assert loop is not None, "Event loop not captured yet — server not fully started"
        asyncio.run_coroutine_threadsafe(self._sio.emit(event, payload), loop)
```

**Why deferred `_loop_getter`:** the loop doesn't exist until `lifespan` runs at server start. Passing a callable lets us instantiate `SocketIOEmitter` at module import time and resolve the loop lazily on first emit.

**Pass it into the loop:** in `_run_conversation_background` and `compute_and_emit_projection`, replace the `socketio=socketio` argument with `socketio=emitter`. The signature of `src/conversation_loop.py:run_conversation` does not change.

**Commit:** `feat(web): add SocketIOEmitter shim bridging worker thread to async loop`

> **Note:** Tasks 2 and 3 land in **separate commits** even though they touch the same file. Reviewer can read the shim independently from the route translation.

---

### Task 4 — Migrate `tests/phase2/test_app.py`

**Goal:** Replace Flask test client with FastAPI TestClient. Preserve assertions verbatim.

**Replacement fixture:**
```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from web.app import app  # the bare FastAPI app, NOT asgi_app — TestClient handles HTTP only
    with TestClient(app) as c:
        yield c
```

**Rename `flask_client` → `client` in all 5 test functions.** Assertions on `response.status_code` and `response.data` work identically (TestClient uses httpx under the hood; `response.data` becomes `response.content`).

**Specific replacements:**
- `flask_client.get("/")` → `client.get("/")`
- `flask_client.post("/upload", data=data, content_type="multipart/form-data")` → `client.post("/upload", files={"file": ("test.csv", csv_content, "text/csv")})`
- `response.data` → `response.content` (or use `response.json()` for the status/sessions/resume endpoints)
- Drop `app.config["TESTING"] = True` — FastAPI has no equivalent (TestClient handles test mode automatically).

**Commit:** `test(phase2): migrate test_app.py to FastAPI TestClient`

---

### Task 5 — Migrate `tests/phase2/test_sessions.py`

**Goal:** Same fixture migration.

**Replacement fixture:**
```python
@pytest.fixture
def client():
    from web.app import app
    with TestClient(app) as c:
        yield c
```

**Drop entirely:** `app.config["TESTING"] = True` and `app.config["PROPAGATE_EXCEPTIONS"] = False` (Flask-specific). FastAPI's TestClient already converts unhandled exceptions to 500 responses, which is what the existing `test_resume_endpoint_exists` expects (`assert response.status_code in (400, 404, 500)`).

**Verify monkey-patch endpoints still work:** `monkeypatch.setattr(app_module, "SESSIONS_DIR", tmpdir)` continues to work — it's a Python attribute swap, framework-agnostic.

**Verify form upload tests:** if any use `content_type="multipart/form-data"`, convert to TestClient's `files=` parameter.

**Commit:** `test(phase2): migrate test_sessions.py to FastAPI TestClient`

---

### Task 6 — Run the full test suite + manual smoke

**Goal:** Confirm zero regressions before touching docs.

**Steps:**
1. `pytest tests/ -q` — must show same pass count as before migration (106 from Phase 3 + any new). No new failures, no new skips.
2. **Manual UI smoke:**
   ```
   uvicorn web.app:asgi_app --host 0.0.0.0 --port 5000 --reload
   ```
   Then open `http://localhost:5000/`:
   - Page loads, status banner shows "Connected"
   - Upload `dataset/ag_news_working.csv` (or any small CSV from `dataset/`)
   - Confirm cluster cards appear, UMAP projection renders, sessions list updates
   - Refresh page → on-connect re-emit re-populates the UI with current state
   - Click a past session → resume works and `turn_index` displays correctly
3. Tail `sessions/<latest>/audit_log.jsonl` and verify per-turn JSONL writes still happen.

**Failure handling:** if anything breaks, **do not paper over with try/except** (CLAUDE.md fail-loudly). Diagnose the root cause and fix in a follow-up commit before proceeding to Task 7.

**Commit (if any fixes needed):** `fix(web): <specific issue found during smoke>`

---

### Task 7 — Update `.planning/` and `CLAUDE.md`

**Goal:** Make the new web stack discoverable to future sessions and contributors.

#### 7a. `CLAUDE.md` — add this section before "## Planning Artifacts"

```markdown
## Web Stack

Debug UI runs on **FastAPI + uvicorn + python-socketio (ASGI)**.

**Run:** `uvicorn web.app:asgi_app --host 0.0.0.0 --port 5000 --reload`

**Deps:** `fastapi`, `uvicorn[standard]`, `python-socketio` (see `requirements.txt`)

**NEVER install or import `eventlet` or `gevent`.** They monkey-patch the
stdlib at import time (threading, socket, time.sleep), which corrupts
numpy/sklearn/HDBSCAN/UMAP internal locking. uvicorn is native asyncio
and needs no monkey-patching. CPU-bound work runs in `threading.Thread`;
emits cross from worker threads to the event loop via
`asyncio.run_coroutine_threadsafe` (see `SocketIOEmitter` in `web/app.py`).
```

#### 7b. `.planning/STATE.md`

Under `### Architecture Constraints` (after the "Sessions persist in sessions/..." bullet), append:

```markdown
- Web layer is FastAPI + uvicorn + python-socketio (ASGI). Migrated from Flask + Flask-SocketIO on 2026-05-15. CPU-bound clustering work runs in worker threads; SocketIOEmitter bridges thread→loop via run_coroutine_threadsafe. eventlet and gevent are explicitly forbidden (would corrupt numpy/sklearn).
```

Bump frontmatter `last_updated: "2026-05-15T..."` and the "**Last updated:**" line.

#### 7c. `.planning/ROADMAP.md`

After the Phase 2 "Delivered" block, append:

```markdown
**Web stack migration (2026-05-15):** Flask + Flask-SocketIO replaced by FastAPI + python-socketio (ASGI). UI-01 and UI-02 requirement wording unchanged; only the underlying framework switched. See `.planning/fastAPI_plan.md`.
```

Update header `**Updated:** 2026-05-15`.

#### 7d. `.planning/REQUIREMENTS.md`

Update only the trailer:
```markdown
*Last updated: 2026-05-15 — web stack migrated to FastAPI (UI-01/UI-02 wording unchanged)*
```

**Commit:** `docs: record FastAPI migration in CLAUDE.md and .planning/`

---

## 5. Verification checklist (end-of-migration gate)

Run all of these before declaring done. Any failure = stop and fix.

- [ ] `pytest tests/ -q` — green; pass count ≥ pre-migration count
- [ ] `pip show eventlet gevent` — both report "Package not found"
- [ ] `pip show fastapi uvicorn python-socketio` — all present
- [ ] `pip show flask flask-socketio` — both report "Package not found" (or version is no longer pinned in `requirements.txt`)
- [ ] Manual smoke: server starts cleanly with `uvicorn web.app:asgi_app`
- [ ] Manual smoke: upload → cluster cards render → UMAP renders → sessions list updates
- [ ] Manual smoke: refresh page → state re-syncs via `connect` handler
- [ ] Manual smoke: click past session → resume restores correctly
- [ ] `sessions/<latest>/audit_log.jsonl` contains per-turn JSONL writes
- [ ] `git diff src/` — no changes outside `src/` except the parent dir (i.e. `src/conversation_loop.py` truly untouched)
- [ ] `git log --oneline FastAPI ^main` — 7 commits in the order above

---

## 6. Rollback

If the migration must be aborted mid-flight:

```bash
git checkout main -- web/ tests/phase2/test_app.py tests/phase2/test_sessions.py requirements.txt
pip install flask flask-socketio
pip uninstall -y fastapi uvicorn python-socketio
```

`src/conversation_loop.py` is untouched by design, so no source rollback needed there.

---

## 7. Out of scope (explicitly NOT in this plan)

- Switching the conversation loop to `async def` — see decision #2; CPU-bound work doesn't benefit from async.
- Adding new endpoints (streaming, LLM token relay, multi-session) — future Phase 4+ work.
- Replacing python-socketio with native FastAPI WebSocket — would require frontend rewrite; not justified for a developer debug UI.
- Authentication / TLS — this is a localhost developer tool (see existing comment in `web/app.py:204`).
- Moving from `threading.Thread` to a process pool — single-session-per-server (D-15) makes one thread sufficient.

---

## 8. Reference: the cross-thread emit pattern (why it's safe)

```
[worker thread]                          [asyncio event loop]
       │                                          │
       │  SocketIOEmitter.emit("state_update",   │
       │      payload)                            │
       │                                          │
       ├──► run_coroutine_threadsafe(            │
       │      sio.emit(...), loop)               │
       │      returns concurrent.futures.Future  │
       │                                          │
       │                                          ◄──── awakens, picks up the coro
       │                                          │
       │                                          ├──► sio.emit() runs as a coroutine,
       │                                          │    encodes payload, writes to socket
       │                                          │
       │  (worker thread already moved on)       │
       ▼                                          ▼
```

No monkey-patching. No GIL surprises. Standard library only.
