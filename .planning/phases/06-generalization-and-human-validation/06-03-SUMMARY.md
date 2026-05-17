---
phase: "06"
plan: "03"
subsystem: "study-ui"
tags: [human-study, fastapi, socketio, umap, canvas, satisfaction-detection, exp-v2-01]
dependency_graph:
  requires:
    - web/app.py                   # Extended with study routes and background worker
    - src/db/experiments.py        # ExperimentCreate with oracle_type='human'
    - src/db/connection.py         # connect(), init_schema()
    - src/db/turns.py              # TurnCreate, create()
    - src/clustering.py            # build_initial_clustering_state, backends
    - src/embedding_store.py       # EmbeddingStore.load(), compute_and_save()
    - src/feedback_parser.py       # parse_feedback() — human free-text -> FeedbackDelta list
    - src/agent_functions.py       # f_next_state()
    - src/serialization.py         # append_to_audit_log(), _write_session_state()
    - src/logging_setup.py         # deviation() for satisfaction_detection_failed
  provides:
    - web/app.py                   # POST /study/sessions, GET /study/{session_id}, study_feedback event
    - web/templates/study.html     # Study page: three-panel layout
    - web/static/study.js          # SocketIO client: cluster cards, mini-plot canvases, feedback
  affects:
    - EXP-V2-01                    # Human study data collection surface
tech_stack:
  added:
    - HTML5 Canvas API (plain vanilla — no Chart.js or D3)
  patterns:
    - threading.Event for worker thread synchronization (human blocks on event.wait())
    - SocketIOEmitter bridge (asyncio.run_coroutine_threadsafe) — same as main session
    - One DB connection per study session; closed in _end_study_session (Phase 4 D-04)
    - LLM satisfaction detection: claude-haiku-4-5, max_tokens=4, YES/NO only (D-12)
    - Faceted UMAP: K mini-canvases sharing global bounds (D-10)
    - textContent (never innerHTML) for user-provided text (T-06-03-05 threat mitigation)
key_files:
  created:
    - web/templates/study.html
    - web/static/study.js
  modified:
    - web/app.py
decisions:
  - "study_feedback SocketIO event unblocks worker via threading.Event — same pattern as main session's per-turn blocking"
  - "Satisfaction detection uses try/except anthropic.APIError only; all other errors propagate (fail loudly); treats APIError as NOT satisfied and calls deviation()"
  - "_end_study_session seals DB experiment row with total_turns + convergence_reason + end_timestamp, then calls db.close() — one connection per run (D-04)"
  - "Per-cluster mini-plots share globalBounds computed once from all embedding coords — spatial consistency across K canvases (D-10)"
  - "study.js uses plain HTML5 Canvas (getContext('2d')) with no external charting library — vanilla JS only per plan constraint"
  - "textContent used for all user-provided text rendering in study.js — prevents XSS (T-06-03-05)"
  - "Hover highlight redraws the specific cluster's canvas at radius 6 with +60 RGB brightness — no full-page re-render"
  - "Participant dataset upload deferred (D-11 in 06-CONTEXT.md); researcher provides dataset_path in POST /study/sessions"
metrics:
  duration: "~40 minutes"
  completed: "2026-05-17T12:45:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 1
---

# Phase 6 Plan 03: Human Study UI — /study Route, Study Page, Satisfaction Detection, Session Management

**One-liner:** Human study data-collection surface with POST /study/sessions API, faceted per-cluster UMAP mini-plots, per-turn LLM satisfaction detection via claude-haiku-4-5, 30-turn hard cap, and full DB writes with oracle_type='human'.

## What Was Built

### T-06-03-01: web/app.py — study backend

**New constants (module-level):**
- `STUDY_MAX_TURNS: int = int(os.environ.get("STUDY_MAX_TURNS", "30"))` — configurable cap
- `_study_sessions: dict = {}` — session_id -> session state dict

**New routes:**
- `POST /study/sessions` — validates `dataset_path` (os.path.exists assert) + `backend` (hdbscan|kmeans assert), loads records, creates experiment row with `oracle_type='human'`, initialises session dict with `threading.Event` + `feedback_queue: []`, launches `_run_study_background` daemon thread, returns `{"session_id": ..., "study_url": "/study/<id>"}`
- `GET /study/{session_id}` — serves `study.html` template with `session_id` in context; 404 if neither in-memory nor on-disk session directory found

**New functions:**
- `_run_study_background(session_id)` — worker thread: computes embeddings (reuses cache if same size), builds initial clustering, computes UMAP projection, emits `study_projection` (coords + cluster_colors + global_bounds + per_cluster) and `study_state` (clusters with 8 sample items each), then enters the feedback loop with threading.Event synchronization
- `_get_item_preview(item_id, records)` — returns first 80 chars of item text
- `_detect_satisfaction(human_text, client)` — claude-haiku-4-5 YES/NO call; catches `anthropic.APIError` only, calls `deviation("satisfaction_detection_failed")` and returns False on error
- `_end_study_session(session_id, convergence_reason)` — updates DB experiment row with `total_turns + convergence_reason + end_timestamp`, closes DB, emits `study_ended`, sets `sess["ended"] = True` and wakes the blocked worker
- `@sio.event study_feedback` — appends `data["text"]` to `feedback_queue`, calls `feedback_event.set()`

**Feedback loop in `_run_study_background`:**
1. Emit `study_awaiting_feedback`
2. Block on `feedback_event.wait()`
3. Pop `feedback_queue[0]` -> `human_text`
4. `_detect_satisfaction(human_text)` — if YES: emit `study_satisfaction_detected`, wait for confirmation, handle yes/no
5. `parse_feedback(human_text, state, client)` -> deltas
6. `f_next_state(state, deltas, ...)` -> new_state
7. `_write_session_state` + `append_to_audit_log`
8. DB: `turns.create(TurnCreate(..., action_type="human_feedback"))`
9. Increment `turn_index`, emit `study_state`
10. If `turn_index >= STUDY_MAX_TURNS`: `_end_study_session(session_id, "turn_budget")`

**CLAUDE.md compliance verified:**
- No eventlet or gevent imports
- No sqlite3.execute outside src/db/
- datetime.now(timezone.utc) for all new timestamps
- One DB connection per study session, closed in `_end_study_session`
- CPU-bound work in threading.Thread(daemon=True)
- emit cross-thread via emitter (asyncio.run_coroutine_threadsafe)

### T-06-03-02: web/templates/study.html + web/static/study.js

**study.html:**
- Three-panel layout: left (cluster cards, 300px), center (UMAP mini-plots, flex 1), right (feedback, 280px)
- Cluster cards: collapsible `.study-cluster-card` with bold name, italic description, expandable item list (`.study-item` spans with `data-item-id` + `data-cluster-id`)
- Satisfaction banner (`#satisfaction-banner`): hidden by default; shown on `study_satisfaction_detected`; Yes/No buttons emit confirmation feedback
- Session complete banner (`#session-complete-banner`): shown on `study_ended`
- Elements: `#study-feedback`, `#send-feedback`, `#study-status`, `#satisfaction-banner`, `#session-complete-banner`
- Scripts: CDN socket.io 4.6.2 + `/static/study.js` (no new libraries)

**study.js:**
- Session ID extracted from `window.location.pathname`
- `socket.on('study_state')`: builds collapsible cluster cards via DOM API; hover on `.study-item` calls `highlightItem(itemId, clusterId)` -> redraws that cluster's canvas
- `socket.on('study_projection')`: stores `_allCoords`, `_globalBounds`, `_clusterColors`, `_perCluster`; calls `renderMiniPlots()`
- `socket.on('study_awaiting_feedback')`: enables textarea + send button
- `socket.on('study_satisfaction_detected')`: shows satisfaction banner with message
- `socket.on('study_ended')`: disables input, shows session-complete banner
- `sendFeedback()`: emits `study_feedback` event; clears textarea; disables input pending response
- `drawMiniPlot(canvas, clusterItemIds, allCoords, clusterColor, globalBounds, highlightItemId)`:
  - Pass 1: all non-cluster items as gray circles (radius 2, opacity 0.15)
  - Pass 2: cluster items as colored circles (radius 3, full opacity)
  - Highlighted item: radius 6, color+60 RGB brightness, dark stroke
  - Scales via `globalBounds` -> same viewport across all K canvases
- XSS mitigation: all user text via `textContent` (never `innerHTML`) — T-06-03-05

## Verification Results

Acceptance criteria verified via automated checks:

**T-06-03-01 (9/9 pass):**
- `@app.post.*study/sessions` — PASS
- `@app.get.*study` — PASS
- `_study_sessions: dict` module-level — PASS
- `STUDY_MAX_TURNS` constant from `os.environ` — PASS
- `@sio.event study_feedback` handler — PASS
- No eventlet or gevent import — PASS
- No sqlite3.execute in web/app.py — PASS
- `os.path.exists(params.dataset_path)` assertion — PASS
- `datetime.datetime.now(_timezone.utc)` for all timestamps — PASS

**T-06-03-02 (10/10 pass):**
- study.html valid HTML5 — PASS
- study.js exists — PASS
- socket.io.min.js script tag — PASS
- /static/study.js script tag — PASS
- `#study-feedback`, `#send-feedback`, `#study-status`, `#satisfaction-banner` — PASS (all 4)
- Handles `study_state`, `study_projection`, `study_awaiting_feedback`, `study_satisfaction_detected`, `study_ended` — PASS (all 5)
- Emits `study_feedback` event — PASS
- `getContext('2d')` present — PASS
- No Chart.js or D3 — PASS
- No ES6 `import` statements — PASS

## Deviations from Plan

### Auto-fixed Issues

None.

### Implementation Notes

**LLM provider assertion in study background:** `_run_study_background` asserts `provider == "anthropic"` before creating the Anthropic client. The plan specifies anthropic for satisfaction detection and feedback parsing; non-Anthropic providers would require a refactor. This is a loud failure rather than silent fallback — consistent with CLAUDE.md fail-loudly philosophy.

**Study background loop uses `parse_feedback` with the Anthropic client directly** (not via `run_conversation`): the plan's step 4f calls `parse_feedback(human_text, state)` in the background function. Since `parse_feedback` requires an LLM client and the study session already has one, the client is passed directly. This avoids the overhead of spinning up a full `run_conversation` call (which assumes a full oracle agent and stopping criteria not applicable to the human study).

**`TurnCreate` action_type set to `"human_feedback"`:** The plan says "insert turn row via src/db/turns.create(...)" but does not specify the `action_type`. Used `"human_feedback"` to distinguish human study turns from oracle-driven turns in the DB.

## Known Stubs

None — all cluster data flows from server-side clustering to client via SocketIO events.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes beyond what the plan's threat model specified. All 5 mitigations implemented:

| Threat ID | Mitigation |
|-----------|-----------|
| T-06-03-02 | `os.path.exists(params.dataset_path)` assertion before loading |
| T-06-03-04 | `parse_feedback("")` returns `[]`; `STUDY_MAX_TURNS` cap limits total turns |
| T-06-03-05 | All user text rendered via `textContent` (never `innerHTML`) in study.js |
| T-06-03-SC | No new npm/pip packages; CDN socket.io already used by index.html |

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| T-06-03-01 + T-06-03-02 | Human study UI: /study routes, study.html, study.js | 579680a |

## Self-Check

- [x] `web/app.py` modified with study routes and backend
- [x] `web/templates/study.html` created
- [x] `web/static/study.js` created
- [x] Commit `579680a` exists in git log
- [x] All acceptance criteria pass (9/9 + 10/10)
- [x] No stubs — all data flows live from server
- [x] No sqlite3.execute outside src/db/
- [x] No eventlet or gevent imports

## Self-Check: PASSED
