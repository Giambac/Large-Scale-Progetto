---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Experimentation Flexibility & Scale
status: planning
last_updated: "2026-05-21T09:17:42.355Z"
last_activity: 2026-05-21
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State: Conversational Clustering

**Last updated:** 2026-05-20
**Updated by:** Phase 6 UAT complete (11/11 pass). Fixed during UAT: f_next_state delta validation (blocker — was crashing study worker), test_mapping monkeypatch path (major), study UI turn counter, STUDY_MAX_TURNS 30→15. Committed b57f735. Remaining: human study (N≥10) + compare_oracle_types — both human-action.

---

## Project Reference

**Core value:** The interaction loop converges toward oracle-accepted clusterings efficiently, with every design decision — what to show, what to ask, when to stop — measured against cognitive load and information gain.

**Current focus:** Phase 6 — Generalization and Human Validation

---

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-05-21 — Milestone v2.0 started

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 6/6 (code) |
| Plans complete | 28/28 |
| Phase 6 UAT | 11/11 pass |
| Blockers | 0 |
| Pending (human-action) | human study N≥10; compare_oracle_types |

---

## Accumulated Context

### Key Decisions (from research)

| Decision | Rationale | Status |
|----------|-----------|--------|
| LangGraph vs. plain Python for orchestrator | LangGraph adds HITL checkpointing; plain Python is simpler for fixed sequential graph. ARCHITECTURE.md recommends plain Python. | **Resolved (Phase 2): plain Python while-loop** |
| ClusteringBackend Protocol shape | ABC vs. typing.Protocol for backend contract | **Resolved (Plan 06): runtime_checkable Protocol — duck typing, no inheritance required** |
| KMeansBackend K initialization timing | __init__ vs. first fit() call for BIC K selection | **Resolved (Plan 06): lazy init at first fit() call; _k can be overridden for tests** |
| BIC GMM covariance type | full vs. diag covariance for 768-dim embeddings | **Resolved (Plan 06): diag + max_iter=50 for startup speed; acceptable for research tool** |
| UMAP random_state for projection | Fixed random_state=42 ensures identical coords across runs on same dataset | **Resolved (Plan 07): random_state=42 in _compute_projection** |
| AMBIGUOUS_P_MIN threshold | Items qualify as ambiguous for BoundaryDrivenStrategy when P(A) >= 0.3 AND P(B) >= 0.3 | **Resolved (05-03): 0.3 per CONTEXT.md threshold spec** |
| Bootstrap CI oracle_type filter | --oracle-type filtered in Python post-query rather than adding SQL param to exp_db.query() | **Resolved (05-05): Python-side filter; adding to query() is Phase 6 concern** |
| Bootstrap seed default | seed defaults to 0 for reproducibility; same code reruns produce identical CI | **Resolved (05-05): explicit seed=0 default per T-05-12 mitigation** |
| MAX_SUBSET_SIZE cap | show_subset item_ids capped at 8 for cognitive-load reasons; sampled deterministically by state.turn_index | **Resolved (05-03): MAX_SUBSET_SIZE=8, random.Random(state.turn_index)** |
| Projection recompute trigger | Expensive UMAP refit should not happen every turn | **Resolved (Plan 07): recompute only on SplitFeedback or MergeFeedback (D-22)** |
| post_turn_callback hook design | How to wire per-turn side effects without coupling the loop to projection logic | **Resolved (Plan 07): Optional[Callable] parameter, default None, called after AuditLog write** |
| Session directory format and persistence | Timestamp dirs under sessions/ with state.json per-turn snapshots | **Resolved (Plan 08): sessions/<YYYY-MM-DDTHH-MM-SS>/ with state.json, audit_log.jsonl, embeddings.npy** |
| Flask PROPAGATE_EXCEPTIONS in tests | AssertionError in Flask routes must return 500 to test client, not propagate | **Resolved (Plan 08): PROPAGATE_EXCEPTIONS=False in test fixture** |
| _per_turn_callback combining D-27 and D-22 | Single callback for both state.json write and UMAP projection recompute | **Resolved (Plan 08): _per_turn_callback composes both concerns** |
| sklearn HDBSCAN `probabilities_` vs. standalone `hdbscan` full multinomial vectors | Determines SoftAssignment data structure and `f_uncertainty` computation; cascading if retrofitted | **Resolved (Phase 1): standalone `hdbscan` 0.8.42** |
| Primary dataset (Amazon Reviews 2023, IMDB, or support tickets) | Held-out split must be locked before any code runs | Unresolved — decide at Phase 0/Phase 1 |
| Oracle cognitive-load weight parameters | Cognitive load is both a design constraint and primary metric; wrong weights produce broken metric | Unresolved — decide at Phase 3 |
| "Oracle satisfaction" operationalization for stopping signal | Must not be circular (oracle both gives feedback and decides when to stop) | Unresolved — must be decided in Phase 1 (PRE-02) |

### Architecture Constraints

- Embeddings computed once at startup, stored in read-only EmbeddingStore — never re-embedded per turn
- LangGraph state (or plain Python ClusteringState) is the single source of truth — no direct agent-to-agent calls
- JSON-first logging from Phase 1; migrate to MLflow at Phase 5
- Inject only a structured state summary (under 500 tokens) into context window, not full history
- Test state integrity at turn 20, 30, 50 with synthetic oracle before any human study
- Sessions persist in sessions/<timestamp>/ directories; server restart does not lose state
- Web layer is FastAPI + uvicorn + python-socketio (ASGI). Migrated from Flask + Flask-SocketIO on 2026-05-15. CPU-bound clustering work runs in worker threads; SocketIOEmitter bridges thread→loop via run_coroutine_threadsafe. eventlet and gevent are explicitly forbidden (would corrupt numpy/sklearn).

### Research Flags by Phase

| Phase | Flag |
|-------|------|
| Phase 1 | HDBSCAN soft-assignment sufficiency must be decided at gate |
| Phase 3 | Oracle cognitive-load weight parameters may need targeted literature check |
| Phase 5 | Information-gain estimation for `f_next_best_step` may need implementation spike |
| Phase 6 | Human study protocol must be written in Phase 1 (PRE-02); no additional research needed at execution |

### Todos

- [x] Select primary dataset and lock held-out split with hash (Phase 1, PRE-01)
- [x] Write the three stopping criteria as code-ready specifications (Phase 1, PRE-02)
- [x] Decide LangGraph vs. plain Python before Phase 2 begins — plain Python while-loop
- [x] Decide sklearn HDBSCAN vs. standalone `hdbscan` package at Phase 1 gate — standalone hdbscan 0.8.42
- [x] Install umap-learn and hdbscan packages in environment
- [ ] **Run human validation study (N≥10) via /study UI** — Phase 6 SC-2, human-action
- [ ] **Run `compare_oracle_types` after DB has both LLM + human rows** — Phase 6 SC-3, human-action
- [ ] (Optional) `/gsd-secure-phase 6` — security gate skipped this session
- [ ] Capture the Phase 6 "design issues" the user noted during UAT (see 06-UAT.md Open Design Notes)

### Blockers

None.

---

## Session Continuity

**Last session:** 2026-05-20
**Stopped at:** Phase 6 UAT complete — 11/11 pass. Resumed paused UAT from test 6, found and fixed two bugs: (1) blocker — f_next_state applied LLM-parsed deltas without validating cluster ids against live state, crashing the study worker thread on a hallucinated/retired id; now skips invalid deltas via deviation(); (2) major — test_mapping monkeypatch targeted a nonexistent src.mapping.anthropic path; now patches resolve_llm_key/build_client/chat. Added study UI turn counter; lowered STUDY_MAX_TURNS 30→15 per user. All 14 mapping/study tests pass. Committed b57f735. Dev server left running in background at port 5000 (STUDY_MAX_TURNS=15).

**Prior session:** 2026-05-19 — Phase 6 code complete, UAT paused at test 6/11.

**To resume:** Phase 6 engineering is done. Next real action is the human study (N≥10 via /study), then `python -m examples.compare_oracle_types` + notebooks/llm_vs_human.ipynb. Optional: `/gsd-secure-phase 6` (skipped this session), then `/gsd-complete-milestone`.

**New canonical reference for Phase 4:** wazzup "how to build simple applications" recipe (user-provided this session). Planner should ask user to commit the recipe markdown somewhere stable (e.g. `private/recipes/`) before execute begins — the recipe is the source of design discipline for the DB layer shape and docs discipline.

**Existing repo artifacts:**

- `Conversational Clustering Script.txt` — prototype/reference script (review before Phase 2)
- `Multi Agent Personalities Script.txt` — multi-agent persona reference (review before Phase 3)

**Critical constraint:** The held-out evaluation split must be locked (PRE-01) before any experiment code is written. Contamination is irreversible.

---

*State initialized: 2026-04-29*
