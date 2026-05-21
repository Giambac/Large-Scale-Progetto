---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Experimentation Flexibility & Scale
status: roadmapped
last_updated: "2026-05-21T09:17:42.355Z"
last_activity: 2026-05-21
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State: Conversational Clustering

**Last updated:** 2026-05-21
**Updated by:** Roadmapper — Milestone v2.0 roadmap created. Phases 7–12 appended to ROADMAP.md (v1 phases 1–6 preserved). All 17 v2.0 requirements mapped, 100% coverage. REQUIREMENTS.md v2.0 traceability filled. Next action: `/gsd-plan-phase 7`.

---

## Project Reference

**Core value:** The interaction loop converges toward oracle-accepted clusterings efficiently, with every design decision — what to show, what to ask, when to stop — measured against cognitive load and information gain.

**Current focus:** Milestone v2.0 — Phase 7 (Pluggable Embedding Backends + Dynamic Dimension) [FOUNDATIONAL]

---

## Current Position

Phase: Phase 7 — Pluggable Embedding Backends + Dynamic Dimension (not started)
Plan: —
Status: Roadmapped — ready to plan Phase 7
Last activity: 2026-05-21 — Milestone v2.0 roadmap created (Phases 7–12)

## v2.0 Phase Map

| Phase | Goal | Requirements | Notes |
|-------|------|--------------|-------|
| 7 | Pluggable embedding backends + dynamic dimension | EMB-V2-01/02/03 | FOUNDATIONAL — unblocks all embedding work; defines provenance/manifest schema |
| 8 | Colab compute-only artifact pipeline | COL-V2-01/02 | Consumes Phase 7 manifest; HF Hub handoff; local imports zero Colab deps |
| 9 | Interactive UMAP recolor + real-time chat | VIZ-V2-02, UX-V2-01/02 | Recolor mechanism built here; correctness validated after Phase 11 alignment |
| 10 | Oracle-initiated flow + KMeans-only + oracle YAML config | CLUST-V2-01, FLOW-V2-01/02, OCFG-V2-01/02 | Drop HDBSCAN from interactive path; intro to events.jsonl sidecar |
| 11 | Re-fit KMeans per query + cluster-ID alignment + query filter | CLUST-V2-02, FILT-V2-01 | ENGINEERING CORE; linchpin Hungarian ID alignment; research spike likely |
| 12 | Coordination agent | COORD-V2-01/02 | LAST; highest risk; flagged for own research spike |

**Coverage:** 17/17 v2.0 requirements mapped, 0 unmapped.

## Performance Metrics

| Metric | Value |
|--------|-------|
| v1 phases complete | 6/6 (code) |
| v1 plans complete | 28/28 |
| Phase 6 UAT | 11/11 pass |
| v2.0 phases | 0/6 started |
| Blockers | 0 |
| Pending (human-action, v1) | human study N≥10; compare_oracle_types |

---

## Accumulated Context

### Key Decisions (v2.0 — locked at milestone start)

| Decision | Rationale | Status |
|----------|-----------|--------|
| Per-query: re-FIT KMeans, NOT re-embed | Preserves read-only EmbeddingStore invariant; avoids cost/geometry churn | **Locked (milestone start)** |
| Colab is compute-only; sockets/UI run locally | Colab cannot host the FastAPI/socketio interactive UI | **Locked (milestone start)** |
| Oracle prompts/configs in versioned YAML, not DB | Clean git diffs; human-editable; ruamel.yaml round-trip | **Locked (milestone start)** |
| Coordination agent built LAST (Phase 12) | Highest risk; breaks single-state/single-writer cardinality; needs the rest stable | **Locked (milestone start)** |
| Dynamic embedding dim before any second backend | EMBEDDING_DIM=384 constant gates pluggable backends + silently gates cache reuse | **Locked (research, Phase 7)** |
| Provenance/manifest schema is the shared contract | Both Colab artifacts (P8) and embedding backends (P7) validate against it | **Locked (research)** |
| Recolor-not-refit UMAP, correctness gated on P11 alignment | Recolor with churned KMeans IDs paints wrong colors | **Open — sequence validation after P11 (P9/P11 coupling)** |
| Cluster-ID reconciliation policy (preserve vs. renumber) | Affects merge/split history, names, recolor correctness | Unresolved — settle in Phase 11 planning |
| Hierarchy lineage on wholesale re-fit | record_split/merge assume incremental edits; re-fit has no clean lineage | Unresolved — settle in Phase 11 planning |
| Coordination merge semantics | Cross-session state-merge contract unresolved | Unresolved — Phase 12 research spike |

### Key Decisions (v1 — for reference)

| Decision | Rationale | Status |
|----------|-----------|--------|
| LangGraph vs. plain Python for orchestrator | Plain Python is simpler for fixed sequential graph | **Resolved (Phase 2): plain Python while-loop** |
| ClusteringBackend Protocol shape | ABC vs. typing.Protocol | **Resolved (Plan 06): runtime_checkable Protocol** |
| UMAP random_state for projection | Fixed random_state=42 ensures identical coords | **Resolved (Plan 07): random_state=42** |
| Projection recompute trigger | Expensive UMAP refit should not happen every turn | **Resolved (Plan 07): recompute only on Split/Merge (D-22) — superseded by P9 recolor-not-refit** |
| Session directory format | Timestamp dirs under sessions/ with state.json per turn | **Resolved (Plan 08)** |
| sklearn vs. standalone hdbscan | Soft-assignment vector shape | **Resolved (Phase 1): standalone hdbscan 0.8.42 — note: HDBSCAN dropped from interactive path in P10** |

### Architecture Constraints

- Embeddings computed once, stored in read-only EmbeddingStore — never re-embedded per turn (re-FIT only in v2.0)
- ClusteringState is the single source of truth — no direct agent-to-agent calls (must survive coordination agent in P12)
- JSONL audit_log is the source of truth for replay; the DB is the queryable cross-run index
- Web layer is FastAPI + uvicorn + python-socketio (ASGI). eventlet/gevent forbidden (corrupts numpy/sklearn/UMAP)
- v2.0: `EMBEDDING_DIM` becomes dynamic (P7); provenance manifest validates all embedding artifacts on load
- v2.0: local torch is CPU-only (2.11.0+cpu) — motivation for Colab GPU offload (P8)
- v2.0: only genuinely new dependency is ruamel.yaml; reuse huggingface_hub + openai + pydantic already installed

### Research Flags by Phase (v2.0)

| Phase | Flag |
|-------|------|
| Phase 7 | Standard pattern (Protocol mirrors ClusteringBackend); likely skip research-phase. Clean up EMBEDDING_DIM/docstring drift + bump requirements.txt floors |
| Phase 9 | Cache-once/recolor is standard and half-exists; likely skip. CROSS-PHASE: recolor correctness depends on P11 alignment |
| Phase 10 | Orchestration around existing loop; likely skip. Decide intro-turn audit semantics (suggested: events.jsonl sidecar) |
| Phase 11 | RESEARCH SPIKE LIKELY — cluster-ID reconciliation policy + hierarchy-lineage gap are genuine design decisions |
| Phase 12 | RESEARCH SPIKE REQUIRED — cross-session state-merge, contradiction recombination, partial-failure rollback, BLAS contention (all 3 researchers flagged) |

### Todos

- [ ] **Plan Phase 7** — `/gsd-plan-phase 7` (FOUNDATIONAL, do first)
- [ ] (v1 pending) Run human validation study (N≥10) via /study UI — Phase 6 SC-2, human-action
- [ ] (v1 pending) Run `compare_oracle_types` after DB has both LLM + human rows — Phase 6 SC-3, human-action
- [ ] During Phase 7: bump stale requirements.txt floors (sentence-transformers>=5.4, openai>=2.26,<3, pin huggingface_hub>=1.7,<2); clean up 768-vs-384 docstring drift

### Blockers

None.

---

## Session Continuity

**Last session:** 2026-05-21 — Milestone v2.0 roadmapped.

**Stopped at:** ROADMAP.md Phases 7–12 written (v1 phases 1–6 preserved); REQUIREMENTS.md v2.0 traceability filled (17/17 mapped); this STATE.md updated. No code changes this session.

**Prior session:** 2026-05-20 — Phase 6 UAT complete (11/11 pass), committed b57f735.

**To resume:** Next real action is `/gsd-plan-phase 7` (Pluggable Embedding Backends + Dynamic Dimension — FOUNDATIONAL, gates all embedding work). v1 human-action items (human study N≥10, compare_oracle_types) remain open but are independent of the v2.0 build.

**Critical cross-phase coupling to carry into planning:** The Phase 9 UMAP recolor is only fully CORRECT once the Phase 11 cluster-ID alignment lands — build recolor in P9 but defer its visual-correctness validation until P11.

**Existing repo artifacts:**

- `Conversational Clustering Script.txt` — prototype/reference script
- `Multi Agent Personalities Script.txt` — multi-agent persona reference

**Critical constraint:** The held-out evaluation split (PRE-01, SHA-256 ea53dfa1) remains locked. v2.0 re-FIT operates on fixed embeddings; never re-embed per query.

---

*State initialized: 2026-04-29*
*Milestone v2.0 roadmapped: 2026-05-21*
