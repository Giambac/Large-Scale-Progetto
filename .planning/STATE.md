# Project State: Conversational Clustering

**Last updated:** 2026-04-29
**Updated by:** Roadmap initialization

---

## Project Reference

**Core value:** The interaction loop converges toward oracle-accepted clusterings efficiently, with every design decision — what to show, what to ask, when to stop — measured against cognitive load and information gain.

**Current focus:** Phase 1 — Pre-Code Obligations and Foundation

---

## Current Position

**Milestone:** v1
**Current phase:** 1 — Pre-Code Obligations and Foundation
**Current plan:** Not started
**Status:** Planning complete; ready to begin Phase 1

**Progress:**
```
Phase 1 [          ] 0%   Pre-Code Obligations and Foundation
Phase 2 [          ] 0%   Clustering Agent Core
Phase 3 [          ] 0%   Oracle Agent
Phase 4 [          ] 0%   Judge Agent
Phase 5 [          ] 0%   Ablation Harness and Strategies
Phase 6 [          ] 0%   Generalization and Human Validation
```

**Overall:** 0/6 phases complete

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 0/6 |
| Plans complete | 0/? |
| Requirements satisfied | 0/28 |
| Blockers | 0 |

---

## Accumulated Context

### Key Decisions (from research)

| Decision | Rationale | Status |
|----------|-----------|--------|
| LangGraph vs. plain Python for orchestrator | LangGraph adds HITL checkpointing; plain Python is simpler for fixed sequential graph. ARCHITECTURE.md recommends plain Python. | Unresolved — decide before Phase 2 |
| sklearn HDBSCAN `probabilities_` vs. standalone `hdbscan` full multinomial vectors | Determines SoftAssignment data structure and `f_uncertainty` computation; cascading if retrofitted | Unresolved — decide at Phase 1 gate |
| Primary dataset (Amazon Reviews 2023, IMDB, or support tickets) | Held-out split must be locked before any code runs | Unresolved — decide at Phase 0/Phase 1 |
| Oracle cognitive-load weight parameters | Cognitive load is both a design constraint and primary metric; wrong weights produce broken metric | Unresolved — decide at Phase 3 |
| "Oracle satisfaction" operationalization for stopping signal | Must not be circular (oracle both gives feedback and decides when to stop) | Unresolved — must be decided in Phase 1 (PRE-02) |

### Architecture Constraints

- Embeddings computed once at startup, stored in read-only EmbeddingStore — never re-embedded per turn
- LangGraph state (or plain Python ClusteringState) is the single source of truth — no direct agent-to-agent calls
- JSON-first logging from Phase 1; migrate to MLflow at Phase 5
- Inject only a structured state summary (under 500 tokens) into context window, not full history
- Test state integrity at turn 20, 30, 50 with synthetic oracle before any human study

### Research Flags by Phase

| Phase | Flag |
|-------|------|
| Phase 1 | HDBSCAN soft-assignment sufficiency must be decided at gate |
| Phase 3 | Oracle cognitive-load weight parameters may need targeted literature check |
| Phase 5 | Information-gain estimation for `f_next_best_step` may need implementation spike |
| Phase 6 | Human study protocol must be written in Phase 1 (PRE-02); no additional research needed at execution |

### Todos

- [ ] Select primary dataset and lock held-out split with hash (Phase 1, PRE-01)
- [ ] Write the three stopping criteria as code-ready specifications (Phase 1, PRE-02)
- [ ] Write human validation study protocol (N=5-10, within-subject, consented) — needed before Phase 6
- [ ] Decide LangGraph vs. plain Python before Phase 2 begins
- [ ] Decide sklearn HDBSCAN vs. standalone `hdbscan` package at Phase 1 gate

### Blockers

None.

---

## Session Continuity

**To resume:** Read ROADMAP.md for phase structure and success criteria. Read REQUIREMENTS.md for full requirement list with phase assignments. Check this STATE.md for current position, open decisions, and todos.

**Existing repo artifacts:**
- `Conversational Clustering Script.txt` — prototype/reference script (review before Phase 2)
- `Multi Agent Personalities Script.txt` — multi-agent persona reference (review before Phase 3)

**Critical constraint:** The held-out evaluation split must be locked (PRE-01) before any experiment code is written. Contamination is irreversible.

---

*State initialized: 2026-04-29*
