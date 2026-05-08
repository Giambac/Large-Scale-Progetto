# Roadmap: Conversational Clustering (Trio — updated)

**Project:** Conversational Clustering — Multi-Agent Human-in-the-Loop System
**Tier:** Trio
**Headline research questions:**
1. Does conversational refinement converge toward oracle-accepted clusterings, and how fast (turns, cognitive load)?
2. Do LLM-simulated oracles converge in patterns comparable to human oracles, or systematically differ?
**Deadline:** early June 2026
**Updated:** 2026-05-08

---

## Phases

- [x] **Phase 1: Pre-Code Obligations and Foundation** — COMPLETE
- [ ] **Phase 2: Clustering Agent Core** — Conversational loop, f_* functions, feedback types, hierarchy, web UI, multiple backends
- [ ] **Phase 3: Oracle Agent** — Configurable LLM oracle with noise, cognitive load, drift detection
- [ ] **Phase 4: Judge Agent** — Convergence detection, per-turn metrics, no-dialogue baseline, database
- [ ] **Phase 5: Ablation Harness and Strategies** — 3 strategies, N×M experiment runner, bootstrap CI
- [ ] **Phase 6: Generalization and Human Validation** — Mapping function, held-out evaluation, human study N≥10

---

## Phase Details

### ✅ Phase 1: Pre-Code Obligations and Foundation
**Status:** COMPLETE — 2026-05-05
**Delivered:**
- dataset/held_out.jsonl sealed with SHA-256 ea53dfa1
- dataset/train.jsonl (12,000 records)
- embeddings/embeddings.npy (12,000×768 float32, ~37 MB)
- src/state.py, src/clustering.py, src/cluster_naming.py
- src/embedding_store.py, src/data_loader.py, src/stopping.py, src/serialization.py
- scripts/setup_phase1.py — end-to-end pipeline working
- 41/41 tests green
- audit_log.jsonl generated with 9 clusters

---

### Phase 2: Clustering Agent Core
**Goal:** The conversational loop works end-to-end with all feedback types, the web UI is accessible, and multiple clustering backends are available
**Depends on:** Phase 1
**Target:** week 1–2

**v1 Requirements** (complete — 2026-05-07):
- ✅ CLUS-01: f_output always returns complete clustering, no partial states
- ✅ CLUS-02: f_uncertainty — ranked list of boundary points, split/merge candidates from soft probs
- ✅ CLUS-03: f_next_best_step — pluggable Strategy interface (RandomStrategy implemented); 30+ turn loop verified
- ✅ CLUS-04: f_next_state — applies oracle feedback, latest intent wins on contradictions
- ✅ FB-01: global feedback ("too many clusters", "focus on X")
- ✅ FB-02: cluster-level feedback (split, merge, too large)
- ✅ FB-03: point-level feedback (move item X, X and Y together)
- ✅ HIER-01: navigable cluster hierarchy
- ✅ HIER-02: hierarchy grows incrementally, not pre-computed upfront
- ✅ UI-01: web UI showing cluster assignments, soft probs, conversation history, per-turn metrics (partial — contradiction count and convergence signal deferred to Phase 4)
- ✅ UI-02: dataset upload via web UI

**Trio requirements** (pending):
- BACK-V2-01: k-means backend alongside HDBSCAN (LLM-first optional if time allows)
- VIZ-V2-01: UMAP/t-SNE 2D projection in web UI with color-coded cluster membership
- UI-V2-01: persistent sessions — state saved to disk and resumable across server restarts

**Success Criteria:**
1. f_output returns complete clustering assignment even mid-conversation ✅
2. f_uncertainty produces ranked boundary point list from calibrated soft probs ✅
3. f_next_best_step selects among show/ask/stop via Strategy interface; RandomStrategy runs 30+ turns without errors ✅
4. f_next_state applies all 4 feedback types; latest intent wins on contradictions ✅
5. Hierarchy is navigable and grows incrementally ✅
6. Web UI accessible during session with UMAP visualization, dataset upload, persistent sessions *(UMAP and persistence pending)*
7. k-means available as an alternative backend to HDBSCAN *(pending)*

---

### Phase 3: Oracle Agent
**Goal:** The LLM Oracle Agent behaves as a configurable, measurable stand-in for a human — with noise, cognitive fatigue, and drift — in a way that supports comparison with real humans in Phase 6
**Depends on:** Phase 2
**Target:** week 2

**v1 Requirements:**
- ORC-01: structured OracleReply objects, not bare text
- ORC-02: consistency_rate, drift_probability, sycophancy_resistance parameters produce measurably different behavior across runs
- ORC-03: per-turn cognitive-load score → visibly simpler replies above threshold
- ORC-04: drift detection — contradictions detected, logged to drift_history, surfaced to Clustering Agent
- FB-04: instructional feedback parsed into structured constraints

**Added for research question 2:**
- ORC-02 parameters must be logged per run so Phase 6 can correlate LLM parameter settings with observed human behavior patterns. Every experiment run records consistency_rate and drift_probability used.

**Success Criteria:**
1. OracleReply is a structured object driven by explicit preference spec and persona
2. Varying consistency_rate/drift_probability produces measurably different convergence curves
3. Cognitive load score visibly affects reply quality above threshold
4. Instructional feedback parsed into structured constraints applied on the next turn
5. Contradictions detected and logged to drift_history with reference to the conflicting prior turn

---

### Phase 4: Judge Agent
**Goal:** The system stops correctly under all three conditions and records per-turn metrics in the database — metrics that directly answer research question 1
**Depends on:** Phase 3
**Target:** week 2–3

**v1 Requirements:**
- DB-01: experiments table (strategy_id, persona_id, seed, dataset, total_turns, convergence_reason, headline metrics)
- DB-02: turns table (turn_index, action_type, cognitive_load_score, contradiction_count, convergence_signal)
- DB-03: oracle_feedback table (feedback_type, raw_text, parsed_delta, is_contradiction)
- JUDG-01: f_eval terminates loop on all 3 stopping criteria
- JUDG-02: per-turn metric bundle (turns-to-convergence counter, cognitive load, contradiction count, pairwise validation accuracy sample)
- JUDG-03: no-dialogue baseline — same metric bundle on initial clustering without any conversation

**Added for research question 1:**
- The no-dialogue baseline (JUDG-03) is critical — it is the comparison point that demonstrates conversation adds value over the default clustering. Every experiment must include a paired baseline run.

**Success Criteria:**
1. f_eval terminates loop on all 3 criteria; each triggerable independently by a synthetic oracle
2. Every turn appends metric bundle to log
3. No-dialogue baseline produces same metric format
4. Database writable and queryable for cross-run analysis
5. Compound oracle feedback produces multiple rows in oracle_feedback table

---

### Phase 5: Ablation Harness and Strategies
**Goal:** 3 comparable strategies with bootstrap CI on turns-to-convergence — direct answer to research question 1
**Depends on:** Phase 4
**Target:** week 3

**v1 Requirements:**
- ALAB-01: RandomStrategy, UncertaintyDrivenStrategy, BoundaryDrivenStrategy in the pluggable Strategy interface
- ALAB-02: experiment harness runs N strategies × M oracle personas × K seeds automatically; fully reproducible from config file and AuditLog alone
- ALAB-03: bootstrap 95% CI on turns-to-convergence computable with a single analysis script

**Trio requirements added (EXP-V2-01 partial):**
- Harness supports both LLM oracles (for scale) and human sessions (for Phase 6)
- Every run logged with strategy_id, persona_id, oracle_type (llm/human)

**Success Criteria:**
1. 3 strategies run through a complete conversation loop without modifying orchestrator code
2. Harness executes N×M×K automatically and produces reproducible results from config file
3. Bootstrap CI computable from AuditLog with a single script
4. oracle_type logged to separate LLM runs from human runs in downstream analysis

---

### Phase 6: Generalization and Human Validation
**Goal:** Oracle preferences are codified into a mapping function evaluated on held-out data, and LLM oracles are quantitatively compared with real humans — direct answer to research question 2
**Depends on:** Phase 5
**Target:** week 3–4

**v1 Requirements:**
- GEN-01: mapping function (prompt, classifier, or rule set) that assigns new items to clusters without oracle interaction
- GEN-02: mapping function evaluated on frozen held-out split; accuracy validated by oracle on 20–30 items

**Trio requirements added (EXP-V2-01 complete):**
- Human study: N≥10, within-subject design, randomized task order, written protocol, consented recording
- Humans run the same sessions as LLM oracles (same dataset, same tasks)
- Reported finding: quantified LLM-vs-human gap on turns-to-convergence and oracle satisfaction with CI

**Added for research question 2:**
- The LLM-vs-human comparison is the headline finding. Report: (a) are convergence patterns similar or systematically different? (b) which Oracle Agent parameter settings (consistency_rate, drift_probability) best approximate observed human behavior?

**Success Criteria:**
1. Mapping function produces reported accuracy on held-out split
2. Human study N≥10 completed with within-subject protocol
3. LLM-vs-human gap quantified with CI on at least one headline metric
4. Honest discussion of what the oracle signal can and cannot tell us

---

## Progress Table

| Phase | Status | Target week |
|-------|--------|-------------|
| 1. Pre-Code Obligations and Foundation | ✅ COMPLETE | — |
| 2. Clustering Agent Core | 🔄 In progress — v1 done, Trio reqs pending | 1–2 |
| 3. Oracle Agent | ⬜ Not started | 2 |
| 4. Judge Agent | ⬜ Not started | 2–3 |
| 5. Ablation Harness and Strategies | ⬜ Not started | 3 |
| 6. Generalization and Human Validation | ⬜ Not started | 3–4 |

---

## Requirements Coverage

| Requirement | Phase | Category |
|-------------|-------|----------|
| PRE-01 | Phase 1 ✅ | Pre-Code |
| PRE-02 | Phase 1 ✅ | Pre-Code |
| FOUND-01 | Phase 1 ✅ | Foundation |
| FOUND-02 | Phase 1 ✅ | Foundation |
| FOUND-03 | Phase 1 ✅ | Foundation |
| FOUND-04 | Phase 1 ✅ | Foundation |
| CLUS-01 | Phase 2 ✅ | Clustering Agent |
| CLUS-02 | Phase 2 ✅ | Clustering Agent |
| CLUS-03 | Phase 2 ✅ | Clustering Agent |
| CLUS-04 | Phase 2 ✅ | Clustering Agent |
| FB-01 | Phase 2 ✅ | Feedback |
| FB-02 | Phase 2 ✅ | Feedback |
| FB-03 | Phase 2 ✅ | Feedback |
| HIER-01 | Phase 2 ✅ | Hierarchy |
| HIER-02 | Phase 2 ✅ | Hierarchy |
| UI-01 | Phase 2 ✅ | Web UI |
| UI-02 | Phase 2 ✅ | Web UI |
| BACK-V2-01 | Phase 2 | Multiple backends |
| VIZ-V2-01 | Phase 2 | UMAP/t-SNE |
| UI-V2-01 | Phase 2 | Persistent sessions |
| ORC-01 | Phase 3 | Oracle Agent |
| ORC-02 | Phase 3 | Oracle Agent |
| ORC-03 | Phase 3 | Oracle Agent |
| ORC-04 | Phase 3 | Oracle Agent |
| FB-04 | Phase 3 | Feedback |
| DB-01 | Phase 4 | Database |
| DB-02 | Phase 4 | Database |
| DB-03 | Phase 4 | Database |
| JUDG-01 | Phase 4 | Judge Agent |
| JUDG-02 | Phase 4 | Judge Agent |
| JUDG-03 | Phase 4 | Judge Agent |
| ALAB-01 | Phase 5 | Ablation |
| ALAB-02 | Phase 5 | Ablation |
| ALAB-03 | Phase 5 | Ablation |
| EXP-V2-01 | Phase 5+6 | N×M experiment |
| GEN-01 | Phase 6 | Generalization |
| GEN-02 | Phase 6 | Generalization |

**Total v1 requirements:** 33
**Trio requirements added:** 4 (BACK-V2-01, VIZ-V2-01, UI-V2-01, EXP-V2-01)
**Total:** 37
**Mapped:** 37
**Unmapped:** 0

---

*Roadmap created: 2026-04-29*
*Updated: 2026-05-08 — replaced with Trio roadmap; Phase 2 v1 requirements marked complete; Trio reqs (BACK-V2-01, VIZ-V2-01, UI-V2-01) added as pending*
*Headline questions: Q1 (convergence speed) + Q2 (LLM vs human oracles)*
*Tier: Trio*
