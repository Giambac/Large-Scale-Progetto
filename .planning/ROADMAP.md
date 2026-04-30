# Roadmap: Conversational Clustering

**Project:** Conversational Clustering — Multi-Agent Human-in-the-Loop System
**Created:** 2026-04-29
**Granularity:** Standard (5-8 phases)
**Coverage:** 29/29 v1 requirements mapped

---

## Phases

- [ ] **Phase 1: Pre-Code Obligations and Foundation** - Lock held-out data, operationalize stopping criteria, build embedding infrastructure and initial clustering
- [ ] **Phase 2: Clustering Agent Core** - Implement all `f_*` functions, feedback types, and hierarchy
- [ ] **Phase 3: Oracle Agent** - Build LLM-simulated oracle with noise parameters, cognitive-load modeling, and drift detection
- [ ] **Phase 4: Judge Agent** - Implement convergence detection, per-turn metrics, and no-dialogue baseline
- [ ] **Phase 5: Ablation Harness and Strategies** - Add non-random strategies, multi-run experiment runner, and bootstrap confidence intervals
- [ ] **Phase 6: Generalization and Human Validation** - Codify oracle preferences, evaluate on held-out set, run human validation study

---

## Phase Details

### Phase 1: Pre-Code Obligations and Foundation
**Goal**: The project's irreversible data and evaluation decisions are locked, and the embedding + state infrastructure is operational
**Depends on**: Nothing (first phase)
**Requirements**: PRE-01, PRE-02, FOUND-01, FOUND-02, FOUND-03, FOUND-04
**Success Criteria** (what must be TRUE):
  1. A frozen held-out dataset split exists with a locked hash file that will never be modified before experiments run
  2. The three stopping conditions (turn budget, state-change diminishing returns, explicit oracle satisfaction token) are written as code-ready specifications, not prose
  3. Running the pipeline on the selected dataset produces a `ClusteringState` with HDBSCAN cluster labels, LLM-generated names/descriptions, and per-point soft assignment probability distributions
  4. Any turn's `ClusteringState` can be serialized to JSONL and deserialized back without data loss
**Plans**: TBD

### Phase 2: Clustering Agent Core
**Goal**: The Clustering Agent's pure functions operate correctly on real state and the full range of oracle feedback types is parsed and applied
**Depends on**: Phase 1
**Requirements**: CLUS-01, CLUS-02, CLUS-03, CLUS-04, FB-01, FB-02, FB-03, HIER-01, HIER-02, UI-01
**Success Criteria** (what must be TRUE):
  1. `f_output` always returns a complete clustering assignment with no partial states, even mid-conversation
  2. `f_uncertainty` produces a ranked list of high-entropy boundary points, split candidates, and merge candidates derived from calibrated soft assignments
  3. `f_next_best_step` selects among show/ask/stop actions via a pluggable Strategy interface (RandomStrategy implemented); a mocked oracle loop runs correctly for 30+ turns with state integrity verified
  4. `f_next_state` applies global, cluster-level, and point-level oracle feedback and updates clustering state so that latest oracle intent wins on any contradiction
  5. The cluster hierarchy is navigable and grows incrementally as oracle feedback arrives — not pre-computed in one shot
  6. A web-based debug UI is accessible during a running session, showing current cluster assignments with soft probabilities, turn-by-turn conversation history, and per-turn metrics (cognitive-load, contradiction count, convergence signal)
**Plans**: TBD

### Phase 3: Oracle Agent
**Goal**: The LLM-simulated oracle behaves as a configurable, realistic stand-in for a human — with noise, cognitive fatigue, and preference drift
**Depends on**: Phase 2
**Requirements**: ORC-01, ORC-02, ORC-03, ORC-04, FB-04
**Success Criteria** (what must be TRUE):
  1. The Oracle Agent issues structured `OracleReply` objects (not bare text) driven by an explicit preference specification and persona, not a bare LLM call
  2. Changing `consistency_rate`, `drift_probability`, or `sycophancy_resistance` parameters produces measurably different behavior across simulation runs
  3. The Oracle Agent receives a pre-computed cognitive-load score each turn and produces visibly simpler or more superficial replies when cumulative load exceeds the configured threshold
  4. Instructional feedback ("treat X and Y as synonyms", "prioritize feature F") is parsed into structured constraints and applied to the next clustering state update
  5. Preference contradictions with prior intents are detected, logged to `drift_history`, and surfaced to the Clustering Agent
**Plans**: TBD

### Phase 4: Judge Agent
**Goal**: The system has a complete stopping mechanism that terminates conversations correctly under all three convergence conditions and records per-turn evaluation metrics
**Depends on**: Phase 3
**Requirements**: JUDG-01, JUDG-02, JUDG-03
**Success Criteria** (what must be TRUE):
  1. `f_eval` terminates the turn loop when any of the three stopping criteria fires (turn budget exhausted, feedback magnitude decay below threshold, explicit oracle satisfaction signal) and each can be triggered independently by a synthetic oracle
  2. Every turn appends a metric bundle to the AuditLog containing: turns-to-convergence counter, cognitive-load score, contradiction count, and a pairwise validation accuracy sample
  3. Running the system with no dialogue (one-shot initial clustering, no oracle turns) produces the same metric bundle, enabling isolation of dialogue contribution in downstream analysis
**Plans**: TBD

### Phase 5: Ablation Harness and Strategies
**Goal**: Non-random interaction strategies are implemented and a multi-run experiment harness produces reproducible, strategy-attributed results with confidence intervals
**Depends on**: Phase 4
**Requirements**: ALAB-01, ALAB-02, ALAB-03
**Success Criteria** (what must be TRUE):
  1. At least three strategies (RandomStrategy, UncertaintyDrivenStrategy, BoundaryDrivenStrategy) are registered in the pluggable Strategy interface and run through a complete conversation loop without modification to orchestrator code
  2. The experiment harness automatically executes N strategies x M oracle personas x K seeds, logs every run to AuditLog with `strategy_id` and `persona_id` fields, and is fully reproducible from a config file and the AuditLog alone
  3. Bootstrap 95% confidence intervals on the headline metric (turns-to-convergence or information-gain-per-cognitive-load ratio) are computable from the AuditLog with a single analysis script
**Plans**: TBD

### Phase 6: Generalization and Human Validation
**Goal**: Oracle preferences are codified into a transferable function evaluated on held-out data, and LLM-simulated oracle results are validated against real human oracles
**Depends on**: Phase 5
**Requirements**: GEN-01, GEN-02
**Success Criteria** (what must be TRUE):
  1. After an oracle-accepted clustering session, the system produces a reusable mapping function (prompt, classifier, or rule set) that assigns new items to clusters without oracle interaction
  2. The generalization function is evaluated on the frozen held-out split and the oracle validates a small sample (20-30 items), producing a reported accuracy figure
**Plans**: TBD
**Note**: Human validation study (N = 5-10, within-subject protocol) is a project constraint defined in PRE-02 and executed in this phase against the pre-written protocol. Results must quantify the simulated-vs-human gap as a reported finding.

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pre-Code Obligations and Foundation | 0/? | Not started | - |
| 2. Clustering Agent Core | 0/? | Not started | - |
| 3. Oracle Agent | 0/? | Not started | - |
| 4. Judge Agent | 0/? | Not started | - |
| 5. Ablation Harness and Strategies | 0/? | Not started | - |
| 6. Generalization and Human Validation | 0/? | Not started | - |

---

## Coverage Validation

| Requirement | Phase | Category |
|-------------|-------|----------|
| PRE-01 | Phase 1 | Pre-Code Obligations |
| PRE-02 | Phase 1 | Pre-Code Obligations |
| FOUND-01 | Phase 1 | Foundation |
| FOUND-02 | Phase 1 | Foundation |
| FOUND-03 | Phase 1 | Foundation |
| FOUND-04 | Phase 1 | Foundation |
| CLUS-01 | Phase 2 | Clustering Agent |
| CLUS-02 | Phase 2 | Clustering Agent |
| CLUS-03 | Phase 2 | Clustering Agent |
| CLUS-04 | Phase 2 | Clustering Agent |
| FB-01 | Phase 2 | Feedback Types |
| FB-02 | Phase 2 | Feedback Types |
| FB-03 | Phase 2 | Feedback Types |
| HIER-01 | Phase 2 | Hierarchy |
| HIER-02 | Phase 2 | Hierarchy |
| UI-01 | Phase 2 | Debug UI |
| ORC-01 | Phase 3 | Oracle Agent |
| ORC-02 | Phase 3 | Oracle Agent |
| ORC-03 | Phase 3 | Oracle Agent |
| ORC-04 | Phase 3 | Oracle Agent |
| FB-04 | Phase 3 | Feedback Types |
| JUDG-01 | Phase 4 | Judge Agent |
| JUDG-02 | Phase 4 | Judge Agent |
| JUDG-03 | Phase 4 | Judge Agent |
| ALAB-01 | Phase 5 | Ablation |
| ALAB-02 | Phase 5 | Ablation |
| ALAB-03 | Phase 5 | Ablation |
| GEN-01 | Phase 6 | Generalization |
| GEN-02 | Phase 6 | Generalization |

**Total v1 requirements:** 29
**Mapped:** 29
**Unmapped:** 0

---

*Roadmap created: 2026-04-29*
*Last updated: 2026-04-30 — added UI-01 (debug web UI) to Phase 2*
