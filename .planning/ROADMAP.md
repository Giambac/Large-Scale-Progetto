# Roadmap: Conversational Clustering (Trio — updated)

**Project:** Conversational Clustering — Multi-Agent Human-in-the-Loop System
**Tier:** Trio
**Headline research questions:**
1. Does conversational refinement converge toward oracle-accepted clusterings, and how fast (turns, cognitive load)?
2. Do LLM-simulated oracles converge in patterns comparable to human oracles, or systematically differ?
**Deadline:** early June 2026
**Updated:** 2026-05-21

---

## Phases

- [x] **Phase 1: Pre-Code Obligations and Foundation** — COMPLETE
- [x] **Phase 2: Clustering Agent Core** — Conversational loop, f_* functions, feedback types, hierarchy, web UI, multiple backends, UMAP projection, persistent sessions — COMPLETE 2026-05-10
- [x] **Phase 3: Oracle Agent** — Configurable LLM oracle with noise, cognitive load, drift detection — COMPLETE 2026-05-12
- [x] **Phase 4: Judge Agent** — Convergence detection, per-turn metrics, no-dialogue baseline, database — COMPLETE 2026-05-16
- [ ] **Phase 5: Ablation Harness and Strategies** — 3 strategies, N×M experiment runner, bootstrap CI
- [ ] **Phase 6: Generalization and Human Validation** — Mapping function, held-out evaluation, human study N≥10

### Milestone v2.0 — Experimentation Flexibility & Scale (Phases 7–12)

- [ ] **Phase 7: Pluggable Embedding Backends + Dynamic Dimension** — Replace hardcoded EMBEDDING_DIM, EmbeddingBackend Protocol (ST + OpenAI), provenance manifest + content-hash cache [FOUNDATIONAL]
- [ ] **Phase 8: Colab Compute-Only Artifact Pipeline** — Colab notebook computes embeddings + initial clustering, HF Hub handoff, manifest-validated local import
- [ ] **Phase 9: Interactive UMAP Recolor + Real-Time Chat** — Cache-once/recolor projection, end-to-end human chat, live LLM-oracle transcript
- [ ] **Phase 10: Oracle-Initiated Flow + KMeans-Only + Oracle YAML Config** — Dataset intro → oracle first query → first fit; drop HDBSCAN; selectable oracle model + versioned YAML config
- [ ] **Phase 11: Re-Fit KMeans Per Query + Cluster-ID Alignment + Query Filter** — Per-query re-fit on fixed embeddings, Hungarian ID alignment, NL query normalization [ENGINEERING CORE]
- [ ] **Phase 12: Coordination Agent** — Decompose complex ops into pairwise sub-operations across N clusterer sessions, recombine into one authoritative state [LAST — research spike]

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

### ✅ Phase 2: Clustering Agent Core — COMPLETE 2026-05-10
**Goal:** The conversational loop works end-to-end with all feedback types, the web UI is accessible, and multiple clustering backends are available
**Depends on:** Phase 1
**Completed:** 2026-05-10 — all 8 plans done

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

**Trio requirements** (all complete):
- ✅ BACK-V2-01: k-means backend alongside HDBSCAN — ClusteringBackend Protocol, HDBSCANBackend, KMeansBackend, --backend CLI flag (plan 06 complete)
- ✅ VIZ-V2-01: UMAP 2D projection in web UI with color-coded cluster membership — server-side UMAP, projection_update SocketIO event, canvas scatter plot (plan 07 complete)
- ✅ UI-V2-01: persistent sessions — timestamped session dirs, per-turn state.json snapshots, GET /sessions, POST /resume, Sessions section in sidebar (plan 08 complete)

**Success Criteria:**
1. f_output returns complete clustering assignment even mid-conversation ✅
2. f_uncertainty produces ranked boundary point list from calibrated soft probs ✅
3. f_next_best_step selects among show/ask/stop via Strategy interface; RandomStrategy runs 30+ turns without errors ✅
4. f_next_state applies all 4 feedback types; latest intent wins on contradictions ✅
5. Hierarchy is navigable and grows incrementally ✅
6. Web UI accessible during session with UMAP visualization, dataset upload, persistent sessions ✅
7. k-means available as an alternative backend to HDBSCAN ✅

---

### ✅ Phase 3: Oracle Agent — COMPLETE 2026-05-12
**Goal:** The LLM Oracle Agent behaves as a configurable, measurable stand-in for a human — with noise, cognitive fatigue, and drift — in a way that supports comparison with real humans in Phase 6
**Depends on:** Phase 2
**Completed:** 2026-05-12 — all 5 plans done

**Delivered:**
- src/oracle_agent.py — OracleAgent, OracleSpec, NoiseParams; satisfies OracleProtocol via structural subtyping
- src/cognitive_load.py — f_cognitive_load pure function; COG_LOAD_THRESHOLD=0.7; OVERLOAD injected into prompt above threshold
- Drift detection — update_delta_window() + _contradicts(); deque(maxlen=10) delta window; contradiction_detected/contradicted_turn on OracleReply
- Loop integration — cognitive_load forwarded from run_conversation() into oracle.reply(); global_instructions (FB-04) forwarded each turn; oracle_init + drift_event to events.jsonl sidecar
- 21 Phase 3 tests green (106 total including Phase 2)

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

**Web stack migration (2026-05-15):** Flask + Flask-SocketIO replaced by FastAPI + python-socketio (ASGI). UI-01 and UI-02 requirement wording unchanged; only the underlying framework switched. See `.planning/fastAPI_plan.md`.

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

**Plans:** 1/6 plans executed
Plans:
- [ ] 04-01-PLAN.md — DB foundation: src/db/ subpackage, schema init, Pydantic triples (DB-01/02/03)
- [x] 04-02-PLAN.md — Stopping criteria fill-in, src/logging_setup.py, OracleProtocol signature update (JUDG-01)
- [ ] 04-03-PLAN.md — src/judge.py: PairBag, pairwise accuracy, turn metrics, run_baseline (JUDG-02/03)
- [ ] 04-04-PLAN.md — Loop wiring: DB writes per turn, isinstance branch removal, state_update extension (DB-01/02/03 integration)
- [ ] 04-05-PLAN.md — Tests: DB layer, judge, run_baseline, STRICT_MODE (all req IDs)
- [ ] 04-06-PLAN.md — CLI wrapper, UI panels, docs/MODEL.md, CLAUDE.md, README.md (JUDG-03 surface)

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


**Plans:** 2/5 plans executed
Plans:
- [x] 05-01-PLAN.md — DB schema: oracle_type column (EXP-V2-01)
- [x] 05-02-PLAN.md — _format_message enrichment for targeted payloads (ALAB-01 support)
- [ ] 05-03-PLAN.md — UncertaintyDrivenStrategy + BoundaryDrivenStrategy (ALAB-01)
- [ ] 05-04-PLAN.md — N×M×K harness + STRATEGY_REGISTRY + harness.yaml (ALAB-02, EXP-V2-01)
- [ ] 05-05-PLAN.md — Bootstrap CI: src/analysis.py + compute_ci.py + notebooks/analysis.ipynb (ALAB-03)

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

**Plans:** 5 plans
Plans:
- [ ] 06-01-PLAN.md — src/mapping.py: OracleRuleSet, MappingProtocol, LLM + Centroid strategies (GEN-01)
- [ ] 06-02-PLAN.md — examples/evaluate_mapping.py: held-out evaluation, both strategies, bootstrap CI (GEN-02)
- [ ] 06-03-PLAN.md — Human study UI: /study route, study.html, satisfaction detection, DB writes (EXP-V2-01)
- [ ] 06-04-PLAN.md — tests/test_mapping.py + tests/test_study_ui.py (GEN-01, GEN-02, EXP-V2-01)
- [ ] 06-05-PLAN.md — LLM-vs-human analysis: examples/compare_oracle_types.py + notebooks/llm_vs_human.ipynb (EXP-V2-01)

---

## Milestone v2.0 — Experimentation Flexibility & Scale

**Defined:** 2026-05-21
**Goal:** Make the system configurable and scalable for research — Colab compute offload, swappable embedding/clustering backends, oracle-initiated flow, query filter, and a coordination agent.
**Locked decisions:** re-cluster (re-fit KMeans), not re-embed, per query · Colab is compute-only · oracle configs in versioned YAML · coordination agent built last.
**Build order:** dependency-locked (per research/SUMMARY.md "Implications for Roadmap"). Two foundational refactors (dynamic dim, provenance manifest) gate everything embedding-related; the coordination agent is last.

### Phase 7: Pluggable Embedding Backends + Dynamic Dimension
**Goal:** Embedding dimension is learned from data (no hardcoded constant) and the system can embed with either a local SentenceTransformer (384) or OpenAI `text-embedding-3-small` (1536), chosen per run, with provenance-validated, content-hash-cached artifacts. [FOUNDATIONAL — unblocks all embedding-related work]
**Depends on:** Phase 6 (v1 system shipped)
**Requirements:** EMB-V2-01, EMB-V2-02, EMB-V2-03
**Success Criteria** (what must be TRUE):
  1. A run can select either the SentenceTransformer or OpenAI embedding backend and the system embeds the dataset correctly with either, deriving the dimension (384 or 1536) from the chosen backend — no `EMBEDDING_DIM = 384` literal remains, and fail-loudly asserts now check the dynamic expected dimension.
  2. Re-running an identical (dataset, model) embedding request reuses the content-hash cache instead of re-embedding (no second OpenAI charge), verifiable from logs.
  3. Loading an embedding artifact whose manifest (model, dim, normalized flag, library versions, input hash, n_items) mismatches the expected values crashes loudly rather than silently proceeding.
  4. Every produced embedding vector passes a unit-norm assertion at the backend boundary (normalize-at-boundary contract).
**Plans:** TBD
**Notes:** `EMBEDDING_DIM` is asserted in 4+ sites in `embedding_store.py` and 3 cache-shape sites in `web/app.py`; a stale 1536-dim cache could be silently reused as 384 until this lands. The provenance/manifest schema defined here is the shared contract consumed by Phase 8 (Colab artifacts). Clean up pre-existing docstring drift (docstrings say 768, constant is 384) and bump stale `requirements.txt` floors. Standard pattern (Protocol mirrors existing `ClusteringBackend`) — likely no research spike needed.

### Phase 8: Colab Compute-Only Artifact Pipeline
**Goal:** Heavy embedding + initial-clustering compute is offloaded to a Colab GPU notebook that exports a validated artifact bundle; the local app imports it via HuggingFace Hub and runs the interactive UI locally, importing zero Colab-only dependencies. [Colab is compute-ONLY]
**Depends on:** Phase 7 (manifest/dim schema must exist first)
**Requirements:** COL-V2-01, COL-V2-02
**Success Criteria** (what must be TRUE):
  1. A Colab notebook computes embeddings + the initial clustering and exports a bundle (embeddings.npy, initial_state.json, manifest.json) recording `embedding_dim` and full provenance.
  2. The local app downloads that bundle from HuggingFace Hub and loads it after validating it against the manifest (dtype float32, shape (N, dim), model/dim/lib-version match, input hash, `n_items` count).
  3. A manifest mismatch or truncated/short artifact causes a loud failure on load rather than a silent partial import.
  4. The local runtime imports zero Colab-only dependencies (the boundary is a directory of files, not shared code).
**Plans:** TBD
**Notes:** Local torch is CPU-only (`2.11.0+cpu`) — the concrete motivation for GPU offload. Use a `requirements-colab.txt` mirroring local versions; pin sklearn `n_init`/seed so Colab and local produce matching initial clusterings. Write artifacts atomically (temp → fsync → rename).

### Phase 9: Interactive UMAP Recolor + Real-Time Chat
**Goal:** The UMAP projection caches its 2D coordinates once (coords are a fixed function of the immutable embeddings) and recolors on any cluster change instead of refitting; human chat works end-to-end in the study UI; and the LLM-oracle conversation is viewable in real time.
**Depends on:** Phase 8
**Requirements:** VIZ-V2-02, UX-V2-01, UX-V2-02
**Success Criteria** (what must be TRUE):
  1. The UMAP scatter computes 2D coordinates once per session and, on a cluster change, re-emits only a recolor payload (points keep their positions; no per-turn refit / no layout jitter).
  2. A human can hold a complete clustering conversation end-to-end in the study UI (send a message, see the system reply and updated clustering, continue to convergence).
  3. An observer can watch the LLM-oracle conversation transcript update live, turn by turn, without reloading.
**Plans:** TBD
**UI hint**: yes
**Notes (CROSS-PHASE COUPLING):** The recolor MECHANISM is built here, but recolor is only fully CORRECT once the Phase 11 cluster-ID alignment lands — recoloring with churned KMeans IDs paints points the wrong colors. Build and ship the recolor mechanism in this phase, but DEFER final recolor-correctness validation until Phase 11's alignment layer exists (the planner must sequence the recolor visual-correctness check after Phase 11). Chat view is a presentation layer over the existing event stream — never a parser-bypass side channel.

### Phase 10: Oracle-Initiated Flow + KMeans-Only + Oracle YAML Config
**Goal:** The loop is reframed to be oracle-initiated — dataset introduction → oracle's first query → first clustering → conversational session — running on KMeans only (HDBSCAN dropped from the interactive path), with the oracle's LLM model selectable per run and its prompt/persona/noise config stored in versioned YAML.
**Depends on:** Phase 9
**Requirements:** CLUST-V2-01, FLOW-V2-01, FLOW-V2-02, OCFG-V2-01, OCFG-V2-02
**Success Criteria** (what must be TRUE):
  1. On session start the user/oracle sees a dataset introduction/summary BEFORE any clustering runs.
  2. The oracle issues an initial query that triggers the first clustering, after which the conversational session begins; "autonomous-first" clustering is retained as a selectable ablation condition.
  3. The interactive path uses KMeans only — HDBSCAN is no longer used for interactive clustering.
  4. A run can select the oracle's LLM model (Anthropic or OpenAI), and the oracle's prompt/persona/noise configuration loads from a versioned YAML file, validated into a typed config object (fail-loudly on malformed config).
**Plans:** TBD
**UI hint**: yes
**Notes:** Bootstrap via `build_initial_clustering_state(defer=True)` placeholder → intro → first query → first fit. Intro should be written to an `events.jsonl` sidecar, NOT an audit state line (intro-turn audit semantics: sidecar, not turn 0/-1). Re-fit semantics are KMeans-specific (HDBSCAN has no fixed-K re-fit), so KMeans-only must settle here before Phase 11. Document a default for an empty/degenerate first query and assert-before-first-clustering. Oracle config: `ruamel.yaml` for round-trip + `pydantic` typed validation; configs live in git, never `experiments.db`. Orchestration-around-existing-loop — likely no research spike.

### Phase 11: Re-Fit KMeans Per Query + Cluster-ID Alignment + Query Filter
**Goal:** Each oracle query re-fits KMeans on the fixed embeddings (no re-embedding), a cluster-ID alignment step preserves cluster IDs/names across re-fits, K never auto-reoptimizes (changes only via explicit oracle split/merge intent), and a query filter normalizes oracle natural language into simple, contradiction-free clusterer instructions. [THE engineering core]
**Depends on:** Phase 10 (oracle-initiated first fit) and Phase 9 (recolor mechanism)
**Requirements:** CLUST-V2-02, FILT-V2-01
**Success Criteria** (what must be TRUE):
  1. An oracle query re-fits KMeans on `store.get_all()` (embeddings stay fixed — the read-only EmbeddingStore invariant is preserved; no re-embedding occurs).
  2. After a re-fit, clusters that correspond to prior clusters keep their IDs and names (Hungarian alignment on item-set overlap / centroid cosine); new IDs are minted only when K genuinely increased via oracle intent.
  3. K stays constant across a re-fit unless the oracle's feedback was a split/merge — an assertion enforces `new_k == prev_k` otherwise; BIC K-selection runs only at the first clustering.
  4. The query filter normalizes a natural-language oracle query into deduped, latest-intent-wins, contradiction-free clusterer instructions (extending `feedback_parser`/`_contradicts`) WITHOUT semantically second-guessing oracle intent.
  5. The same query against the same state produces identical re-fit results (determinism test passes).
**Plans:** TBD
**Notes (CROSS-PHASE COUPLING):** This alignment layer is the linchpin that makes the Phase 9 UMAP recolor correct — once it lands, validate that recolor shows stable colors across re-fits. Open design decisions to settle in planning (FLAGGED for a focused research pass): (a) cluster-ID reconciliation policy — preserve-stable-IDs vs. renumber; (b) hierarchy lineage on a wholesale re-fit — `record_split/merge` assume incremental edits, so decide whether to reset the hierarchy on re-fit or skip hierarchy recording for re-fit turns. The filter normalizes structure only; the feedback/contradiction layer arbitrates intent (avoid double-handling).

### Phase 12: Coordination Agent
**Goal:** A coordination agent decomposes a complex clustering operation into pairwise sub-operations, fans them out to N clusterer sessions, and recombines the results into a single authoritative `ClusteringState` — preserving the single-source-of-truth / single-writer / single-replay-log invariants, with whole-operation abort on partial failure. [LAST — highest risk]
**Depends on:** Phase 11 (re-fit + alignment stable)
**Requirements:** COORD-V2-01, COORD-V2-02
**Success Criteria** (what must be TRUE):
  1. A complex clustering operation is decomposed into pairwise sub-operations, each runnable as an independent clusterer session.
  2. Pairwise sub-operations fan out to N clusterer sessions (running in the existing no-I/O mode: `db_conn=None`, `socketio=None`) and recombine into ONE authoritative `ClusteringState`, with exactly one audit line per merged turn.
  3. On any sub-operation failure the WHOLE operation aborts (no partial merge) and recovery falls back to the last good audit-log state.
**Plans:** TBD
**Notes (RESEARCH SPIKE FLAGGED):** All three technical researchers flagged this for its own `/gsd-research-phase` spike — cross-session state-merge semantics, contradiction recombination across N sessions, partial-failure rollback, and BLAS/thread-pool contention under N concurrent KMeans fits are unresolved DESIGN problems, not mechanics. Sub-sessions run via `threading.Thread` (mind BLAS oversubscription); NEVER reach for eventlet/gevent. Pure `merge(authoritative, [sub_results])` reduce applied sequentially through the single-writer pipeline.

---

## Progress Table

| Phase | Status | Target week |
|-------|--------|-------------|
| 1. Pre-Code Obligations and Foundation | ✅ COMPLETE | — |
| 2. Clustering Agent Core | ✅ COMPLETE — v1 + BACK-V2-01 + VIZ-V2-01 + UI-V2-01 all done (2026-05-10) | 1–2 |
| 3. Oracle Agent | ✅ COMPLETE — ORC-01..04 + FB-04 all done (2026-05-12) | 2 |
| 4. Judge Agent | ⬜ Not started | 2–3 |
| 5. Ablation Harness and Strategies | ⬜ Not started | 3 |
| 6. Generalization and Human Validation | ⬜ Not started | 3–4 |

### Milestone v2.0 Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 7. Pluggable Embedding Backends + Dynamic Dimension | 0/? | Not started | - |
| 8. Colab Compute-Only Artifact Pipeline | 0/? | Not started | - |
| 9. Interactive UMAP Recolor + Real-Time Chat | 0/? | Not started | - |
| 10. Oracle-Initiated Flow + KMeans-Only + Oracle YAML Config | 0/? | Not started | - |
| 11. Re-Fit KMeans Per Query + Cluster-ID Alignment + Query Filter | 0/? | Not started | - |
| 12. Coordination Agent | 0/? | Not started | - |

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
| BACK-V2-01 | Phase 2 ✅ | Multiple backends |
| VIZ-V2-01 | Phase 2 ✅ | UMAP/t-SNE |
| UI-V2-01 | Phase 2 ✅ | Persistent sessions |
| ORC-01 | Phase 3 ✅ | Oracle Agent |
| ORC-02 | Phase 3 ✅ | Oracle Agent |
| ORC-03 | Phase 3 ✅ | Oracle Agent |
| ORC-04 | Phase 3 ✅ | Oracle Agent |
| FB-04 | Phase 3 ✅ | Feedback |
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

### Milestone v2.0 Requirements Coverage

| Requirement | Phase | Category |
|-------------|-------|----------|
| EMB-V2-01 | Phase 7 | Embedding & Compute |
| EMB-V2-02 | Phase 7 | Embedding & Compute |
| EMB-V2-03 | Phase 7 | Embedding & Compute |
| COL-V2-01 | Phase 8 | Compute Offload (Colab) |
| COL-V2-02 | Phase 8 | Compute Offload (Colab) |
| VIZ-V2-02 | Phase 9 | Visualization & Chat |
| UX-V2-01 | Phase 9 | Visualization & Chat |
| UX-V2-02 | Phase 9 | Visualization & Chat |
| CLUST-V2-01 | Phase 10 | Clustering |
| FLOW-V2-01 | Phase 10 | Flow & Onboarding |
| FLOW-V2-02 | Phase 10 | Flow & Onboarding |
| OCFG-V2-01 | Phase 10 | Oracle Configurability |
| OCFG-V2-02 | Phase 10 | Oracle Configurability |
| CLUST-V2-02 | Phase 11 | Clustering |
| FILT-V2-01 | Phase 11 | Query Filter |
| COORD-V2-01 | Phase 12 | Coordination Agent |
| COORD-V2-02 | Phase 12 | Coordination Agent |

**Total v2.0 requirements:** 17
**Mapped:** 17
**Unmapped:** 0

---

*Roadmap created: 2026-04-29*
*Updated: 2026-05-08 — replaced with Trio roadmap; Phase 2 v1 requirements marked complete; Trio reqs (BACK-V2-01, VIZ-V2-01, UI-V2-01) added as pending*
*Updated: 2026-05-21 — appended Milestone v2.0 (Phases 7–12, 17 requirements across 7 categories); v1 phases 1–6 preserved unchanged; dependency-locked build order per research/SUMMARY.md*
*Headline questions: Q1 (convergence speed) + Q2 (LLM vs human oracles)*
*Tier: Trio*
