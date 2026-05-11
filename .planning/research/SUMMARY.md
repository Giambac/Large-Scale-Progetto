# Research Summary: Conversational Clustering

**Project:** Conversational Clustering — Multi-Agent Human-in-the-Loop System
**Synthesized:** 2026-04-29
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md
**Overall confidence:** HIGH for core system design; MEDIUM for oracle cognitive-load modeling and LLM sycophancy mitigations

---

## Executive Summary

This project builds a three-agent system (Clustering Agent, Oracle Agent, Judge Agent) in which a simulated human oracle iteratively refines text cluster proposals through natural-language conversation, and the system learns to extract codified preferences efficiently. The research contribution is not a better clustering algorithm but a better interaction strategy: the headline question is whether a guided `f_next_best_step` policy (uncertainty-driven, information-gain-driven, or hybrid) reaches oracle-accepted clusterings faster and at lower cognitive cost than a random baseline. The literature on interactive clustering (COBRAS, COBRA, Prodigy) confirms this is a well-scoped, publishable research question — but only if turn-efficiency and cognitive load are primary metrics from day one, not derivations added at evaluation time.

The critical risk is not technical: the core stack (LangGraph + sentence-transformers + HDBSCAN + structured Python state) is well-understood and stable. The risks are methodological. LLM-simulated oracles are systematically over-consistent, over-cooperative, and overconfident in ways that produce optimistic convergence results that do not transfer to real humans. Evaluation that measures final clustering quality (NMI, silhouette) rather than turn efficiency cannot isolate what the dialogue contributes. Soft assignments from HDBSCAN are uncalibrated, making `f_uncertainty` unreliable if raw probabilities are used without temperature scaling. All three of these risks are pre-empted by design decisions made before the first agent is coded.

The recommended approach is to build in strict dependency order — data structures and embedding infrastructure first, then agent logic one agent at a time with gates between phases, then ablation harness — while treating evaluation design, oracle noise parameterization, state serialization, and the held-out split lock as Phase 0 obligations that cannot be deferred.

---

## 1. Recommended Stack

The definitive technology choices with rationale.

| Layer | Choice | Version | Rationale |
|-------|--------|---------|-----------|
| Agent orchestration | LangGraph | ~=1.1.10 | Only framework where stateful graph + HITL interrupts + durable checkpointing are first-class, not workarounds. Reached 1.0 stability in late 2025. |
| LLM abstraction | LangChain Core + langchain-openai | ~=0.3.x | Model-agnostic wrappers; swapping gpt-4o-mini for gpt-4o (or Anthropic) for ablations requires changing one config value. |
| LLM provider | OpenAI API | gpt-4o-mini (Oracle, Judge), gpt-4o (Clustering Agent) | gpt-4o-mini at $0.15/M tokens enables hundreds of ablation runs without budget pressure; gpt-4o for proposal quality. |
| Sentence embeddings | sentence-transformers, model: all-mpnet-base-v2 | ~=5.3.0 | Local inference, fully deterministic, reproducible across runs. No API cost. all-mpnet-base-v2 outperforms all-MiniLM-L6-v2 by 3-4% on MTEB clustering benchmarks; speed is not a bottleneck. |
| Clustering algorithm | scikit-learn HDBSCAN | >=1.6 | Native in sklearn since 1.3; provides `probabilities_` for soft assignments; automatically determines cluster count; outputs hierarchy. Use standalone `hdbscan` package only if full multinomial probability vectors are required (evaluate at Phase 1). |
| State schema | Python dataclasses + Pydantic v2 | stdlib / 2.x | TypedDict/dataclass for LangGraph state; Pydantic for structured LLM output parsing and oracle reply validation. |
| Experiment tracking | JSON-first, migrate to MLflow at Phase 5 | mlflow >=2.19 | JSON is sufficient for early phases. MLflow adds run comparison and parameter search when ablation experiments begin (20+ runs). Design JSON schema to be MLflow-compatible from day one. |
| CLI / interface | Typer + Rich | typer >=0.12, rich >=13 | Typer generates `--help` from type hints; Rich panels and tables are the right display primitive for "show cluster proposal, ask oracle" turns. Typer + Rich is the 2025-2026 standard for Python research CLI tooling. |
| Supporting libraries | numpy, pandas, scipy, pytest, python-dotenv, datasets (HuggingFace) | see STACK.md | Core numeric substrate, dataset loading, cosine distance, unit testing, API key management. |

**Python version:** 3.11 (maximum compatibility) or 3.12. All pinned packages support both.

**Non-negotiable stack decisions:**
1. LangGraph is the single state authority. All agent communication passes through the shared TypedDict state object — never direct agent-to-agent calls.
2. Embeddings are computed once at startup and stored in a read-only `EmbeddingStore`. Never re-embedded per turn. Never inlined in `ClusteringState`.
3. JSON-first logging from Phase 1. MLflow from Phase 5 (ablation harness). Schema must be compatible across the migration.

---

## 2. Table Stakes Features

Features whose absence makes the core research question unanswerable.

| Feature | Why It Is Table Stakes |
|---------|----------------------|
| Initial clustering with LLM-generated names and descriptions | Oracle has nothing to react to without it; anchor for the first turn |
| Soft assignments (per-point probability distribution over K clusters) | Required for calibration, boundary detection, and `f_uncertainty`; hard labels discard the information the system needs |
| `f_uncertainty`: ranked list of high-entropy boundary points, split candidates, merge candidates | Prerequisite for any non-random query strategy; broken if soft assignments are uncalibrated |
| `f_next_best_step`: show / ask / stop decision with pluggable strategy interface | The core algorithmic contribution; must be a Strategy interface, not a switch statement, for ablation to work |
| Global, cluster-level, point-level, and instructional feedback parsing | Covers the full range of oracle actions; point-level and instructional are higher complexity but needed for the paper's feedback taxonomy |
| State persistence: typed `ClusteringState` serialized to disk after every turn | Without this, sessions cannot be reproduced, ablations cannot be crossed, and contradiction handling is unreliable after 20+ turns |
| Contradiction / preference-drift tracking with active use in `f_next_best_step` | Explicitly required by research question; must be an active signal (triggers clarification question), not a passive log |
| Stopping signal: three independently operationalized criteria (turn budget, diminishing returns on state change, structured oracle satisfaction token) | Without an explicit stopping contract, Judge Agent and Oracle Agent become circularly dependent |
| Turn-by-turn experiment log (JSONL AuditLog) | Without structured logs there is no data; all primary metrics (turns-to-convergence, cognitive load, contradiction rate) are computed from this log alone |
| LLM-simulated Oracle Agent with configurable consistency, drift, sycophancy-resistance, and fatigue parameters | Human oracles are too expensive for ablation scale; simulation with explicit noise parameters is the stated method — and must be realistic enough to stress-test contradiction handling |
| Held-out frozen evaluation subset locked and hashed before any code runs | Required for valid generalization measurement; contamination is irreversible |
| Ablation runner: 3-5 interaction strategies x oracle personas x dataset seeds | The headline experiment; without it the paper has no result |
| Turns-to-convergence and cognitive load per turn with bootstrap 95% CIs | Primary efficiency measures; project constraint requires CIs on all headline claims |
| Generalization function: codified oracle preference applied to held-out items | Explicitly in scope; tests whether extracted preferences transfer to unseen data |
| Human validation study (N = 5-10) with within-subject design | Project constraint: "Simulated oracle results must be validated against humans before any quantitative claim." This is not optional. |

**Build priority order (from FEATURES.md MVP recommendation):**
1. Initial clustering + names/descriptions
2. Soft assignments + `f_uncertainty`
3. Global and cluster-level feedback
4. State persistence + AuditLog
5. Stopping signal (turn budget first)
6. Oracle Agent (one persona, explicit noise params)
7. `f_next_best_step` (random baseline + uncertainty-driven)
8. Ablation runner (two strategies, N sessions)
9. Metrics + bootstrap CIs
10. Point-level feedback
11. Instructional feedback
12. Generalization function
13. Human validation study

**Defer to v2:** Hierarchy navigation, UMAP/t-SNE visualization, web UI, multiple clustering backends, real-time embedding updates from oracle feedback, inter-annotator agreement scoring, noise-tolerant constraint propagation (nCOBRAS-style).

---

## 3. Architecture Overview

### System diagram

```
                     ORCHESTRATOR
                (owns ClusteringState, routes turns, writes AuditLog, enforces turn budget)
                        |
          +-------------+-------------+
          |             |             |
   ClusteringAgent  OracleAgent   JudgeAgent
   (read/write)     (read-only)   (read-only)
```

The Orchestrator is a thin coordinator — no LLM calls, no strategic decisions. Only the ClusteringAgent proposes state mutations. Oracle and Judge are read-only consumers. This makes each component independently testable and ablation-reproducible.

### Turn flow (each iteration)

```
1. ClusteringAgent.f_next_best_step(state)  ->  Action {show|ask|stop}
2. OracleAgent.respond(action, state)       ->  OracleReply {feedback_type, structured_delta, load_estimate}
3. ClusteringAgent.f_next_state(state, reply) -> NEW ClusteringState (immutable update)
4. Orchestrator applies new state
5. JudgeAgent.f_eval(state)                ->  EvalResult {continue, convergence_signal, metrics}
6. AuditLog.append(TurnRecord)
7. If not continue OR budget exhausted -> ExperimentComplete
```

### Key data structures

- `ClusteringState`: clusters (id, name, description, members), soft assignments (per-point distribution + entropy + is_boundary flag), PreferenceModel (constraints, synonyms, feature weights, drift_history), turn_history, turn_budget, codified_mapping
- `Action`: type (show_full | show_subset | show_boundary | ask_pairwise | ask_merge_confirm | ask_name | stop), payload, rationale, strategy_id
- `OracleReply`: feedback_type, content, structured_delta, load_estimate, persona_consistency_score
- `EvalResult`: continue_loop, convergence_signal (none | weak | strong | explicit_stop), metrics bundle, stopping_reason
- `TurnRecord`: turn index, Action, OracleReply, EvalResult, load_spent, cumulative_load

### Ablation axis: Strategy interface

`f_next_best_step` dispatches to a Strategy object. All strategies implement `select_action(state) -> Action`. The Orchestrator never references strategy names. Planned strategies: RandomStrategy (baseline), UncertaintyStrategy, InformationGainStrategy, BudgetAwareStrategy, HybridStrategy.

### Cognitive load model (computed deterministically before LLM call)

```
load(action) = w_items * count(items_shown) + w_clust * count(clusters_shown)
             + w_text * text_length_tokens + w_q_type * question_complexity
```

Weights are hyperparameters (defaults from literature). Oracle Agent receives cumulative load as a system prompt fact; responds more superficially when load exceeds 0.7 of budget.

### Framework note

ARCHITECTURE.md recommends plain Python (dataclasses + while loop) over LangGraph for the turn loop because the conversation graph is fixed and sequential. LangGraph's graph serialization adds dependency weight without benefit when the ablation axis is a Strategy object, not a graph topology change. Use LangGraph if a 4th agent with conditional branching is added. This is a deliberate trade-off: prioritize reproducibility and debuggability over framework features.

---

## 4. Critical Pitfalls

The top 5 mistakes that will invalidate research claims or force a rewrite.

### Pitfall 1: Evaluation measures clustering quality, not dialogue contribution

If NMI / ARI / silhouette are the primary metrics, the dialogue is decorative. Any improvement can come from the embedding + algorithm without oracle interaction. The system becomes an expensive wrapper around HDBSCAN.

**Prevention:** Define three baseline conditions before writing any interaction code: (A) no dialogue (one-shot clustering), (B) random interaction (uninformative oracle), (C) full system. Primary metrics are turn-efficiency (turns to oracle satisfaction threshold), cognitive load per turn, and satisfaction at fixed budgets. Run the ablation where `f_next_best_step` is replaced with random action selection and verify degradation. If metrics do not degrade, the policy is not contributing. The Judge Agent's `f_eval` must be fully specified before Phase 2 begins.

### Pitfall 2: LLM oracle is too consistent, too cooperative, and too helpful

LLMs exhibit ~58% sycophancy rates in simulated evaluation (SycEval 2025). An oracle prompted to be "satisfied" will converge faster and more smoothly than any real human. The contradiction-handling and preference-drift components will never be stress-tested. The human validation study will reveal the gap at the worst possible time.

**Prevention:** Oracle Agent must have explicit configurable parameters from the first working prototype: `consistency_rate`, `preference_drift_probability`, `sycophancy_resistance`, `fatigue_model` (increasing brevity after N turns). Include at least one "adversarial oracle" persona: contradictory preferences, high drift, low cooperation. Run the human validation study before finalizing quantitative claims — treat the simulated-vs-human gap as a reported finding, not a footnote.

### Pitfall 3: Soft assignments are uncalibrated — `f_uncertainty` is broken

HDBSCAN `probabilities_` and softmax-over-distances are systematically overconfident. A point assigned with 0.9 confidence is not actually a 90% likely cluster member. `f_uncertainty` built on raw soft assignments will surface the wrong points for oracle clarification.

**Prevention:** Apply temperature scaling post-hoc to soft assignments as a standard calibration step. Include a reliability diagram (or at minimum expected calibration error, ECE) as a reported metric. Maintain a pairwise validation set of 50-100 point pairs where the oracle has stated membership judgments; verify that soft assignment ordering is consistent with oracle judgments. This must be part of Phase 1 evaluation, not added later.

### Pitfall 4: State management complexity collapses across turns

Multi-turn LLM agents show a documented 39% average performance drop vs. single-turn (GPT-4o drops to ~14% accuracy in complex multi-turn scenarios). State stored only in conversation context will silently degrade after 20+ turns. Cascading updates (merge two clusters -> must update soft assignments, descriptions, hierarchy, uncertainty surface) introduce bugs that are invisible until late sessions.

**Prevention:** Define a typed `ClusteringState` schema as the very first code artifact. The state schema is a data structure, not a prompt. Serialize state to disk after every turn (enables session replay, surfaces state bugs early). Inject only a structured state summary (under 500 tokens) into the context window — not the full history. Test state integrity at turn 20, 30, 50 with a synthetic oracle before any human study runs.

### Pitfall 5: Contradiction / preference-drift handling is logged but never validated

The "latest intent wins" policy requires that when the oracle contradicts a prior preference, all downstream state (soft assignments, descriptions, cluster membership, hierarchy) updates to reflect the new intent — not just the preference log. The common failure mode is partial state updates that produce the appearance of contradiction handling while silently accumulating stale, conflicting state.

**Prevention:** After each state update, run a programmatic consistency check: all active constraints must be satisfiable with the current partition. Write contradiction injection tests: force "put X in cluster A" then "put X in cluster B" and assert that final state, soft assignments, and descriptions all reflect the second instruction. Make preference drift an active `f_next_best_step` signal: when drift exceeds a threshold, ask a clarifying question. This is also what makes the drift metric meaningful and measurable.

---

## 5. Phase Build Order Recommendation

Derived from dependency chains in FEATURES.md, build order in ARCHITECTURE.md, and phase-timing warnings in PITFALLS.md.

### Phase 0 — Obligations before first line of code (1-2 days)

**Rationale:** Four decisions are irreversible once development begins. Deferring them creates either contaminated evaluation data or invalidated claims.

- Lock and hash the held-out evaluation subset. Write its path and hash to a file that is never modified.
- Define the primary quantified claim with required sample size for a meaningful 95% CI: e.g., "uncertainty-driven strategy reaches oracle satisfaction in fewer turns than random baseline (N = 27 runs: 3 strategies x 3 personas x 3 seeds)."
- Write the human validation study protocol (within-subject, counterbalanced, session duration limit 30-45 min, NASA-TLX or simplified Likert, consent procedures).
- Specify the three independently operationalized stopping criteria (turn budget hard cap, state-change diminishing returns threshold, structured oracle satisfaction token).

**Research flag:** This phase has no standard pattern. Requires team alignment, not implementation.

### Phase 1 — Foundation: data structures and embedding infrastructure (3-5 days)

**Rationale:** Everything downstream depends on EmbeddingStore and ClusteringState. Building agents before the state schema is stable causes cascading rewrites.

- EmbeddingStore: load dataset (Amazon Reviews / IMDB via HuggingFace `datasets`), compute sentence-transformers embeddings once, expose nearest-neighbor queries. Read-only singleton after init.
- ClusteringState dataclasses: all data structures from ARCHITECTURE.md. Pydantic validation for structured fields.
- Initial HDBSCAN pass: populate first ClusteringState with cluster labels, soft assignments, descriptions (LLM-generated). Verify `probabilities_` calibration with reliability diagram.
- AuditLog: append-only JSONL writer + reader.

**Gate:** Can serialize a valid ClusteringState from a real dataset and read it back without loss.
**Research flag:** HDBSCAN soft-assignment sufficiency (sklearn `probabilities_` vs. standalone `hdbscan` full multinomial vectors) must be decided here. Do not defer.

### Phase 2 — Clustering Agent core logic (no LLM yet) (3-5 days)

**Rationale:** The agent's pure functions (`f_uncertainty`, `f_next_state`, RandomStrategy) can be written and tested without any LLM calls. Validating them in isolation prevents LLM variability from masking logic bugs.

- `f_output`: return current clustering (trivial initially).
- `f_uncertainty`: compute calibrated entropy from soft assignments; rank boundary points, split candidates, merge candidates, unresolved contradictions.
- `RandomStrategy`: `f_next_best_step` with random action selection. Testable with mocked oracle replies.
- `f_next_state`: parse a hardcoded OracleReply and produce a new ClusteringState.

**Gate:** Full turn loop runs with random strategy and mocked oracle replies. State evolves correctly across 30+ turns. State integrity tests pass.
**Research flag:** Standard patterns apply; no deeper research needed.

### Phase 3 — Oracle Agent (4-6 days)

**Rationale:** The Oracle is the most research-sensitive component. Its noise parameterization must be correct before any simulation results are recorded, or all ablation data is invalid.

- OracleAgent base: LLM call with system prompt (persona spec + preference history + cognitive-load state), returns structured OracleReply.
- Cognitive load calculator: deterministic `load(action)` from payload. Computed before LLM call; injected as system prompt fact.
- Preference drift detection: parse `DRIFT:` markers; update PreferenceModel.drift_history. Latest intent overwrites constraint; history preserved.
- Persona config loader: JSON persona spec including `consistency_rate`, `preference_drift_probability`, `sycophancy_resistance`, `cognitive_decay_rate`.
- Adversarial oracle persona: contradictory preferences, high drift, low cooperation. Run loop with this persona to stress-test contradiction handling.

**Gate:** Oracle produces valid OracleReply for each Action type; drift markers are parsed and logged; consistency_rate and drift_probability params produce measurably different behavior in simulation runs.
**Research flag:** Oracle cognitive-load modeling is MEDIUM confidence (active research area). May need literature check on load weight parameters.

### Phase 4 — Judge Agent and Orchestrator (3-4 days)

**Rationale:** With three independent stopping criteria operationalized (Phase 0), the Judge can be built to test each independently. Budget enforcement belongs in the Orchestrator.

- JudgeAgent.f_eval: convergence signal logic (explicit satisfaction, feedback magnitude decay, preference stability, diminishing returns, budget exhaustion). Metric bundle per turn.
- Feedback magnitude tracker: `|structured_delta|` across last N turns.
- Budget enforcement in Orchestrator: turn count, budget check, ExperimentComplete termination.
- Oracle validity checks: persona_consistency_score drops, oracle_stuck warnings.

**Gate:** Judge terminates the loop correctly under all five stopping conditions. Each stopping criterion can be triggered independently by a synthetic oracle.
**Research flag:** Standard patterns apply.

### Phase 5 — Real Strategies and Ablation-Ready Loop (4-6 days)

**Rationale:** Once Oracle, Judge, and RandomStrategy are validated, non-random strategies can be added against a known baseline. Full integration test required before ablation runner.

- UncertaintyStrategy: select highest-entropy (calibrated) point; prefer ask_pairwise.
- InformationGainStrategy: estimate entropy reduction per action type; select max.
- BudgetAwareStrategy: modulate action type by remaining turn budget.
- HybridStrategy: combine uncertainty + cluster representativeness.
- Full integration test: all strategies through a complete loop with the adversarial oracle persona.

**Gate:** All strategies run through full loop. AuditLog shows strategy_id on every turn. Metrics are strategy-attributable.
**Research flag:** Information-gain estimation for LLM-based actions may need deeper research — how to estimate entropy reduction before making the oracle call.

### Phase 6 — Ablation Harness and Evaluation (5-8 days)

**Rationale:** Harness must enforce full crossing (strategies x personas x seeds) to avoid confounded conditions. Analysis must read AuditLog only; no internal agent state.

- Experiment config loader (JSON): strategy, oracle_persona, dataset, turn_budget, random_seed, replications.
- Multi-run executor: N replications per condition, consistent seeds, parallelizable via multiprocessing.
- Migrate to MLflow for run tracking and comparison.
- Analysis scripts: read AuditLog only; compute turns-to-convergence, cognitive load per turn, contradiction rate, preference stability; bootstrap 95% CIs.
- Introduce pairwise validation probes (straightforward given AuditLog).
- Persona-varied oracle experiments (parameter sweep using existing Oracle Agent).

**Gate:** Ablation table is reproducible from AuditLog + config file alone. Primary quantified claim with 95% CI is computable.
**Research flag:** Minimum crossing: 3 strategies x 3 oracle personas x 3 dataset seeds = 27 runs. Feasible with LLM oracles at gpt-4o-mini cost.

### Phase 7 — Generalization and Human Validation (5-7 days)

**Rationale:** Both depend on all prior phases being stable and the held-out set being clean (locked in Phase 0).

- Generalization function: derive codified preference classifier/rule set from accumulated preference log; evaluate accuracy on frozen held-out subset.
- Preference codification quality metric: oracle re-labels 20-30 held-out items; compare to preference-function predictions.
- Human validation study: execute pre-written protocol (Phase 0); 2-person pilot first; compare simulated vs. human convergence patterns as a reported finding.
- Confidence interval sweep: ensure all headline claims meet 95% CI requirement.

**Gate:** Human study is complete; simulated-vs-human gap is quantified and reported; all headline claims have defensible CIs.
**Research flag:** Human study protocol written in Phase 0; no additional research needed. Generalization function design may need one implementation spike.

---

## 6. Open Questions

Decisions that are unresolved and must be answered early to avoid rework.

| Question | Why It Must Be Answered Early | When to Decide |
|----------|------------------------------|---------------|
| Does sklearn HDBSCAN `probabilities_` (scalar membership strength) satisfy the soft-assignment requirement, or is the full multinomial distribution from the standalone `hdbscan` package's `all_points_membership_vectors()` needed? | Determines which package is used; changes the SoftAssignment data structure and `f_uncertainty` computation. Retrofitting mid-project cascades through state schema. | Phase 1 gate |
| What are the oracle cognitive-load weight parameters (w_items, w_clust, w_text, w_q_type)? Are the literature defaults empirically defensible for this interaction design? | Cognitive load is both a design constraint (per-turn cap) and a primary metric. Wrong weights produce a broken metric that is invisible until the human study. | Phase 3; may need targeted literature search |
| Is the information-gain estimate for `f_next_best_step` computable cheaply (entropy over current soft assignments) or does it require a model-dependent prior (information-theoretic estimate of what a pairwise query reveals)? | Determines whether InformationGainStrategy is implementable within the project scope or must be approximated. | Phase 5 start |
| What dataset will be used as the primary ablation target — Amazon Reviews 2023, IMDB, or support tickets — and is a ground-truth label set available for calibration and generalization evaluation? | The held-out split must be locked before code runs. Ground truth (even noisy) enables reliability diagram computation and NMI as a secondary diagnostic metric. | Phase 0 |
| Should the Orchestrator be implemented in LangGraph or plain Python? ARCHITECTURE.md recommends plain Python for the fixed sequential graph; STACK.md recommends LangGraph for HITL interrupts and durable checkpointing. | This is a build-vs-buy trade-off. The decision affects how HITL pausing is implemented and whether the system can checkpoint mid-session for the human study. | Before Phase 2 |
| What is the operationalized definition of "oracle satisfaction" for the stopping signal — a structured `DONE` token in OracleReply, a turn where `|structured_delta| == 0`, or an LLM self-report? | Three stopping criteria are required; this determines one of them. Circular definitions (oracle both gives feedback and determines when to stop) invalidate the convergence metric. | Phase 0 |
| What is the maximum acceptable turns-to-convergence budget for the human study, given a 30-45 minute session limit and cognitive-load cap per turn? | Determines the turn budget hyperparameter used across all ablation conditions. Inconsistent turn budgets between simulated and human conditions make comparison invalid. | Phase 0 / human study protocol |

---

## Confidence Assessment

| Research Area | Confidence | Basis |
|---------------|------------|-------|
| Stack (LangGraph, sentence-transformers, HDBSCAN, Typer+Rich) | HIGH | Ecosystem defaults as of 2026; multiple corroborating sources |
| Features (table stakes, dependency order, anti-features) | HIGH | Grounded in COBRAS, Prodigy, active learning literature; HIGH-confidence primary sources |
| Architecture (supervisor pattern, data structures, ablation axis) | HIGH | Well-established patterns; supervisor/coordinator is documented in production multi-agent systems |
| Oracle cognitive-load modeling | MEDIUM | Active research area; load weight parameters are literature defaults, not empirically validated for this design |
| LLM sycophancy mitigation parameters | MEDIUM | SycEval 2025 documents the problem; specific mitigation parameters (consistency_rate thresholds) are heuristic |
| Soft-assignment calibration | MEDIUM | Temperature scaling is well-validated for supervised classifiers; direct application to HDBSCAN probabilities is underexplored |
| Preference drift handling | MEDIUM-LOW | Under-studied: "No IDM technique has been explicitly designed to handle preference drift" (PITFALLS.md source) |

**Gaps requiring attention during planning or early implementation:**
- sklearn HDBSCAN soft-assignment sufficiency (resolve at Phase 1)
- Oracle load weight calibration (resolve at Phase 3, possibly with a targeted literature search)
- Orchestrator framework decision: LangGraph vs. plain Python (resolve before Phase 2)
- Primary dataset selection and ground-truth availability (resolve at Phase 0)

---

## Sources (aggregated)

**HIGH confidence:**
- LangGraph GitHub (v1.1.10, April 2026): https://github.com/langchain-ai/langgraph
- scikit-learn HDBSCAN docs (1.8.0): https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
- COBRAS paper PDF (DTAI): https://dtai.cs.kuleuven.be/software/cobras/cobras_ida_cameraready.pdf
- Large Language Models Enable Few-Shot Clustering (TACL/MIT Press): https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00648/
- Semi-supervised constrained clustering review (Springer AI Review 2024): https://link.springer.com/article/10.1007/s10462-024-11103-8
- Prodigy annotation tool: https://explosion.ai/blog/prodigy-annotation-tool-active-learning
- HITL machine learning state of the art (Springer, 2022): https://link.springer.com/article/10.1007/s10462-022-10246-w
- Multi-agent architecture patterns: https://arxiv.org/html/2601.13671v1
- Text Clustering with LLM Embeddings: https://arxiv.org/html/2403.15112v5

**MEDIUM confidence:**
- SycEval: LLM Sycophancy Evaluation (2025): https://arxiv.org/html/2502.08177v2
- LLMs Get Lost In Multi-Turn Conversation: https://arxiv.org/html/2505.06120v1
- Towards Calibrated Deep Clustering Network: https://arxiv.org/html/2403.02998v2
- Cognitive load in LLM conversations: https://arxiv.org/pdf/2505.10742
- Interactive Clustering comprehensive review (ACM, 2020, 105 papers): https://dl.acm.org/doi/fullHtml/10.1145/3340960
- Do We Still Need Humans in the Loop? (arXiv 2604.13899): https://arxiv.org/html/2604.13899

**LOW confidence (abstract only or blog):**
- Handling concept drift in preference learning: https://www.researchgate.net/publication/228967853_Handling_concept_drift_in_preference_learning_for_interactive
- LLM Perception Drift: https://www.stridec.com/blog/llm-perception-drift-why-matters-ai-applications/
