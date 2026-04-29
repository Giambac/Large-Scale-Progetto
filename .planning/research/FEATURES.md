# Feature Landscape

**Domain:** Conversational human-in-the-loop clustering research system
**Project:** Conversational Clustering (3-agent: Clustering Agent, Oracle Agent, Judge Agent)
**Researched:** 2026-04-29
**Interface:** CLI / Jupyter notebook

---

## Table Stakes

Features where absence makes the core research question unanswerable. Missing any of these
means the study cannot produce defensible quantitative claims.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Initial clustering with names and natural-language descriptions | The oracle has nothing to react to without it; every HITL clustering system starts here | Low | Sentence embeddings + LLM label generation; one representation only |
| Global feedback acceptance | Canonical feedback type from COBRAS, interactive clustering literature — "too many clusters", "merge everything" | Low | Free-text parsing to structured intent; must map to `f_next_state` |
| Cluster-level feedback acceptance | Split / merge / rename / reweight a specific cluster; this is the most common oracle action in constrained clustering research | Medium | Requires cluster identity persistence across turns; cluster IDs must be stable |
| Point-level feedback acceptance | "x belongs in B", "x and y should be together"; must-link / cannot-link constraints are the most studied primitive in semi-supervised clustering (COBRAS, COBRA, nCOBRAS) | Medium | Requires point-addressable interface; oracle references items by index, content snippet, or short ID |
| Instructional / declarative feedback acceptance | "treat 'error' and 'fail' as synonyms", "weight feature F more"; this is the most powerful feedback type for preference codification | High | LLM parses instruction into a preference constraint; hardest to operationalize cleanly |
| Soft assignments (per-point distribution over K clusters) | Required for calibration testing, boundary detection, and uncertainty sampling; hard labels discard information needed for the research questions | Medium | Softmax over embedding distances or LLM confidence scores; must be storable per-turn |
| `f_uncertainty`: identify low-confidence and boundary points | Needed for uncertainty-driven query strategies; central to the ablation experiment | Medium | Margin between top-2 cluster probabilities; must flag boundary cases before asking oracle |
| `f_next_best_step`: show / ask / stop decision each turn | The core algorithmic contribution of the system; differentiates it from passive label collection | High | Implements the interaction strategy being ablated; each strategy variant must be swappable |
| State persistence across turns (conversation memory) | Without this the oracle's earlier preferences disappear; every IML system must maintain state | Medium | Structured JSON state object: current clustering, all feedback turns, preference log, contradiction flags |
| Contradiction / preference-drift tracking | Explicitly required by the research question; "latest intent wins" is a policy that must be implemented and logged | Medium | Compare current instruction against prior conflicting instructions; surface the conflict in the log |
| Stopping signal | Required for convergence measurement; without it the experiment has no endpoint | Medium | Three variants needed for ablation: oracle explicit accept, diminishing-returns heuristic, turn budget hard cap |
| Turn-by-turn experiment log | Needed to compute turns-to-convergence, cognitive-load-per-turn, and contradiction rate; without structured logs no quantitative claims are possible | Medium | JSON-L format per experiment run: turn index, feedback type, state delta, soft assignments, uncertainty score |
| LLM-simulated oracle (Oracle Agent) | Human oracles are too expensive for ablation at scale; LLM simulation is the stated method in the project scope | High | Oracle Agent needs a preference specification JSON, a persona, a cognitive-load budget; output must be parseable feedback |
| Ablation across interaction strategies | The headline experiment: 3-5 strategies compared on convergence efficiency; without this the paper has no result | High | Strategies must be hot-swappable; random / uncertainty-driven / boundary-driven / full-display baselines |
| Evaluation metrics: turns to convergence, cognitive load per turn | Primary efficiency measures; absence makes the headline claim unmeasurable | Low | Cognitive load = items shown + clusters referenced + question complexity score |
| Generalization function: codified oracle preference for new items | Explicitly in scope; tests whether extracted preferences transfer | High | Prompt-based classifier or rule set derived from logged preferences; validated on frozen held-out subset |
| Held-out frozen evaluation subset | Required for valid generalization measurement; must be set aside before any oracle interaction begins | Low | Slice dataset before first run; do not allow oracle to see hold-out items during training sessions |
| Human validation study (N = 5-10) | Required by the project constraint: "Simulated oracle results must be validated against humans before any quantitative claim"; without this no paper claim is defensible | High | Within-subject protocol, consented, scripted prompts; compare LLM-oracle convergence patterns to human-oracle patterns |
| Confidence intervals on headline claims | Stated project constraint: "All headline quantitative claims must include confidence intervals" | Medium | Bootstrap CIs on turns-to-convergence; report mean + 95% CI, not means alone |

---

## Differentiators

Features that give the system and resulting paper a competitive academic advantage. Not
expected by reviewers, but valued because they strengthen claims or enable novel findings.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Hierarchy navigation (zoom in / out) | Enables multi-resolution feedback; oracle can approve at coarse level, drill into ambiguous sub-cluster; used in COBRAS super-instances and Clustergrammer dendrogram-sliders | High | CLI representation as indented tree; oracle can reference level by depth; requires hierarchical clustering step |
| Anytime behavior: valid clustering at every turn | COBRAS was the first system to achieve anytime behavior, query efficiency, and time efficiency simultaneously; makes the system useful during the session, not just at convergence | Medium | Ensures `f_output` always returns a complete assignment even mid-session; no "wait for convergence" state |
| Pairwise validation probes | "Are x and y correctly grouped?" — behavioral probe independent of satisfaction self-report; adds a second measurement channel | Low | Sample 10-20 pairs per evaluation round; compute agreement rate; calibrate against soft-assignment boundary scores |
| Soft-assignment calibration test | Tests whether boundary-flagged points are actually ambiguous to the oracle; if calibration is poor, the uncertainty model is wrong | Medium | Administer oracle judgment on top-N highest-uncertainty points; compute calibration error (ECE equivalent) |
| Contradiction surface and clarification request | When the system detects a direct conflict with earlier stated preference, it can explicitly surface the contradiction to the oracle ("three turns ago you said X, now Y — which applies?") instead of silently applying latest-intent | Medium | Requires diff against preference log; adds a clarification turn type to the interaction loop |
| Persona-varied LLM oracle experiments | Running the same dataset through multiple LLM oracle personas (consistent, contradictory, cognitively conservative) surfaces how persona variation affects convergence; publishable finding | Medium | Oracle Agent persona spec as a parameter; experiment runner sweeps N personas |
| Cognitive load budget enforcement in Oracle Agent | Simulates realistic human fatigue by limiting items displayed per turn; forces the system to be selective; validates the cognitive-load measurement methodology | Medium | Oracle Agent refuses to process more than K items per turn; returns partial feedback signal |
| Preference codification quality metric | Measures how well the learned preference function generalizes: accuracy on hold-out + oracle agreement rate on a sample; stronger than just reporting hold-out accuracy | Medium | Oracle re-labels 20-30 hold-out items; compare to preference-function predictions |
| Information gain per cognitive-load unit | Normalizes turn value: which question type (pairwise check vs. full display vs. targeted merge suggestion) delivers the most clustering improvement per unit of oracle attention | High | Requires defining information gain operationally (entropy reduction in soft assignments); plots turn type vs. normalized gain |
| Comparison of feedback types at identical decision points | Controlled experiment: at the same decision point, send three oracle agents different question types and measure the downstream clustering difference; isolates the value of each feedback level | High | Requires branching experiment runner; one of the optional research questions in the project brief |

---

## Anti-Features

Features to explicitly NOT build in v1. Each has been seen to cause scope creep, rewrite
pressure, or validity problems in similar academic systems.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Web UI / dataset upload / persistent sessions | Explicitly out of scope in PROJECT.md; building a UI shifts effort from the research question to software engineering; trio/quartet scope only | CLI or notebook interface; use JSON files for session state |
| UMAP / t-SNE visualization | Explicitly deferred in PROJECT.md; adds a dependency that requires visual display and is hard to quantify in a text-based study | Track soft-assignment distributions numerically; defer visualization |
| Multiple clustering backends (k-means / HDBSCAN / LLM-first) | Backend comparison is a separate research question from conversational interaction efficiency; mixing them makes ablation uninterpretable | Fix one representation (sentence embeddings) for the entire study |
| Real-time model fine-tuning / embedding updates from oracle feedback | Expensive, slow, and conflates "better representations" with "better interaction strategies"; the oracle feedback should act on the grouping layer, not the embedding layer | Oracle feedback updates cluster assignments and preference log only; embeddings stay frozen |
| Full N x M human oracle study | Requires larger team; explicitly deferred to trio/quartet scope | Small human validation study (N = 5-10) sufficient to validate LLM oracle behavior |
| Noise-tolerant constraint handling (nCOBRAS-style) | nCOBRAS adds redundant queries to detect noisy user labels; the project treats preference evolution as signal, not noise — this conflicts with the "latest intent wins" policy | Log contradictions and surface them; do not silently absorb or retry |
| Confidence-weighted must-link / cannot-link constraint propagation | Semi-supervised clustering constraint propagation (COP-k-means, PCKMeans) is a full sub-field; implementing it correctly requires significant algorithmic work unrelated to the conversational interface | Use soft assignments and direct oracle feedback to cluster assignments; treat constraints as preference signals, not hard algorithmic constraints |
| Automatic K selection / K optimization loop | Optimizing number of clusters without oracle input bypasses the oracle as objective function; it introduces a secondary optimization target that conflicts with the research premise | K can change only via oracle global or cluster-level feedback; no automatic K selection |
| Inter-annotator agreement scoring | Relevant only when multiple oracles rate the same item simultaneously; the study uses within-subject or single-oracle protocol | Not applicable to the research design; would add complexity with no payoff |
| Rich cluster metadata UI (confidence bars, embedding distances displayed to user) | Cognitive overload; Prodigy's key design insight is "ask as little as possible"; showing internals to the oracle inflates cognitive load and biases their feedback | Surface only what is needed for the specific question being asked each turn |
| Asynchronous / batch oracle sessions | Breaks the conversational loop; the research question is about turn-by-turn interaction dynamics | Keep the loop synchronous; LLM oracle is fast enough for synchronous ablation |

---

## Feature Dependencies

```
Initial clustering
    |
    +-- Soft assignments
    |       |
    |       +-- f_uncertainty (boundary / low-confidence detection)
    |               |
    |               +-- f_next_best_step (show / ask / stop)
    |                       |
    |                       +-- Stopping signal
    |
    +-- Cluster names + descriptions
            |
            +-- Cluster-level feedback acceptance
            |       |
            |       +-- Contradiction / preference-drift tracking
            |
            +-- Global feedback acceptance
            +-- Point-level feedback acceptance
            +-- Instructional feedback acceptance

State persistence (conversation memory)
    |
    +-- Contradiction tracking (needs history)
    +-- Experiment log (structured per-turn record)
    +-- Generalization function (built from accumulated preference log)

LLM-simulated Oracle Agent
    |
    +-- Ablation experiment runner (sweeps strategies against Oracle Agent)
    +-- Human validation study (validates Oracle Agent behavior)

Experiment log
    |
    +-- Turns-to-convergence metric
    +-- Cognitive load per turn metric
    +-- Confidence intervals on headline claims

Held-out frozen evaluation subset
    |
    +-- Generalization function evaluation
    +-- Preference codification quality metric (differentiator)
```

---

## MVP Recommendation

The MVP must answer the headline research question: "Does conversational refinement converge
toward oracle-accepted clusterings, and how efficiently compared to a baseline?"

**Build in this order:**

1. Initial clustering + cluster names/descriptions (unblocks everything)
2. Soft assignments + `f_uncertainty` (needed for any non-random query strategy)
3. Global and cluster-level feedback acceptance (covers 80% of oracle actions in practice)
4. State persistence + experiment log (without logs there is no data to analyze)
5. Stopping signal: turn-budget hard cap first, satisfaction signal second
6. LLM-simulated Oracle Agent with one preference specification
7. `f_next_best_step` with two strategies: random baseline + uncertainty-driven
8. Ablation runner: compare two strategies across N simulated sessions
9. Turns-to-convergence + cognitive-load metrics with bootstrap CIs
10. Point-level feedback (adds constraint expressiveness for later ablation rounds)
11. Instructional feedback (highest complexity; add after core loop is stable)
12. Generalization function on held-out subset
13. Human validation study (N = 5-10) as final validation gate

**Defer to later phases:**
- Hierarchy navigation: implement after core feedback loop is stable; adds complexity
- Contradiction surface / clarification request: implement after preference-drift tracking is in place
- Pairwise validation probes: straightforward to add once experiment log exists
- Persona-varied oracle experiments: add after baseline Oracle Agent is validated
- Information gain per cognitive-load unit: compute from existing logs; no new infrastructure needed

---

## Lessons From Similar Systems

**COBRAS / COBRA (KU Leuven, interactive clustering):**
- Key insight: anytime behavior (valid clustering at every turn) is essential for usability; systems that require completing a full session before producing output frustrate users.
- Key lesson: query efficiency and time efficiency must both be measured; a system that asks few questions but takes 10 seconds per turn is not better than a faster system asking slightly more questions.
- Applicable: the `f_next_best_step` decision function directly addresses query efficiency.

**Prodigy (Explosion AI, active learning annotation):**
- Key insight: "ask as little as possible" is the right interface philosophy; binary questions outperform complex multi-attribute forms for annotation quality per unit of cognitive load.
- Key lesson: showing model internals (confidence scores, embeddings) to the annotator increases cognitive load without proportionally increasing annotation quality.
- Applicable: each turn's output should be the minimal display needed for that specific question.

**nCOBRAS:**
- Key lesson: adding noise-tolerance via redundant constraints improved clustering quality but increased query count; tradeoff is not always favorable.
- Applicable: the "latest intent wins" policy is a deliberate design choice that avoids redundant constraint overhead, but requires logging contradictions explicitly.

**LLM-as-oracle research (2024-2025 literature):**
- Key finding: LLM annotators and human annotators produce comparable label distributions in many NLP tasks, but LLMs show systematic bias (more consistent, less cognitively fatigued, different distribution of contradictions).
- Key lesson: you cannot publish LLM-oracle results as a proxy for human-oracle results without explicit validation; validation against humans (N = 5-10) is the minimum defensible gate.
- Applicable: the human validation study is table stakes, not optional.

**Failure modes observed in academic IML systems:**
- Scope expansion to UI before the core loop was validated — common in research prototypes that become engineering projects.
- Evaluation measuring the underlying clustering quality (silhouette, NMI against ground truth) rather than what the interaction influenced — produces results that could have been obtained without any oracle.
- Cognitive load ignored until study design, then discovered to be confounded with interaction strategy — budget it from turn one.
- LLM oracle that is too consistent (no contradictions, always accepts refinements) — does not stress-test the contradiction handling and preference-drift components; Oracle Agent persona must include some level of inconsistency.

---

## Sources

- [Interactive Clustering: A Comprehensive Review (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3340960) — MEDIUM confidence (paywalled; accessed via search summaries)
- [COBRAS: Interactive Clustering with Pairwise Queries (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-01768-2_29) — HIGH confidence
- [COBRAS paper PDF (DTAI)](https://dtai.cs.kuleuven.be/software/cobras/cobras_ida_cameraready.pdf) — HIGH confidence
- [Clustering With User Feedback (DTAI Stories)](https://dtai.cs.kuleuven.be/stories/post/jonas-soenen/clustering-with-user-feedback/) — MEDIUM confidence
- [Large Language Models Enable Few-Shot Clustering (TACL/MIT Press)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00648/120476/Large-Language-Models-Enable-Few-Shot-Clustering) — HIGH confidence
- [Do We Still Need Humans in the Loop? LLM vs Human Annotation in Active Learning (arXiv 2604.13899)](https://arxiv.org/html/2604.13899) — MEDIUM confidence
- [Human-LLM Collaborative Annotation (CHI 2024)](https://dl.acm.org/doi/10.1145/3613904.3641960) — MEDIUM confidence
- [Validating LLM Simulations as Behavioral Evidence (arXiv 2602.15785)](https://arxiv.org/html/2602.15785v1) — MEDIUM confidence
- [Semi-supervised constrained clustering review (Springer AI Review 2024)](https://link.springer.com/article/10.1007/s10462-024-11103-8) — HIGH confidence
- [Prodigy: A new tool for radically efficient machine teaching (Explosion)](https://explosion.ai/blog/prodigy-annotation-tool-active-learning) — HIGH confidence
- [Human-in-the-loop machine learning: a state of the art (Springer AI Review)](https://link.springer.com/article/10.1007/s10462-022-10246-w) — HIGH confidence
- [Handling concept drift in preference learning for interactive systems (ResearchGate)](https://www.researchgate.net/publication/228967853_Handling_concept_drift_in_preference_learning_for_interactive) — LOW confidence (abstract only)
- Project reference scripts: `Conversational Clustering Script.txt`, `Multi Agent Personalities Script.txt`
- Project specification: `.planning/PROJECT.md`
