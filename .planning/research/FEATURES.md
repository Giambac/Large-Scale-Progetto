# Feature Landscape

**Domain:** Conversational human-in-the-loop clustering research system
**Project:** Conversational Clustering (3-agent: Clustering Agent, Oracle Agent, Judge Agent)

> This file has two parts. **Part A** is the v2.0 milestone feature research (NEW features only —
> the active research for the current milestone "Experimentation Flexibility & Scale"). **Part B**
> below the divider is the original v1 feature research, preserved as historical reference. The
> roadmapper for v2.0 should consume **Part A**.

---

# PART A — v2.0 Milestone: Experimentation Flexibility & Scale

**Researched:** 2026-05-21
**Confidence:** MEDIUM-HIGH (orchestrator/onboarding/intent patterns verified against multiple sources; UMAP-recolor and chat-view detail are LOW-MEDIUM — verified against UMAP docs, remaining detail is implementation common-sense)

## Scope Note (read first)

This is a SUBSEQUENT milestone on an existing system. The conversational loop, `feedback_parser`,
`OracleAgent`/`OracleSpec`/`NoiseParams`, structural contradiction detection (`_contradicts` +
rolling delta window), and a server-side UMAP projection already exist. This research covers ONLY
the five NEW v2 features and explicitly marks each as **EXTEND** (modifies existing code) or
**NEW** (a new subsystem).

| Feature | EXTEND / NEW | Touches |
|---------|-------------|---------|
| 1. Oracle-initiated onboarding + initial query | NEW orchestration around existing loop | `conversation_loop.py` entry, web routes, new "intro/query" pre-phase |
| 2. Query filter (free-text → simple instructions) | EXTEND `feedback_parser` + `_contradicts` | `feedback_parser.py`, contradiction layer |
| 3. Coordination agent (pairwise decomposition + fan-out) | NEW subsystem (HIGH complexity — last phase) | new module; orchestrates N `run_conversation` sessions |
| 4. Interactive UMAP recolor (cache coords) | EXTEND existing projection | `web/app.py` projection hook, client JS |
| 5. Real-time chat view (human + LLM) | EXTEND existing `/study` + `/watch` | web routes, socket events |

## Table Stakes (Users Expect These)

For a HITL clustering research tool of this kind, these are the baseline expectations for the
milestone. Their absence makes the v2 system feel half-built relative to the milestone goal.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Dataset introduction / summary before interaction | HITL tools (AstronomicAL, IPBC, active-learning dashboards) always orient the human BEFORE asking for input — you cannot query a corpus you have not seen. Without it the oracle's first query is uninformed. | LOW–MEDIUM | Summary = corpus size, sample items, optional cheap auto-topic preview. LLM-generated 1-paragraph précis + N sampled rows. NEW pre-phase, reuses dataset load. |
| Oracle's initial free-text query drives the FIRST clustering | The milestone's central reframe: cluster on demand, not autonomously. Active-learning UIs let the human declare intent ("group by sentiment") before the model commits. | MEDIUM | Re-fit KMeans on query (embeddings fixed — decision locked). Query becomes the seed `GlobalFeedback`/`InstructionalFeedback` fed into the first `f_next_state`. |
| Query validated/normalized before it hits the clusterer | A raw human query is messy/compound/contradictory. The clusterer needs clean, executable instructions — the role `feedback_parser` already plays mid-conversation, now at the front door. | MEDIUM | EXTENDS the existing parser + `_contradicts`. See Feature 2 detail below. |
| Live cluster view that updates as clusters change | Existing UMAP + `state_update` emit already does this; oracle expects the picture to track the conversation. v2 just makes it cheaper (recolor not refit). | LOW (extend) | Cache coords once; recolor on assignment change. Refit ONLY on split/merge (already the trigger in current code). |
| Real-time chat transcript of the conversation | Both `/study` (human types) and `/watch` (observe LLM oracle) need the turn-by-turn dialogue, not just metrics. | LOW–MEDIUM | Append-only message list driven by socket events. Human path adds an input box; watch path is read-only stream. |
| Latest-intent-wins on contradictory queries | Already a locked project decision; the query filter must honor it at query time too, not only mid-loop. | LOW (reuse) | The contradiction layer already encodes "latest wins"; query filter calls into it. |

## Differentiators (Competitive Advantage)

These align with the Core Value ("converge efficiently under a cognitive-load budget") and set this
research system apart from a generic clustering UI.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Query filter emits SIMPLE, contradiction-free instruction set | Most NL-query systems stop at "parse intent → run." This one normalizes (dedupe + resolve internal contradictions + simplify compound asks) before execution, cutting wasted clusterer turns — directly serves the efficiency thesis. | MEDIUM | EXTENDS `parse_feedback`. Add a normalization pass over the delta list. Emits a "filtered_query" audit record (original vs normalized). |
| Oracle-initiated flow as an experimental condition | Lets the study compare "autonomous-first" vs "oracle-initiated" clustering as a measurable factor (turns-to-convergence, satisfaction). Research signal, not just UX. | MEDIUM | NEW. Structure onboarding as a discrete pre-phase the loop can run or skip — keep both modes for ablation. |
| Coordination agent: decompose complex op → pairwise sub-ops across N sessions | Scales to larger K / harder operations by fanning a "reorganize everything" request into independent pairwise split/merge sub-tasks run in parallel sessions, then recombined. Classic orchestrator-worker (≈70% of production multi-agent deployments use this shape). | HIGH | NEW, **deferred to last phase** (locked decision; highest risk). See dedicated detail below. |
| Versioned YAML oracle config (model + structured prompts) | Reproducible experiments: prompt/persona/model pinned per run, diffable, not buried inline in code (current `_build_system_prompt` is inline). | LOW–MEDIUM | EXTENDS `OracleAgent`: load `OracleSpec`/`NoiseParams`/prompt sections from YAML. Decision locked: configs in YAML, not DB. |
| Projection stability across recolors (no jitter) | HITL viz research (IPBC) shows users lose trust when the map "jumps." Caching coords and only recoloring keeps the mental map stable turn-to-turn — a real cognitive-load win. | LOW–MEDIUM | The win is precisely NOT refitting. Refit (accept relayout) only on genuine topology change (split/merge), ideally with a brief transition. |

## Anti-Features (Commonly Requested, Often Problematic)

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| Re-embed the corpus on each query | "Query changed, re-encode for relevance." | Breaks the read-only `EmbeddingStore` invariant; expensive; non-reproducible. Resolved AGAINST at milestone start. | Re-FIT KMeans on fixed embeddings (locked). Query changes *clustering*, never *embeddings*. |
| Query filter that "fixes"/second-guesses oracle intent semantically | "Make the AI smooth over what the human meant." | Silently overriding the oracle corrupts the objective function — the oracle IS ground truth. Violates fail-loudly + latest-intent-wins. | Filter only NORMALIZES structure (dedupe, drop self-contradiction by latest-wins, split compound). Never invents/reinterprets intent. Unparseable query → crash/clarify, don't guess. |
| Automatic K optimization triggered by the query | "Pick the best K for this query." | Project anti-feature — K changes ONLY through oracle intent. A query saying "~5 groups" is intent; silhouette/BIC is not. | Take K from the oracle's stated query; if unspecified, carry prior K. Never run an internal K-search. |
| Always-on parallel coordination (every op as fan-out) | "Parallelism is faster, use it everywhere." | Only helps GENUINELY decomposable ops; forcing it adds recombination bugs, cross-session contradiction merging, and overhead for simple edits. | Coordination agent fires ONLY for complex multi-cluster reorganizations; simple ops stay in the single existing loop. Gate behind a complexity check. |
| Animated UMAP refit every turn | "Smooth motion looks polished." | Constant relayout destroys the stable mental map and adds per-turn compute — opposite of the cognitive-load goal. | Recolor in place (no motion) for assignment changes; relayout only on split/merge, with a brief transition so the user can track it. |
| Free-form chat that bypasses the parser/filter | "Let the human just talk to the clusterer directly." | Unstructured text reaching the clusterer skips validation, hallucinated-cluster-ID guards, and contradiction handling — reintroduces silent-wrong-answer risk. | ALL human/LLM text routes through query filter → `feedback_parser` → validated `FeedbackDelta`. Chat view is presentation, not a side channel. |

## Feature Dependencies

```
[Versioned YAML oracle config]
    └──enables──> [Oracle-initiated onboarding] (intro/query prompts live in YAML)

[Dataset introduction/summary]
    └──precedes──> [Oracle initial query]
                       └──requires──> [Query filter]  (normalize before first clustering)
                                          └──extends──> [feedback_parser + _contradicts]  (EXISTING)
                                          └──feeds──> [First KMeans re-fit on fixed embeddings]
                                                          └──produces──> [Initial ClusteringState]
                                                                            └──enters──> [run_conversation loop]  (EXISTING)

[Interactive UMAP recolor] ──extends──> [server-side UMAP projection]  (EXISTING)
    └──depends on──> [state_update socket emit]  (EXISTING)

[Real-time chat view] ──extends──> [/study + /watch routes]  (EXISTING)
    └──shares──> [state_update / message socket events]

[Coordination agent]  (LAST PHASE — HIGH complexity)
    └──requires──> [Query filter]      (decompose a normalized complex op)
    └──requires──> [run_conversation]  (each pairwise sub-op runs a session)
    └──requires──> [contradiction layer across sessions]  (recombine without re-introducing conflicts)
    └──conflicts──> [single-session assumptions in conversation_loop]  (PairBag, audit log, DB experiment_id are per-run)
```

### Dependency Notes

- **Oracle initial query requires the Query filter:** the first clustering must be driven by a clean instruction set, exactly as mid-conversation turns are. Build the filter before/with onboarding so the first turn isn't a special-cased un-validated path.
- **Query filter extends `feedback_parser` + `_contradicts`:** do NOT build a parallel parser. Add (a) a batch-level normalization pass (dedupe + intra-batch contradiction resolution by latest-wins + compound-splitting) reusing `_contradicts`, and (b) accept the front-door query as just another `raw_text` into `parse_feedback`. The only genuinely new code is normalization over the returned delta list.
- **YAML oracle config enables onboarding prompts:** the intro-summary prompt and the "ask for initial query" prompt are oracle-facing prompts — version them in the same YAML the milestone already mandates, so onboarding wording is reproducible per run.
- **Interactive UMAP recolor depends on existing projection + state_update:** coords are cached from the existing server-side fit; the client recolors using the per-point cluster assignment already in the `state_update` payload (`soft_probs`/clusters). Refit trigger already exists (split/merge) — reuse it.
- **Coordination agent conflicts with single-session assumptions:** `conversation_loop.run_conversation` currently assumes ONE PairBag, ONE audit log path, ONE `experiment_id`. Fanning into N sessions needs per-session logs + a recombination step that re-validates against the contradiction layer. This coupling is the main reason it is HIGH complexity and last.

## Five-Feature Concrete Behavior (downstream consumer detail)

### 1. Oracle-initiated onboarding + initial query (NEW orchestration)

How comparable HITL / active-learning tools structure this onboarding step:
1. **Orient** — show the human a summary of the data before any model commitment (corpus stats, sampled items, optional cheap auto-preview). HITL dashboards (AstronomicAL) and projection tools (IPBC) all front-load orientation.
2. **Declare intent** — the human issues an initial query/goal ("group these by topic, ~5 groups"). The active-learning analogue of choosing what to label first.
3. **Commit** — only now does the system produce the FIRST clustering, seeded by that intent.
4. **Converge** — hand off to the existing turn-by-turn refinement loop.

Concrete wiring: insert a pre-loop phase that (a) builds a dataset summary, (b) collects one oracle query (human via `/study`, or LLM via a new `OracleAgent` onboarding prompt), (c) routes it through query filter → `parse_feedback`, (d) re-fits KMeans on the fixed embeddings using the resulting intent/K, (e) builds the initial `ClusteringState`, then (f) calls `run_conversation` as today. Keep "autonomous-first" available as an ablation condition.

### 2. Query filter (EXTEND `feedback_parser` + `_contradicts`)

What translation/normalization layers look like in practice: intent recognition → entity extraction → mapping to structured/executable form, with a normalization stage resolving ambiguity and underspecification. For this repo the filter is a thin EXTENSION:
- Reuse `parse_feedback(raw_text, state, client)` to turn the query into `FeedbackDelta`s (it already validates cluster IDs and crashes on hallucinations — keep that fail-loud behavior).
- Add a `normalize(deltas) -> deltas` pass: (1) drop exact duplicates; (2) run `_contradicts` pairwise within the batch and, on conflict, keep the latest (latest-intent-wins, already the project rule); (3) split compound asks into the minimal independent delta set; (4) emit a "filtered_query" audit record showing original vs normalized.
- It NORMALIZES structure only — never reinterprets meaning (see anti-feature). Front-door queries with no current clusters (onboarding) reduce to global/instructional + a target K.

### 3. Coordination agent (NEW, HIGH complexity, LAST PHASE)

How orchestrator/planner agents decompose+fan-out: the dominant production pattern is **orchestrator-worker** (≈70% of multi-agent deployments) — a planner interprets intent, decomposes a high-level goal into atomic subtasks, surfaces dependencies, dispatches to workers (fan-out for independent tasks), then fan-ins/recombines results. For this repo:
- **Decompose:** take one complex normalized operation (e.g. "reorganize the whole space") and break it into PAIRWISE sub-ops (split clusterA; merge clusterB+clusterC) — atomic, independent where possible.
- **Fan-out:** dispatch each pairwise sub-op to its own `run_conversation` session (worker), each with its own audit log / PairBag / `experiment_id`.
- **Fan-in/recombine:** merge resulting assignments back into one `ClusteringState`, re-validating with the contradiction layer so parallel edits don't reintroduce conflicts; latest-intent-wins arbitrates collisions.
- **Gate:** invoke only for genuinely decomposable complex ops; simple single-pair edits stay in the single existing loop (see anti-feature). FLAG for a dedicated research/spike phase before implementation — this is the milestone's highest-risk item.

### 4. Interactive UMAP recolor (EXTEND existing projection)

Common UX pattern: fit the 2D projection ONCE, cache the coordinates, and on cluster-membership change just **recolor** points by their cached coordinate — never refit, because refitting jitters the layout and breaks the user's mental map (a known trust problem in projection-based HITL tools). Refit (accepting relayout) ONLY when topology genuinely changes — split/merge — which is already the recompute trigger in current code. Client receives cluster assignments via the existing `state_update` payload and restyles markers in place.

### 5. Real-time chat view (EXTEND `/study` + `/watch`)

Two modes already exist: `/study` (human oracle types feedback) and `/watch` (observe the LLM oracle). The chat view is an append-only, turn-keyed transcript driven by socket events: each turn renders the system's message and the oracle's reply. `/study` adds an input box whose submissions route through query filter → parser (never a bypass side channel). `/watch` is read-only streaming. Keep it a presentation layer over the existing event stream.

## MVP Definition (this milestone)

### Build first (earlier phases)

- [ ] **Dataset introduction/summary** — orientation is a precondition for an informed initial query.
- [ ] **Oracle initial query → first KMeans re-fit (fixed embeddings)** — the milestone's core reframe.
- [ ] **Query filter (extend parser + contradiction layer)** — first clustering must use clean instructions; reuses existing code.
- [ ] **Versioned YAML oracle config** — reproducibility precondition; cheap; unblocks configurable onboarding prompts.
- [ ] **Interactive UMAP recolor** — extends existing projection; high UX value, low cost.
- [ ] **Real-time chat view (human + LLM)** — extends existing routes; needed for human study + watch mode.

### Add after core is stable (mid-milestone)

- [ ] **Oracle-initiated vs autonomous-first ablation condition** — expose both modes as a measurable factor once onboarding is stable.
- [ ] **Pluggable embedding backends (ST + OpenAI), KMeans-only** — orthogonal infra; land once the loop reshape is stable.

### Defer (last phase)

- [ ] **Coordination agent (pairwise decompose + N-session fan-out)** — deferred by locked decision; highest risk; build only after simpler upgrades stabilize. Needs its own research spike on cross-session contradiction recombination.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Dataset introduction/summary | HIGH | LOW | P1 |
| Oracle initial query → first re-fit | HIGH | MEDIUM | P1 |
| Query filter (extend parser/contradiction) | HIGH | MEDIUM | P1 |
| Versioned YAML oracle config | MEDIUM | LOW | P1 |
| Interactive UMAP recolor | MEDIUM | LOW | P1 |
| Real-time chat view (human + LLM) | HIGH | MEDIUM | P1 |
| Oracle-initiated vs autonomous ablation | MEDIUM | LOW (once onboarding exists) | P2 |
| Pluggable embedding backends / KMeans-only | MEDIUM | MEDIUM | P2 |
| Coordination agent (pairwise fan-out) | HIGH (at scale) | HIGH | P3 (last phase) |

**Priority key:** P1 = must have for milestone core; P2 = add when core is stable; P3 = deferred, highest risk, last phase.

## Prior-Art Feature Analysis (v2 features)

| Feature | Prior art A | Prior art B | Our Approach |
|---------|-------------|-------------|--------------|
| HITL onboarding + intent declaration | AstronomicAL (orient → label highest-info points) | IPBC (orient → lasso must/cannot-link constraints) | Orient via dataset summary → single free-text initial query → first KMeans re-fit on fixed embeddings |
| NL query → structured ops | Text-to-SQL (intent → entities → structured query) | Intent-classification + LLM hybrid NLQ | Reuse `parse_feedback`; add structural normalization (dedupe/contradiction/compound) — never semantic reinterpretation |
| Decompose + fan-out | Orchestrator-worker (≈70% of prod multi-agent) | Fan-out/fan-in for independent parallel tasks | Pairwise sub-ops → N `run_conversation` sessions → contradiction-validated recombination; gated to complex ops |
| Stable projection recolor | UMAP transform on precomputed embedding (avoid refit) | Plotly/scanpy categorical recolor by label | Cache coords from one fit; recolor by cluster on the client; refit only on split/merge |

## Sources (Part A)

- [IPBC: Interactive Projection-Based HITL Semi-Supervised Clustering](https://arxiv.org/pdf/2601.18828) — orientation + projection stability + constraint declaration — MEDIUM
- [AstronomicAL: interactive active-learning labelling dashboard](https://arxiv.org/pdf/2109.05207) — onboarding/orient-then-query workflow — MEDIUM
- [Beyond Labels: Information-Efficient HITL with rich query types](https://arxiv.org/pdf/2602.15738) — rich initial-query/intent patterns — MEDIUM
- [6 Multi-Agent Orchestration Patterns for Production (2026)](https://beam.ai/agentic-insights/multi-agent-orchestration-patterns-production) — orchestrator-worker, fan-out/fan-in — MEDIUM
- [Agent Orchestration Patterns: Swarm vs Mesh vs Hierarchical](https://gurusup.com/blog/agent-orchestration-patterns) — decomposition + recombination — MEDIUM
- [OpenAI Agents SDK — Agent orchestration](https://openai.github.io/openai-agents-python/multi_agent/) — planner/worker decomposition — HIGH
- [Intent-Driven NL Interface: Hybrid LLM + Intent Classification](https://medium.com/data-science-collective/intent-driven-natural-language-interface-a-hybrid-llm-intent-classification-approach-e1d96ad6f35d) — intent normalization layer — MEDIUM
- [From Natural Language to SQL: LLM-based Text-to-SQL Systems](https://arxiv.org/html/2410.01066v1) — NL → structured-command translation — MEDIUM
- [Plotting UMAP results — umap docs](https://umap-learn.readthedocs.io/en/latest/plotting.html) — recolor-by-label; precompute + transform vs refit — HIGH

---
---

# PART B — v1 Feature Research (historical reference)

**Researched:** 2026-04-29
**Interface:** CLI / Jupyter notebook

## Table Stakes (v1)

Features where absence makes the core research question unanswerable. Missing any of these
means the study cannot produce defensible quantitative claims.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Initial clustering with names and natural-language descriptions | The oracle has nothing to react to without it; every HITL clustering system starts here | Low | Sentence embeddings + LLM label generation; one representation only |
| Global feedback acceptance | Canonical feedback type from COBRAS, interactive clustering literature — "too many clusters", "merge everything" | Low | Free-text parsing to structured intent; must map to `f_next_state` |
| Cluster-level feedback acceptance | Split / merge / rename / reweight a specific cluster; the most common oracle action in constrained clustering research | Medium | Requires cluster identity persistence across turns; cluster IDs must be stable |
| Point-level feedback acceptance | "x belongs in B", "x and y should be together"; must-link / cannot-link constraints are the most studied primitive | Medium | Requires point-addressable interface; oracle references items by index, content snippet, or short ID |
| Instructional / declarative feedback acceptance | "treat 'error' and 'fail' as synonyms"; the most powerful feedback type for preference codification | High | LLM parses instruction into a preference constraint; hardest to operationalize cleanly |
| Soft assignments (per-point distribution over K clusters) | Required for calibration testing, boundary detection, uncertainty sampling | Medium | Softmax over embedding distances or LLM confidence scores; storable per-turn |
| `f_uncertainty`: identify low-confidence/boundary points | Needed for uncertainty-driven query strategies; central to the ablation | Medium | Margin between top-2 cluster probabilities |
| `f_next_best_step`: show/ask/stop decision each turn | The core algorithmic contribution; differentiates from passive label collection | High | Implements the interaction strategy being ablated; swappable |
| State persistence across turns | Without this the oracle's earlier preferences disappear | Medium | Structured JSON state: current clustering, feedback turns, preference log, contradiction flags |
| Contradiction / preference-drift tracking | Explicitly required; "latest intent wins" must be implemented and logged | Medium | Compare current vs prior conflicting instructions; surface conflict in the log |
| Stopping signal | Required for convergence measurement | Medium | Three variants for ablation: explicit accept, diminishing-returns, turn budget |
| Turn-by-turn experiment log | Needed for turns-to-convergence, cognitive-load-per-turn, contradiction rate | Medium | JSON-L per run: turn index, feedback type, state delta, soft assignments, uncertainty |
| LLM-simulated oracle (Oracle Agent) | Human oracles too expensive for ablation at scale | High | Needs preference spec, persona, cognitive-load budget; parseable feedback output |
| Ablation across interaction strategies | The headline experiment: 3-5 strategies on convergence efficiency | High | Strategies hot-swappable; random/uncertainty/boundary/full-display baselines |
| Evaluation metrics: turns to convergence, cognitive load per turn | Primary efficiency measures | Low | Cognitive load = items shown + clusters referenced + question complexity |
| Generalization function: codified oracle preference for new items | Explicitly in scope; tests preference transfer | High | Prompt-based classifier or rule set; validated on frozen held-out subset |
| Held-out frozen evaluation subset | Required for valid generalization measurement | Low | Slice before first run; oracle never sees hold-out during training |
| Human validation study (N=5-10) | Required: simulated results validated against humans before any claim | High | Within-subject, consented, scripted; compare LLM-oracle to human-oracle patterns |
| Confidence intervals on headline claims | Stated constraint | Medium | Bootstrap CIs on turns-to-convergence; mean + 95% CI |

## Differentiators (v1)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Hierarchy navigation (zoom in/out) | Multi-resolution feedback; coarse approve, drill into ambiguous sub-cluster (COBRAS super-instances) | High | CLI indented tree; reference level by depth; needs hierarchical clustering step |
| Anytime behavior: valid clustering every turn | COBRAS achieved anytime + query + time efficiency simultaneously; useful during the session | Medium | `f_output` always returns a complete assignment mid-session |
| Pairwise validation probes | "Are x and y correctly grouped?" — behavioral probe independent of self-report | Low | Sample 10-20 pairs per round; agreement rate vs soft-assignment boundary |
| Soft-assignment calibration test | Tests whether boundary-flagged points are actually ambiguous to the oracle | Medium | Oracle judges top-N uncertainty points; compute calibration error |
| Contradiction surface + clarification request | Explicitly surface a conflict instead of silently applying latest-intent | Medium | Diff against preference log; adds a clarification turn type |
| Persona-varied LLM oracle experiments | Same dataset across multiple personas surfaces convergence effects; publishable | Medium | Persona spec as a parameter; runner sweeps N personas |
| Cognitive load budget enforcement in Oracle Agent | Simulates fatigue by limiting items per turn; validates the cognitive-load methodology | Medium | Oracle refuses >K items per turn; returns partial feedback |
| Preference codification quality metric | How well the learned preference generalizes: hold-out accuracy + oracle agreement | Medium | Oracle re-labels 20-30 hold-out items vs preference-function predictions |
| Information gain per cognitive-load unit | Which question type delivers most improvement per unit of oracle attention | High | Define information gain operationally (entropy reduction); plot turn type vs normalized gain |
| Comparison of feedback types at identical decision points | Send different question types at the same point; isolate value of each feedback level | High | Branching experiment runner |

## Anti-Features (v1)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Web UI / dataset upload / persistent sessions | Was out of scope in v1 PROJECT.md (note: v2 milestone reverses this — UI is now in scope) | (v1) CLI/notebook + JSON session files |
| UMAP / t-SNE visualization | (v1) deferred — note v2 makes this in scope | (v1) track soft-assignment distributions numerically |
| Multiple clustering backends | (v1) backend comparison is a separate research question — note v2 adds pluggable backends/KMeans-only | (v1) fix one representation |
| Real-time model fine-tuning / embedding updates from feedback | Conflates better representations with better interaction strategy | Feedback updates cluster assignments + preference log only; embeddings frozen |
| Full N×M human oracle study | Requires larger team; deferred | Small validation study (N=5-10) |
| Noise-tolerant constraint handling (nCOBRAS) | Adds redundant queries; conflicts with "latest intent wins" (signal not noise) | Log + surface contradictions; do not silently absorb/retry |
| Confidence-weighted ML/CL constraint propagation | A full sub-field (COP-k-means, PCKMeans); off-topic algorithmic work | Soft assignments + direct feedback; constraints as preference signals |
| Automatic K selection/optimization loop | Bypasses oracle as objective function | K changes only via oracle feedback |
| Inter-annotator agreement scoring | Only for simultaneous multi-oracle; study is within-subject/single-oracle | Not applicable to the design |
| Rich cluster metadata UI (confidence bars, distances) | Cognitive overload (Prodigy: "ask as little as possible") | Surface only what the specific question needs |
| Asynchronous / batch oracle sessions | Breaks the conversational loop | Keep synchronous; LLM oracle is fast enough |

## Sources (Part B)

- [Interactive Clustering: A Comprehensive Review (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3340960) — MEDIUM
- [COBRAS: Interactive Clustering with Pairwise Queries (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-01768-2_29) — HIGH
- [COBRAS paper PDF (DTAI)](https://dtai.cs.kuleuven.be/software/cobras/cobras_ida_cameraready.pdf) — HIGH
- [Large Language Models Enable Few-Shot Clustering (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00648/120476/Large-Language-Models-Enable-Few-Shot-Clustering) — HIGH
- [Prodigy: radically efficient machine teaching (Explosion)](https://explosion.ai/blog/prodigy-annotation-tool-active-learning) — HIGH
- [Human-in-the-loop ML: state of the art (Springer AI Review)](https://link.springer.com/article/10.1007/s10462-022-10246-w) — HIGH
- Project specification: `.planning/PROJECT.md`

---
*Feature research for: HITL conversational clustering — Part A (v2.0 milestone, 2026-05-21), Part B (v1, 2026-04-29)*
