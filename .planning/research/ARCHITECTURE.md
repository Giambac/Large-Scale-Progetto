# Architecture Patterns

**Domain:** Multi-agent conversational clustering research system
**Researched:** 2026-04-29
**Overall confidence:** HIGH (core patterns well-established; oracle cognitive-load modeling is MEDIUM — active research area)

---

## Recommended Architecture

The system is a **turn-based, supervisor-coordinated multi-agent loop** with a single shared state object as the authority. Three agents — Clustering, Oracle, Judge — never call each other directly. All communication passes through a thin Orchestrator that owns the state and the turn schedule. This is the architecture used in production LangGraph / AutoGen systems and is the right fit here because:

1. The conversation is naturally sequential (one turn at a time).
2. All three agents need to read the same state; write access must be serialized.
3. Ablation experiments swap strategies inside agents, not between them — the orchestrator stays unchanged.

```
                         ┌─────────────────────────────────────────────┐
                         │                  ORCHESTRATOR                │
                         │   owns ClusteringState, routes turns,        │
                         │   writes audit log, enforces turn budget      │
                         └───────┬──────────┬───────────────┬───────────┘
                                 │          │               │
                          read/  │    read/ │         read/ │
                          write  │    write │         read  │
                                 ▼          ▼               ▼
                      ┌──────────────┐  ┌──────────┐  ┌──────────┐
                      │  Clustering  │  │  Oracle  │  │  Judge   │
                      │    Agent     │  │  Agent   │  │  Agent   │
                      └──────────────┘  └──────────┘  └──────────┘
```

**Turn flow:**
```
Turn N:
  1. Orchestrator calls ClusteringAgent.f_next_best_step(state)
     → returns Action: {type: show|ask|stop, payload}
  2. If stop → skip to 4 (Judge makes final eval)
  3. Orchestrator delivers Action to OracleAgent.respond(action, state)
     → returns OracleReply: {feedback_type, content, load_estimate}
  4. Orchestrator calls ClusteringAgent.f_next_state(state, oracle_reply)
     → returns updated ClusteringState
  5. Orchestrator calls JudgeAgent.f_eval(state)
     → returns EvalResult: {continue, metrics, convergence_signal}
  6. If not continue → terminate loop
  7. Increment turn counter, log everything, repeat
```

---

## Component Boundaries

| Component | Responsibility | Reads | Writes |
|-----------|---------------|-------|--------|
| **Orchestrator** | Turn scheduling, state authority, logging, turn-budget enforcement | ClusteringState | ClusteringState (delegated), AuditLog |
| **ClusteringAgent** | `f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state` | ClusteringState | Returns new ClusteringState (never mutates directly) |
| **OracleAgent** | Simulated human response with persona + cognitive-load constraint | ClusteringState (read-only), Action payload | Returns OracleReply (never mutates state) |
| **JudgeAgent** | `f_eval`, convergence detection, metric accumulation | ClusteringState, full TurnHistory | Returns EvalResult (never mutates state) |
| **EmbeddingStore** | Precomputed sentence embeddings, nearest-neighbor queries | Dataset | Read-only after init |
| **AuditLog** | Append-only turn trace, reproducibility | — | Appended each turn |

**Key constraint:** Oracle and Judge are read-only consumers of state. Only the ClusteringAgent proposes state mutations; the Orchestrator applies them. This prevents race conditions and makes replay trivial.

---

## Data Structures

### ClusteringState (the canonical object)

```python
@dataclass
class Cluster:
    id: str                             # stable UUID, never reused
    name: str                           # human-readable label
    description: str                    # natural language summary
    member_ids: list[str]               # point IDs (hard assignment fallback)
    representative_ids: list[str]       # 3-5 centrally positioned examples

@dataclass
class SoftAssignment:
    point_id: str
    distribution: dict[str, float]      # cluster_id -> probability, sums to 1.0
    entropy: float                      # derived: uncertainty of this point
    is_boundary: bool                   # entropy > threshold

@dataclass
class PreferenceModel:
    constraints_merge: list[tuple[str,str]]     # (cluster_a, cluster_b) must merge
    constraints_split: list[str]                # cluster_id should be split
    constraints_must_link: list[tuple[str,str]] # (point_a, point_b) same cluster
    constraints_cannot_link: list[tuple[str,str]]
    feature_weights: dict[str, float]           # instructional feedback
    synonyms: list[tuple[str,str]]              # ("error","fail") = same
    drift_history: list[PreferenceDelta]        # ordered log of all contradictions

@dataclass
class PreferenceDelta:
    turn: int
    old_constraint: str
    new_constraint: str
    delta_type: str     # "contradiction" | "refinement" | "extension"

@dataclass
class ClusteringState:
    # Core clustering
    clusters: list[Cluster]
    soft_assignments: list[SoftAssignment]
    k: int                              # current number of clusters
    embeddings_ref: str                 # path/key to EmbeddingStore (not inlined)

    # Preference model (oracle's accumulated intent)
    preference_model: PreferenceModel

    # Conversation
    turn_history: list[TurnRecord]
    turn_number: int
    turn_budget: int

    # Internal metrics (diagnostic, not objective)
    silhouette_score: float | None
    avg_intra_cluster_distance: float | None

    # Generalization
    codified_mapping: dict | None       # available after oracle satisfaction
```

### TurnRecord (conversation history entry)

```python
@dataclass
class TurnRecord:
    turn: int
    action: Action                  # what Clustering Agent decided to show/ask
    oracle_reply: OracleReply       # oracle's response
    eval_result: EvalResult         # judge's assessment
    load_spent: float               # estimated cognitive load this turn
    cumulative_load: float          # running total
```

### Action (what Clustering Agent proposes each turn)

```python
@dataclass
class Action:
    type: Literal["show_full", "show_subset", "show_boundary", "ask_pairwise",
                  "ask_merge_confirm", "ask_name", "stop"]
    payload: dict               # type-specific: which clusters/points to show
    rationale: str              # why this action was chosen (logged)
    strategy_id: str            # which ablation strategy produced this (e.g. "uncertainty-driven")
```

### OracleReply

```python
@dataclass
class OracleReply:
    feedback_type: Literal["global", "cluster_level", "point_level", "instructional",
                           "satisfaction", "no_comment"]
    content: str                # natural language, parsed by ClusteringAgent
    structured_delta: dict      # parsed intent: merges, splits, must-links, etc.
    load_estimate: float        # 0-1 scale, computed by OracleAgent from action payload
    persona_consistency_score: float  # self-reported; used by Judge for oracle validity
```

### EvalResult

```python
@dataclass
class EvalResult:
    continue_loop: bool
    convergence_signal: Literal["none", "weak", "strong", "explicit_stop"]
    metrics: dict[str, float]   # turns_to_now, load_total, drift_count, etc.
    stopping_reason: str | None # populated when continue_loop=False
```

---

## Agent Implementation Patterns

### Clustering Agent: f_next_best_step Decision Logic

The decision between show/ask/stop is a policy — the primary ablation axis. The architecture must make this policy pluggable via a Strategy interface.

```
ClusteringAgent.f_next_best_step(state) dispatches to:
  self.strategy.select_action(state) → Action

Strategy implementations (ablation targets):
  - RandomStrategy:          random choice from valid actions
  - UncertaintyStrategy:     prioritize highest-entropy boundary points → ask_pairwise
  - InformationGainStrategy: estimate IG per action type; pick max
  - BudgetAwareStrategy:     modulate based on remaining turn budget
  - HybridStrategy:          combine uncertainty + representativeness
```

**Decision heuristics (within any strategy):**
- If `turn_number == 1`: always `show_full` (oracle needs an anchor)
- If `avg_entropy > threshold AND remaining_budget > 5`: prefer `ask_pairwise` on highest-entropy point
- If `preference_model.drift_history` has recent contradiction: `ask_merge_confirm` to surface it
- If oracle gave `satisfaction` last turn AND judge says `convergence_signal == "strong"`: `stop`
- If `remaining_budget <= 2`: `show_subset` of most-changed clusters, then `stop`

**f_uncertainty** produces a ranked list of:
1. Points with highest soft-assignment entropy (boundary candidates)
2. Clusters with most internal distance variance (split candidates)
3. Cluster pairs with closest inter-cluster distance (merge candidates)
4. Unresolved preference contradictions (clarification candidates)

### Oracle Agent: Cognitive Load and Preference Drift

The Oracle Agent is an LLM called with a system prompt that encodes persona + preference spec + load constraint. This is the "simulated human" for ablations.

**Cognitive load model.** Load is computed from the Action payload before calling the LLM:

```
load(action) =
  w_items  * count(items_shown)    # number of data points displayed
+ w_clust  * count(clusters_shown) # number of clusters shown
+ w_text   * text_length_tokens    # length of descriptions
+ w_q_type * question_complexity   # pairwise < merge < name < global
```

Weights are hyperparameters (defaults from literature; tunable). The Oracle Agent tracks `cumulative_load` and is prompted to behave more superficially when load is high:

```
System prompt fragment:
  "You have reviewed [N] clusters across [T] turns. Your cumulative mental effort
   score is [L/L_max]. When this exceeds 0.7, begin giving shorter, less nuanced
   responses; prioritize major disagreements over fine-grained corrections."
```

**Preference drift handling.** The Oracle is prompted with its own `preference_model` history so it stays internally consistent but can evolve:

```
System prompt fragment:
  "Your stated preferences so far: [preference_model summary].
   If your current response contradicts a prior preference, explicitly flag it
   as a change of intent with: DRIFT: [old intent] → [new intent]."
```

The ClusteringAgent parses `DRIFT:` markers and logs them to `PreferenceModel.drift_history`. Latest intent always overwrites the constraint — never discards history.

**Persona spec format** (drives LLM oracle for each experimental condition):

```json
{
  "persona_id": "domain_expert_skeptical",
  "background": "10 years in customer support; cares about actionability",
  "preference_bias": "prefers fewer, broader clusters; dislikes fine-grained splits",
  "attention_pattern": "skimmer — reads labels + 1 example per cluster",
  "feedback_style": "terse; uses yes/no + one correction per turn",
  "cognitive_decay_rate": 0.15
}
```

### Judge Agent: Convergence Detection

The Judge runs every turn but its `continue_loop` signal only terminates the loop if the Orchestrator's budget is also satisfied. This separation prevents premature stopping.

**Convergence signals:**
1. **Explicit oracle satisfaction** — Oracle's reply includes `feedback_type: "satisfaction"` with no structural changes requested. Strong signal.
2. **Feedback magnitude decay** — Track `|structured_delta|` (number of changes requested) across turns. If 3 consecutive turns have `|delta| <= 1`, emit `"weak"` convergence.
3. **Preference stability** — `drift_count` in last 3 turns == 0 AND `preference_model` changes are minor. Strengthens signal.
4. **Diminishing returns** — Soft-assignment entropy averaged over all points is below threshold AND silhouette score is stable (< 2% change last 3 turns). Secondary; not sole criterion.
5. **Turn budget exhaustion** — Hard stop regardless of convergence quality. Logged as `stopping_reason: "budget_exhausted"`.

**f_eval produces metric bundle per turn:**
```
turns_to_now, total_load, load_per_turn,
drift_events_total, drift_events_last_3,
feedback_magnitude_last_3, convergence_signal,
silhouette_history, entropy_history,
oracle_satisfaction_flag
```

The Judge also implements **oracle validity checks** for the simulated setting:
- `persona_consistency_score` drops below threshold → log warning (oracle drifted from persona)
- Oracle gives identical response 3 turns in a row → log `oracle_stuck` warning

---

## Multi-Agent Coordination Pattern

**Pattern: Turn-Based Supervisor (not event-driven)**

Rationale: The conversation is fundamentally sequential. Event-driven coordination (where agents fire independently on message receipt) adds concurrency complexity with no benefit — only one thing happens per turn. A simple synchronous turn loop is more reproducible, easier to trace, and simpler to replay for ablation analysis.

The Orchestrator is a **thin coordinator, not a smart planner**. It does not make strategic decisions — that belongs to the ClusteringAgent. The Orchestrator:
- Advances the turn counter
- Applies ClusteringState mutations returned by ClusteringAgent
- Routes the Action to OracleAgent
- Routes the updated state to JudgeAgent
- Checks `EvalResult.continue_loop` and budget
- Appends to AuditLog
- Raises a clean `ExperimentComplete` exception to terminate

No LLM call is made in the Orchestrator itself.

**Why not LangGraph or AutoGen?**

Both frameworks are valid but add dependency weight for a research system where the conversation graph is fixed (not dynamic). The 3-agent sequential loop maps naturally to a plain Python `while` loop with dataclasses. Use LangGraph only if the graph needs to become conditional mid-project (e.g., adding a 4th data generation agent with branching). For ablations, a plain strategy registry is simpler to reproduce than a graph serialization format.

---

## Ablation Experiment Design

Ablation experiments swap the strategy inside ClusteringAgent. Everything else is held fixed. This requires:

**1. Strategy Registry**
```python
STRATEGY_REGISTRY = {
    "random":           RandomStrategy,
    "uncertainty":      UncertaintyStrategy,
    "information_gain": InformationGainStrategy,
    "budget_aware":     BudgetAwareStrategy,
    "hybrid":           HybridStrategy,
}
```

**2. Experiment Config (JSON/YAML)**
```json
{
  "experiment_id": "ablation_002",
  "strategy": "uncertainty",
  "oracle_persona": "domain_expert_skeptical",
  "dataset": "amazon_reviews_frozen_v1",
  "turn_budget": 20,
  "random_seed": 42,
  "replications": 5
}
```

**3. Fixed interfaces** — All strategies implement the same `select_action(state) -> Action` signature. Swapping strategies never touches Orchestrator, OracleAgent, or JudgeAgent code.

**4. Deterministic replay** — AuditLog records the full state at every turn + random seed. Any run can be replayed exactly by re-feeding the log through the Orchestrator.

**5. Metric collection** — All metrics are emitted through JudgeAgent.f_eval, not scattered through the codebase. Analysis scripts read AuditLog only — never internal agent state.

---

## Data Flow

```
[Dataset files]
       │
       ▼
[EmbeddingStore]  ←── sentence-transformers / precomputed at init
       │
       ▼ (read-only after init)
[Orchestrator initializes ClusteringState from embeddings + initial k-means pass]
       │
       └──────────────────────────── TURN LOOP ────────────────────────────────┐
                                                                                │
  ClusteringState ──────────────────────────────────────────────────────────┐  │
       │                                                                    │  │
       ▼                                                                    │  │
  ClusteringAgent.f_next_best_step(state)                                  │  │
       │ returns Action                                                     │  │
       ▼                                                                    │  │
  OracleAgent.respond(action, state)  ← reads oracle persona config        │  │
       │ returns OracleReply                                                │  │
       ▼                                                                    │  │
  ClusteringAgent.f_next_state(state, oracle_reply)                        │  │
       │ returns NEW ClusteringState (immutable update)                    │  │
       ▼                                                                    │  │
  Orchestrator applies new state ──────────────────────────────────────────┘  │
       │                                                                        │
       ▼                                                                        │
  JudgeAgent.f_eval(state) → EvalResult                                        │
       │                                                                        │
       ▼                                                                        │
  AuditLog.append(TurnRecord)                                                   │
       │                                                                        │
       ├── if continue_loop AND budget remaining ──────────────────────────────┘
       │
       └── else → ExperimentComplete(final_state, audit_log)
```

**Information flows one direction per turn.** Nothing feeds backward (OracleAgent never writes to ClusteringState; JudgeAgent never calls ClusteringAgent). This makes each component independently testable.

---

## Suggested Build Order

Dependencies determine build order. Items in each phase can be built in parallel within a pair.

### Phase 1 — Foundation (no agents yet)

1. `EmbeddingStore` — load dataset, compute sentence embeddings, expose nearest-neighbor queries
2. `ClusteringState` dataclasses — all data structures, Pydantic or dataclass validation
3. Initial k-means pass → populate `ClusteringState` with first clustering
4. `AuditLog` — append-only JSONL writer + reader

**Gate:** Can produce and serialize an initial `ClusteringState` from a real dataset.

### Phase 2 — Clustering Agent (core loop logic, no LLM yet)

5. `f_output` — return best-guess clustering from current state (trivial: return `state.clusters`)
6. `f_uncertainty` — compute entropy from soft assignments; rank boundary points
7. `RandomStrategy` — `f_next_best_step` with random action selection (testable without LLM)
8. `f_next_state` — parse a hardcoded oracle reply and mutate state (tests state update logic)

**Gate:** Full turn loop runs with random strategy + mocked oracle replies. State evolves correctly.

### Phase 3 — Oracle Agent

9. `OracleAgent` base — LLM call with system prompt, returns structured OracleReply
10. Cognitive load calculator — `load(action)` function
11. Preference drift detection — parse `DRIFT:` markers, update `PreferenceModel`
12. Persona config loader — JSON persona spec → system prompt

**Gate:** Oracle produces valid OracleReply for each Action type; drift is logged.

### Phase 4 — Judge Agent

13. `JudgeAgent.f_eval` — convergence signal logic, metric bundle
14. Feedback magnitude tracker — `|structured_delta|` across last N turns
15. Budget enforcement in Orchestrator

**Gate:** Judge terminates the loop correctly under all five stopping conditions.

### Phase 5 — Real Strategies (ablation-ready)

16. `UncertaintyStrategy` — select highest-entropy point, prefer `ask_pairwise`
17. `InformationGainStrategy` — estimate IG per action, select max
18. `BudgetAwareStrategy` — modulate by remaining budget
19. `HybridStrategy` — combine uncertainty + representativeness

**Gate:** All strategies run through full loop; AuditLog shows strategy_id on every turn.

### Phase 6 — Experiment Harness + Evaluation

20. Experiment config loader (JSON/YAML)
21. Multi-run executor (N replications per condition, consistent seeds)
22. Analysis scripts (read AuditLog only; compute CIs)
23. Human validation protocol implementation (pairwise probe)

**Gate:** Ablation table reproducible from AuditLog + config file alone.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Agents Sharing Mutable State Directly
**What:** Oracle or Judge hold a reference to ClusteringState and mutate it in place.
**Why bad:** Destroys reproducibility; race conditions if any async is added; silent state corruption.
**Instead:** Return new state objects (immutable update pattern); Orchestrator is sole writer.

### Anti-Pattern 2: Cognitive Load Computed Inside the LLM Call
**What:** Oracle LLM is asked to estimate its own cognitive load in the same call.
**Why bad:** LLMs self-report load unreliably; mixes evaluation with generation; not reproducible.
**Instead:** Compute load deterministically from Action payload before the LLM call; pass the result as a system prompt fact.

### Anti-Pattern 3: Strategy Logic Embedded in Orchestrator
**What:** `if strategy == "uncertainty": ... elif strategy == "random": ...` in the main loop.
**Why bad:** Cannot add strategies without modifying the loop; strategies are not independently testable.
**Instead:** Strategy registry + interface; Orchestrator never references strategy names.

### Anti-Pattern 4: Convergence Detection in Clustering Agent
**What:** ClusteringAgent decides when to stop by emitting `stop` action based on its own assessment.
**Why bad:** Conflates optimization (what to show/ask) with evaluation (is it good enough). Judge loses authority; ablation experiments cannot independently vary convergence threshold.
**Instead:** ClusteringAgent can emit `stop` as a suggestion; Orchestrator only terminates on JudgeAgent confirmation + budget check. Both signals required.

### Anti-Pattern 5: Metrics Computed Scattered Across Agents
**What:** Each agent logs its own metrics to separate files.
**Why bad:** Analysis requires joining multiple logs; easy to misalign turns.
**Instead:** All metrics flow through JudgeAgent.f_eval → AuditLog. One source of truth.

### Anti-Pattern 6: Inlining Embeddings in ClusteringState
**What:** Store the full embedding matrix (N x 768) inside ClusteringState, serialized each turn.
**Why bad:** AuditLog balloons to GB; serialization becomes bottleneck; embeddings never change.
**Instead:** EmbeddingStore is a separate, read-only singleton. ClusteringState stores only a reference key.

---

## Scalability Considerations

This is a research system, not a production service. Scalability targets are for experimental scale (many ablation runs), not user scale.

| Concern | At 1 run | At 100 ablation runs | Notes |
|---------|----------|---------------------|-------|
| LLM API cost | ~$0.01-0.10/run | ~$1-10 total | Cache OracleAgent responses per (action, state_hash) |
| State serialization | Trivial | Trivial | JSONL, one record per turn |
| Embedding computation | ~5-30s for 1K points | Precomputed once | EmbeddingStore init is one-time |
| Analysis | Instant | Seconds | Read AuditLog, compute metrics; pandas/numpy sufficient |
| Parallelism | Sequential | Trivially parallelizable | Each ablation run is independent; use multiprocessing |

---

## Sources

- Multi-agent architecture patterns: [The Orchestration of Multi-Agent Systems](https://arxiv.org/html/2601.13671v1) — HIGH confidence
- LangGraph stateful agent workflows: [LangGraph documentation](https://www.langchain.com/langgraph) — HIGH confidence
- AutoGen conversation framework: [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) — HIGH confidence
- Uncertainty sampling and boundary points: [Active Learning Overview](https://medium.com/data-science/active-learning-overview-strategies-and-uncertainty-measures-521565e0b0b) — MEDIUM confidence (WebSearch verified by multiple sources)
- Active learning stopping criteria: [Hitting the target: stopping active learning at the cost-based optimum](https://link.springer.com/article/10.1007/s10994-022-06253-1) — MEDIUM confidence
- Cognitive load in LLM conversations: [Precision Proactivity: Measuring Cognitive Load in AI-Assisted Work](https://arxiv.org/pdf/2505.10742) — MEDIUM confidence
- LLM simulation of human biases: [Emulating Aggregate Human Choice Behavior with GPT Agents](https://arxiv.org/html/2602.05597v1) — MEDIUM confidence
- Preference drift detection: [LLM Perception Drift](https://www.stridec.com/blog/llm-perception-drift-why-matters-ai-applications/) — LOW confidence (WebSearch only)
- Ablation study design: [Ablation Studies: The Operating System for Trustworthy AI](https://medium.com/@adnanmasood/ablation-studies-the-operating-system-for-trustworthy-ai-decisions-b99300d3bd32) — MEDIUM confidence
- Supervisor/coordinator pattern: [Agent system design patterns](https://docs.databricks.com/aws/en/generative-ai/guide/agent-system-design-patterns) — HIGH confidence
- Sentence embeddings for clustering: [Text Clustering with LLM Embeddings](https://arxiv.org/html/2403.15112v5) — HIGH confidence
