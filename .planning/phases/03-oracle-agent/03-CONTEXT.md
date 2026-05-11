# Phase 3: Oracle Agent - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the LLM Oracle Agent — a configurable, measurable stand-in for a human oracle. The agent implements `OracleProtocol.reply(state, message) → OracleReply` without changing the conversation loop. It takes a preference specification and noise parameters at construction, computes a per-turn cognitive load score, generates replies that simulate human fatigue and noise, detects contradictions with prior feedback using structural FeedbackDelta comparison, and logs drift events to the AuditLog. Phase 3 replaces `MockOracle` everywhere in the system.

Requirements in scope: ORC-01, ORC-02, ORC-03, ORC-04, FB-04.

</domain>

<decisions>
## Implementation Decisions

### Preference Specification (ORC-01)

- **D-01:** **`OracleSpec` dataclass** — three fields: `preferred_k: int` (target cluster count the oracle aims for), `semantic_axes: list[str]` (dimensions it groups by, e.g. `["topic", "sentiment"]`), `persona_description: str` (free-text flavor injected into the LLM system prompt). No additional fields for v1.
- **D-02:** **Injected at construction** — `OracleAgent.__init__(spec: OracleSpec, noise_params: NoiseParams, ...)`. The spec is fixed for the lifetime of the agent. Different experiment runs instantiate different `OracleAgent` objects with different specs. No mutation mid-run.
- **D-03:** `OracleSpec` and `NoiseParams` are separate dataclasses. `NoiseParams` carries the three noise parameters (ORC-02). Keeping them separate allows ablations that vary noise without touching the preference spec.

### Noise Parameters (ORC-02)

- **D-04:** **Prompt-injected natural-language behavioral rules** — each parameter becomes a behavioral instruction in the LLM system prompt. No post-processing, no second LLM call, no temperature manipulation.
  - `consistency_rate` → "You agree with the proposed clustering X% of the time. The other Y% of the time you request minor adjustments."
  - `drift_probability` → "With probability X you introduce a new preference per turn that may contradict a prior one."
  - `sycophancy_resistance` → "You maintain your stated position even when the system pushes back, at rate X."
- **D-05:** **`NoiseParams` dataclass, fixed at construction** — `consistency_rate: float`, `drift_probability: float`, `sycophancy_resistance: float`. All three are injected into `OracleAgent.__init__`. Values are logged to the AuditLog at run start (as a `oracle_init` event) so every run is reproducible. ORC-02 requires this logging for cross-run correlation in Phase 6.

### Cognitive Load (ORC-03)

- **D-06:** **Standalone `f_cognitive_load(state, message) → float`** in a new `src/cognitive_load.py`. The conversation loop calls it before `oracle.reply()` and passes the score into the oracle's system prompt. Consistent with the `f_*` function decomposition pattern (D-02 from Phase 2). The oracle does not compute its own load.
- **D-07:** **Formula:** `load = (len(state.clusters) / MAX_K) * w1 + (items_shown / total_items) * w2 + (len(message) / MAX_MSG_LEN) * w3`, normalized to `[0, 1]`. Default weights `w1 = w2 = w3 = 1/3` (equal weighting). `MAX_K`, `MAX_MSG_LEN` are named constants in `cognitive_load.py`.
  - `items_shown`: estimated from `len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER` (a constant, since `_format_message` shows top-5 per cluster by default).
- **D-08:** **Hard threshold `COG_LOAD_THRESHOLD = 0.7`** (named constant). Above 0.7, the oracle's system prompt instructs it: "Focus on a single cluster or item. Acknowledge only one issue in this reply." Below 0.7, no constraint on reply scope.

### Drift Detection (ORC-04)

- **D-09:** **Structural `FeedbackDelta` comparison** — no extra LLM call. After the oracle generates a reply and `parse_feedback` produces deltas, compare each delta against a rolling window of prior deltas (last N turns, default N=10). Contradiction rules:
  - `MergeFeedback(A, B)` contradicts a prior `SplitFeedback(cluster_id=A)` or `SplitFeedback(cluster_id=B)` on the same cluster IDs.
  - `SplitFeedback(A)` contradicts a prior `MergeFeedback` that produced A.
  - `MoveItemFeedback(item, target=B)` contradicts a prior `MoveItemFeedback(item, target=C)` where C ≠ B.
  Ignores `GlobalFeedback` and `InstructionalFeedback` (too semantic for structural comparison).
- **D-10:** **Drift logged to AuditLog only** — no in-memory `drift_history` field on `OracleAgent`. Each detected contradiction is written as a `drift_event` JSONL record alongside the turn's normal AuditLog entry. Keeps `OracleAgent` stateless beyond its delta window.
- **D-11:** **`OracleReply` extended with `contradiction_detected: bool` and `contradicted_turn: int | None`** — fields added to the existing `OracleReply` dataclass (NOT frozen per D-03 in Phase 2). The loop receives these and emits them via SocketIO `state_update` for the debug UI. The Clustering Agent sees them as part of the standard reply object.

### Instructional Constraints (FB-04)

- **D-12:** `InstructionalFeedback` is already parsed to a structured dataclass by `feedback_parser.py` (Phase 2). Phase 3 wires it into the oracle's system prompt via the existing `global_instructions` accumulator in `conversation_loop.py`. Each `InstructionalFeedback.instruction_text` appended to `global_instructions` is injected into the oracle's prompt context on the next turn. No new ConstraintStore needed.

### Claude's Discretion

- Which Anthropic model to use for the oracle's `reply()` call — researcher picks appropriate cost/quality tradeoff (Haiku for speed in ablations, Sonnet for quality in human-comparison runs).
- Exact system prompt structure (ordering of spec, noise params, cognitive load instructions, global instructions, current state summary) — planner decides.
- Rolling window size N for structural drift comparison (default suggested: 10 turns) — planner decides.
- Named constants: `MAX_K`, `MAX_MSG_LEN`, `TOP_K_ITEMS_PER_CLUSTER` — planner picks reasonable defaults.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning Artifacts
- `.planning/REQUIREMENTS.md` §Oracle Agent — ORC-01 through ORC-04 (structured oracle, noise params, cognitive load, drift detection)
- `.planning/REQUIREMENTS.md` §Feedback Types — FB-04 (instructional feedback → structured constraints)
- `.planning/ROADMAP.md` §Phase 3 — goal, success criteria, note about ORC-02 logging for Phase 6
- `.planning/PROJECT.md` — fail-loudly philosophy, architecture constraints

### Prior Phase Decisions
- `.planning/phases/02-clustering-agent-core/02-CONTEXT.md` — D-01 (plain Python loop), D-02 (standalone modules), D-03 (OracleProtocol interface — DO NOT CHANGE), D-04 (loop owns JSONL write), D-05 (feedback_parser.py is the only LLM parsing point), D-06/D-07 (FeedbackDelta types), D-11 (cluster ID monotonic counter)

### Phase 3 Integration Points (source files to read before implementing)
- `src/oracle_protocol.py` — `OracleProtocol`, `OracleReply`, `MockOracle`. Phase 3 replaces MockOracle; extends OracleReply with `contradiction_detected` and `contradicted_turn`.
- `src/conversation_loop.py` — `run_conversation()`. Phase 3 adds `f_cognitive_load` call before `oracle.reply()` and logs drift from `reply.contradiction_detected`. `global_instructions` accumulator already wired (FB-04 uses it).
- `src/feedback_parser.py` — `parse_feedback()` and `InstructionalFeedback`. FB-04 constraint injection uses the existing `global_instructions` path.
- `src/feedback.py` — all `FeedbackDelta` types. D-09 structural comparison operates on these types.
- `src/serialization.py` — `append_to_audit_log()`. Drift events are appended as special JSONL records using the same function.

### Reference Script
- `Conversational Clustering Script.txt` — contains oracle persona description patterns and the cognitive-load budget framing that informed D-07 and D-08.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `OracleProtocol` / `OracleReply` (`src/oracle_protocol.py`): The interface is locked. `OracleReply` accepts new fields (not frozen). Phase 3 adds `contradiction_detected: bool = False` and `contradicted_turn: int | None = None`.
- `parse_feedback()` (`src/feedback_parser.py`): Already handles `InstructionalFeedback`. Phase 3 uses the parsed output directly; no changes to the parser.
- `global_instructions: list[str]` (`src/conversation_loop.py` line ~106): Already accumulated per turn and passed to `f_next_state`. Phase 3 also passes it into the oracle's system prompt.
- `append_to_audit_log()` (`src/serialization.py`): Use for `oracle_init` event (D-05) and `drift_event` records (D-10).

### Established Patterns
- **Fail loudly:** No `try/except` except at the LLM API call boundary inside `OracleAgent.reply()`. Assert all preconditions.
- **Dataclass-first:** `OracleSpec`, `NoiseParams`, `DriftEvent` (if needed for structuring the JSONL record) all as `@dataclass`.
- **Named constants:** `COG_LOAD_THRESHOLD = 0.7`, `MAX_K`, `MAX_MSG_LEN` in `cognitive_load.py`. Same convention as `ORACLE_MOVE_CONFIDENCE = 0.95` and `KMEANS_SOFTMAX_TEMP = 1.0`.
- **Pure functions:** `f_cognitive_load` is a pure function (no I/O, no global state). Takes `state` and `message`, returns `float`.

### Integration Points
- `f_cognitive_load(state, message)` → called in `run_conversation()` before `oracle.reply()` → score injected into oracle system prompt
- `oracle.reply()` → returns `OracleReply` with `contradiction_detected` → loop logs drift event to AuditLog and emits via SocketIO
- `global_instructions` accumulator → injected into oracle system prompt each turn (FB-04 constraint propagation)
- `oracle_init` JSONL event → written once at start of `run_conversation()` with `spec` and `noise_params` values (ORC-02 logging)

</code_context>

<specifics>
## Specific Ideas

- **`NoiseParams` logging format:** `{"event": "oracle_init", "consistency_rate": 0.8, "drift_probability": 0.1, "sycophancy_resistance": 0.9, "preferred_k": 5, "semantic_axes": ["topic", "sentiment"], "turn": 0}` — mirrors the `backend_init` event format already in `web/app.py`.
- **Structural contradiction window:** A `deque(maxlen=10)` of `(turn_index, delta)` pairs inside `OracleAgent`. Checked after each `parse_feedback` call. Zero LLM cost.
- **Cognitive load in SocketIO:** The loop already emits `"cognitive_load": reply.turn_cognitive_load` in `state_update`. Phase 3 fills this in properly via `f_cognitive_load` — no SocketIO changes needed.
- **`COG_LOAD_THRESHOLD = 0.7`:** When load > threshold, append a behavioral instruction at the END of the system prompt: "OVERLOAD: Focus on one thing only."

</specifics>

<deferred>
## Deferred Ideas

- **Mutable noise params (fatigue simulation):** Allowing `consistency_rate` to degrade turn-by-turn to simulate accumulating fatigue. Deferred to Phase 5 ablation harness.
- **LLM-based semantic contradiction detection:** Using an LLM call to detect semantic drift (not just structural). Higher recall but extra cost per turn. Deferred — structural comparison is sufficient for Phase 3; Phase 5 ablation can compare approaches.
- **Per-persona `cog_load_threshold` in `NoiseParams`:** Making the overload threshold configurable per persona. Deferred — `COG_LOAD_THRESHOLD = 0.7` is a good fixed default; add to NoiseParams if ablation reveals it matters.
- **BACK-V2-02 (representation choice):** Out of scope for all phases so far.
- **4th synthetic data agent:** Deferred per PROJECT.md.

</deferred>

---

*Phase: 3-Oracle-Agent*
*Context gathered: 2026-05-11*
