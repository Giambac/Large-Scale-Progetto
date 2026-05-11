# Phase 3: Oracle Agent - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-11
**Phase:** 03-oracle-agent
**Areas discussed:** Preference spec format, Noise parameter mechanics, Cognitive load formula, Drift detection approach

---

## Preference Spec Format

| Option | Description | Selected |
|--------|-------------|----------|
| Semantic axes + preferred K | Structured dataclass with semantic_axes, preferred_k, domain desc | |
| Free-text persona prompt | Single NL paragraph — simpler but harder to vary | |
| Both: structured fields + persona text | Dataclass with axes + K, plus optional persona_description string | ✓ |

**User's choice:** Both — `OracleSpec(preferred_k, semantic_axes, persona_description)`
**Notes:** —

---

| Option | Description | Selected |
|--------|-------------|----------|
| preferred_k, semantic_axes, persona_description | Minimal structured fields | ✓ |
| Add domain + strictness level | More tunable but more experiment parameters | |
| You decide | Researcher/planner picks | |

**User's choice:** `preferred_k: int`, `semantic_axes: list[str]`, `persona_description: str`

---

| Option | Description | Selected |
|--------|-------------|----------|
| Injected at construction | OracleAgent.__init__(spec, noise_params) — fixed for lifetime of agent | ✓ |
| Passed per-turn | OracleAgent.reply(state, message, spec) — changes OracleProtocol signature | |

**User's choice:** Injected at construction. OracleProtocol signature unchanged.

---

## Noise Parameter Mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| Prompt-injected instructions | Each param becomes a behavioral line in the system prompt | ✓ |
| Post-process the reply | Generate consistent reply first, then flip probabilistically | |
| You decide | Researcher/planner picks | |

**User's choice:** Prompt-injected natural-language behavioral rules. No post-processing.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Natural-language behavioral rules | "You agree X% of the time..." — lines in system prompt | ✓ |
| Numeric thresholds with coin flip logic | Tell LLM to simulate a Bernoulli draw each turn | |

**User's choice:** Natural-language rules per parameter.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed at construction, logged per run | NoiseParams injected at init, logged as oracle_init JSONL event | ✓ |
| Mutable mid-run | Params change turn-by-turn to simulate fatigue | |

**User's choice:** Fixed at construction. Logged to AuditLog as `oracle_init` event at run start.

---

## Cognitive Load Formula

| Option | Description | Selected |
|--------|-------------|----------|
| Cluster count + items shown + message length | Weighted average of 3 signals, normalized 0–1 | ✓ |
| Cluster count only | load = num_clusters / max_k — minimal but incomplete | |
| You decide | Researcher/planner picks | |

**User's choice:** Three-signal weighted formula with equal weights (1/3 each).

---

| Option | Description | Selected |
|--------|-------------|----------|
| Threshold 0.7 — acknowledge one thing only | COG_LOAD_THRESHOLD = 0.7, hard cutoff | ✓ |
| Threshold in NoiseParams | Per-persona configurable threshold | |
| Soft degradation — quality degrades linearly | No hard threshold; gradual quality drop | |

**User's choice:** `COG_LOAD_THRESHOLD = 0.7` (named constant). Above: single-focus constraint injected into prompt.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone function, called by the loop | f_cognitive_load(state, message) -> float in src/cognitive_load.py | ✓ |
| Computed inside oracle.reply() | Oracle calculates its own load | |

**User's choice:** Standalone `f_cognitive_load` — pure function, called by conversation loop before `oracle.reply()`.

---

## Drift Detection Approach

| Option | Description | Selected |
|--------|-------------|----------|
| LLM comparison against recent history | Extra LLM call per turn; catches semantic contradictions | |
| Structural FeedbackDelta comparison | Compare delta types on same cluster IDs; zero LLM cost | ✓ |
| Embedding similarity of replies | Cosine similarity per turn; noisy for short replies | |

**User's choice:** Structural FeedbackDelta comparison. Rolling window of prior deltas, check for type-level contradictions (merge vs. split on same cluster, move vs. prior move on same item).

---

| Option | Description | Selected |
|--------|-------------|----------|
| List on OracleAgent, appended each turn | self.drift_history: list[DriftEvent] | |
| Written to AuditLog only | drift_event JSONL record, no in-memory list | ✓ |
| Stored on ClusteringState | Mutates frozen state schema — conflicts with D-13 | |

**User's choice:** AuditLog only. No in-memory drift_history. Keeps OracleAgent lean.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Flag in OracleReply | contradiction_detected: bool, contradicted_turn: int\|None — no protocol change | ✓ |
| Separate method on OracleAgent | oracle.get_drift_events() — breaks separation of concerns | |
| You decide | Researcher/planner picks | |

**User's choice:** `OracleReply` extended with `contradiction_detected: bool = False` and `contradicted_turn: int | None = None`.

---

## Claude's Discretion

- LLM model for oracle.reply() — haiku for ablations, sonnet for quality runs
- System prompt structure (ordering of spec, noise, cog load, global instructions, state summary)
- Rolling window size N for structural drift comparison (suggested: 10)
- Named constants: MAX_K, MAX_MSG_LEN, TOP_K_ITEMS_PER_CLUSTER

## Deferred Ideas

- Mutable noise params (turn-by-turn fatigue simulation) — Phase 5 ablation
- LLM-based semantic drift detection — deferred; structural sufficient for Phase 3
- Per-persona cog_load_threshold in NoiseParams — deferred; fixed 0.7 sufficient
