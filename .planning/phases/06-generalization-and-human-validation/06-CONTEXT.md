# Phase 6: Generalization and Human Validation - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Codify oracle preferences into a pluggable mapping function (GEN-01) evaluated on the frozen held-out split with LLM oracle as ground-truth labeler (GEN-02). Run a human study (N≥10, within-subject) through a minimal `/study` web UI with per-cluster UMAP + linked item lists. Quantify the LLM-vs-human gap on turns-to-convergence with bootstrap CI — the headline finding for research question 2.

Requirements in scope: **GEN-01, GEN-02, EXP-V2-01 (complete)**.

</domain>

<decisions>
## Implementation Decisions

### Mapping Function — Architecture (GEN-01)

- **D-01:** **Both LLMMappingStrategy and CentroidMappingStrategy implemented and compared.** GEN-02 evaluation runs both on the held-out split and reports accuracy side by side. This is a research finding, not just a number. Adding a third ensemble or combined strategy in the future requires only registering a new class — no changes to evaluation code.
- **D-02:** **Pluggable `MappingProtocol` (structural, mirrors `StrategyProtocol` from Phase 2/5).** Registry dict `MAPPING_REGISTRY = {"llm": LLMMappingStrategy, "centroid": CentroidMappingStrategy}` in `src/mapping.py`. Interface: `assign(item_text, state, rule_set, embedding_store) → cluster_id`.
- **D-03:** **`OracleRuleSet` is a Pydantic object.** Extracted from AuditLog FB-04 instructional feedback entries once at session end via `extract_oracle_rules(audit_log_path) → OracleRuleSet`. Shared input to both classifiers. Consistent with Phase 4 Pydantic-in-src/ discipline.
- **D-04:** **`LLMMappingStrategy` uses extracted rules as context.** Prompt = final cluster descriptions + `OracleRuleSet` fields + new item text → cluster assignment. One LLM API call per item. Robust to edge cases; picks up oracle semantic intent.
- **D-05:** **`CentroidMappingStrategy` uses cosine similarity.** Compute cluster centroids from `embeddings.npy` (items in each cluster at session end), embed new item, assign to nearest centroid. No LLM call at inference; ignores explicit oracle rules but captures positional signal.

### Mapping Function — GEN-02 Evaluation

- **D-06:** **LLM oracle is the ground-truth labeler.** Show the same OracleAgent 20–30 held-out items and ask it to assign each to a cluster. Compare oracle labels vs. mapping function predictions — accuracy = agreement rate. No participant burden; reproducible. Human participants do NOT do GEN-02 labeling. Participant dataset upload for study participants is deferred.
- **D-07:** **Both mapping strategies evaluated on the same 20–30 item sample.** Results reported as a comparison table (same format as Phase 5's CI table): strategy, accuracy, N items.

### Human Study UI (EXP-V2-01)

- **D-08:** **New `/study` route** — minimal dedicated page, separate from the debug UI at `/`. Researcher pre-loads a session (same dataset, same initial clustering for all participants); participant joins via URL/session ID.
- **D-09:** **Study page shows full cluster detail** — cluster names, descriptions, and expandable item list (5–10 items per cluster visible). No conversation history shown (current clustering state only). Free text input for participant feedback.
- **D-10:** **Faceted UMAP representation** — K per-cluster UMAP mini-plots sharing a single global coordinate space (same viewport bounds, same embedding). Interacting with an item in the list (hover or click) highlights its exact position in the corresponding mini-plot. Preserves spatial context without visual overlap between clusters.
- **D-11:** **Researcher starts session; participant joins by URL/session ID.** All participants see the same initial clustering for controlled conditions. Participant-side dataset upload deferred to future phase.

### Oracle Satisfaction for Human Sessions

- **D-12:** **LLM parses human free-text for satisfaction signal per turn.** After each human feedback turn, a lightweight LLM call checks whether the message contains satisfaction intent (e.g., "looks good", "I'm happy with this", "this is fine"). When detected, the system prompts the participant to confirm before stopping.
- **D-13:** **Hard safety cap: 30 turns for human sessions (configurable).** If satisfaction never detected, session terminates at turn 30 with `convergence_reason = "turn_budget"`. The 30-turn cap is a config value, not hardcoded.
- **D-14:** **Confirmation prompt before stopping.** When satisfaction is detected: system shows "It looks like you're satisfied with the clustering. End session?" (Yes/No). Prevents false positives; adds one interaction before logging `convergence_reason = "oracle_satisfied"`. `oracle_type = 'human'` in DB (Phase 5 column).

### Claude's Discretion

- Exact `OracleRuleSet` Pydantic field schema — planner designs based on FB-04 constraint types observed in prior phases (synonyms, focus areas, exclusions, etc.)
- Whether `extract_oracle_rules()` uses a structured prompt or JSON mode — planner decides based on Anthropic API capabilities
- Cosine similarity implementation — scipy or numpy; planner picks based on existing imports
- Number of held-out items for GEN-02 sample — 20–30 per ROADMAP.md; planner picks a fixed default (e.g., 25) configurable via CLI flag
- `/study` page frontend stack — reuses existing FastAPI + SocketIO + HTML/JS pattern from `web/`; planner decides layout
- How cluster centroids are computed — mean of item embeddings in the cluster, weighted by soft-assignment probabilities, or hard-assignment mean; planner picks based on signal quality

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Mapping function design
- `src/strategy.py` — `StrategyProtocol`, `STRATEGY_REGISTRY` pattern. **Mirror exactly** for `MappingProtocol` + `MAPPING_REGISTRY` in `src/mapping.py`
- `src/uncertainty.py` — `f_uncertainty`, `UncertaintyReport`. Centroid strategy uses cluster membership from state; this shows what per-item signals are available
- `src/embedding_store.py` — `EmbeddingStore`. Centroid strategy reads item embeddings here
- `src/feedback_parser.py` — FB-04 instructional feedback parsing. `extract_oracle_rules()` reads the AuditLog for parsed FB-04 entries to build `OracleRuleSet`

### Held-out data and evaluation
- `dataset/held_out.jsonl` — SHA-256 `ea53dfa1`. **Never modify.** GEN-02 draws 20–30 items from here
- `src/data_loader.py` — data loading; use existing loader for held-out items
- `src/analysis.py` — `compute_bootstrap_ci()`. Reuse for GEN-02 accuracy CI if sample size warrants it
- `examples/compute_ci.py` — CLI pattern to follow for GEN-02 evaluation script

### Oracle agent (for GEN-02 labeling and satisfaction detection)
- `src/oracle_agent.py` — `OracleAgent`, `OracleSpec`, `NoiseParams`. GEN-02 uses the same agent to label held-out items
- `src/oracle_protocol.py` — `OracleProtocol`. GEN-02 labeling reuses the same reply interface

### Human study UI
- `web/app.py` — existing FastAPI + python-socketio app. `/study` route added here; reuses `SocketIOEmitter` and session management patterns
- `web/` directory — existing HTML/JS templates and static assets to extend for study page

### DB and experiment tracking
- `src/db/connection.py` — `connect()`, `init_schema()`. Human sessions write to same DB with `oracle_type='human'`
- `src/db/experiments.py` — `ExperimentCreate`. `oracle_type='human'` field already in schema (Phase 5 D-11)
- `docs/MODEL.md` — DB schema spec. **Update first** before any schema change (Phase 4 D-31 discipline)

### Project Planning Artifacts
- `.planning/REQUIREMENTS.md` §Generalization — GEN-01 (mapping function), GEN-02 (held-out evaluation)
- `.planning/REQUIREMENTS.md` §Headline Experiment — EXP-V2-01 (human study N≥10, within-subject, LLM-vs-human gap)
- `.planning/ROADMAP.md` §Phase 6 — goal, success criteria, research question 2
- `.planning/PROJECT.md` — fail-loudly philosophy, "at least one defensible quantified claim with CI"
- `CLAUDE.md` — Key Constraints (src/db/ only SQL layer, WAL mode, deviation() usage, datetime.now(timezone.utc), no eventlet/gevent)

### Prior Phase Decisions That Carry Forward
- `.planning/phases/05-ablation-harness-and-strategies/05-CONTEXT.md` — D-10 (examples/ script pattern), D-11 (oracle_type column), D-12 (oracle_type indexed), D-13 (src/analysis.py pure function pattern), D-14 (compute_ci.py CLI pattern)
- `.planning/phases/04-judge-agent/04-CONTEXT.md` — D-01 (SQLite + WAL), D-04 (one connection per run), D-09 (src/db/ layout), D-10 (Pydantic triples), D-25 (examples/ pattern), D-31 (MODEL.md first)
- `.planning/phases/02-clustering-agent-core/02-CONTEXT.md` — D-01 (plain Python while-loop — DO NOT switch to LangGraph)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `StrategyProtocol` + `STRATEGY_REGISTRY` (`src/strategy.py`): Copy this pattern verbatim for `MappingProtocol` + `MAPPING_REGISTRY` in `src/mapping.py`
- `EmbeddingStore` (`src/embedding_store.py`): Centroid strategy reads item embeddings from here; already loaded at startup
- `compute_bootstrap_ci()` (`src/analysis.py`): Reuse for GEN-02 accuracy CI
- `OracleAgent.reply()` (`src/oracle_agent.py`): GEN-02 labeling calls this to get oracle cluster assignments for held-out items
- `feedback_parser.py`: FB-04 parsed entries in AuditLog are the input to `extract_oracle_rules()`
- `run_conversation()` (`src/conversation_loop.py`): Human sessions route through this; study UI triggers it via SocketIO background thread (same as existing debug UI)
- `SocketIOEmitter` (`web/app.py`): Existing bridge between worker thread and asyncio event loop — reuse for `/study` route

### Established Patterns
- **examples/ script pattern (Phase 4 D-25):** `src/` holds importable function; `examples/` holds ~30-line CLI wrapper. Follow for GEN-02 evaluation script
- **One DB connection per run (Phase 4 D-04):** Human sessions get their own connection; `oracle_type='human'`
- **Fail loudly:** No try/except in mapping/evaluation code except at LLM API boundary
- **deviation() for unexpected branches:** Empty AuditLog, no FB-04 entries, zero-item cluster — all logged via `deviation()`, not silently handled
- **datetime.now(timezone.utc).isoformat() everywhere** — never `datetime.utcnow()`
- **Pydantic only inside src/db/ and for data contracts** — `OracleRuleSet` is a Pydantic model in `src/mapping.py`; harness configs use dataclasses

### Integration Points
- `/study` route wired in `web/app.py` alongside existing `/` debug UI
- `oracle_type='human'` flows from study session → `ExperimentCreate` → `experiments` table — already supported by Phase 5 schema
- `extract_oracle_rules(audit_log_path)` reads the session's `audit_log.jsonl` from `sessions/<ts>/` — same path used by existing serialization
- Both mapping strategies registered in `MAPPING_REGISTRY`; GEN-02 evaluation script loops over registry entries

</code_context>

<specifics>
## Specific Ideas

- **Faceted UMAP layout:** K mini-plots on the `/study` page, all sharing global coordinate bounds computed once from all embeddings. Each mini-plot renders only its cluster's items at full opacity; other cluster items either hidden or shown at low opacity for spatial context. Hover/click on item in the expandable list triggers a highlight event on the corresponding mini-plot.
- **Satisfaction detection prompt:** Lightweight LLM call with a classification prompt: "Does this message indicate the user is satisfied with the clustering? Reply YES or NO only. Message: {text}". Use the same `anthropic.Anthropic` client; do NOT spin up a separate OracleAgent for this.
- **GEN-02 evaluation output format:**
  ```
  Mapping strategy    | Accuracy | N items | 95% CI
  --------------------|----------|---------|--------
  llm                 |   0.87   |    25   | [0.72–0.96]
  centroid            |   0.74   |    25   | [0.57–0.87]
  ```
- **`OracleRuleSet` sketch (planner fleshes out field types):** `OracleRuleSet(synonyms: list[list[str]], focus_areas: list[str], exclusions: list[str], cluster_rules: list[str])` — derived from FB-04 parsed deltas in AuditLog
- **Study session start:** Researcher calls `POST /study/sessions` with dataset path + backend → system creates session and returns session URL. Participant opens URL; no researcher present during session.

</specifics>

<deferred>
## Deferred Ideas

- **Participant-side dataset upload on the study page** — user noted interest in future. Not needed for Phase 6 controlled study; add when study design requires it.
- **Oracle-initiated turns** — human pushes a message before the agent acts. Deferred from Phase 5; still deferred. Not blocking for Phase 6 study.
- **Mutable noise params / fatigue simulation** — deferred from Phase 3/5. Still not needed.
- **Soft-assignment calibration (EVAL-V2-01)** — reliability diagram. Not in Phase 6 scope.
- **Ensemble MappingStrategy** — `MappingProtocol` is designed to support it; implementation deferred until Phase 6 results show whether combining strategies adds value.
- **Human GEN-02 labeling** — participants label held-out items for a human-oracle accuracy comparison. Deferred; GEN-02 uses LLM oracle only for Phase 6.

</deferred>

---

*Phase: 6-Generalization-and-Human-Validation*
*Context gathered: 2026-05-17*
