# Phase 5: Ablation Harness and Strategies - Context

**Gathered:** 2026-05-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the two remaining strategies (UncertaintyDrivenStrategy, BoundaryDrivenStrategy), build the N×M×K experiment harness that runs all combinations automatically in parallel, add `oracle_type` to the DB schema, and ship a bootstrap CI analysis layer (src/analysis.py + CLI script + notebook). This phase directly answers research question 1: which strategy converges fastest, and by how much?

Requirements in scope: **ALAB-01, ALAB-02, ALAB-03, EXP-V2-01 (partial — oracle_type column)**.

Phase 5 also enriches `_format_message()` in `conversation_loop.py` so that targeted actions (`show_subset`, `ask_question` with payload) include actual item text — currently a placeholder that sends no useful content to the oracle.

</domain>

<decisions>
## Implementation Decisions

### Strategy: UncertaintyDrivenStrategy (ALAB-01)

- **D-01:** **Targets the highest-uncertainty cluster** — the cluster with the lowest mean soft-assignment confidence from `f_uncertainty`'s output. Payload carries `cluster_id`. Selects `show_subset` or `ask_question` targeting that cluster. Meaningfully different from Random (no targeting) and BoundaryDriven (cluster pair vs. single cluster).
- **D-02:** **Degenerate fallback = `ask_question` with generic payload.** When `f_uncertainty` gives no valid target (only 1 cluster, or all items equally confident), the strategy emits `ask_question` with an empty/generic payload — handing initiative to the oracle ("what should we focus on?"). Logged as `deviation()`. Does NOT crash; does NOT fall back to random. Semantically meaningful: the agent admits it doesn't see a clear next step.

### Strategy: BoundaryDrivenStrategy (ALAB-01)

- **D-03:** **Targets the most-confused cluster pair as a group.** Algorithm:
  1. Find cluster pair (A, B) with highest mutual soft-assignment overlap (most items with high P(A) and P(B) simultaneously dominating their distribution).
  2. Collect all items in that "ambiguous zone" — items where both P(A) and P(B) are high and they dominate.
  3. Emit `show_subset` or `ask_question` targeting that group as a coherent unit.
  4. Payload: `{cluster_a: int, cluster_b: int, item_ids: list[int]}`.
  
  The oracle sees the group of ambiguous items and can give merge/split/rule feedback on the pattern — not isolated point-level decisions.
- **D-04:** **Same degenerate fallback as D-02** — `ask_question` with generic payload when no confusing pair can be found. Logged as `deviation()`.
- **D-05:** **Subset only in the oracle message.** When a strategy targets a subset, `_format_message()` includes only the targeted items with their actual text — NOT the full clustering. Lower cognitive load, more focused feedback. Consistent with Phase 3 cognitive load budget design. `show_full` retains current format (cluster names + sizes).

### Oracle conversational initiative

- **D-06:** **Full freedom within response — no architecture change.** The oracle can express any feedback type in response to any agent action. The parser (`feedback_parser.py`) does not filter by action type. Oracle-initiated turns (sending a message before the agent acts) are not needed for Phase 5 LLM oracle runs and deferred to Phase 6 human study design.

### Harness design (ALAB-02)

- **D-07:** **3×3×3 = 27 runs default, fully tunable.** N=3 strategies, M=3 oracle personas (e.g. curious, skeptical, drifty), K=3 seeds. All three dimensions configurable in the harness YAML config. Enough for meaningful CIs at research scale; manageable LLM cost.
- **D-08:** **Parallel execution via `ThreadPoolExecutor`.** Runs execute concurrently with a configurable `max_workers` to respect LLM rate limits. Each run gets its own `sqlite3.Connection` (one per run, per Phase 4 D-04). SQLite WAL mode (`PRAGMA journal_mode=WAL`) supports concurrent connections without blocking. Fail loudly on thread exceptions — no swallowed errors.
- **D-09:** **Cross-product YAML config with optional `exclude` list.** The harness config lists `strategies`, `personas`, and `seeds` separately; the harness generates the full cross-product automatically. An optional `exclude` list allows skipping specific `{strategy, persona, seed}` combos. Example shape:
  ```yaml
  harness:
    strategies: [random, uncertainty_driven, boundary_driven]
    personas: [curious, skeptical, drifty]
    seeds: [1, 2, 3]
    exclude: []          # optional: [{strategy: random, persona: drifty, seed: 3}]
    max_workers: 4
  personas:
    curious:
      preferred_k: 5
      semantic_axes: [topic, sentiment]
      persona_description: "..."
      noise:
        consistency_rate: 0.9
        drift_probability: 0.05
        sycophancy_resistance: 0.7
    skeptical: ...
    drifty: ...
  ```
- **D-10:** **Entry point: `examples/run_harness.py`** — consistent with `examples/run_baseline.py` pattern (Phase 4 D-25). Importable `run_harness(config_path, db) → list[ExperimentRead]` function in `src/harness.py`; CLI wrapper is ~30 lines. Harness imports `run_conversation()` (for interactive LLM runs) and `run_baseline()` (for no-dialogue baseline paired runs) directly — no HTTP, no socketio.

### oracle_type DB field (EXP-V2-01 partial)

- **D-11:** **New column added to `CREATE TABLE experiments` DDL in `init_schema()`.** Column: `oracle_type TEXT NOT NULL DEFAULT 'llm'`. Values: `'llm'` (automated LLM oracle run) or `'human'` (Phase 6 human session). Added directly to the schema definition — no `ALTER TABLE` migration needed since Phase 5 starts with a fresh `experiments.db`. `docs/MODEL.md` updated first per Phase 4 D-31 discipline.
- **D-12:** **Indexed column** — directly filterable in SQL: `WHERE oracle_type = 'human'`. Phase 6 LLM-vs-human comparison analysis (`SELECT strategy_id, oracle_type, AVG(total_turns)...`) works cleanly without `json_extract()`.

### Bootstrap CI analysis (ALAB-03)

- **D-13:** **`src/analysis.py`** — CI computation logic as a pure importable function: `compute_bootstrap_ci(values: list[float], n_bootstrap: int = 10000, ci: float = 0.95) → tuple[float, float]`. No I/O, no SQL. Shared by CLI and notebook.
- **D-14:** **`examples/compute_ci.py`** — ~30-line CLI script. Queries `experiments` table grouped by `strategy_id` (and optionally `oracle_type`, `dataset`), calls `compute_bootstrap_ci()`, prints a comparison table by default. `--json` flag for machine-readable output. Consistent with `examples/run_baseline.py` pattern.
- **D-15:** **`notebooks/analysis.ipynb`** — Jupyter notebook for interactive exploration. Queries the same DB directly; calls `src/analysis.py` for CI computation. Useful for reviewing past runs, exploring persona effects, visualising bootstrap distributions for the paper. Register existing venv as Jupyter kernel once (`python -m ipykernel install --user --name=conversational-clustering`).

### _format_message() enrichment

- **D-16:** **Extend `_format_message()` signature to accept `id_to_text: dict[int, str]`** — needed so targeted actions can include actual item text in the oracle message. When `action_type == "show_subset"` with a non-empty payload, format the item texts directly. When `action.payload` is empty (Phase 2 legacy), fall back to current placeholder. `show_full` retains current format (cluster names + sizes) — planner enriches further if needed.

### Claude's Discretion

- Exact threshold for "items confused between cluster pair" in BoundaryDriven — e.g., `P(A) > 0.3 AND P(B) > 0.3` vs. top-N by `|P(A) - P(B)|` proximity. Planner picks based on `f_uncertainty`'s existing output shape.
- Action-type selection logic within each strategy (when to pick `show_subset` vs. `ask_question` for a given target) — planner decides; should differ between strategies for ablation signal clarity.
- `max_workers` default value — planner estimates based on turn budget and typical LLM latency.
- Exact persona configs (`OracleSpec` + `NoiseParams` values for curious/skeptical/drifty) — planner picks values that produce meaningfully different convergence behaviour.
- Whether `show_full` message is also enriched with sample item texts — planner decides based on oracle reply quality observed during Phase 4 testing.
- SQL index on `oracle_type` column — planner adds if Phase 6 query patterns warrant it.
- `n_bootstrap` default (10,000 is standard; planner may lower for fast CI checks during development).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Strategy interface and existing implementation
- `src/strategy.py` — `StrategyProtocol`, `Action`, `RandomStrategy`, `_enumerate_valid_actions()`. Phase 5 adds two new strategy classes to this file. Comments on lines 4 and 49 explicitly note Phase 5 extensions.
- `src/uncertainty.py` — `f_uncertainty`, `UncertaintyReport`. New strategies consume this output. Read before implementing D-01 and D-03 to understand available signals.
- `src/agent_functions.py` — `f_next_best_step`. Passes `uncertainty_report` to `strategy.select()` — unchanged in Phase 5.

### Conversation loop and message formatting
- `src/conversation_loop.py` — `_format_message()` (line 60). Phase 5 extends this function (D-16). Read the full function and threading notes before modifying.

### Phase 4 DB layer (carry forward)
- `src/db/connection.py` — `init_schema()`, `connect()`. D-11 adds `oracle_type` column to the `CREATE TABLE experiments` DDL here.
- `src/db/experiments.py` — `ExperimentCreate`, `create()`, `query()`. Harness calls `create()` per run; `compute_ci.py` calls `query()`.
- `src/judge.py` — `run_baseline()`. Harness imports this directly for baseline paired runs (D-10).
- `docs/MODEL.md` — DB schema spec. **Update this first before any schema change** (Phase 4 D-31 discipline).

### Project Planning Artifacts
- `.planning/REQUIREMENTS.md` §Ablation & Evaluation — ALAB-01 (3 strategies), ALAB-02 (N×M×K harness), ALAB-03 (bootstrap CI)
- `.planning/REQUIREMENTS.md` §Headline Experiment — EXP-V2-01 (oracle_type column, LLM+human support)
- `.planning/ROADMAP.md` §Phase 5 — goal, success criteria, "oracle_type logged" Trio requirement
- `.planning/PROJECT.md` — fail-loudly philosophy, "at least one defensible quantified claim with CI" requirement
- `CLAUDE.md` — Key Constraints (src/db/ only SQL layer, WAL mode, deviation() usage, datetime.now(timezone.utc))

### Prior Phase Decisions That Carry Forward
- `.planning/phases/04-judge-agent/04-CONTEXT.md` — D-01 (SQLite + raw SQL, WAL mode), D-04 (one connection per run, check_same_thread=False), D-09 (src/db/ subpackage layout), D-10 (Pydantic triples — ExperimentCreate/Read), D-16 (YAML config format — extend for harness), D-25 (examples/ script pattern — run_baseline.py), D-29 (deviation() for unexpected branches), D-31 (MODEL.md updated first on schema changes)
- `.planning/phases/02-clustering-agent-core/02-CONTEXT.md` — D-01 (plain Python while-loop — DO NOT switch to LangGraph), D-02 (standalone modules, pure f_* functions), D-04 (loop owns JSONL write)
- `.planning/phases/03-oracle-agent/03-CONTEXT.md` — D-06 (f_cognitive_load pure function), D-03/D-04/D-05 (NoiseParams, OracleSpec, OracleAgent construction)

### Experiment config
- `experiments/configs/default.yaml` — existing stopping + judge config shape. Phase 5 extends this format with `harness:` and `personas:` sections (D-09).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `RandomStrategy` (`src/strategy.py:71`): seeded `random.Random` instance pattern — copy for determinism in new strategies. `_enumerate_valid_actions()` (line 42) builds the valid action list used by all strategies.
- `f_uncertainty` / `UncertaintyReport` (`src/uncertainty.py`): existing output. UncertaintyDrivenStrategy uses cluster-level uncertainty ranking; BoundaryDrivenStrategy uses per-item soft-prob vectors to find confused cluster pairs.
- `run_baseline()` (`src/judge.py`): importable function, one DB connection per call. Harness calls this for no-dialogue paired runs (D-10).
- `run_conversation()` (`src/conversation_loop.py`): importable, accepts `strategy` parameter. Harness calls this for interactive LLM oracle runs.
- `connect()` / `init_schema()` (`src/db/connection.py`): called once per harness run in each thread. WAL mode already configured.
- `ExperimentCreate` / `ExperimentRead` (`src/db/experiments.py`): Pydantic models; harness builds `ExperimentCreate` per run.
- `experiments/configs/default.yaml`: existing YAML shape to extend for harness config (D-09).

### Established Patterns
- **examples/ script pattern (Phase 4 D-25):** `src/` module holds the importable function; `examples/` holds the ~30-line CLI wrapper that parses args, opens DB, calls the function, prints JSON or table. Follow for `run_harness.py` and `compute_ci.py`.
- **One DB connection per run (Phase 4 D-04):** `check_same_thread=False` + WAL mode. Each parallel harness thread opens its own connection — do not share across threads.
- **Fail loudly:** No `try/except` in harness except at LLM API boundary. Thread exceptions must propagate, not be swallowed. Use `ThreadPoolExecutor` with `as_completed()` so exceptions surface immediately.
- **deviation() for unexpected branches (Phase 4 D-29):** degenerate strategy fallback (D-02/D-04), thread that can't acquire a connection, config key missing with fallback applied — all logged via `deviation()`.
- **Pydantic only inside src/db/ (Phase 4 D-10):** harness config parsing uses dataclasses, not Pydantic. `OracleSpec`, `NoiseParams` stay as dataclasses.
- **datetime.now(timezone.utc).isoformat() everywhere** — never `datetime.utcnow()`.

### Integration Points
- `_format_message(action, state, id_to_text)` — Phase 5 extends the signature to include `id_to_text`. Caller (`run_conversation()`) already has `id_to_text`; pass it through. Non-enriched payload falls back to current behaviour.
- New strategies slot into `f_next_best_step(state, strategy, uncertainty_report)` without any changes to `agent_functions.py` or `conversation_loop.py` — `StrategyProtocol` is structural, no inheritance required.
- `oracle_type` value flows from harness config → `ExperimentCreate(oracle_type='llm')` → `src/db/experiments.create()` → `experiments` table. Human sessions (Phase 6) pass `oracle_type='human'` at construction.
- `compute_bootstrap_ci()` in `src/analysis.py` is called by both `examples/compute_ci.py` and `notebooks/analysis.ipynb` — shared logic, two entry points.

</code_context>

<specifics>
## Specific Ideas

- **Harness YAML personas section** — each persona entry maps to an `OracleSpec` + `NoiseParams` construction. Strategy strings map to class names via a registry dict in `src/harness.py`: `{"random": RandomStrategy, "uncertainty_driven": UncertaintyDrivenStrategy, "boundary_driven": BoundaryDrivenStrategy}`.
- **Bootstrap CI output table format** (D-14):
  ```
  Strategy              | Mean turns | 95% CI         | N runs
  ----------------------|------------|----------------|-------
  random                |       14.2 | [12.1 – 16.3]  |     9
  uncertainty_driven    |       10.8 | [9.1 – 12.5]   |     9
  boundary_driven       |       11.4 | [9.8 – 13.0]   |     9
  no_dialogue (baseline)|        1.0 | [1.0 – 1.0]    |     9
  ```
- **BoundaryDriven group payload example** — `Action(action_type="show_subset", payload={"cluster_a": 2, "cluster_b": 5, "item_ids": [42, 87, 103, 211]})` — message becomes: `"These items are ambiguous between cluster 'Electronics' and cluster 'Gadgets': 'iPhone 14 case', 'Android charger', 'laptop bag', 'USB hub'. How would you distinguish them?"`
- **UncertaintyDriven payload example** — `Action(action_type="ask_question", payload={"cluster_id": 3})` — message becomes: `"Cluster 'Peripherals' has the most ambiguous assignments. Do you want to split it, move items out, or rename it?"`
- **Deterministic sampling in BoundaryDriven** — seed item selection by `random.Random(state.turn_index)` to mirror PairBag determinism from Phase 4 D-19.

</specifics>

<deferred>
## Deferred Ideas

- **Oracle-initiated turns** — human can push a message at any point without waiting for the agent. Deferred to Phase 6 human study UI design. Not needed for Phase 5 LLM oracle runs.
- **Mutable noise params (fatigue simulation)** — `consistency_rate` degrading turn-by-turn. Noted in Phase 3 deferred. Still deferred; can be added as a fourth strategy variant if Phase 5 ablations reveal it matters.
- **LLM-based semantic contradiction detection** in strategies — deferred per Phase 3. Structural comparison is sufficient.
- **`docs/DEMO.md` + `examples/seed.py`** — deferred per Phase 4 D-33. Phase 5 `run_harness.py` is the seeded comparison-demo flow; add DEMO.md if warranted.
- **CI enforcement with STRICT_MODE matrix** — deferred per Phase 4 D-30. Reconsider after Phase 5 or when team grows.
- **Soft-assignment calibration (EVAL-V2-01)** — reliability diagram. Out of scope for Phase 5; revisit in Phase 6.
- **show_full enrichment with sample item texts** — whether to also enrich the `show_full` message format with actual item content. Left to planner based on oracle reply quality from Phase 4 testing.

</deferred>

---

*Phase: 5-Ablation-Harness-and-Strategies*
*Context gathered: 2026-05-16*
