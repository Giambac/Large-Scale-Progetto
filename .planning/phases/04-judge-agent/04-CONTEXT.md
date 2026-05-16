# Phase 4: Judge Agent - Context

**Gathered:** 2026-05-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Add the third agent — convergence detection, per-turn metric capture, database persistence, and a no-dialogue baseline mode. The system already records per-turn `ClusteringState` to `audit_log.jsonl` and oracle events to `events.jsonl`. Phase 4 layers a structured SQLite database on top for cross-run analysis (DB-01/02/03), fills in the Phase 1 stopping-criteria stubs (`src/stopping.py` has placeholders `nan`/`-1` waiting for JUDG-01), adds the per-turn metric bundle (JUDG-02 — cognitive load, contradiction count, pairwise validation accuracy, convergence signal), and ships a no-dialogue baseline run mode (JUDG-03) so Phase 5 ablations can isolate dialogue contribution.

Requirements in scope: **DB-01, DB-02, DB-03, JUDG-01, JUDG-02, JUDG-03**.

Phase 4 also fixes Phase 3 REVIEW.md **WR-04** as part of loop wiring: `OracleProtocol.reply()` signature gains `global_instructions` and `cognitive_load` so the `isinstance(oracle, OracleAgent)` branch in `run_conversation()` can be removed.

Phase 4 follows the **wazzup recipe** (user-provided this session) for the new DB layer shape, code organization, error conventions, and documentation discipline. Recipe location: see canonical refs.

</domain>

<decisions>
## Implementation Decisions

### Database choice and lifecycle

- **D-01:** **SQLite + raw SQL**, single shared `experiments.db` file at repo root (gitignored). One file across all runs; each experiment run inserts experiment + turn + oracle_feedback rows into the same DB. Cross-run analysis (Phase 5 bootstrap CI) is a single SELECT. Tests use `sqlite3.connect(":memory:", check_same_thread=False)` via a `conftest.py` fixture — production code never knows which connection it got.
- **D-02:** **Both DB and JSONL coexist.** Per-turn `ClusteringState` continues to be written to `sessions/<ts>/audit_log.jsonl` unchanged (preserves Phase 2 D-04: "loop owns the JSONL write"). DB stores DB-01 experiments + DB-02 turns + DB-03 oracle_feedback rows for cross-run query. JSONL is the source of truth for replay; DB is the queryable index.
- **D-03:** **Write order per turn:** append `audit_log.jsonl` FIRST → insert turns + oracle_feedback rows → DB commit → `post_turn_callback` → socket emit. JSONL-first preserves the Phase 2 D-04 contract; if a DB insert crashes mid-turn, the JSONL still has the truth and DB rows can be re-derived from it.
- **D-04:** **One `sqlite3.Connection` per run.** Opened at `run_conversation()` start, committed after each turn (matching JSONL flush cadence), closed at run end. Caller owns the unit of work (recipe §6). `check_same_thread=False` is MANDATORY (`run_conversation` runs in a background thread via `socketio.start_background_task` — CLAUDE.md). `PRAGMA journal_mode=WAL` enabled so a separate FastAPI read-only connection can query the DB mid-run without blocking writes.

### Schema shape and relationships

- **D-05:** **`experiments` table = full recipe shape:** `id` + `name` + `slug` (unique among live rows via partial index `WHERE deleted_at IS NULL`) + `created_at` + `updated_at` + `deleted_at` + indexed cols + `details` JSON. **`turns` and `oracle_feedback` = minimal shape:** `id` + FKs + indexed cols + `created_at` + `details` JSON. No name/slug/updated_at/deleted_at on turns/oracle_feedback — they live and die with their experiment, and the recipe §2 explicitly endorses dropping name/slug for entities like `message` that have no natural human name. Timestamps stored as ISO-8601 UTC strings, server-set via `datetime.now(timezone.utc).isoformat()` (also fixes Phase 3 REVIEW.md CR-02 naive-datetime issue in `oracle_agent.py:131`).
- **D-06:** **Dedicated FK columns on each child table:** `turns.experiment_id INTEGER REFERENCES experiments(id)`; `oracle_feedback.turn_id INTEGER REFERENCES turns(id)`. `PRAGMA foreign_keys=ON` enforces integrity at the DB level. No polymorphic rels table (recipe §3 explicitly endorses dedicated FK columns for "frequent rels" — these run on every analytics query). Direct joins, no src_type/tgt_type dispatch.
- **D-07:** **Cascade rule:** soft-delete on an experiment cascades to its turns and oracle_feedback rows (set `deleted_at` on all three). Hard delete uses `ON DELETE CASCADE` on the FK. Cascade is orchestrated from a single `src/db/deletion.py` per recipe §4.2 (three roles per entity: `_delete_primary` in the entity module, `cascade_delete` orchestrator, 4-line public `delete()` wrapper). Live reads filter `WHERE deleted_at IS NULL`. Rule documented in `docs/MODEL.md` (D-29).
- **D-08:** **`experiments` indexed columns:** `strategy_id`, `persona_id`, `seed`, `dataset`, `total_turns`, `convergence_reason`, `start_timestamp`, `end_timestamp`. **`experiments.details` JSON:** headline metrics (`mean_cognitive_load`, `mean_pairwise_accuracy`, `contradiction_count`, `turns_to_convergence`) + run metadata (config snapshot from D-16, `oracle_init` params, git commit if available). "Promote to a column only when you actually query by it" — recipe §2. `turns.details` and `oracle_feedback.details` JSON: any extra parsed_delta data not promoted to indexed columns.

### DB code layer organization

- **D-09:** **New `src/db/` subpackage** following recipe §5 `wazzup/api/` layout 1:1:
  - `src/db/__init__.py` — `NotFound` exception, shared types
  - `src/db/connection.py` — `connect()` + `init_schema()` + WAL/foreign-keys setup
  - `src/db/experiments.py` — per-entity CRUD: `create / get / get_by_slug / update / delete / query`
  - `src/db/turns.py` — per-entity CRUD
  - `src/db/oracle_feedback.py` — per-entity CRUD
  - `src/db/deletion.py` — cascade orchestrator (recipe §4.2)
  - `src/db/slugs.py` — `slugify()` + `make_slug(db, table, name, override=None)`
  
  **`src/db/` is the ONLY layer that touches SQL.** Anywhere else, sqlite3.execute is a bug (recipe §6).
- **D-10:** **Pydantic triples per entity** (recipe §7): `ExperimentCreate / ExperimentRead / ExperimentUpdate`, `TurnCreate / TurnRead`, `OracleFeedbackCreate / OracleFeedbackRead`. **NOTE:** this is the entry point for Pydantic in the codebase — Phase 1–3 use plain dataclasses (per Phase 2 D-06, Phase 3 D-05). Pydantic stays INSIDE `src/db/`; the rest of the codebase keeps dataclasses. Choosing Pydantic here for: free validation, `model_dump_json()`/`model_validate_json()` for the `details` column, and future HTTP-layer reuse.
- **D-11:** **Recipe §6 error conventions:** `src/db/__init__.py` declares `class NotFound(Exception)`. `get()` / `get_by_slug()` return `None` on miss; `query()` returns `[]`. `create()` raises `sqlite3.IntegrityError` on slug or FK conflict. `update()` and `delete()` raise `NotFound` when the id isn't there. Consistent shape across all three entity modules so a future HTTP layer (Phase 5+) gets free 404/409 translation.
- **D-12:** **Slug generation via `src/db/slugs.py`** with `make_slug(db, table, name, override=None)` per recipe §6. For experiments: `name = f"{strategy_id}-{persona_id}-seed{seed}"` (e.g. `"random-trump-seed42"`); `slug = slugify(name)`; collision suffix `-2`, `-3`, … backed by a partial unique index `CREATE UNIQUE INDEX experiment_slug_live ON experiments(slug) WHERE deleted_at IS NULL`. Slugs are immutable once written. Reproducible, greppable, makes filenames and CLI args ergonomic (`python -m examples.run_baseline --name random-trump-seed42`).

### Diminishing-returns concrete values (JUDG-01)

Phase 1's `src/stopping.py` left these as `float("nan")` and `-1` placeholders explicitly waiting for Phase 4. ALL three values are tunable per-experiment via the YAML config (D-16), and the actual values used are recorded in `experiments.details` JSON for every run.

- **D-13:** **Magnitude weights — geometric falloff:** `global=1.0`, `cluster=0.5`, `point=0.2`, `instructional=0.1`. Strong emphasis on global feedback (10x instructional). Overrides Phase 1 placeholder nan values in `FeedbackMagnitudeWeights`. **Tunable per-experiment.**
- **D-14:** **`magnitude_threshold_epsilon = 0.05`** (overrides Phase 1 nan). Effectively a "true-silence" threshold since the smallest weighted magnitude is 1 instructional = 0.1 > 0.05. **Tunable per-experiment.**
- **D-15:** **`magnitude_fallback_turns = 3`** (overrides Phase 1 `-1`). Three consecutive silent turns triggers `DIMINISHING_RETURNS`. **Tunable per-experiment.**
- **D-16:** **YAML config file per experiment** at `experiments/configs/<name>.yaml`. CLI accepts `--config path`. Adds **PyYAML** dependency. Config snapshot copied into `experiments.details` JSON at run start for reproducibility. Defaults from D-13/14/15 if no `--config` provided.

### Pairwise validation accuracy (JUDG-02)

- **D-17:** **Ground truth = accumulated oracle feedback** (never an external label set). A pair is `(item_x, item_y, expected_same: bool)`. No new oracle calls per turn — zero cognitive-load cost. Honest with the project's "oracle is the objective function" framing (PROJECT.md).
- **D-18:** **Pair extraction (same-cluster + different-cluster):**
  - `MoveItemFeedback(x, cluster_C)` → same-cluster pairs `(x, y, True)` for every `y` currently in `C`.
  - `SplitFeedback(cluster_id, seed_item_ids=[a, b])` → different-cluster pair `(a, b, False)`.
  - `MergeFeedback(A, B)` → same-cluster pairs `(x, y, True)` for every `x in A`, `y in B`.
  - `GlobalFeedback` and `InstructionalFeedback` contribute NO pairs (too semantic — mirrors Phase 3 D-09 structural-comparison philosophy).
- **D-19:** **Sample size `N = min(50, bag_size)` per turn**, **tunable** via the same YAML config as D-16. Sampler is seeded by `turn_index` for deterministic replay — re-running the same experiment from `audit_log.jsonl` produces the exact same `pairwise_accuracy` per turn. Compute = `matches / N`. Stored in `turns.details` JSON each turn as `pairwise_accuracy`. End-of-run mean written to `experiments.details` JSON as `mean_pairwise_accuracy`. Hard cap of 50 prevents big merges (which can add thousands of pairs in one shot) from dominating later turns and from blowing up cost in Phase 5 N×M×K ablations.
- **D-20:** **Contradiction handling: overwrite on contradiction.** Honors CLUS-04 "latest intent wins". When a contradicting feedback arrives, drop pairs involving the affected items from older feedback, then add the new ones. The bag always reflects the oracle's CURRENT preferences. `pairwise_accuracy` measures "do we match what the oracle currently wants", not "did we ever match anything they ever said". Phase 3 `drift_event` records in `events.jsonl` preserve the contradiction history separately.
- **D-21:** **`PairBag` lives in memory inside the Judge module**, owned by a run. On session resume (Phase 2 UI-V2-01), the bag is **reconstructed from `audit_log.jsonl` + `events.jsonl`** by replaying feedback events sequentially and applying overwrites in order. No new persistence surface; the bag is derived state. Reconstruction is deterministic per D-04.

### No-dialogue baseline (JUDG-03)

- **D-22:** **Oracle runs ONCE on the initial clustering.** Build initial clustering (FOUND-02 HDBSCAN or KMeans backend). Show it to the oracle exactly once. Capture the one-shot feedback into the pair bag. Compute `pairwise_accuracy` of the initial clustering against THAT bag. Also capture `oracle.satisfied` (would they accept this default?). One turn, one entry in `turns` table, all metric bundle fields filled. Directly answers "how good is the default clustering before the conversation adds value?" — the comparison point JUDG-03 requires.
- **D-23:** **`strategy_id = "no_dialogue"`** distinguishes baseline runs in the `experiments` table. Baseline is just another strategy. Same schema, same columns. Phase 5 analysis: `SELECT mean_pairwise_accuracy FROM experiments WHERE dataset=? GROUP BY strategy_id` shows baseline alongside random/uncertainty_driven/boundary_driven naturally.
- **D-24:** **Pairing by `(dataset, persona_id, seed)` tuple — implicit join.** No explicit FK between interactive and baseline rows. The CLI flags for the baseline script mirror the interactive entry point (`--dataset`, `--persona`, `--seed`) so a matched pair is one command away. Phase 5 harness fans out paired runs automatically.
- **D-25:** **Entry point split:** `src/judge.py` exposes `run_baseline(dataset, persona_spec, noise_params, seed, db, criteria) → ExperimentRead` — the reusable function. `examples/run_baseline.py` is a ~20-line CLI wrapper (recipe §10): parses args, opens DB connection per recipe pattern, calls `run_baseline()`, prints the resulting `ExperimentRead` as JSON. Imports the internal Python API directly — no HTTP, no socketio. Phase 5 harness imports `run_baseline` for batch runs.

### Judge structure and WR-04 cleanup

- **D-26:** **`src/stopping.py` is `f_eval`.** Phase 1's `check_stopping()` IS the JUDG-01 `f_eval` — already a pure function. Phase 4 just fills in the D-13/14/15 numbers. **New `src/judge.py`** holds everything else (recipe §6: "if a layer leaks, fix that before adding the next"):
  - `PairBag` dataclass + extraction/sampling logic (D-17 through D-20)
  - `compute_pairwise_accuracy(state, bag, rng) → float`
  - `assemble_turn_metrics(state, reply, pair_acc, stop_reason) → TurnCreate` (Pydantic, recipe §7)
  - `run_baseline(...)` from D-25
  
  All pure functions matching the f_* decomposition pattern (Phase 2 D-02). `conversation_loop.py` calls these per turn after `f_next_state`. **`src/judge.py` MUST NOT touch SQL directly** — it builds Pydantic models and hands them to `src/db/turns.py` / `src/db/oracle_feedback.py`.
- **D-27:** **WR-04 fix bundled with Phase 4 wiring:** update `OracleProtocol.reply()` signature to `(state, message, global_instructions=None, cognitive_load=None)`. Update `MockOracle.reply()` to accept (and ignore) both new params. Remove the `isinstance(oracle, OracleAgent)` branch in `conversation_loop.py:183-188` — call `oracle.reply(..., global_instructions=..., cognitive_load=...)` unconditionally for all oracle types. Three small edits; one plan task. Required for Phase 5's planned baseline-strategy oracle variant.
- **D-28:** **Web UI panels** (deferred from Phase 2 D-14): add to the existing metrics sidebar — **Contradictions count** (from `reply.contradiction_detected`, already in `state_update` SocketIO event), **Convergence signal** indicator (one of `ORACLE_SATISFIED` / `TURN_BUDGET` / `DIMINISHING_RETURNS` / `running`, from `check_stopping` result), **Pairwise accuracy** value + a tiny 10-turn sparkline (from `turns.details` JSON). All three already flow through `state_update`; the JS just renders them. ~30 lines of vanilla JS, no new dependencies (consistent with Phase 2 D-13/24).

### STRICT_MODE + deviation() (CI deferred)

- **D-29:** **`deviation()` helper in new code only** (recipe §11). Add `src/logging_setup.py` with `deviation(msg, **kwargs)` + `UnexpectedDeviation(RuntimeError)`. Behavior: `STRICT_MODE=1` env var → raise; otherwise → `log.warning(msg, extra=kwargs)`. **Phase 4 code (src/db/, src/judge.py, examples/run_baseline.py) uses it for every "this shouldn't normally happen" branch** (LLM retry exhausted, soft-delete that matched no rows, FK lookup that returns None unexpectedly, config value falling back to default). **Phase 1–3 code is NOT mass-rewritten** — existing asserts already match the fail-loudly philosophy and serve as deviation markers for must-be-true conditions. `deviation()` is complementary, not replacement: assertions = must-be-true invariants; `deviation()` = may-happen-but-unexpected.
- **D-30:** **CI deferred entirely.** No `.github/workflows/ci.yml` in Phase 4. No `pyproject.toml` lint/test/coverage config. `STRICT_MODE` is a runtime flag with no automated enforcement; contributors who run `STRICT_MODE=1 pytest` locally get the strict-mode signal, but nothing forces it. Reconsider at end of Phase 5 or when team grows.

### Documentation discipline

- **D-31:** **New `docs/MODEL.md`** as the living spec of the DB layer. Contents: column-by-column table for `experiments` / `turns` / `oracle_feedback`, the cascade rules from D-07 in list form, the slug-uniqueness invariant from D-12, the ISO-8601 UTC timestamp convention from D-05. ~40 lines per recipe §15. **`docs/MODEL.md` changes FIRST when the schema changes; SQL follows** — drift between spec and reality becomes a workflow violation, not a quiet inconsistency. Phase 4 plan tasks include an "edit `docs/MODEL.md` first" step before each schema task.
- **D-32:** **CLAUDE.md additive updates** — do NOT replace or weaken the existing "Coding Philosophy — Fail Loudly" section. New entries added under "Key Constraints" and adjacent to "Web Stack":
  - `src/db/` is the ONLY layer that touches SQL — if you find `sqlite3.execute` outside `src/db/`, fix it.
  - `experiments.db` lives at repo root, gitignored, opened with `check_same_thread=False` + `PRAGMA journal_mode=WAL` + `PRAGMA foreign_keys=ON`.
  - `deviation()` in `src/logging_setup.py` marks unexpected branches; `STRICT_MODE=1` raises; complementary to `assert`, not a replacement.
  - JSONL `audit_log.jsonl` remains the source of truth for replay; DB is the queryable index.
- **D-33:** **README.md updates only** (no `docs/DEMO.md` for now). Document: how to run `examples/run_baseline.py`, new env vars (`STRICT_MODE`), where `experiments.db` lives, the YAML config flag, quickstart for the baseline-vs-interactive comparison. Skip `docs/DEMO.md` — recipe pairs it with a `seed.py` which doesn't exist (and isn't in Phase 4 scope). Add `docs/DEMO.md` when Phase 5 introduces a real seeded comparison-demo flow.

### Claude's Discretion

- Exact Pydantic field validators and types for `ExperimentCreate` / `TurnCreate` / `OracleFeedbackCreate` (e.g., int constraints, JSON schema for `details`) — planner decides based on Phase 4 schema.
- Exact YAML config schema (field names, structure for the `weights` sub-object) — planner decides; should mirror `StoppingCriteria` + `FeedbackMagnitudeWeights` field names.
- SQL indexes beyond the FKs and the partial slug-uniqueness index — planner decides based on Phase 5 query patterns (likely `(strategy_id, persona_id, dataset)` composite for the cross-run join).
- `src/logging_setup.py` log format (JSON-line per recipe §11 vs. simpler key=value) — planner decides.
- `RotatingFileHandler` config (recipe §11 suggests 10MB × 5 backups when `LOG_FILE_PATH` is set) — opt-in, planner picks defaults.
- Exact SocketIO event shape additions for D-28 (likely extend `state_update` rather than add new events).
- Sparkline rendering details for D-28 (canvas vs. tiny SVG vs. inline divs) — planner decides.
- HDBSCAN `min_cluster_size` / KMeans BIC search range for `run_baseline()` initial clustering — reuse Phase 1/Phase 2 defaults unless ablations need otherwise.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Recipe (user-provided this session — MUST read; informs entire DB layer design)
- `private/recipes/how-to-build-simple-applications.md` — wazzup recipe. Specifically §2 (entity shape), §3 (rels vs FK), §4 (cascade orchestrator), §5 (folder structure), §6 (api/ as only SQL layer + slug helper + error conventions), §7 (Pydantic triples), §9 (TestClient + check_same_thread=False fixture), §10 (examples/ scripts), §11 (deviation + STRICT_MODE), §15 (MODEL.md + CLAUDE.md). **Note:** physical file path TBD — user shared the recipe inline this session. Planner should ask the user to commit the recipe markdown somewhere stable (e.g., `private/recipes/`) before execute.

### Project Planning Artifacts
- `.planning/REQUIREMENTS.md` §Database — DB-01, DB-02, DB-03 (experiments / turns / oracle_feedback schemas)
- `.planning/REQUIREMENTS.md` §Judge Agent — JUDG-01 (3-criteria f_eval), JUDG-02 (per-turn metric bundle), JUDG-03 (no-dialogue baseline)
- `.planning/REQUIREMENTS.md` §Pre-Code Obligations — PRE-02 (3 stopping conditions; structure locked Phase 1, numbers locked here)
- `.planning/ROADMAP.md` §Phase 4 — goal, success criteria, "no-dialogue baseline critical for research Q1" note
- `.planning/PROJECT.md` — fail-loudly philosophy (D-29 deviation is COMPLEMENTARY not replacement), "no intrinsic ground truth" (D-17 reason), 3-agent architecture
- `CLAUDE.md` — Web Stack section (D-04 WAL/threading constraints), Key Constraints (AuditLog jsonl every turn — D-02 preserves this)

### Prior Phase Decisions That Carry Forward
- `.planning/phases/01-pre-code-obligations-and-foundation/01-CONTEXT.md` — D-08/09/10 (stopping criteria STRUCTURE: oracle_satisfied + turn_budget=15 + diminishing_returns). D-13 (ClusteringState schema FROZEN — never widened). D-14 (JSONL AuditLog every turn).
- `.planning/phases/02-clustering-agent-core/02-CONTEXT.md` — D-01 (plain Python while-loop — DO NOT switch to LangGraph). D-02 (standalone modules, f_* are pure functions imported by the loop). D-03 (`OracleProtocol` — D-27 updates its signature). D-04 (loop owns the JSONL write — D-03 here preserves this).  D-06 (dataclasses for feedback types — D-10 here adds Pydantic ONLY in src/db/, NOT for FeedbackDelta types). D-14/24 (existing metrics sidebar — D-28 here extends it). D-26/27 (sessions/<ts>/ dirs — D-02 here keeps these unchanged).
- `.planning/phases/03-oracle-agent/03-CONTEXT.md` — D-06 (f_cognitive_load is a pure function — D-26 here uses it). D-09 (structural FeedbackDelta comparison — D-18 here uses the same FeedbackDelta types). D-10 (events.jsonl sidecar separate from audit_log.jsonl — D-21 here reads BOTH for pair bag reconstruction). D-11 (OracleReply.contradiction_detected — D-28 here surfaces it in UI).

### Phase 3 REVIEW.md Items Resolved by Phase 4
- `.planning/phases/03-oracle-agent/03-REVIEW.md` WR-04 — `OracleProtocol.reply()` signature update (resolved by D-27).
- `.planning/phases/03-oracle-agent/03-REVIEW.md` CR-02 — `datetime.utcnow()` deprecation (resolved by D-05 ISO-8601 UTC timestamp convention applied to all new datetime usage).

### Phase 4 Integration Points (source files to read before implementing)
- `src/stopping.py` — `check_stopping()`, `StopReason`, `StoppingCriteria`, `FeedbackMagnitudeWeights`. D-13/14/15 fill the `nan`/`-1` placeholders; D-26 confirms this IS `f_eval`.
- `src/conversation_loop.py` — `run_conversation()`. D-03 (DB write after JSONL), D-21 (PairBag updates from deltas), D-27 (remove isinstance branch), D-28 (extend state_update SocketIO event).
- `src/oracle_protocol.py` — `OracleProtocol`, `OracleReply`, `MockOracle`. D-27 updates signatures.
- `src/feedback.py` — `FeedbackDelta` types. D-18 pair extraction operates on these.
- `src/serialization.py` — `append_to_audit_log()`, `load_audit_log()`. D-02 preserves JSONL contract; D-21 calls `load_audit_log()` on resume.
- `src/cognitive_load.py` — `f_cognitive_load`, `COG_LOAD_THRESHOLD`. D-26 calls this from the per-turn metric assembler.
- `web/app.py` — Background thread (`socketio.start_background_task`), `SocketIOEmitter`. D-04 (background-thread connection), D-28 (extend `state_update` payload).
- `web/templates/index.html` + `web/static/main.js` — D-28 sidebar panels + sparkline.
- `requirements.txt` — D-10 adds `pydantic`; D-16 adds `pyyaml`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `check_stopping()` / `StoppingCriteria` / `FeedbackMagnitudeWeights` (`src/stopping.py`): Phase 1 left explicit `nan` / `-1` placeholders for Phase 4 to fill (lines 51-52, 72-75). D-13/14/15 fill them.
- `f_cognitive_load()` (`src/cognitive_load.py`): Pure function; called by `conversation_loop.py:180` and by `OracleAgent.reply()`. After D-27 the loop's value is the sole computation; OracleAgent receives it as a param (recipe-clean ownership).
- `FeedbackDelta` types (`src/feedback.py`): `MoveItemFeedback`, `SplitFeedback`, `MergeFeedback`, `GlobalFeedback`, `InstructionalFeedback`. D-18 extracts pairs from the first three.
- `append_to_audit_log()` / `load_audit_log()` (`src/serialization.py`): D-02 preserves the JSONL contract unchanged. D-21 uses `load_audit_log()` for pair bag reconstruction on resume.
- `_write_event()` (`src/conversation_loop.py:70-88`): Phase 3 sidecar writer. NOT extended for DB events — DB writes go through `src/db/`.
- `OracleReply.contradiction_detected` + `contradicted_turn` (`src/oracle_protocol.py`): Phase 3 added these. D-28 surfaces them in the UI.
- `SocketIOEmitter` thread→loop bridge (`web/app.py`): Pattern for emitting from worker threads via `asyncio.run_coroutine_threadsafe`. D-28 reuses it for the extended `state_update` payload.

### Established Patterns
- **Fail loudly + recipe `deviation()` complement:** D-29 introduces `deviation()` for unexpected branches in NEW code; existing `assert` statements stay. CLAUDE.md "Coding Philosophy — Fail Loudly" is unchanged (D-32).
- **Pure f_* functions:** `f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state`, `f_cognitive_load`, and now `f_eval` (via `check_stopping`) all take state as input and return new state — no I/O, no global mutation. D-26 keeps `src/judge.py` pure (no SQL); it hands Pydantic models to `src/db/`.
- **Dataclass-first for in-memory state; Pydantic only inside `src/db/`** (D-10). FeedbackDelta types, ClusteringState, OracleReply, etc. stay dataclasses.
- **Loop owns the JSONL write (Phase 2 D-04):** D-03 preserves this; DB write is AFTER JSONL write in the same loop step.
- **Sessions persist in `sessions/<ts>/` with state.json + audit_log.jsonl + events.jsonl** (Phase 2 D-26, Phase 3 D-10): D-02 keeps these unchanged; DB is additive.
- **Named constants in module:** `COG_LOAD_THRESHOLD = 0.7`, `KMEANS_SOFTMAX_TEMP = 1.0`. D-13/14/15 follow the convention if any default needs naming in code (StoppingCriteria field defaults already provide this).

### Integration Points
- `f_cognitive_load(state, message)` → called once in `run_conversation()` line 180 → passed to `oracle.reply(..., cognitive_load=...)` (D-27 removes isinstance branch) → also passed to `assemble_turn_metrics()` for the turns.details JSON.
- `f_next_state(...)` → returns `new_state` AND `deltas` → both feed `PairBag.update(deltas, new_state)` per D-18/19/20 → which updates the in-memory bag.
- `check_stopping(...)` → returns `StopReason | None` → fed to `assemble_turn_metrics()` (the `convergence_signal` field for D-28) and consumed by the loop's break condition.
- `assemble_turn_metrics(state, reply, pair_acc, stop_reason)` → builds `TurnCreate` Pydantic model → handed to `src/db/turns.create(db, turn_create)` per D-09.
- `parse_feedback()` deltas → for each delta: `OracleFeedbackCreate(...)` Pydantic model → `src/db/oracle_feedback.create(db, fb_create)`. Compound oracle messages produce multiple rows per turn (DB-03 explicitly requires this).
- Run start: `src/db/experiments.create(db, ExperimentCreate(...))` → returns `ExperimentRead` with id+slug → loop uses `.id` as the FK for all turn/feedback writes.
- Run end: `src/db/experiments.update(db, exp_id, ExperimentUpdate(total_turns=..., convergence_reason=..., end_timestamp=..., details=mean_metrics_dict))` → seals the summary row.
- Resume: `src/db/experiments.get_by_slug(db, slug)` → if found, replay `load_audit_log(path)` → rebuild `PairBag` via D-21 → continue loop.

### Forbidden Patterns (recipe rules)
- SQL anywhere outside `src/db/` (recipe §6) → fix immediately.
- `try: ... except: pass` or `except Exception:` outside the 4 recipe-legitimate shapes (recipe §11) — covered by existing CLAUDE.md fail-loudly.
- Writing oracle_init / drift_event / DB records to `audit_log.jsonl` (Phase 3 Pitfall 5 — only ClusteringState lines go there).
- Mixing soft and hard delete in a single cascade (recipe §4.6 — separate code paths).
- `datetime.utcnow()` anywhere new (Phase 3 CR-02) — use `datetime.now(timezone.utc).isoformat()`.

</code_context>

<specifics>
## Specific Ideas

- **Experiment slug pattern:** `f"{strategy_id}-{persona_id}-seed{seed}"` → `"random-trump-seed42"`, `"no_dialogue-curie-seed7"`. Greppable in logs, usable as a CLI arg (`python -m examples.run_baseline --name random-trump-seed42`).
- **YAML config example shape** (D-16) — planner should ship one starter config:
  ```yaml
  # experiments/configs/default.yaml
  stopping:
    epsilon: 0.05
    n_fallback: 3
    weights:
      global: 1.0
      cluster: 0.5
      point: 0.2
      instructional: 0.1
  judge:
    pairwise_sample_size: 50
  ```
- **Deterministic pairwise sampling seed:** `random.Random(turn_index).sample(bag, k)` — no global RNG, no test contamination. Replay produces identical samples.
- **WAL mode setup (D-04):** `conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA foreign_keys=ON"); conn.execute("PRAGMA synchronous=NORMAL")` — the third is the recipe-implicit perf/safety tradeoff acceptable for non-financial workloads.
- **Partial unique index for slug (D-12):** `CREATE UNIQUE INDEX experiment_slug_live ON experiments(slug) WHERE deleted_at IS NULL;` — soft-deleted slugs become reusable.
- **`run_baseline` minimal CLI surface (D-25):** `--dataset`, `--persona`, `--seed`, `--config`, optional `--name` override.
- **JSON Lines AuditLog cache for pair bag rebuild (D-21):** stream-read `audit_log.jsonl` once; for each line decode `deltas` if present; replay through `PairBag.update()`. O(N_turns) at session resume; negligible cost for 15-turn budget.
- **STRICT_MODE in tests:** test that deliberately exercises a `deviation()` path uses `with pytest.raises(UnexpectedDeviation): ...` and runs under `STRICT_MODE=1`. Per recipe §11 "Testing deviation paths in strict mode".

</specifics>

<deferred>
## Deferred Ideas

- **CI with STRICT_MODE matrix** (recipe §12) — deferred per D-30. Reconsider at end of Phase 5 or when team grows beyond solo.
- **Local lint/test tooling config** (`[tool.ruff]`, `pytest -W error`, branch coverage) — deferred entirely per D-30. Reconsider with CI.
- **Backfill `deviation()` to Phase 1–3 code** — explicitly NOT in scope (D-29). Migrate opportunistically when modifying old code.
- **Mass-rewriting Phase 1–3 dataclasses to Pydantic** — explicitly NOT in scope (D-10). Pydantic stays inside `src/db/`.
- **Polymorphic rels table** (recipe §3 purist form) — deferred per D-06. Add only if a future phase needs ad-hoc cross-experiment relationships.
- **`docs/DEMO.md` + `examples/seed.py`** (recipe §10/§15) — deferred to Phase 5 when a seeded comparison-demo flow becomes real.
- **`UPDATE` flow on experiments rows** — only `update()` we ship is for end-of-run summary write (D-09). Full mutation API deferred until needed.
- **`ExperimentUpdate` Pydantic model** — declared per D-10 for completeness/symmetry, but no caller mutates experiments mid-run; only the end-of-run summary write uses it. Could be slimmed to just `total_turns`, `convergence_reason`, `end_timestamp`, `details` fields.
- **Health/readiness HTTP endpoints + real auth + migrations** (recipe "What we haven't built") — all explicitly out of scope for v1 per PROJECT.md tier.
- **Real-time websocket pushes for pair bag updates** — not needed; existing `state_update` SocketIO event already fires per turn, D-28 extends its payload.
- **Async DB driver / Postgres** — out of scope; SQLite is in-process and sync per D-01.
- **Project-level pending todo "Install umap-learn and hdbscan packages"** (from STATE.md) — not Phase 4 scope; pre-existing env issue from Phase 1.
- **Project-level pending todo "Human validation study protocol"** (from STATE.md, for Phase 6) — not Phase 4 scope.

</deferred>

---

*Phase: 4-Judge-Agent*
*Context gathered: 2026-05-16*
*Recipe followed: wazzup "how to build simple applications" (user-provided this session)*
