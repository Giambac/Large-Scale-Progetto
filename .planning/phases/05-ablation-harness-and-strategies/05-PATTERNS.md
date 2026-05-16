# Phase 5: Ablation Harness and Strategies - Pattern Map

**Mapped:** 2026-05-17
**Files analyzed:** 12 (6 CREATE + 6 MODIFY)
**Analogs found:** 11 / 12 (notebook has no in-repo analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/harness.py` (CREATE) | new module — orchestrator | producer (writes DB rows) | `src/judge.py` `run_baseline()` | role-match (no parallel orchestrator exists) |
| `src/analysis.py` (CREATE) | new module — pure compute | transformer (pure function) | `src/uncertainty.py` `f_uncertainty()` | role-match (pure stats, no I/O) |
| `examples/run_harness.py` (CREATE) | CLI wrapper | consumer (CLI → fn) | `examples/run_baseline.py` | exact (Phase 4 D-25 pattern) |
| `examples/compute_ci.py` (CREATE) | CLI wrapper | consumer (DB → table/JSON) | `examples/run_baseline.py` | exact (D-25 pattern) |
| `notebooks/analysis.ipynb` (CREATE) | notebook | consumer (DB → plots) | *(none — no notebooks in repo)* | no analog |
| `experiments/configs/harness.yaml` (CREATE) | config | producer (consumed by harness) | `experiments/configs/default.yaml` | exact (extend, do not replace) |
| `src/strategy.py` (MODIFY — add 2 classes) | extension | transformer (state → Action) | `RandomStrategy` in same file | exact (same protocol target) |
| `src/uncertainty.py` (READ-ONLY) | reference | — | n/a — no change | n/a |
| `src/conversation_loop.py` (MODIFY `_format_message`) | extension | transformer (Action → str) | `_format_message` existing signature | exact (extend in place) |
| `src/db/connection.py` (MODIFY `init_schema`) | extension — DDL | producer (schema) | existing DDL in same `init_schema()` | exact (add one column) |
| `src/db/experiments.py` (MODIFY Pydantic models) | extension — model | transformer (row ↔ Pydantic) | `ExperimentCreate` / `ExperimentRead` in same file | exact |
| `docs/MODEL.md` (MODIFY — document new column) | docs | — | existing experiments table doc block | exact (add row to existing table) |

## Pattern Assignments

---

### `src/strategy.py` — add `UncertaintyDrivenStrategy` and `BoundaryDrivenStrategy`

**Analog:** `src/strategy.py` `RandomStrategy` (lines 71–86)

**Seeded RNG init pattern** (lines 79–80):
```python
def __init__(self, seed: int | None = None) -> None:
    self._rng = random.Random(seed)
```
Both new strategies MUST adopt this exact init shape (per D-19 "Deterministic sampling in BoundaryDriven — seed item selection by `random.Random(state.turn_index)`").

**`select()` signature pattern** (lines 82–86):
```python
def select(self, state: ClusteringState, uncertainty_report: object) -> Action:
    """Select a uniformly random valid action for the current state."""
    valid_actions = _enumerate_valid_actions(state, uncertainty_report)
    assert len(valid_actions) > 0, "BUG: no valid actions to select from"
    return self._rng.choice(valid_actions)
```
Same signature for both new strategies. They MUST NOT inherit from `StrategyProtocol` (line 28–39: structural subtyping; runtime-checkable Protocol). Type hint `uncertainty_report: object` is consistent across the file; cast/narrow inside `select()`.

**Action payload extension pattern** (lines 15–26 — `Action` dataclass):
```python
@dataclass
class Action:
    action_type: Literal["show_full", "show_subset", "ask_question", "stop"]
    payload: dict = field(default_factory=dict)
```
New strategies populate `payload` per D-01 (`{"cluster_id": int}`) and D-03 (`{"cluster_a": int, "cluster_b": int, "item_ids": list[int]}`). The `Action` dataclass already supports arbitrary payload dicts — no change to the dataclass itself.

**UncertaintyReport consumption pattern** (`src/uncertainty.py` lines 17–28):
```python
@dataclass
class UncertaintyReport:
    boundary_items: list[tuple[int, float]]        # (item_id, normalized_entropy), descending
    split_candidates: list[tuple[int, float]]      # (cluster_id, mean_entropy), descending
    merge_candidates: list[tuple[int, int, float]] # (cluster_a_id, cluster_b_id, distance), ascending
```
- `UncertaintyDrivenStrategy` (D-01) → uses `split_candidates[0]` (highest mean_entropy cluster) → `cluster_id` payload.
- `BoundaryDrivenStrategy` (D-03) → uses `merge_candidates[0]` (closest cluster pair), then scans `state.soft_probs` to assemble `item_ids` whose top-2 dominant cluster indices match the pair.

**Degenerate fallback pattern** (D-02 / D-04) — use `deviation()` from `src/logging_setup.py` line 41:
```python
from src.logging_setup import deviation
# When no valid uncertainty target:
deviation(
    "UncertaintyDrivenStrategy: no valid target — falling back to generic ask_question",
    n_clusters=len(state.clusters),
)
return Action(action_type="ask_question", payload={})
```
Matches the `deviation()` calls in `src/judge.py` lines 143–147 (kwargs-style key/value pairs, NOT a crash).

---

### `src/harness.py` — `run_harness(config_path, db) → list[ExperimentRead]` + strategy registry + ThreadPoolExecutor

**Analog:** `src/judge.py` `run_baseline()` (lines 297–463)

**Function-level lazy imports pattern** (`src/judge.py` lines 328–343):
```python
from datetime import datetime, timezone

# NOTE: AnthropicClusterNamer intentionally NOT imported here ...
from src.clustering import HDBSCANBackend, build_initial_clustering_state
from src.conversation_loop import _format_message
from src.db import experiments as exp_db
from src.db import oracle_feedback as fb_db
from src.db import turns as turn_db
from src.db.connection import init_schema
from src.db.experiments import ExperimentCreate, ExperimentUpdate
from src.oracle_protocol import MockOracle, OracleReply
from src.stopping import StoppingCriteria, compute_magnitude, FeedbackMagnitudeWeights
from src.strategy import RandomStrategy
from src.uncertainty import f_uncertainty
from src.agent_functions import f_next_best_step
```
Harness body uses this same lazy-import idiom — keep heavy imports inside the function. Add the new strategy registry near the top of the module:
```python
# Module-level registry (D-09 spec — referenced in <specifics>)
STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "uncertainty_driven": UncertaintyDrivenStrategy,
    "boundary_driven": BoundaryDrivenStrategy,
}
```

**ExperimentCreate construction pattern** (`src/judge.py` lines 353–363):
```python
exp_name = f"no_dialogue-{persona_id}-seed{seed}"
exp_create = ExperimentCreate(
    name=exp_name,
    strategy_id="no_dialogue",
    persona_id=persona_id,
    seed=seed,
    dataset=dataset_name,
    start_timestamp=start_ts,
    details={},
)
experiment = exp_db.create(db, exp_create)
```
Harness per-run worker builds the same shape but with `strategy_id=<from registry key>` and the new `oracle_type="llm"` field (D-11). Slug pattern stays `{strategy}-{persona}-seed{seed}`.

**Per-run DB connection pattern** (Phase 4 D-04 — `src/db/connection.py` lines 21–33):
```python
def connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
```
Each `ThreadPoolExecutor` worker calls `connect()` (NOT shared from the parent thread per D-08). `check_same_thread=False` + WAL mode already make this safe. Each worker closes its own connection in a `finally` block (mirror `examples/run_baseline.py` lines 73–84).

**ThreadPoolExecutor pattern** (D-08 — NO existing analog in `src/`; the canonical fail-loudly shape, to be implemented per CLAUDE.md):
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
# Fail-loudly: as_completed surfaces thread exceptions immediately — no swallowing.
with ThreadPoolExecutor(max_workers=harness_cfg["max_workers"]) as pool:
    futures = {pool.submit(_run_one, combo): combo for combo in combos}
    results = []
    for fut in as_completed(futures):
        results.append(fut.result())  # raises if worker raised
return results
```
No `try/except` around `fut.result()` — let exceptions propagate per CLAUDE.md "Fail Loudly" rule and Phase 5 D-08.

**Sealing/finalization pattern** (`src/judge.py` lines 449–461):
```python
end_ts = datetime.now(timezone.utc).isoformat()
experiment = exp_db.update(db, experiment.id, ExperimentUpdate(
    total_turns=1,
    convergence_reason=stop_reason.value if stop_reason else None,
    end_timestamp=end_ts,
    details={
        "mean_cognitive_load": reply.turn_cognitive_load,
        "mean_pairwise_accuracy": pair_acc,
        ...
    },
))
```
Harness `_run_one` calls `run_conversation()` (for non-baseline strategies) or `run_baseline()` (for `no_dialogue`); both already seal the row. Harness just returns the `ExperimentRead`.

**OracleAgent construction pattern** (`src/oracle_agent.py` lines 32–60 — `OracleSpec` + `NoiseParams` are dataclasses, Phase 4 D-10):
```python
@dataclass
class OracleSpec:
    preferred_k: int
    semantic_axes: list[str]
    persona_description: str

@dataclass
class NoiseParams:
    consistency_rate: float
    drift_probability: float
    sycophancy_resistance: float
```
Harness parses YAML `personas:` section into these dataclasses (NOT Pydantic — D-10 carries forward: Pydantic ONLY inside `src/db/`).

---

### `src/analysis.py` — `compute_bootstrap_ci(values, n_bootstrap=10000, ci=0.95) → tuple[float, float]`

**Analog:** `src/uncertainty.py` `f_uncertainty()` (lines 31–88)

**Pure-function module header pattern** (`src/uncertainty.py` lines 1–8):
```python
"""
uncertainty.py — f_uncertainty pure function + UncertaintyReport dataclass (CLUS-02).

Computes normalized Shannon entropy per item and aggregates into ranked lists for
boundary detection (split candidates) and proximity detection (merge candidates).
No I/O. No global state. Pure function only.
"""
```
`src/analysis.py` MUST use the same docstring shape — "No I/O. No SQL. Pure function only." matching D-13.

**Numpy import + signature pattern** (`src/uncertainty.py` lines 12, 31–40):
```python
import numpy as np
...
def f_uncertainty(state: ClusteringState) -> UncertaintyReport:
    """
    Pure function. No I/O. No global state.
    ...
    """
```
`compute_bootstrap_ci` uses `numpy.random.default_rng(seed)` for deterministic resampling (mirror RNG-isolation pattern from `PairBag.sample()` `src/judge.py` line 127: `rng = random.Random(turn_index)` — local RNG, never global module state).

**Assert-on-invariants pattern** (`src/uncertainty.py` line 42):
```python
assert K > 0, "f_uncertainty called on empty ClusteringState (no clusters)"
```
Apply to `compute_bootstrap_ci`: `assert len(values) > 0, "compute_bootstrap_ci called with empty values"` and `assert 0 < ci < 1`. Fail loudly per CLAUDE.md.

---

### `examples/run_harness.py` — ~30-line CLI wrapper

**Analog:** `examples/run_baseline.py` (lines 1–91, entire file)

**Argparse + dataset-load + config-load + run pattern** (`examples/run_baseline.py` lines 24–87):
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-dialogue baseline (JUDG-03)")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--persona", required=True, help="Oracle persona identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--config", default=None, help="Path to YAML config file (default: experiments/configs/default.yaml)")
    args = parser.parse_args()
    ...
    # Open DB and run baseline
    from src.db.connection import connect, init_schema
    from src.judge import run_baseline
    db = connect()
    init_schema(db)
    try:
        experiment = run_baseline(records=records, persona_id=args.persona, ...)
    finally:
        db.close()
    print(experiment.model_dump_json(indent=2))
```
`run_harness.py` adopts the same skeleton but its only required arg is `--config` (default `experiments/configs/harness.yaml`); calls `run_harness(config_path, db)`, prints `json.dumps([e.model_dump() for e in results], indent=2)`.

**Lazy import below argparse pattern** (`examples/run_baseline.py` lines 66–68):
```python
# Open DB and run baseline
from src.db.connection import connect, init_schema
from src.judge import run_harness  # ← harness import here
```
Import of `run_harness` is delayed until after argparse — fast `--help`.

---

### `examples/compute_ci.py` — ~30-line CLI, queries experiments, prints table or `--json`

**Analog:** `examples/run_baseline.py` (same Phase 4 D-25 pattern)

**Argparse + DB query pattern** — combine baseline CLI skeleton with `src/db/experiments.py` `query()` (lines 156–180):
```python
def query(
    conn: sqlite3.Connection,
    *,
    strategy_id: str | None = None,
    persona_id: str | None = None,
    dataset: str | None = None,
    include_deleted: bool = False,
) -> list[ExperimentRead]:
```
`compute_ci.py` calls `exp_db.query(db, dataset=args.dataset)` then groups by `strategy_id` in Python (NOT SQL), calls `compute_bootstrap_ci(values)` per group, prints a table.

**Output table format** (from CONTEXT.md `<specifics>` lines 162–170 — copy verbatim):
```
Strategy              | Mean turns | 95% CI         | N runs
----------------------|------------|----------------|-------
random                |       14.2 | [12.1 – 16.3]  |     9
uncertainty_driven    |       10.8 | [9.1 – 12.5]   |     9
boundary_driven       |       11.4 | [9.8 – 13.0]   |     9
no_dialogue (baseline)|        1.0 | [1.0 – 1.0]    |     9
```
`--json` flag emits `json.dumps([{"strategy": s, "mean": m, "ci_lo": lo, "ci_hi": hi, "n": n}, ...])`.

**Open/close DB pattern** (`examples/run_baseline.py` lines 70–84) — identical:
```python
db = connect()
init_schema(db)
try:
    rows = exp_db.query(db, dataset=args.dataset)
    ...
finally:
    db.close()
```

---

### `notebooks/analysis.ipynb` — Jupyter exploration notebook

**Analog:** *(none — no notebooks exist in the repo)*

Use these patterns from existing modules as cell templates:
- Cell 1: imports — copy module imports from `examples/compute_ci.py` (connect + query).
- Cell 2: DB query — `exp_db.query(db, dataset=...)` (same as CLI).
- Cell 3: CI computation — `from src.analysis import compute_bootstrap_ci` (D-15: shared with CLI).
- Cell 4+: plot bootstrap distributions with matplotlib (no in-repo analog; standard practice).

Kernel registration step documented per CONTEXT.md D-15: `python -m ipykernel install --user --name=conversational-clustering`.

---

### `experiments/configs/harness.yaml` — cross-product config

**Analog:** `experiments/configs/default.yaml` (lines 1–17 — entire file)

**YAML structure pattern** (`experiments/configs/default.yaml` lines 6–16):
```yaml
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
`harness.yaml` EXTENDS — does NOT replace — by adding `harness:` and `personas:` top-level sections per CONTEXT.md D-09 (lines 46–64). Include `stopping:` and `judge:` sections too so a single config drives everything.

**Required new shape** (verbatim from CONTEXT.md D-09):
```yaml
harness:
  strategies: [random, uncertainty_driven, boundary_driven]
  personas: [curious, skeptical, drifty]
  seeds: [1, 2, 3]
  exclude: []            # optional list of {strategy, persona, seed} dicts
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
stopping: ...   # carry forward from default.yaml
judge: ...      # carry forward from default.yaml
```

---

### `src/db/connection.py` — add `oracle_type` column to `experiments` DDL

**Analog:** `src/db/connection.py` `init_schema()` (lines 36–101 — same function, in-place edit)

**DDL extension pattern** (lines 46–62 — existing experiments table):
```python
conn.executescript("""
    CREATE TABLE IF NOT EXISTS experiments (
        id                  INTEGER PRIMARY KEY,
        name                TEXT    NOT NULL,
        slug                TEXT    NOT NULL,
        ...
        dataset             TEXT    NOT NULL,
        total_turns         INTEGER,
        ...
        details             TEXT
    );
```
Add ONE new line per D-11, e.g. immediately after `dataset` (consistent column grouping):
```python
        dataset             TEXT    NOT NULL,
        oracle_type         TEXT    NOT NULL DEFAULT 'llm',
        total_turns         INTEGER,
```
**No `ALTER TABLE`** — D-11 explicit: Phase 5 starts with fresh `experiments.db` (gitignored; CLAUDE.md notes `experiments.db` is at repo root, gitignored).

**Optional index pattern** (lines 64–68 — existing index style):
```python
CREATE INDEX IF NOT EXISTS idx_experiments_strategy
    ON experiments(strategy_id, persona_id, dataset);
```
D-12 calls oracle_type "indexed and directly filterable" — planner adds (per Claude's Discretion line 89): `CREATE INDEX IF NOT EXISTS idx_experiments_oracle_type ON experiments(oracle_type);`

---

### `src/db/experiments.py` — extend Pydantic models with `oracle_type`

**Analog:** `src/db/experiments.py` `ExperimentCreate` / `ExperimentRead` (lines 36–63 — same file)

**ExperimentCreate extension pattern** (lines 36–44):
```python
class ExperimentCreate(BaseModel):
    name: str
    strategy_id: str
    persona_id: str
    seed: int
    dataset: str
    start_timestamp: str
    details: dict[str, Any] = Field(default_factory=dict)
    slug_override: str | None = None
```
Add: `oracle_type: str = "llm"` (default per D-11) immediately after `dataset`.

**ExperimentRead extension pattern** (lines 47–62):
```python
class ExperimentRead(BaseModel):
    id: int
    name: str
    slug: str
    ...
    dataset: str
    total_turns: int | None
    ...
```
Add: `oracle_type: str` (no default — read from DB row; column is `NOT NULL DEFAULT 'llm'` at DDL level).

**create() INSERT extension pattern** (lines 72–87 — same file):
```python
cursor = conn.execute(
    """
    INSERT INTO experiments
        (name, slug, created_at, updated_at, strategy_id, persona_id, seed,
         dataset, start_timestamp, details)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (data.name, slug, now, now, data.strategy_id, data.persona_id, data.seed,
     data.dataset, data.start_timestamp, details_json),
)
```
Add `oracle_type` to column list AND value tuple AND parameter count (10 → 11 placeholders).

**_read() pattern** (lines 183–186 — unchanged signature, Pydantic handles new field automatically because column comes through `dict(row)` in `_row_to_dict`).

---

### `src/conversation_loop.py` — extend `_format_message()` to accept `id_to_text`

**Analog:** `src/conversation_loop.py` `_format_message()` (lines 60–80 — same function, in-place edit)

**Current signature** (line 60):
```python
def _format_message(action: object, state: ClusteringState) -> str:
```
**New signature per D-16**:
```python
def _format_message(
    action: object,
    state: ClusteringState,
    id_to_text: dict[int, str] | None = None,
) -> str:
```
Default `None` for backward compatibility — `src/judge.py` line 408 calls `_format_message(action, initial_state)` (no `id_to_text`) and must keep working without modification.

**Backward-compatible payload check pattern** (lines 67–80 — existing branches):
```python
if action.action_type == "show_full":
    cluster_summary = "; ".join(
        f"Cluster {c.id} '{c.name}' ({len(c.item_ids)} items)"
        for c in state.clusters
    )
    return f"Current clustering: {cluster_summary}"
elif action.action_type == "show_subset":
    return f"Showing a subset of clusters at turn {state.turn_index}."
```
Replace `show_subset` branch with the D-16 / `<specifics>` enrichment:
```python
elif action.action_type == "show_subset":
    if id_to_text and action.payload.get("item_ids"):
        items_text = ", ".join(
            f"'{id_to_text[i]}'" for i in action.payload["item_ids"]
            if i in id_to_text
        )
        cluster_a = action.payload.get("cluster_a")
        cluster_b = action.payload.get("cluster_b")
        # ... format per BoundaryDriven example in CONTEXT.md <specifics> line 171
    return f"Showing a subset of clusters at turn {state.turn_index}."  # fallback (empty payload — Phase 2 legacy)
```
Similar enrichment for `ask_question` when `payload.get("cluster_id")` present (UncertaintyDriven example, CONTEXT.md line 172).

**Caller threading** — `run_conversation()` (line 241) currently calls:
```python
message = _format_message(action, state)
```
Update to:
```python
message = _format_message(action, state, id_to_text)
```
`id_to_text` is already a parameter of `run_conversation()` (line 113); just thread it through.

---

### `docs/MODEL.md` — document `oracle_type` column

**Analog:** `docs/MODEL.md` existing experiments table block (lines 12–28)

**Table row insertion pattern** (line 23 — `dataset` row, current style):
```markdown
| `dataset` | TEXT | NOT NULL | Dataset name used for this run |
```
Add immediately after, matching column style and D-11 spec:
```markdown
| `oracle_type` | TEXT | NOT NULL, DEFAULT `'llm'` | Source of oracle replies: `'llm'` (automated Phase 5) or `'human'` (Phase 6 study). Indexed for cross-run filtering (EXP-V2-01). |
```

**Index entry pattern** (line 124 — existing index row):
```markdown
| `idx_experiments_strategy` | experiments | `(strategy_id, persona_id, dataset)` | — | Phase 5 cross-run GROUP BY queries |
```
Append (if D-12 index added):
```markdown
| `idx_experiments_oracle_type` | experiments | `(oracle_type)` | — | Phase 6 LLM-vs-human comparison filtering (EXP-V2-01) |
```

**Phase 4 D-31 discipline reminder** (lines 144–148 — at end of file):
> Edit this file FIRST when the schema changes. SQL follows.

This phase complies: MODEL.md is updated BEFORE `src/db/connection.py`.

---

## Shared Patterns

### Fail-Loudly + `deviation()` for unexpected branches
**Source:** `src/logging_setup.py` line 41 (`def deviation(msg, **kwargs)`) and `src/judge.py` lines 143–147 (call site).
**Apply to:** Both new strategy classes (D-02, D-04 degenerate fallbacks); `src/harness.py` (config key missing, persona registry miss, thread that can't acquire connection); `src/analysis.py` (n_bootstrap < typical threshold).

```python
from src.logging_setup import deviation
deviation(
    "BoundaryDrivenStrategy: no confused cluster pair found — falling back",
    n_clusters=len(state.clusters),
    n_items=len(state.assignments),
)
```
**Never** silently `return None` or `try/except: pass`. CLAUDE.md "Fail Loudly" + Phase 4 D-29.

### One DB connection per run (`check_same_thread=False`, WAL)
**Source:** `src/db/connection.py` `connect()` lines 21–33.
**Apply to:** Every `ThreadPoolExecutor` worker in `src/harness.py`. Each worker calls `connect()` itself; no connection sharing across threads. Each closes its own connection in `finally`.

```python
def _run_one(combo):
    db = connect()
    try:
        init_schema(db)  # idempotent — safe to call per thread
        ...
    finally:
        db.close()
```

### Datetime UTC ISO format
**Source:** `src/db/experiments.py` line 28 (`_now()`) and `src/judge.py` line 350.
**Apply to:** Every timestamp written in `src/harness.py` and `src/analysis.py` (if any).

```python
from datetime import datetime, timezone
start_ts = datetime.now(timezone.utc).isoformat()
```
**Never** `datetime.utcnow()` (CLAUDE.md rule + MODEL.md "Timestamp Convention" lines 130–139).

### Dataclasses everywhere outside `src/db/`
**Source:** `src/oracle_agent.py` lines 32–60 (`OracleSpec`, `NoiseParams`); `src/strategy.py` lines 15–25 (`Action`); `src/uncertainty.py` lines 17–28 (`UncertaintyReport`).
**Apply to:** `src/harness.py` config-parsing intermediates (any new struct). Phase 4 D-10 carry-forward: Pydantic ONLY inside `src/db/`.

### examples/ script skeleton (Phase 4 D-25)
**Source:** `examples/run_baseline.py` (entire file, 91 lines).
**Apply to:** `examples/run_harness.py` and `examples/compute_ci.py`. ~30 lines each: argparse → load config → `connect()` → call importable function → print JSON/table → close DB in `finally`. All real logic lives in `src/harness.py` / `src/analysis.py`.

### SQL stays inside `src/db/` only
**Source:** `src/db/experiments.py` (only file with `conn.execute(...INSERT/SELECT...)` outside of `connection.py`'s DDL).
**Apply to:** `src/harness.py` (uses `exp_db.create()`, `exp_db.query()` — never raw SQL); `examples/compute_ci.py` (same — calls `exp_db.query()`); `notebooks/analysis.ipynb` (same).
CLAUDE.md: "`src/db/` is the **ONLY** layer that touches SQL — if you find `sqlite3.execute` outside `src/db/`, fix it."

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `notebooks/analysis.ipynb` | notebook | DB-read + plots | No `.ipynb` files exist in repo. Treat CONTEXT.md D-15 + matplotlib/numpy standards as the reference; cells reuse `examples/compute_ci.py` patterns. |
| `ThreadPoolExecutor` usage in `src/harness.py` | concurrency primitive | parallel execution | No existing `ThreadPoolExecutor` or `multiprocessing` usage in `src/`. Phase 5 introduces it; planner follows D-08 spec (max_workers from config, `as_completed`, no exception swallowing) and stdlib docs. |

## Metadata

**Analog search scope:** `src/`, `src/db/`, `examples/`, `experiments/configs/`, `docs/`
**Files scanned:** 8 (strategy.py, judge.py, uncertainty.py, conversation_loop.py, oracle_agent.py, db/connection.py, db/experiments.py, default.yaml, MODEL.md, examples/run_baseline.py, logging_setup.py)
**Pattern extraction date:** 2026-05-17

## PATTERN MAPPING COMPLETE

**Phase:** 05 - ablation-harness-and-strategies
**Files classified:** 12
**Analogs found:** 11 / 12

### Coverage
- Files with exact analog: 9 (run_harness.py, compute_ci.py, harness.yaml, strategy.py extensions, conversation_loop.py edit, db/connection.py edit, db/experiments.py edit, docs/MODEL.md edit, src/uncertainty.py read-only)
- Files with role-match analog: 2 (src/harness.py — like judge.run_baseline but parallel orchestrator; src/analysis.py — like uncertainty.f_uncertainty in purity)
- Files with no analog: 1 (notebooks/analysis.ipynb)

### Key Patterns Identified
- `examples/` CLI wrappers stay ~30 lines and delegate ALL logic to importable `src/` functions (Phase 4 D-25)
- All concurrent code uses one `sqlite3.Connection` per thread with `check_same_thread=False` + WAL (Phase 4 D-04, Phase 5 D-08)
- Strategy classes use seeded `random.Random` instances (never global `random` module) and structural Protocol conformance (no inheritance from `StrategyProtocol`)
- Pydantic is confined to `src/db/`; everywhere else uses `@dataclass` (Phase 4 D-10)
- Unexpected-but-possible branches use `deviation()` with kwargs context, not `try/except: pass` (Phase 4 D-29, CLAUDE.md fail-loudly)
- Schema changes update `docs/MODEL.md` FIRST, then DDL (Phase 4 D-31)
- All timestamps: `datetime.now(timezone.utc).isoformat()` — never `datetime.utcnow()`

### File Created
`.planning/phases/05-ablation-harness-and-strategies/05-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
