---
phase: 4
plan: "04-03"
subsystem: judge
tags: [judge, pairwise-accuracy, pair-bag, baseline, metrics]
dependency_graph:
  requires:
    - "04-01"  # src/db/turns.py TurnCreate
    - "04-02"  # src/db/experiments.py, src/db/oracle_feedback.py, src/stopping.py
  provides:
    - "src/judge.py PairBag"
    - "src/judge.py compute_pairwise_accuracy"
    - "src/judge.py assemble_turn_metrics"
    - "src/judge.py assemble_feedback_rows"
    - "src/judge.py run_baseline"
  affects:
    - "04-04"  # conversation_loop.py will import assemble_turn_metrics, assemble_feedback_rows
    - "04-06"  # examples/run_baseline.py imports run_baseline
tech_stack:
  added: []
  patterns:
    - "PairBag: in-memory pair accumulator with deterministic sampling via random.Random(turn_index)"
    - "TYPE_CHECKING guard: avoids circular imports; all heavy imports deferred inside functions"
    - "_NullNamer pattern: inline class returns dict[str, str] to satisfy ClusterNamer Protocol"
key_files:
  created:
    - path: src/judge.py
      description: "PairBag, compute_pairwise_accuracy, assemble_turn_metrics, assemble_feedback_rows, run_baseline — ~464 lines"
  modified: []
decisions:
  - "D-17: Ground truth = accumulated oracle feedback; no external labels used"
  - "D-18: Pair extraction from MoveItemFeedback/SplitFeedback/MergeFeedback only; Global/Instructional yield no pairs"
  - "D-19: Sample min(50, bag_size) per turn; seed=turn_index for deterministic replay"
  - "D-20: Contradiction → drop affected-item pairs, add fresh ones (overwrite semantics)"
  - "D-22/D-23: Baseline = initial clustering + one oracle turn; strategy_id='no_dialogue'"
  - "D-26: src/judge.py is SQL-free; builds Pydantic models, hands to src/db/ for persistence"
  - "V-4-01: _NullNamer.name_cluster() returns dict[str, str] with 'name'+'description' keys (not plain str)"
  - "V-4-07: AnthropicClusterNamer NOT imported in run_baseline — dead import removed"
metrics:
  duration: "~20 minutes"
  completed: "2026-05-16"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 0
---

# Phase 4 Plan 03: src/judge.py — PairBag, compute_pairwise_accuracy, assemble_turn_metrics, run_baseline Summary

**One-liner:** In-memory PairBag accumulates oracle-derived pairwise constraints with deterministic sampling; pure-function metric assembly produces TurnCreate/OracleFeedbackCreate Pydantic models; run_baseline runs the no-dialogue experimental arm without SQL in judge.py.

## Tasks Completed

| Task | Name | Status | Key output |
|------|------|--------|-----------|
| 1 | PairBag + compute_pairwise_accuracy | DONE | class PairBag, _extract_pairs, compute_pairwise_accuracy |
| 2 | assemble_turn_metrics + run_baseline | DONE | assemble_turn_metrics, assemble_feedback_rows, run_baseline |

## Implementation Notes

### PairBag (D-17 through D-21)

`PairBag` accumulates `(item_x, item_y, expected_same)` tuples from three structural FeedbackDelta types:
- `MoveItemFeedback(x, C)` → same-cluster pairs `(x, y, True)` for all `y` currently in cluster C
- `SplitFeedback([a, b, ...])` → different-cluster pair `(a, b, False)` from first two seed items
- `MergeFeedback(A, B)` → same-cluster pairs `(x, y, True)` for all `x in A`, `y in B` (cross-product)

`GlobalFeedback` and `InstructionalFeedback` yield no pairs (too semantic per D-18).

Contradiction handling (D-20): when `is_contradiction=True`, all pairs involving affected item IDs are dropped before the new delta's pairs are added. This implements the "latest intent wins" semantics from CLUS-04.

Sampling (D-19): `random.Random(turn_index).sample()` — per-turn deterministic RNG that does not mutate global state, ensuring identical accuracy on replay.

### compute_pairwise_accuracy

Returns 0.0 on empty bag (no oracle feedback yet — not an error). For non-empty bags, samples `min(50, bag_size)` pairs and compares against `state.assignments`: a pair `(x, y, True)` matches if `assignments[x] == assignments[y]`.

### assemble_turn_metrics

Builds `TurnCreate` Pydantic model by reading `state.turn_index`, `reply.turn_cognitive_load`, `reply.contradiction_detected`, and the caller-supplied `stop_reason`. Stores `pairwise_accuracy` and `pairwise_sample_size` in the `details` JSON column. No SQL.

### assemble_feedback_rows

Produces one `OracleFeedbackCreate` per `FeedbackDelta` via `dataclasses.asdict()` for the `parsed_delta` JSON column. Empty list for empty `deltas`. No SQL.

### run_baseline (D-22, D-23, D-25, D-26)

Full pipeline in a single function:
1. Create experiment row with `strategy_id="no_dialogue"` (D-23)
2. Encode texts via `EmbeddingStore.compute_and_save()` to a temp file (cleaned up immediately after)
3. Cluster via `HDBSCANBackend` + `build_initial_clustering_state()` using `_NullNamer`
4. Call oracle once via `MockOracle` (default) or caller-provided `OracleProtocol`
5. Parse feedback, compute pair accuracy, assemble and persist turn + feedback rows
6. Seal experiment with summary metrics in `details` JSON

`_NullNamer.name_cluster()` returns `{"name": f"Cluster {cluster_id}", "description": ""}` — the dict form is mandatory because `build_initial_clustering_state()` accesses `naming_result["name"]` and `naming_result["description"]` (V-4-01 fix).

`AnthropicClusterNamer` is NOT imported anywhere in `run_baseline` (V-4-07 fix, dead import).

## Deviations from Plan

None — plan executed exactly as written.

All field names verified against actual source before writing:
- `MoveItemFeedback.item_id`, `MoveItemFeedback.target_cluster_id` — correct
- `SplitFeedback.cluster_id`, `SplitFeedback.seed_item_ids` — correct
- `MergeFeedback.cluster_a_id`, `MergeFeedback.cluster_b_id` — correct
- `ClusteringState.clusters`, `ClusteringState.assignments`, `ClusteringState.turn_index` — correct
- `EmbeddingStore.compute_and_save(texts, save_path)` — correct signature
- `_format_message` exists in `src/conversation_loop.py` at line 47 — confirmed
- `RandomStrategy` in `src/strategy.py` at line 71 — confirmed
- `f_uncertainty` in `src/uncertainty.py` at line 31 — confirmed
- `init_schema` in `src/db/connection.py` at line 36 — confirmed

## Static Verification (Bash unavailable)

Manual grep checks performed:

| Check | Result |
|-------|--------|
| `class PairBag` present | PASS (line 50) |
| `def compute_pairwise_accuracy` present | PASS (line 185) |
| `def _extract_pairs` present | PASS (line 131) |
| `def assemble_turn_metrics` present | PASS (line 214) |
| `def assemble_feedback_rows` present | PASS (line 259) |
| `def run_baseline` present | PASS (line 297) |
| `strategy_id="no_dialogue"` present | PASS (line 356) |
| No `sqlite3.execute` or `conn.execute` | PASS (0 matches) |
| `AnthropicClusterNamer` not imported | PASS (comments only) |
| `_NullNamer.name_cluster()` returns dict | PASS (line 393) |

## Known Stubs

None. All functions are fully implemented with correct behavior.

## Threat Flags

No new threat surface beyond what was planned. All threats from the plan's threat register are mitigated:
- T-04-03-02: `random.Random(turn_index)` deterministic sampling implemented
- T-04-03-03: `sample_size=50` cap in `compute_pairwise_accuracy` implemented

## Self-Check

Files created: `src/judge.py` — confirmed exists.

Note: Bash permissions were unavailable during this execution, so runtime Python verification commands could not be run. The implementation was verified via static analysis of the file contents, dependency inspection, and grep pattern matching. The user should run the plan's verification commands manually:

```bash
python -c "from src.judge import PairBag, compute_pairwise_accuracy; b = PairBag(); assert compute_pairwise_accuracy(None, b, 0) == 0.0; print('empty bag returns 0.0 ok')"
python -c "from src.judge import PairBag, compute_pairwise_accuracy, assemble_turn_metrics, assemble_feedback_rows, run_baseline; print('all exports ok')"
python -c "from src.judge import run_baseline; import inspect; src = inspect.getsource(run_baseline); assert 'AnthropicClusterNamer' not in src; print('no dead import ok')"
```
