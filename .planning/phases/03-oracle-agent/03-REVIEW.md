---
phase: 03-oracle-agent
reviewed: 2026-05-12T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - src/agent_functions.py
  - src/cognitive_load.py
  - src/conversation_loop.py
  - src/oracle_agent.py
  - src/oracle_protocol.py
  - tests/conftest.py
  - tests/phase3/__init__.py
  - tests/phase3/test_cognitive_load.py
  - tests/phase3/test_oracle_agent.py
  - tests/phase3/test_oracle_loop_integration.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-05-12T00:00:00Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Phase 3 added `OracleAgent` (LLM-backed oracle with noise params), `f_cognitive_load` pure function, drift detection via delta window, and wiring of all three into `run_conversation()`. The structural architecture is sound and the fail-loudly philosophy is largely respected. Two correctness bugs stand out: the `cognitive_load` value computed in the loop is silently discarded (never passed to the oracle), and `OracleAgent.__init__` uses the deprecated `datetime.utcnow()` which returns a naive UTC datetime, while every other timestamp in the codebase uses an aware UTC datetime. Four additional warnings cover logic gaps that could produce silently wrong results under normal operation.

---

## Critical Issues

### CR-01: `cognitive_load` computed in loop is never passed to the oracle — ORC-03 wiring is broken

**File:** `src/conversation_loop.py:180-184`

**Issue:** `run_conversation()` computes `cognitive_load = f_cognitive_load(state, message)` on line 180, then immediately discards the value. When the oracle is an `OracleAgent`, `oracle.reply(state, message, global_instructions=global_instructions)` is called on line 184 with no `cognitive_load` argument. Inside `OracleAgent.reply()`, `f_cognitive_load(state, message)` is called a **second time** independently (line 287 of `oracle_agent.py`). The loop's cognitive-load value is never consumed.

This means:
1. The loop variable `cognitive_load` (line 180) is dead code.
2. The socket emit on line 239 emits `reply.turn_cognitive_load`, which correctly reflects the oracle's internal recompute, but the loop's own compute is wasted.
3. If the intent (per ORC-03 / D-06) is that the loop computes cognitive load once and passes it into the oracle, that contract is broken. If the oracle is supposed to compute it independently, the loop's computation is purely redundant dead code that could mask a future bug when callers diverge.

The requirement states "f_cognitive_load computed before oracle.reply() each turn (ORC-03, D-06)" — the word "before" implies the result should be forwarded, not recomputed. The `OracleAgent.reply()` signature does not accept a `cognitive_load` parameter, so there is no path to pass the pre-computed value in.

**Fix:** Either (a) add a `cognitive_load: float | None = None` parameter to `OracleAgent.reply()` and use it when provided, skipping the internal recompute — this is the architecturally clean path that makes the loop the single owner of the value per D-06:

```python
# oracle_agent.py — reply() signature change
def reply(
    self,
    state: "ClusteringState",
    message: str,
    global_instructions: list[str] | None = None,
    cognitive_load: float | None = None,
) -> OracleReply:
    if cognitive_load is None:
        cognitive_load = f_cognitive_load(state, message)
    ...
```

```python
# conversation_loop.py — pass the pre-computed value
reply = oracle.reply(state, message, global_instructions=global_instructions,
                     cognitive_load=cognitive_load)
```

Or (b) remove the `cognitive_load = f_cognitive_load(state, message)` line from the loop entirely and document that `OracleAgent.reply()` owns the computation. Either is acceptable, but the current state has two computations and zero coordination between them.

---

### CR-02: `datetime.utcnow()` in `OracleAgent.__init__` produces a naive datetime — inconsistent with codebase and deprecated

**File:** `src/oracle_agent.py:131`

**Issue:** The `oracle_init` event record uses `datetime.utcnow().isoformat()` to produce a timestamp string. `datetime.utcnow()` is deprecated since Python 3.12 and returns a **naive** datetime object (no timezone info). Every other timestamp in the codebase uses timezone-aware UTC datetimes via `datetime.datetime.now(datetime.timezone.utc).isoformat()` (e.g., `agent_functions.py:526`). This inconsistency means:
- The `oracle_init` record timestamp lacks a `+00:00` / `Z` suffix, making it ambiguous when read back.
- Any downstream code that parses timestamps and expects ISO 8601 with timezone will silently misinterpret this record as local time.
- Python 3.12+ emits a `DeprecationWarning` on every construction.

**Fix:**
```python
# oracle_agent.py line 131
from datetime import datetime, timezone
"timestamp": datetime.now(timezone.utc).isoformat(),
```

Note: the `datetime` import at line 22 already imports `datetime` from the `datetime` module; the fix requires adding `timezone` to that import:
```python
from datetime import datetime, timezone
```

---

## Warnings

### WR-01: `OracleAgent` dual-writes `oracle_init` — once in `__init__` (when `events_path` provided) and once again via `run_conversation()` — resulting in duplicate records

**File:** `src/conversation_loop.py:155-165`, `src/oracle_agent.py:122-139`

**Issue:** `OracleAgent.__init__` writes an `oracle_init` event when `events_path` is provided (lines 122-139 of `oracle_agent.py`). `run_conversation()` then independently checks `isinstance(oracle, _OracleAgent)` and writes its own `oracle_init` event (lines 155-165 of `conversation_loop.py`) **without checking whether the oracle was already constructed with `events_path`**.

If a caller constructs `OracleAgent(events_path=p)` and then passes the same path to `run_conversation(events_path=p)`, the `oracle_init` record is written twice. The integration test `test_oracle_agent_loop_5_turns` uses `oracle_agent_factory` which does NOT pass `events_path` to the constructor, so the test never triggers the duplicate — the bug is invisible to the test suite.

Downstream consumers reading `events.jsonl` and counting or deduplicating `oracle_init` records will see unexpected duplicates in production usage.

**Fix:** Either (a) remove the `oracle_init` write from `run_conversation()` entirely and document that `OracleAgent.__init__` owns it when `events_path` is provided, or (b) remove the write from `__init__` and always write it from `run_conversation()`. Do not support both paths simultaneously without a deduplication guard.

---

### WR-02: `_contradicts` returns `False` for `SplitFeedback(X)` vs prior `SplitFeedback(X)` but the semantic case of re-requesting the same already-applied split is not detected

**File:** `src/oracle_agent.py:81-82`

**Issue:** The `SplitFeedback vs MergeFeedback` contradiction check at line 81 reads:

```python
if isinstance(new_delta, SplitFeedback) and isinstance(prior_delta, MergeFeedback):
    return new_delta.cluster_id in (prior_delta.cluster_a_id, prior_delta.cluster_b_id)
```

This checks whether the new split target was one of the **inputs** to a prior merge. However, after a merge is applied by `_apply_merge`, the source cluster IDs (`cluster_a_id`, `cluster_b_id`) are **retired** and no longer exist in the state (D-11: monotonic counter, IDs never reused). A subsequent `SplitFeedback(cluster_id=X)` where X was retired would crash in `_apply_split` with an assertion error (`cluster_id not found in state`) before the contradiction check even matters.

The case that does NOT crash but IS logically contradictory — splitting the **newly created merged cluster** — is not caught. If `prior_delta = MergeFeedback(a_id=0, b_id=1)` produced `new_id=2`, and then `new_delta = SplitFeedback(cluster_id=2)` arrives, `_contradicts` returns `False` because `2 not in (0, 1)`. The contradiction (merge then immediately re-split the result) goes undetected.

This is a logic gap in the contradiction detection algorithm, not a crash risk, but it means `drift_event` records are missed for a class of real contradictions.

**Fix:** The contradiction check needs access to the merge output ID. One approach: store `(turn_index, delta, output_id)` in the window for merge operations, or add a `result_id` field to `MergeFeedback`. Alternatively, document the known gap in a `# KNOWN LIMITATION` comment so downstream consumers of `drift_event` records don't assume completeness.

---

### WR-03: `_apply_split` validates only the first two `seed_item_ids` but passes only `seed_item_ids[:2]` to KMeans — seeds beyond index 1 are silently ignored without validation

**File:** `src/agent_functions.py:220-229`

**Issue:** The validation loop (lines 220-225) iterates `delta.seed_item_ids[:2]` — only the first two seeds. If a caller provides 3+ seeds, the extra seeds pass through unvalidated. The KMeans call at line 228 then slices to `[:2]`, silently ignoring any seeds beyond the first two. This is not a crash but a silent data loss: the caller believes all their seeds were used, but only two were.

The assertion message at line 224 says "is not in cluster" which is correct for what it checks, but the structure misleads readers into thinking all seeds are validated.

**Fix:** Either assert `len(delta.seed_item_ids) <= 2` before the validation loop, or validate all seeds (not just `[:2]`). Consistent with the fail-loudly philosophy:
```python
if delta.seed_item_ids:
    assert len(delta.seed_item_ids) <= 2, (
        f"_apply_split: expected at most 2 seed_item_ids, got {len(delta.seed_item_ids)}"
    )
    target_item_set = set(target.item_ids)
    for sid in delta.seed_item_ids:
        assert sid in target_item_set, ...
```

---

### WR-04: `OracleProtocol.reply()` signature does not include `global_instructions` — structural subtype check via `isinstance` will pass for `MockOracle` but the loop calls `oracle.reply(..., global_instructions=...)` only for `OracleAgent`, creating a protocol/dispatch split

**File:** `src/oracle_protocol.py:47`, `src/conversation_loop.py:183-186`

**Issue:** `OracleProtocol` declares `reply(self, state, message) -> OracleReply` with no `global_instructions` parameter (line 47 of `oracle_protocol.py`). The loop checks `isinstance(oracle, _OracleAgent)` to decide whether to pass `global_instructions` (lines 183-186), rather than relying on the protocol. This means:
1. Any future custom oracle that wants `global_instructions` must be an `OracleAgent` subclass or get special-cased in the loop — the protocol doesn't express the capability.
2. If a developer adds a non-`OracleAgent` oracle that accepts `global_instructions`, the loop will silently never pass them.
3. The `OracleProtocol` interface is now partially stale as a contract.

This is a maintainability defect that will cause hard-to-debug integration failures when adding new oracle implementations.

**Fix:** Update `OracleProtocol.reply()` to include `global_instructions: list[str] | None = None` in its signature, and update `MockOracle.reply()` to accept (and ignore) it. The loop can then call `oracle.reply(state, message, global_instructions=global_instructions)` unconditionally for all oracle types.

---

## Info

### IN-01: Dead import `cognitive_load` variable in `run_conversation` — variable assigned, never used by caller

**File:** `src/conversation_loop.py:180`

**Issue:** `cognitive_load = f_cognitive_load(state, message)` produces a value that is never read after line 180. It is not passed to `oracle.reply()`, not stored in any record, and not used in any condition. It is purely dead code as the implementation stands today (see CR-01 for the root cause).

**Fix:** Resolve CR-01. Until then, at minimum add a `# noqa: F841` comment or remove the line to silence the unused-variable linter warning and signal intent clearly.

---

### IN-02: `_contradicts` has unreachable comment after `return False`

**File:** `src/oracle_agent.py:90`

**Issue:** Line 90 contains a comment `# GlobalFeedback and InstructionalFeedback: ignored (D-09)` placed AFTER `return False` on line 89. The comment is unreachable (dead comment after a return statement). While harmless, it is misleading — it appears to document code that follows it, but nothing follows it.

**Fix:** Move the comment to before the `return False` statement:
```python
    # GlobalFeedback and InstructionalFeedback are ignored — too semantic for structural comparison (D-09)
    return False
```

---

### IN-03: `test_cognitive_load.py` — `_make_state` mutates `Cluster.item_ids` after construction

**File:** `tests/phase3/test_cognitive_load.py:26`

**Issue:** The test helper `_make_state` builds `Cluster` objects with `item_ids=[]` and then appends to `clusters[cluster_id].item_ids` in the loop (line 26). `Cluster` is not a frozen dataclass, so mutation works, but this pattern relies on the fact that `clusters[cluster_id]` and the list in `clusters` are the same object. If `Cluster` were ever made frozen or if the list were copied at construction, the helper would silently produce empty clusters.

This is a fragility in a test helper, not a production bug.

**Fix:** Build `item_ids` lists before constructing `Cluster` objects:
```python
items_per_cluster: dict[int, list[int]] = {i: [] for i in range(n_clusters)}
for item_id in range(n_items):
    items_per_cluster[item_id % n_clusters].append(item_id)
clusters = [Cluster(id=i, ..., item_ids=items_per_cluster[i]) for i in range(n_clusters)]
```

---

_Reviewed: 2026-05-12T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
