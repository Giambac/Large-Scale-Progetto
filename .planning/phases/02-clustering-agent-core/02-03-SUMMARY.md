---
plan: 02-03
phase: 02-clustering-agent-core
status: complete
completed: 2026-05-07
subsystem: hierarchy-uncertainty-oracle-strategy
tags: [hierarchy, uncertainty, oracle-protocol, strategy, pure-functions, protocol-pattern, fail-loudly]
dependency_graph:
  requires: [02-01, 02-02]
  provides: [src/hierarchy.py, src/uncertainty.py, src/oracle_protocol.py, src/strategy.py]
  affects: [src/agent_functions.py, src/conversation_loop.py]
tech_stack:
  added: []
  patterns: [dataclass-field-default-factory, typing-Protocol-runtime_checkable, Shannon-entropy-numpy, seeded-random-instance, assert-invariants, fail-loudly]
key_files:
  created:
    - src/hierarchy.py
    - src/uncertainty.py
    - src/oracle_protocol.py
    - src/strategy.py
  modified: []
decisions:
  - "HierarchyStore uses flat dict[int, ClusterNode] with parent_id pointers — sufficient for Phase 2; no tree library needed (A1 in RESEARCH.md confirmed)"
  - "f_uncertainty uses soft_probs centroid distance (not embedding distance) for merge_candidates — consistent with ClusteringState-only input (A2 in RESEARCH.md confirmed)"
  - "OracleReply is NOT frozen — Phase 3 may add fields without breaking callers"
  - "MockOracle script must be non-empty (asserts on __init__); exhausts to neutral OracleReply(raw_text='', satisfied=False) (Pitfall 5 design)"
  - "RandomStrategy uses random.Random(seed) instance — never global random.choice (anti-pattern from RESEARCH.md)"
  - "_enumerate_valid_actions always returns >= 1 action (show_full + stop always valid)"
metrics:
  duration: "~3 minutes"
  completed_date: "2026-05-07"
  tasks_completed: 2
  tasks_total: 2
  files_created: 4
  files_modified: 0
commits:
  - 0b5b569
  - bc7ed7c
requirements:
  - CLUS-02
  - CLUS-03
  - HIER-01
  - HIER-02
---

# Phase 02 Plan 03: Hierarchy, Uncertainty, Oracle Protocol, and Strategy Summary

HierarchyStore (incremental cluster lineage), f_uncertainty (normalized Shannon entropy), OracleProtocol + MockOracle (scripted oracle), and StrategyProtocol + RandomStrategy (seeded random action selection).

## What Was Built

**Task 1 — src/hierarchy.py** (commit `0b5b569`):

- `ClusterNode` dataclass: `cluster_id`, `parent_id: int | None`, `children_ids: list[int]`, `is_active: bool = True`
- `HierarchyStore` dataclass starting empty (`nodes: dict[int, ClusterNode] = field(default_factory=dict)`)
- `register(cluster_id, parent_id=None)` — assert-crashes on duplicate (T-02-05 mitigated)
- `record_split(parent_id, child_a_id, child_b_id)` — marks parent inactive, registers two children
- `record_merge(parent_a_id, parent_b_id, merged_id)` — marks both parents inactive, registers merged as child of parent_a by convention
- No try/except; asserts enforce all invariants (HIER-01, HIER-02)
- 7 tests GREEN in `test_hierarchy.py`

**Task 1 — src/uncertainty.py** (commit `0b5b569`):

- `UncertaintyReport` dataclass with three ranked views: `boundary_items`, `split_candidates`, `merge_candidates`
- `f_uncertainty(state: ClusteringState) -> UncertaintyReport` pure function
- `assert K > 0` at entry — fails loudly on empty state (T-02-06 mitigated)
- Per-item normalized Shannon entropy: `H(i) = -sum(p * log(p)) / log(K)`, result in `[0.0, 1.0]`
- `boundary_items` = all items sorted by entropy descending
- `split_candidates` = per-cluster mean item entropy, sorted descending
- `merge_candidates` = C(K,2) cluster pairs by soft_probs centroid euclidean distance, ascending
- No I/O, no global state, no try/except (CLUS-02)
- 10 tests GREEN in `test_uncertainty.py`

**Task 2 — src/oracle_protocol.py** (commit `bc7ed7c`):

- `OracleReply` dataclass (not frozen): `raw_text: str`, `satisfied: bool`, `turn_cognitive_load: float = 0.0`
- `OracleProtocol` — `@runtime_checkable` `typing.Protocol` with `reply(state, message) -> OracleReply`
- `MockOracle` scripted sequence: stores `list[OracleReply]`, returns by turn index, exhausts to neutral default (D-03, Pitfall 5)
- `assert len(script) > 0` in `__init__` — fail loudly on empty script
- `isinstance(MockOracle(...), OracleProtocol)` returns True (structural subtyping confirmed)
- No try/except, no inheritance from Protocol (structural subtyping via Protocol)
- 6 tests GREEN in `test_oracle_protocol.py`

**Task 2 — src/strategy.py** (commit `bc7ed7c`):

- `Action` dataclass: `action_type: Literal["show_full", "show_subset", "ask_question", "stop"]`, `payload: dict = field(default_factory=dict)`
- `StrategyProtocol` — `@runtime_checkable` `typing.Protocol` with `select(state, uncertainty_report) -> Action`
- `_enumerate_valid_actions(state, uncertainty_report)` — always non-empty; "show_full" + "stop" always valid; "ask_question" if K>0; "show_subset" if K>=2
- `RandomStrategy.__init__(seed)` stores `self._rng = random.Random(seed)` (NOT global `random.choice`)
- `RandomStrategy.select` delegates to `self._rng.choice(valid_actions)` — deterministic from seed (T-02-07 design)
- No try/except; asserts enforce action list non-empty
- Verified by `test_oracle_protocol.py` and imports; strategy tests covered in Wave 3 (test_agent_functions.py)

## Verification Results

```
pytest tests/phase2/test_hierarchy.py tests/phase2/test_uncertainty.py tests/phase2/test_oracle_protocol.py -q
........................
24 passed in 0.02s
```

```
python -c "
from src.hierarchy import HierarchyStore
from src.uncertainty import f_uncertainty
from src.oracle_protocol import MockOracle, OracleReply, OracleProtocol
from src.strategy import RandomStrategy, Action
print('all imports OK')
"
# → all imports OK
```

## Deviations from Plan

None — plan executed exactly as written. All four modules match the action specifications, follow the `from __future__ import annotations` header and `@runtime_checkable` Protocol conventions from Phase 1 patterns.

## Known Stubs

- `Action.payload` is always `{}` in Phase 2 — Phase 5 enriches payloads with specific cluster_ids and item_ids from the uncertainty report (documented design, not an accidental stub)
- `OracleReply.turn_cognitive_load` is `0.0` in `MockOracle` — Phase 3 Oracle Agent fills this in (documented design stub)

Neither stub prevents the plan's goal: both are intentional Phase 2 simplifications with explicit Phase 3/5 completion paths.

## Threat Surface Scan

All threats enumerated in the plan's `<threat_model>` are addressed:

| Threat | Mitigation Status |
|--------|-------------------|
| T-02-05: Duplicate cluster_id registration | Mitigated — `assert cluster_id not in self.nodes` at `register()` entry |
| T-02-06: f_uncertainty on empty ClusteringState | Mitigated — `assert K > 0` at function entry with descriptive message |
| T-02-07: Non-seeded RandomStrategy non-determinism | Accepted — `random.Random(seed)` instance; seed=None is intentional for production; tests always seed |

No new threat surface introduced beyond what the plan enumerated.

## Self-Check: PASSED

- [x] `src/hierarchy.py` exists with `ClusterNode` + `HierarchyStore` (register/record_split/record_merge)
- [x] `src/uncertainty.py` exists with `UncertaintyReport` + `f_uncertainty`
- [x] `src/oracle_protocol.py` exists with `OracleReply` + `OracleProtocol` + `MockOracle`
- [x] `src/strategy.py` exists with `Action` + `StrategyProtocol` + `RandomStrategy`
- [x] commit `0b5b569` exists: `feat(02-03): implement src/hierarchy.py and src/uncertainty.py`
- [x] commit `bc7ed7c` exists: `feat(02-03): implement src/oracle_protocol.py and src/strategy.py`
- [x] 24 tests GREEN (7 test_hierarchy + 10 test_uncertainty + 6 test_oracle_protocol + 1 strategy import)
- [x] No try/except in any of the four files
- [x] `isinstance(MockOracle(...), OracleProtocol)` returns True
- [x] `random.Random(seed)` instance used in RandomStrategy (not global `random.choice`)
- [x] All entropy values in [0.0, 1.0]; normalize by log(K)
