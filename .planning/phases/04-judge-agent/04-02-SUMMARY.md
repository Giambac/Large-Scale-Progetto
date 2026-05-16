---
phase: 4
plan: "04-02"
subsystem: stopping-criteria, logging, oracle-protocol
tags: [stopping, deviation, oracle, yaml-config, WR-04, D-13, D-14, D-15, D-27, D-29]
dependency_graph:
  requires: []
  provides:
    - src/logging_setup.py (deviation() + UnexpectedDeviation + STRICT_MODE)
    - src/stopping.py (filled placeholders + diminishing-returns branch + compute_magnitude)
    - src/oracle_protocol.py (updated reply() signatures — WR-04)
    - experiments/configs/default.yaml (default YAML config)
  affects:
    - src/conversation_loop.py (isinstance branch can now be removed — Plan 04)
    - src/judge.py (can use deviation() for unexpected-but-possible branches)
tech_stack:
  added: [pyyaml (experiments/configs/default.yaml)]
  patterns: [STRICT_MODE env-var gate, dataclass frozen config, weighted magnitude]
key_files:
  created:
    - src/logging_setup.py
    - experiments/configs/default.yaml
  modified:
    - src/stopping.py
    - src/oracle_protocol.py
decisions:
  - "D-13: FeedbackMagnitudeWeights global=1.0, cluster=0.5, point=0.2, instructional=0.1"
  - "D-14: magnitude_threshold_epsilon = 0.05"
  - "D-15: magnitude_fallback_turns = 3"
  - "D-27/WR-04: OracleProtocol.reply() + MockOracle.reply() gain global_instructions and cognitive_load keyword params"
  - "D-29: deviation() raises UnexpectedDeviation when STRICT_MODE=1, logs WARNING otherwise"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-16"
  tasks_completed: 3
  files_changed: 4
---

# Phase 4 Plan 02: Stopping Criteria Filled + deviation() + Oracle Protocol Fix (WR-04) Summary

**One-liner:** Filled Phase 1 NaN/(-1) stubs in stopping.py (D-13/14/15), implemented diminishing-returns branch, added deviation() helper (D-29), and extended OracleProtocol reply() signatures to remove the isinstance branch in conversation_loop.py (WR-04/D-27).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create src/logging_setup.py with deviation() + UnexpectedDeviation (D-29) | 526adba | src/logging_setup.py |
| 2 | Fill src/stopping.py placeholders + implement diminishing-returns branch (D-13/14/15) | 526adba | src/stopping.py |
| 3 | Update OracleProtocol + MockOracle reply() signatures (D-27/WR-04) + create default.yaml (D-16) | 526adba | src/oracle_protocol.py, experiments/configs/default.yaml |

## What Was Built

### Task 1 — src/logging_setup.py (new)

New module providing the `deviation()` helper for Phase 4 code (D-29):
- `UnexpectedDeviation(RuntimeError)` — raised when `STRICT_MODE=1`
- `deviation(msg, **kwargs)` — logs WARNING by default, raises when `STRICT_MODE=1`
- Phase 1-3 `assert` statements untouched (deviation is complementary, not a replacement)

### Task 2 — src/stopping.py (updated)

Filled all Phase 1 placeholder stubs:
- `FeedbackMagnitudeWeights`: `global_feedback=1.0`, `cluster_level=0.5`, `point_level=0.2`, `instructional=0.1` (D-13)
- `StoppingCriteria`: `magnitude_threshold_epsilon=0.05` (D-14), `magnitude_fallback_turns=3` (D-15)
- `check_stopping()` diminishing-returns branch implemented: checks last N magnitudes against epsilon (D-10)
- `compute_magnitude()` function added: applies `FeedbackMagnitudeWeights` to a list of `FeedbackDelta` objects; to be used by `conversation_loop.py` in Plan 04

### Task 3 — src/oracle_protocol.py (updated) + experiments/configs/default.yaml (new)

WR-04 fix: Extended `OracleProtocol.reply()` and `MockOracle.reply()` to accept `global_instructions: list[str] | None = None` and `cognitive_load: float | None = None` keyword params. `OracleAgent.reply()` already had these params (Phase 3). Now the `isinstance(oracle, OracleAgent)` branch in `conversation_loop.py` is no longer needed (Plan 04 removes it).

Created `experiments/configs/default.yaml` with matching values for `stopping` (epsilon, n_fallback, weights) and `judge` (pairwise_sample_size=50).

## Verification Results

All plan verification commands passed:

```
stopping ok
all 3 stop conditions ok
oracle protocol ok
deviation warning ok
strict mode ok: test strict
yaml ok
```

## Deviations from Plan

None — plan executed exactly as written. OracleAgent.reply() already had the `global_instructions` and `cognitive_load` params from Phase 3, confirming no changes needed there.

## Known Stubs

None. All Phase 1 NaN/(-1) stubs have been replaced with real float/int values.

## Threat Flags

No new security-relevant surface introduced beyond what was specified in the plan's threat model. `experiments/configs/default.yaml` is a local research tool config; no user-facing web input flows through it. `STRICT_MODE` env var controls raise vs. log with no PII or security implication.

## Self-Check: PASSED

- src/logging_setup.py exists and importable
- src/stopping.py: no `float("nan")` remaining, no `= -1` for magnitude_fallback_turns
- src/oracle_protocol.py: both reply() signatures have global_instructions and cognitive_load
- experiments/configs/default.yaml: loadable with yaml.safe_load(), epsilon=0.05, pairwise_sample_size=50
- Commit 526adba verified in git log
