---
phase: 03-oracle-agent
plan: "01"
subsystem: oracle-agent
tags: [oracle, llm, noise-simulation, cognitive-load, drift-detection, tdd]
dependency_graph:
  requires:
    - 02-clustering-agent-core
  provides:
    - OracleAgent (src/oracle_agent.py)
    - OracleSpec, NoiseParams dataclasses
    - f_cognitive_load pure function (src/cognitive_load.py)
    - Extended OracleReply with contradiction_detected, contradicted_turn fields
  affects:
    - src/oracle_protocol.py (OracleReply extended)
    - tests/conftest.py (oracle_agent_factory fixture added)
tech_stack:
  added:
    - src/oracle_agent.py (new: OracleAgent LLM-backed oracle, ORC-01/ORC-02/ORC-04)
    - src/cognitive_load.py (new: f_cognitive_load pure function, ORC-03)
  patterns:
    - Structural subtyping via runtime_checkable Protocol (OracleAgent satisfies OracleProtocol)
    - Provider-aware LLM call dispatch (Anthropic system= kwarg vs. OpenAI/Google prepend)
    - Rolling deque(maxlen=10) for structural drift detection window
    - oracle_init JSONL event written at construction time when events_path provided
    - TDD: RED (tests created before src) then GREEN (implementation makes tests pass)
key_files:
  created:
    - src/oracle_agent.py
    - src/cognitive_load.py
    - tests/phase3/__init__.py
    - tests/phase3/test_oracle_agent.py
  modified:
    - src/oracle_protocol.py (two new OracleReply fields with defaults)
    - tests/conftest.py (oracle_agent_factory fixture appended)
decisions:
  - "oracle_init event written in OracleAgent.__init__ when events_path provided — enables unit test without conversation loop"
  - "Wave 1 reply() always returns contradiction_detected=False — Wave 2 Plan 03 adds the deque check"
  - "cognitive_load.py created alongside oracle_agent.py (both Wave 1) since oracle_agent imports COG_LOAD_THRESHOLD"
  - "Provider-aware LLM call: isinstance check for anthropic.Anthropic; OpenAI/Google adapters get system prompt prepended to user message"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-12"
  tasks_completed: 2
  files_created: 4
  files_modified: 2
---

# Phase 3 Plan 01: Oracle Agent Core — Summary

**One-liner:** LLM-backed OracleAgent with OracleSpec preference spec, NoiseParams prompt-injected behavioral rules, per-turn cognitive load gate, and oracle_init JSONL event written at construction time via events_path.

---

## What Was Built

### Task 1: Extend OracleReply + test scaffold (RED phase)

Extended `src/oracle_protocol.py` with two new `OracleReply` fields using defaults for backward compatibility:
- `contradiction_detected: bool = False` (Phase 3 drift detection result)
- `contradicted_turn: int | None = None` (which prior turn was contradicted)

Created the Phase 3 test package:
- `tests/phase3/__init__.py` — empty package init
- `tests/phase3/test_oracle_agent.py` — 13 unit tests, all with MagicMock LLM clients
- Added `oracle_agent_factory` fixture to `tests/conftest.py`

**Commit:** e6b2b23

### Task 2: Implement OracleAgent + cognitive_load.py (GREEN phase)

Created `src/cognitive_load.py` with:
- Named constants: `MAX_K=20`, `MAX_MSG_LEN=500`, `TOP_K_ITEMS_PER_CLUSTER=5`, `COG_LOAD_THRESHOLD=0.7`
- `f_cognitive_load(state, message) -> float` pure function using the D-07 formula

Created `src/oracle_agent.py` with:
- `OracleSpec` dataclass: `preferred_k`, `semantic_axes`, `persona_description`
- `NoiseParams` dataclass: `consistency_rate`, `drift_probability`, `sycophancy_resistance` (all [0,1] with assert guards)
- `OracleAgent` class:
  - `__init__` writes `oracle_init` JSONL event immediately when `events_path` is provided
  - `_build_system_prompt()` assembles 5-section prompt (persona + [SATISFIED] token, behavioral rules, global instructions, state summary, OVERLOAD gate)
  - `_check_contradiction()` structural deque comparison (callable in Wave 1; Wave 2 wires it into reply())
  - `reply()` with provider-aware LLM dispatch, satisfaction token detection, returns `OracleReply`
- Module-level `_contradicts()` helper with D-09 structural contradiction rules

**Commit:** 85f3041

---

## Test Results

```
pytest tests/phase3/test_oracle_agent.py -q -m "not llm"
13 passed, 1 warning in 0.73s
```

Tests that pass (Wave 1 targets):
- test_oracle_agent_satisfies_protocol
- test_reply_returns_oracle_reply
- test_oracle_spec_fields
- test_oracle_init_logged
- test_noise_params_in_prompt
- test_overload_prompt_injected
- test_no_overload_below_threshold
- test_oracle_agent_crashes_on_invalid_noise_params
- test_global_instructions_in_prompt
- test_contradiction_merge_after_split
- test_contradiction_move_item
- test_no_contradiction_empty_window
- test_no_false_positive_contradiction

Phase 2 regression check: 13 failed (pre-existing), 72 passed — zero new failures.

Tests intentionally RED (Wave 2+):
- test_contradiction_merge_after_split — now GREEN (implementation complete)
- test_contradiction_move_item — now GREEN (implementation complete)
- test_no_contradiction_empty_window — now GREEN
- test_no_false_positive_contradiction — now GREEN

---

## Deviations from Plan

### Auto-created Issue

**1. [Rule 3 - Blocking] cognitive_load.py created in Plan 01**

- **Found during:** Task 2 — oracle_agent.py imports `f_cognitive_load` and `COG_LOAD_THRESHOLD` from `src.cognitive_load`
- **Issue:** `src/cognitive_load.py` did not exist (it's Wave 1 Plan 02's scope), causing `ModuleNotFoundError` when importing oracle_agent.py
- **Fix:** Created `src/cognitive_load.py` with the D-07 formula, named constants, and `f_cognitive_load()` pure function as part of Plan 01 Task 2. Plan 02 can either skip creation or update if needed.
- **Files modified:** `src/cognitive_load.py` (new)
- **Commit:** 85f3041

---

## Known Stubs

- `reply()` always returns `contradiction_detected=False` and `contradicted_turn=None` (Wave 1 design). `_check_contradiction()` is implemented and tested but not called from `reply()`. Wave 2 (Plan 03) wires these together.
- `_delta_window` is never populated by `reply()` in Wave 1; Wave 2 adds the window update after `parse_feedback` produces deltas.

These stubs are intentional and documented per plan design.

---

## Threat Flags

No new threat surface beyond what was declared in the plan's `<threat_model>`. The `oracle_init` event records only noise params (no secrets, no PII). The `persona_description` field is researcher-controlled at construction time with no user input path in v1.

---

## Self-Check: PASSED

- FOUND: src/oracle_agent.py
- FOUND: src/cognitive_load.py
- FOUND: src/oracle_protocol.py (modified)
- FOUND: tests/phase3/__init__.py
- FOUND: tests/phase3/test_oracle_agent.py
- FOUND: tests/conftest.py (modified)
- FOUND: .planning/phases/03-oracle-agent/03-01-SUMMARY.md
- Commit e6b2b23 verified in git log
- Commit 85f3041 verified in git log
