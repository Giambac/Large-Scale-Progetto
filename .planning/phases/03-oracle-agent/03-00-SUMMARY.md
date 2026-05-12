---
phase: 03-oracle-agent
plan: "00"
subsystem: test-scaffold
tags: [tdd, wave-0, oracle-agent, test-infrastructure]
dependency_graph:
  requires: []
  provides:
    - tests/phase3/__init__.py
    - tests/phase3/test_oracle_agent.py
    - tests/phase3/test_cognitive_load.py
    - tests/phase3/test_oracle_loop_integration.py
    - tests/conftest.py (oracle_agent_factory fixture)
  affects:
    - Phase 3 Waves 1-3 implementation tasks (each has a pre-written test)
tech_stack:
  added: []
  patterns:
    - MagicMock LLM client builder (_make_oracle_client helper)
    - pytest factory fixture (oracle_agent_factory in conftest.py)
    - lazy imports inside test functions (ImportError expected until Wave 1)
    - direct internal method testing (_check_contradiction, _build_system_prompt)
key_files:
  created:
    - tests/phase3/__init__.py
    - tests/phase3/test_oracle_agent.py
    - tests/phase3/test_cognitive_load.py
    - tests/phase3/test_oracle_loop_integration.py
  modified:
    - tests/conftest.py
decisions:
  - "test_oracle_init_logged tests OracleAgent directly with tmp_path events_path — no pytest.skip guard"
  - "test_drift_event_logged stays RED intentionally until Wave 3 (Plan 04) wires drift logging"
  - "test_instructional_feedback_accumulates creates OracleAgent manually (not via fixture) to access mock_client for assertion"
  - "run_conversation called with events_path= in integration tests — run_conversation does not yet accept this param, will be RED until Wave 3"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-12"
  tasks_completed: 3
  files_created: 4
  files_modified: 1
---

# Phase 3 Plan 00: Wave 0 Test Scaffold Summary

Wave 0 TDD scaffold — 21 pre-written test functions covering OracleAgent, f_cognitive_load, and integration loop. All tests import from src/ lazily; ImportError is expected and acceptable until Wave 1 src files are created.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create tests/phase3/__init__.py and oracle_agent_factory fixture | 65cec58 | tests/phase3/__init__.py, tests/conftest.py |
| 2 | Write tests/phase3/test_oracle_agent.py (13 unit tests) | 1d19089 | tests/phase3/test_oracle_agent.py |
| 3 | Write test_cognitive_load.py and test_oracle_loop_integration.py | cc107c0 | tests/phase3/test_cognitive_load.py, tests/phase3/test_oracle_loop_integration.py |

## Test Coverage

### tests/phase3/test_oracle_agent.py (13 tests)
- `test_oracle_agent_satisfies_protocol` — isinstance(agent, OracleProtocol)
- `test_reply_returns_oracle_reply` — reply() returns OracleReply with non-empty raw_text
- `test_oracle_spec_fields` — spec.preferred_k, semantic_axes, persona_description
- `test_oracle_init_logged` — oracle_init event written to events_path on __init__
- `test_noise_params_in_prompt` — consistency_rate percentage in _build_system_prompt
- `test_overload_prompt_injected` — "OVERLOAD" in prompt when load=0.8
- `test_no_overload_below_threshold` — "OVERLOAD" absent when load=0.5
- `test_contradiction_merge_after_split` — MergeFeedback(A,B) after SplitFeedback(A) → contradiction
- `test_contradiction_move_item` — MoveItem(item=0, target=2) after MoveItem(item=0, target=1) → contradiction
- `test_no_contradiction_empty_window` — empty window → (False, None)
- `test_no_false_positive_contradiction` — SplitFeedback(A) then SplitFeedback(A) → no contradiction
- `test_global_instructions_in_prompt` — global_instructions content in _build_system_prompt
- `test_oracle_agent_crashes_on_invalid_noise_params` — consistency_rate=1.5 → AssertionError

### tests/phase3/test_cognitive_load.py (5 tests)
- `test_load_in_range` — f_cognitive_load returns float in [0.0, 1.0]
- `test_load_above_threshold` — 20 clusters + MAX_MSG_LEN message → load > COG_LOAD_THRESHOLD
- `test_load_zero_for_minimal_state` — 1 cluster, 1 item, short message → load < COG_LOAD_THRESHOLD
- `test_f_cognitive_load_crashes_on_empty_state` — empty ClusteringState → AssertionError
- `test_f_cognitive_load_pure` — same inputs → same output (idempotent)

### tests/phase3/test_oracle_loop_integration.py (3 tests)
- `test_oracle_agent_loop_5_turns` — OracleAgent in run_conversation() for 5 turns without crash
- `test_drift_event_logged` — contradiction triggers drift_event in events.jsonl (RED until Wave 3)
- `test_instructional_feedback_accumulates` — accumulated instruction appears in oracle LLM call args (RED until Wave 3)

## Deviations from Plan

None — plan executed exactly as written. All test files match the function names specified in VALIDATION.md.

## Intentionally RED Tests (Wave 0)

The following tests will remain RED (ImportError or assertion failure) until the specified wave:

| Test | Will Green At |
|------|---------------|
| All tests in test_oracle_agent.py | Wave 1 Plan 01 (src/oracle_agent.py) |
| All tests in test_cognitive_load.py | Wave 1 Plan 02 (src/cognitive_load.py) |
| test_oracle_agent_loop_5_turns | Wave 1 Plan 01 |
| test_drift_event_logged | Wave 3 Plan 04 (drift logging in run_conversation) |
| test_instructional_feedback_accumulates | Wave 3 Plan 04 (global_instructions wired to oracle) |

## Self-Check: PASSED

All 5 files exist and parse without SyntaxError. pytest collects 21 test functions (13 + 5 + 3) across the phase3 directory. oracle_agent_factory appears 2 times in conftest.py (fixture def + inner _factory reference).
