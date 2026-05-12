---
status: complete
phase: 03-oracle-agent
source:
  - 03-00-SUMMARY.md
  - 03-01-SUMMARY.md
  - 03-02-SUMMARY.md
  - 03-03-SUMMARY.md
  - 03-04-SUMMARY.md
started: "2026-05-12T15:00:00Z"
updated: "2026-05-12T15:00:00Z"
---

## Current Test

## Current Test

[testing complete]

## Tests

### 1. Phase 3 test suite passes
expected: Run `pytest tests/phase3/ -v` — all 21 tests pass (0 failures, 0 errors).
result: pass
note: 1 DeprecationWarning — datetime.utcnow() in oracle_agent.py:131 (cosmetic, not a failure)

### 2. OracleAgent import and instantiation
expected: Prints OracleSpec/NoiseParams fields and 'OK' — no error.
result: pass

### 3. Invalid noise params crash loudly
expected: OracleAgent(spec, bad_noise, client) raises AssertionError when consistency_rate=1.5.
result: pass
note: Assert guards are in OracleAgent.__init__, not NoiseParams — UAT command was wrong. Unit test test_oracle_agent_crashes_on_invalid_noise_params passes (confirmed by Test 1).

### 4. f_cognitive_load returns score in [0.0, 1.0]
expected: pytest tests/phase3/test_cognitive_load.py -v — all 5 tests pass.
result: pass

### 5. Contradiction detection
expected: Both contradiction tests pass.
result: pass

### 6. Oracle runs in conversation loop (5 turns without crash)
expected: test_oracle_agent_loop_5_turns passes.
result: pass

### 7. Events.jsonl written on oracle init
expected: test_oracle_init_logged passes.
result: pass
note: datetime.utcnow() DeprecationWarning in oracle_agent.py:131 (cosmetic, recurring)

### 8. Drift event logged when contradiction occurs
expected: test_drift_event_logged passes.
result: pass

### 9. InstructionalFeedback accumulates in global_instructions
expected: test_instructional_feedback_accumulates passes.
result: pass

### 10. Full suite still green (regression check)
expected: All 106+ tests pass — no regressions from Phase 3 changes.
result: issue
reported: "FAILED tests/phase2/test_feedback_parser.py::test_parse_feedback_real_llm - ModuleNotFoundError: no module named 'anthropic'"
severity: minor

## Summary

total: 10
passed: 9
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Full test suite passes with no failures"
  status: failed
  reason: "User reported: FAILED tests/phase2/test_feedback_parser.py::test_parse_feedback_real_llm - ModuleNotFoundError: no module named 'anthropic'"
  severity: minor
  test: 10
  root_cause: "test_parse_feedback_real_llm does `import anthropic` at line 99 BEFORE the ANTHROPIC_API_KEY skip guard at line 103 — crashes with ModuleNotFoundError instead of skipping. Pre-existing Phase 2 bug, not a Phase 3 regression. Also @pytest.mark.llm is unregistered in pyproject.toml."
  fix: "Move `import anthropic` inside the `if not key:` block, or add try/except ImportError → pytest.skip. Register llm mark in pyproject.toml."
  artifacts:
    - tests/phase2/test_feedback_parser.py:99
    - pyproject.toml
