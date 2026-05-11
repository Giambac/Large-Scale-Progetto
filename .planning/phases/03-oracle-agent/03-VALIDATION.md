---
phase: 3
slug: oracle-agent
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-11
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `pytest tests/phase3/ -q -m "not llm"` |
| **Full suite command** | `pytest tests/ -q -m "not llm"` |
| **LLM smoke test** | `pytest tests/phase3/ -q -m llm` (optional — requires API key) |
| **Estimated runtime** | ~5 seconds (mocked) / ~30 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/phase3/ -q -m "not llm"`
- **After every plan wave:** Run `pytest tests/ -q -m "not llm"`
- **Before `/gsd-verify-work`:** Full suite must be green (`pytest tests/ -q -m "not llm"`)
- **Max feedback latency:** 5 seconds (mocked unit tests)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 03-W0-01 | W0 | 0 | — | Test scaffold in place | infra | `pytest tests/phase3/ -q -m "not llm"` | ❌ W0 | ⬜ pending |
| 03-01-01 | 01 | 1 | ORC-01 | OracleAgent satisfies OracleProtocol | unit | `pytest tests/phase3/test_oracle_agent.py::test_oracle_agent_satisfies_protocol -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | ORC-01 | reply() returns OracleReply with non-empty raw_text | unit | `pytest tests/phase3/test_oracle_agent.py::test_reply_returns_oracle_reply -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | ORC-01 | OracleSpec fields accessible from OracleAgent | unit | `pytest tests/phase3/test_oracle_agent.py::test_oracle_spec_fields -x` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 1 | ORC-02 | oracle_init JSONL record written to events log | unit | `pytest tests/phase3/test_oracle_agent.py::test_oracle_init_logged -x` | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 1 | ORC-02 | NoiseParams values appear in assembled system prompt | unit | `pytest tests/phase3/test_oracle_agent.py::test_noise_params_in_prompt -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | ORC-03 | f_cognitive_load returns float in [0, 1] | unit | `pytest tests/phase3/test_cognitive_load.py::test_load_in_range -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | ORC-03 | f_cognitive_load > COG_LOAD_THRESHOLD for max-stress state | unit | `pytest tests/phase3/test_cognitive_load.py::test_load_above_threshold -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | ORC-03 | OVERLOAD instruction in prompt when load > 0.7 | unit | `pytest tests/phase3/test_oracle_agent.py::test_overload_prompt_injected -x` | ❌ W0 | ⬜ pending |
| 03-02-04 | 02 | 1 | ORC-03 | OVERLOAD instruction absent when load <= 0.7 | unit | `pytest tests/phase3/test_oracle_agent.py::test_no_overload_below_threshold -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 2 | ORC-04 | Contradiction: MergeFeedback(A,B) after SplitFeedback(A) | unit | `pytest tests/phase3/test_oracle_agent.py::test_contradiction_merge_after_split -x` | ❌ W0 | ⬜ pending |
| 03-03-02 | 03 | 2 | ORC-04 | Contradiction: MoveItemFeedback(item, B) after MoveItemFeedback(item, C) | unit | `pytest tests/phase3/test_oracle_agent.py::test_contradiction_move_item -x` | ❌ W0 | ⬜ pending |
| 03-03-03 | 03 | 2 | ORC-04 | No contradiction when window is empty | unit | `pytest tests/phase3/test_oracle_agent.py::test_no_contradiction_empty_window -x` | ❌ W0 | ⬜ pending |
| 03-03-04 | 03 | 2 | ORC-04 | No false-positive contradiction on unrelated deltas | unit | `pytest tests/phase3/test_oracle_agent.py::test_no_false_positive_contradiction -x` | ❌ W0 | ⬜ pending |
| 03-04-01 | 04 | 3 | ORC-04 | drift_event JSONL record written when contradiction_detected=True | integration | `pytest tests/phase3/test_oracle_loop_integration.py::test_drift_event_logged -x` | ❌ W0 | ⬜ pending |
| 03-04-02 | 04 | 3 | FB-04 | global_instructions content appears in oracle system prompt | unit | `pytest tests/phase3/test_oracle_agent.py::test_global_instructions_in_prompt -x` | ❌ W0 | ⬜ pending |
| 03-04-03 | 04 | 3 | FB-04 | InstructionalFeedback appended to global_instructions by loop | integration | `pytest tests/phase3/test_oracle_loop_integration.py::test_instructional_feedback_accumulates -x` | ❌ W0 | ⬜ pending |
| 03-05-01 | 04 | 3 | ALL | OracleAgent runs 5 turns in conversation loop without error | integration | `pytest tests/phase3/test_oracle_loop_integration.py::test_oracle_agent_loop_5_turns -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/phase3/__init__.py` — empty init file (new directory)
- [ ] `tests/phase3/test_oracle_agent.py` — OracleSpec, NoiseParams, OracleAgent unit tests (all mocked LLM)
- [ ] `tests/phase3/test_cognitive_load.py` — f_cognitive_load formula, threshold, edge cases
- [ ] `tests/phase3/test_oracle_loop_integration.py` — OracleAgent wired into run_conversation() with mocked LLM; oracle_init and drift_event logging
- [ ] `tests/conftest.py` (update) — add `oracle_agent_factory` fixture (OracleAgent with MagicMock client)

*Wave 0 must be complete before any implementation tasks execute.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Varying consistency_rate produces different convergence curves | ORC-02 | Requires multi-run statistical comparison | Run OracleAgent with consistency_rate=0.3 vs 0.9 for 20 turns each; compare feedback acceptance rates in events.jsonl |
| Oracle replies visibly simpler when cognitive load > 0.7 | ORC-03 | "Simpler reply" is a subjective judgment | Set up a session with MAX_K clusters; verify raw_text is shorter and focused on one cluster |

---

## Validation Sign-Off

- [ ] All tasks have automated verify commands
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING file references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
