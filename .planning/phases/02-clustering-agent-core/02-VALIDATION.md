---
phase: 2
slug: clustering-agent-core
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-07
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (configured in pyproject.toml) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/phase2/ -q -x` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~30 seconds (excluding `@pytest.mark.llm` tests) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/phase2/ -q -x`
- **After every plan wave:** Run `pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| feedback-datamodel | 02-01 | 1 | FB-01,FB-02,FB-03 | — | N/A | unit | `pytest tests/phase2/test_feedback.py -x` | ❌ W0 | ⬜ pending |
| feedback-parser | 02-01 | 1 | FB-01,FB-02,FB-03 | LLM prompt injection | Validate all cluster IDs from parse_feedback before passing to f_next_state; assert on invalid references | unit | `pytest tests/phase2/test_feedback_parser.py -x` | ❌ W0 | ⬜ pending |
| uncertainty | 02-02 | 2 | CLUS-02 | — | N/A | unit | `pytest tests/phase2/test_uncertainty.py -x` | ❌ W0 | ⬜ pending |
| agent-functions-f-output | 02-02 | 2 | CLUS-01 | — | f_output asserts complete assignment; never partial | unit | `pytest tests/phase2/test_agent_functions.py::test_f_output_complete -x` | ❌ W0 | ⬜ pending |
| agent-functions-f-next-state-split | 02-02 | 2 | FB-02,CLUS-04 | — | Old cluster ID retired; two new IDs assigned | unit | `pytest tests/phase2/test_agent_functions.py::test_split_retires_id -x` | ❌ W0 | ⬜ pending |
| agent-functions-f-next-state-merge | 02-02 | 2 | FB-02,CLUS-04 | — | soft_probs rows sum to 1.0 after merge | unit | `pytest tests/phase2/test_agent_functions.py::test_merge_soft_probs_normalized -x` | ❌ W0 | ⬜ pending |
| agent-functions-f-next-state-move | 02-02 | 2 | FB-03,CLUS-04 | — | Target cluster set to 0.95; residual distributed proportionally | unit | `pytest tests/phase2/test_agent_functions.py::test_move_item_0_95 -x` | ❌ W0 | ⬜ pending |
| agent-functions-feedback-priority | 02-02 | 2 | FB-01,CLUS-04 | — | Global applied first in type-priority order | unit | `pytest tests/phase2/test_agent_functions.py::test_feedback_priority_order -x` | ❌ W0 | ⬜ pending |
| agent-functions-strategy | 02-02 | 2 | CLUS-03 | — | RandomStrategy selects valid action only | unit | `pytest tests/phase2/test_agent_functions.py::test_random_strategy -x` | ❌ W0 | ⬜ pending |
| hierarchy | 02-03 | 2 | HIER-01,HIER-02 | — | Hierarchy grows only on split/merge, not at session start | unit | `pytest tests/phase2/test_hierarchy.py -x` | ❌ W0 | ⬜ pending |
| oracle-protocol | 02-03 | 2 | CLUS-03 | — | MockOracle scripted reply sequence deterministic | unit | `pytest tests/phase2/test_oracle_protocol.py -x` | ❌ W0 | ⬜ pending |
| conversation-loop | 02-04 | 3 | CLUS-03,CLUS-04 | — | 30-turn loop state integrity verified at turns 10, 20, 30 | integration | `pytest tests/phase2/test_conversation_loop.py::test_30_turn_loop -x` | ❌ W0 | ⬜ pending |
| web-ui-routes | 02-05 | 4 | UI-01,UI-02 | Flask debug=False | Never run socketio.run() with debug=True | smoke | `pytest tests/phase2/test_app.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/phase2/__init__.py` — empty init for test discovery
- [ ] `tests/phase2/test_feedback.py` — FeedbackDelta dataclass construction and type checks
- [ ] `tests/phase2/test_feedback_parser.py` — parse_feedback with MockLLMClient returning scripted JSON (LLM-gated with `@pytest.mark.llm`)
- [ ] `tests/phase2/test_uncertainty.py` — f_uncertainty entropy computation on known soft_probs
- [ ] `tests/phase2/test_agent_functions.py` — f_output, f_next_state split/merge/move, f_next_best_step RandomStrategy
- [ ] `tests/phase2/test_hierarchy.py` — HierarchyStore register, record_split, record_merge
- [ ] `tests/phase2/test_oracle_protocol.py` — MockOracle scripted reply sequence
- [ ] `tests/phase2/test_conversation_loop.py` — 30-turn loop with MockOracle, state integrity checks
- [ ] `tests/phase2/test_app.py` — Flask routes smoke test (no real WebSocket)
- [ ] `tests/conftest.py` extension — Phase 2 shared fixtures: tiny ClusteringState with 3 clusters, MockOracle factory

**Note:** LLM-dependent tests (e.g., `test_feedback_parser.py` calling the real Anthropic API) must use `@pytest.mark.llm` and be excluded from the default fast run (`-m "not llm"`).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| WebSocket live updates render in browser | UI-01 | Requires browser WebSocket client + running server | Start `python web/app.py`; upload a dataset; verify cluster cards update each turn in browser |
| Dataset upload resets session visually | UI-02 | Requires browser interaction | Upload a second dataset file; verify cluster cards refresh and old session data disappears |
| 30-turn loop state integrity with real LLM | CLUS-03 | Requires real Anthropic API key | Run `python -m src.conversation_loop --mock=False --turns=30`; inspect AuditLog JSONL for valid state each turn |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
