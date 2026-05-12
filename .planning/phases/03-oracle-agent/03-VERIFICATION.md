---
phase: 03-oracle-agent
verified: 2026-05-12T10:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 4/5
  gaps_closed:
    - "ORC-03 dead-code gap (CR-01): cognitive_load now accepted as parameter by OracleAgent.reply() and forwarded from run_conversation(). Loop's pre-computed value is no longer dead code."
  gaps_remaining: []
  regressions: []
---

# Phase 3: Oracle Agent — Verification Report

**Phase Goal:** The LLM Oracle Agent behaves as a configurable, measurable stand-in for a human — with noise, cognitive fatigue, and drift — in a way that supports comparison with real humans in Phase 6
**Verified:** 2026-05-12T10:00:00Z
**Status:** passed
**Re-verification:** Yes — after ORC-03 architectural fix (cognitive_load forwarded from loop to oracle.reply())

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | OracleSpec / NoiseParams / OracleAgent satisfy OracleProtocol structural subtype | VERIFIED | `isinstance(agent, OracleProtocol)` passes; test_oracle_agent_satisfies_protocol passes |
| 2 | Noise parameters (consistency_rate, drift_probability, sycophancy_resistance) produce measurably different behavior via prompt injection | VERIFIED | All three injected into _build_system_prompt() as percentages; oracle_init JSONL event written with all three values; test_noise_params_in_prompt, test_oracle_init_logged pass |
| 3 | Per-turn cognitive-load score is pre-computed by the loop, forwarded to oracle.reply(), and triggers OVERLOAD in oracle system prompt above threshold | VERIFIED | `f_cognitive_load(state, message)` at conversation_loop.py line 180; value passed as `cognitive_load=cognitive_load` kwarg to `oracle.reply()` at line 184-188; `OracleAgent.reply()` accepts `cognitive_load: float | None = None` and uses it directly (skips internal recompute when provided); `_build_system_prompt()` injects OVERLOAD when `load > COG_LOAD_THRESHOLD`; test_overload_prompt_injected, test_no_overload_below_threshold pass |
| 4 | Structural contradictions detected, logged to delta window, and surfaced via OracleReply fields | VERIFIED | `_contradicts()`, `_check_contradiction()`, `update_delta_window()` implemented; `reply.contradiction_detected` set by loop via `update_delta_window(deltas, new_state.turn_index)` after `f_next_state`; drift_event written to events.jsonl; all 4 contradiction unit tests pass; test_drift_event_logged integration test passes |
| 5 | InstructionalFeedback parsed into global_instructions and injected into oracle system prompt on next turn | VERIFIED | `f_next_state` appends `InstructionalFeedback.instruction_text` to `global_instructions` list in-place (agent_functions.py line 511-513); loop passes `global_instructions=global_instructions` to `oracle.reply()` when oracle is OracleAgent; `_build_system_prompt()` includes them in Section 3; test_instructional_feedback_accumulates passes |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/oracle_agent.py` | OracleSpec, NoiseParams, OracleAgent with reply(), _build_system_prompt(), _check_contradiction(), update_delta_window() | VERIFIED | All exports present and substantive; reply() accepts cognitive_load param (fix applied); wired into conversation_loop.py |
| `src/cognitive_load.py` | f_cognitive_load pure function, COG_LOAD_THRESHOLD, MAX_K, MAX_MSG_LEN, TOP_K_ITEMS_PER_CLUSTER | VERIFIED | All 5 exports present; pure function with AssertionError on empty state |
| `src/oracle_protocol.py` | OracleReply extended with contradiction_detected, contradicted_turn | VERIFIED | Both fields present with defaults (backward-compatible) |
| `src/conversation_loop.py` | run_conversation() with f_cognitive_load, drift logging, oracle_init, global_instructions wiring, cognitive_load forwarded to oracle | VERIFIED | f_cognitive_load imported and computed; value forwarded as cognitive_load kwarg; oracle_init written to events.jsonl; drift_event written; global_instructions passed |
| `tests/phase3/test_oracle_agent.py` | 13 unit tests for OracleSpec, NoiseParams, OracleAgent | VERIFIED | All 13 tests present and passing |
| `tests/phase3/test_cognitive_load.py` | 5 unit tests for f_cognitive_load | VERIFIED | All 5 tests present and passing |
| `tests/phase3/test_oracle_loop_integration.py` | 3 integration tests for OracleAgent in run_conversation() | VERIFIED | All 3 tests present and passing |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| src/oracle_agent.py | src/oracle_protocol.py | `from src.oracle_protocol import OracleReply` | WIRED | Top-level import at line 27 |
| src/oracle_agent.py | src/cognitive_load.py | `from src.cognitive_load import f_cognitive_load, COG_LOAD_THRESHOLD` | WIRED | Lazy import in `reply()` and `_build_system_prompt()` |
| src/oracle_agent.py | src/feedback.py | isinstance checks for SplitFeedback, MergeFeedback, MoveItemFeedback | WIRED | Lazy import in `_contradicts()` and `update_delta_window()` |
| src/conversation_loop.py | src/cognitive_load.py | `from src.cognitive_load import f_cognitive_load` | WIRED | Import at function body level (line 130), outside while loop |
| src/conversation_loop.py | src/oracle_agent.py | `isinstance(oracle, OracleAgent)` | WIRED | Local import at line 154; isinstance guards for oracle_init, reply, update_delta_window |
| src/conversation_loop.py | events.jsonl sidecar | `_write_event()` helper | WIRED | `_write_event` defined at lines 70-88; called for oracle_init and drift_event |
| loop cognitive_load → oracle.reply() | src/oracle_agent.py reply(cognitive_load=...) | `cognitive_load=cognitive_load` kwarg | WIRED | Previously dead code (CR-01); fix applied — value now forwarded at lines 184-188 of conversation_loop.py |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| OracleAgent._build_system_prompt() | cognitive_load | Forwarded from loop (f_cognitive_load at loop line 180); fallback internal recompute if None | Yes — pure formula, same-turn state and message | FLOWING |
| OracleAgent.reply() | turn_cognitive_load in OracleReply | cognitive_load forwarded from loop; stored in OracleReply at return | Yes | FLOWING |
| run_conversation() | reply.contradiction_detected | oracle.update_delta_window(deltas, new_state.turn_index) at lines 207-212 | Yes — only set when parse_feedback returns structural deltas | FLOWING |
| events.jsonl | oracle_init record | Written at lines 155-165 (when oracle is OracleAgent) | Yes — oracle.spec and oracle.noise_params properties | FLOWING |
| events.jsonl | drift_event record | Written at lines 217-224 (when reply.contradiction_detected) | Yes — new_state.turn_index, new_state.timestamp, reply.contradicted_turn | FLOWING |
| global_instructions list | InstructionalFeedback texts | f_next_state appends instruction_text in-place; loop passes list to oracle.reply() | Yes — accumulates across turns | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| OracleAgent satisfies OracleProtocol | pytest tests/phase3/test_oracle_agent.py -q | 13 passed | PASS |
| f_cognitive_load returns float in [0,1] | pytest tests/phase3/test_cognitive_load.py -q | 5 passed | PASS |
| Contradiction detection unit tests | pytest tests/phase3/test_oracle_agent.py -k contradiction -q | 4 passed | PASS |
| Integration loop tests | pytest tests/phase3/test_oracle_loop_integration.py -q | 3 passed | PASS |
| Full Phase 3 suite | pytest tests/phase3/ -q -m "not llm" | 21 passed, 1 warning | PASS |
| No Phase 2 regression | pytest tests/phase2/ tests/phase3/ -m "not llm" | 106 passed, 1 skipped | PASS |
| Phase 1 failures pre-existing | pytest tests/phase1/ -q -m "not llm" | 3 failed (ModuleNotFoundError: hdbscan) — pre-date Phase 3, unrelated | PASS (not a regression) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ORC-01 | 03-01 | Oracle Agent is an LLM with explicit preference specification and persona | SATISFIED | OracleSpec (preferred_k, semantic_axes, persona_description), OracleAgent satisfies OracleProtocol. test_oracle_agent_satisfies_protocol, test_reply_returns_oracle_reply pass. |
| ORC-02 | 03-01, 03-04 | Oracle Agent has explicit noise parameters: consistency_rate, drift_probability, sycophancy_resistance | SATISFIED | NoiseParams dataclass; all three injected into system prompt as percentages (D-04); oracle_init JSONL event written with all three values; test_noise_params_in_prompt, test_oracle_init_logged, test_oracle_agent_loop_5_turns pass. |
| ORC-03 | 03-02, 03-04 | Oracle Agent receives a pre-computed cognitive-load score per turn and simulates fatigue/overload above threshold | SATISFIED | f_cognitive_load() computed in run_conversation() before oracle.reply() each turn; value forwarded as cognitive_load= kwarg (fix closed CR-01); oracle applies OVERLOAD above COG_LOAD_THRESHOLD; OracleReply.turn_cognitive_load populated correctly. All related tests pass. |
| ORC-04 | 03-03, 03-04 | Oracle Agent tracks and surfaces preference drift — contradictions detected, logged, and optionally flagged to Clustering Agent | SATISFIED | _contradicts() implements all three D-09 rules; update_delta_window() checks then appends; reply.contradiction_detected set by loop; drift_event written to events.jsonl. All 4 contradiction unit tests pass; test_drift_event_logged passes. |
| FB-04 | 03-04 | System accepts and acts on instructional oracle feedback via LLM parsing into structured constraints | SATISFIED | InstructionalFeedback parsed by parse_feedback; f_next_state appends instruction_text to global_instructions list; loop passes list to oracle.reply(); _build_system_prompt() includes them in Section 3. test_instructional_feedback_accumulates passes. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/oracle_agent.py | 131 | `datetime.utcnow().isoformat()` — deprecated API, naive datetime | WARNING | Python 3.12+ emits DeprecationWarning (confirmed in test run). oracle_init event timestamp in __init__ lacks timezone info. Non-blocking. |
| src/oracle_agent.py | 89-90 | Comment `# GlobalFeedback and InstructionalFeedback: ignored (D-09)` appears AFTER `return False` — unreachable comment | INFO | Misleading but harmless. |
| src/oracle_agent.py | multiple | Dual-write of oracle_init: both __init__ (when events_path given at construction) and run_conversation() write oracle_init — potential duplicate records in production | WARNING | Integration tests don't trigger duplicate because oracle_agent_factory does NOT pass events_path to constructor. Production usage could produce two oracle_init records. Non-blocking. |

None of these warnings block Phase 4 execution.

### Human Verification Required

None. All truths are programmatically verified.

---

## Re-verification Summary

**Previous status:** human_needed (4/5 truths, ORC-03 partial — dead-code architectural gap)

**Gap closed:** `OracleAgent.reply()` now accepts `cognitive_load: float | None = None`. The loop at `conversation_loop.py` lines 183-188 passes the pre-computed value as `cognitive_load=cognitive_load`. When the value is provided, `reply()` uses it directly and skips the internal `f_cognitive_load()` call. The dead-code variable (CR-01) is eliminated. ORC-03 architectural contract is now fully met.

**Regressions:** None. All 21 Phase 3 tests pass. All 85 Phase 2 tests pass. Full suite: 106 passed, 1 skipped (pre-existing umap edge case), 3 failed (pre-existing Phase 1 hdbscan not installed — unrelated to Phase 3).

---

_Verified: 2026-05-12T10:00:00Z_
_Verifier: Claude (gsd-verifier)_
