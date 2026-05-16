---
phase: 05-ablation-harness-and-strategies
plan: 02
subsystem: conversation_loop
tags: [conversation, message, formatting, oracle, payload, enrichment]
dependency_graph:
  requires: [04-03]
  provides: [enriched-oracle-messages]
  affects: [src/conversation_loop.py, tests/phase5/test_format_message.py]
tech_stack:
  added: []
  patterns: [deviation-on-genuine-errors-only, fallback-without-deviation]
key_files:
  created:
    - tests/phase5/test_format_message.py
  modified:
    - src/conversation_loop.py
decisions:
  - W-02 fix: empty payload in show_subset and ask_question is NOT a deviation — RandomStrategy Phase 2 baseline contract; deviation() reserved for unknown cluster_id and missing item_id
  - id_to_text defaults to None for backward-compat with run_baseline 2-arg call
metrics:
  duration: ~10 minutes
  completed: 2026-05-17
  tasks_completed: 2
  files_modified: 2
---

# Phase 05 Plan 02: _format_message Enrichment Summary

One-liner: Enriched `_format_message` with optional `id_to_text` dict so Phase 5 strategies produce oracle messages containing actual item text and cluster names, while Phase 2 empty-payload callers fall back silently (W-02 fix).

## What Was Done

### Task 1: Extend _format_message signature + body

Replaced the `_format_message` function in `src/conversation_loop.py` with an enriched version that:

- Adds `id_to_text: dict[int, str] | None = None` as a third optional parameter (default None preserves backward-compat)
- `show_subset` branch: when `id_to_text` is provided and `item_ids` is non-empty in the payload, renders item texts and cluster names in the oracle message; falls back to Phase 2 placeholder WITHOUT deviation() for empty payloads (W-02 fix)
- `ask_question` branch: when `cluster_id` is in the payload, renders the named cluster in the oracle message; falls back to Phase 2 placeholder WITHOUT deviation() for empty payloads (W-02 fix)
- `deviation()` fires ONLY for genuinely unexpected payload state: unknown `cluster_a`/`cluster_b`, unknown `cluster_id`, or `item_id` missing from `id_to_text`
- Updated call site in `run_conversation` (line ~307 post-expansion): `message = _format_message(action, state, id_to_text)`
- `src/judge.py:run_baseline` 2-arg call `_format_message(action, initial_state)` continues to work unmodified

### Task 2: Add test suite

Created `tests/phase5/test_format_message.py` with 12 tests:

1. `test_show_subset_enriched_with_items_and_cluster_names` — item text and cluster names appear in message
2. `test_ask_question_enriched_with_cluster_name` — cluster name appears in message
3. `test_show_subset_empty_payload_falls_back_to_placeholder` — Phase 2 fallback string returned
4. `test_ask_question_empty_payload_falls_back_to_placeholder` — Phase 2 fallback string returned
5. `test_show_full_unchanged` — show_full behavior unchanged
6. `test_stop_unchanged` — stop behavior unchanged
7. `test_backward_compat_no_id_to_text_kwarg` — 2-arg call works
8. `test_strict_mode_show_subset_empty_payload_does_NOT_raise` — W-02: no UnexpectedDeviation
9. `test_strict_mode_ask_question_empty_payload_does_NOT_raise` — W-02: no UnexpectedDeviation
10. `test_strict_mode_unknown_cluster_id_in_ask_question_DOES_raise` — genuine deviation raises
11. `test_strict_mode_unknown_cluster_in_show_subset_DOES_raise` — genuine deviation raises
12. `test_strict_mode_missing_item_id_in_show_subset_DOES_raise` — genuine deviation raises

## Files Modified

| File | Change |
|------|--------|
| `src/conversation_loop.py` | `_format_message` signature extended + body enriched; call site updated to pass `id_to_text` |
| `tests/phase5/test_format_message.py` | Created — 12 tests covering all acceptance criteria |

## Test Results

```
tests/phase5/test_format_message.py ............ 12 passed in 12.53s
```

Phase 2 regression note: `tests/phase2/test_conversation_loop.py::test_30_turn_loop_completes` was already failing before this plan due to an embedding dimension mismatch in the test fixture (`Expected embedding dim 384, got 768`). This is a pre-existing failure confirmed by reverting changes and re-running the test — it is out of scope for this plan.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — `_format_message` only formats an Action + state into a string. No network, no disk, no auth. Matches T-05-03/T-05-04 accepted threats in the plan's threat model.

## Self-Check: PASSED

- `src/conversation_loop.py` exists and contains `id_to_text: dict[int, str] | None = None` and `message = _format_message(action, state, id_to_text)`
- `tests/phase5/test_format_message.py` exists with 12 tests, all passing
- Commit `42fb2a5` exists (Task 1 production code)
- Commit `ac52bb5` exists (Task 2 tests)
