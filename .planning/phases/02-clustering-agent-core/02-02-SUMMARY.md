---
plan: 02-02
phase: 02-clustering-agent-core
status: complete
completed: 2026-05-07
subsystem: feedback
tags: [feedback, data-model, llm-parsing, structured-types]
dependency_graph:
  requires: [02-01]
  provides: [src/feedback.py, src/feedback_parser.py]
  affects: [src/agent_functions.py, src/conversation_loop.py]
tech_stack:
  added: []
  patterns: [frozen-dataclass, Union-type-alias, LLM-JSON-parse, assert-validation, fail-loudly]
key_files:
  created:
    - src/feedback.py
    - src/feedback_parser.py
  modified: []
decisions:
  - "FeedbackDelta Union ordering (GlobalFeedback first) aligns with D-07 type-priority dispatch order in f_next_state"
  - "parse_feedback fast-path: empty raw_text returns [] without LLM call to avoid unnecessary API cost"
  - "_build_delta() uses if-chains instead of dict-dispatch so KeyError propagates immediately (fail loudly)"
metrics:
  duration: "~15 minutes"
  completed_date: "2026-05-07"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
commits:
  - 664c148
  - b740457
requirements:
  - FB-01
  - FB-02
  - FB-03
---

# Phase 02 Plan 02: Feedback Data Model and LLM Parser Summary

5 frozen FeedbackDelta dataclasses + Union alias + ORACLE_MOVE_CONFIDENCE constant, and parse_feedback() LLM parser with single-boundary try/except and post-parse cluster ID assertion.

## What Was Built

**Task 1 — src/feedback.py** (commit `664c148`):

- 5 frozen dataclasses: `SplitFeedback`, `MergeFeedback`, `MoveItemFeedback`, `GlobalFeedback`, `InstructionalFeedback`
- `FeedbackDelta` Union alias — ordering matches D-07 type-priority dispatch: global → split/merge → move_item → instructional
- `ORACLE_MOVE_CONFIDENCE = 0.95` named constant (D-10; used by `f_next_state` for point-move soft_probs override)
- `UNIFORM_FALLBACK_THRESHOLD = 1e-9` named constant (zero-sum guard for point-move redistribution edge case)
- No try/except, no helper functions — pure data types only
- All 10 `test_feedback.py` tests GREEN

**Task 2 — src/feedback_parser.py** (commit `b740457`):

- `parse_feedback(raw_text, state, client) -> list[FeedbackDelta]` — the only Phase 2 module that calls the LLM for parsing (D-05)
- Fast path: returns `[]` immediately if `raw_text` is empty (no LLM call)
- `_build_cluster_summary(state)` — compact "id: name" string for LLM prompt context
- `_build_delta(item, valid_cluster_ids)` — maps JSON item to FeedbackDelta; asserts valid type + cluster IDs; KeyError propagates on missing fields (fail loudly)
- `VALID_FEEDBACK_TYPES = frozenset({"global", "split", "merge", "move_item", "instructional"})` module constant
- Exactly one `try/except` block wrapping `json.loads` — `json.JSONDecodeError` re-raised (T-02-04 accepted)
- Post-parse assertion validates every cluster_id in every delta against `state.clusters` (T-02-02 mitigated)
- Unknown type assertion crashes immediately (T-02-03 mitigated)
- 8/9 `test_feedback_parser.py` tests GREEN (`@pytest.mark.llm` integration test skipped as designed)

## Verification Results

```
pytest tests/phase2/test_feedback.py tests/phase2/test_feedback_parser.py -q -m "not llm"
18 passed in <1s
```

```
from src.feedback import SplitFeedback, MergeFeedback, MoveItemFeedback, GlobalFeedback, InstructionalFeedback, FeedbackDelta, ORACLE_MOVE_CONFIDENCE
from src.feedback_parser import parse_feedback
# imports OK
# ORACLE_MOVE_CONFIDENCE == 0.95 → constants OK
```

## Deviations from Plan

None — plan executed exactly as written. Both modules match the action specifications in the plan, follow the `from __future__ import annotations` header convention, and replicate the `cluster_naming.py` LLM call + JSON parse + assert schema pattern.

## Known Stubs

None. Both files are fully implemented with no placeholder values, hardcoded empty returns, or TODO markers. The `@pytest.mark.llm` test in `test_feedback_parser.py` is intentionally skipped in non-LLM environments (by design from Wave 0 scaffolding).

## Threat Surface Scan

All threats enumerated in the plan's `<threat_model>` are addressed:

| Threat | Mitigation Status |
|--------|-------------------|
| T-02-02: LLM prompt injection via oracle raw_text | Mitigated — post-parse `assert item["cluster_id"] in valid_cluster_ids` in `_build_delta()` |
| T-02-03: LLM hallucinated feedback type | Mitigated — `assert item["type"] in VALID_FEEDBACK_TYPES` in `_build_delta()` |
| T-02-04: Malformed JSON from LLM | Accepted — `json.JSONDecodeError` re-raised; single permitted try/except boundary |

No new threat surface introduced beyond what the plan enumerated.

## Self-Check: PASSED

- [x] `src/feedback.py` exists with 5 frozen dataclasses + FeedbackDelta + 2 constants
- [x] `src/feedback_parser.py` exists with `parse_feedback()` function
- [x] commit `664c148` exists: `feat(02-02): implement src/feedback.py`
- [x] commit `b740457` exists: `feat(02-02): implement src/feedback_parser.py`
- [x] 18 tests GREEN (10 from test_feedback.py, 8 non-LLM from test_feedback_parser.py)
- [x] Exactly 1 try/except in feedback_parser.py (json.JSONDecodeError only)
- [x] No except Exception, no swallowed exceptions
- [x] No stubs in either file
