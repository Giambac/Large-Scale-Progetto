---
phase: 05-ablation-harness-and-strategies
plan: 03
subsystem: strategy
tags: [strategy, uncertainty, boundary, ablation, deterministic, soft-probs]

# Dependency graph
requires:
  - phase: 05-02
    provides: enriched _format_message that consumes cluster_id / cluster_a / cluster_b / item_ids payloads
  - phase: 04
    provides: UncertaintyReport + f_uncertainty + soft_probs positional invariant
provides:
  - UncertaintyDrivenStrategy in src/strategy.py (ALAB-01, D-01, D-02)
  - BoundaryDrivenStrategy in src/strategy.py (ALAB-01, D-03, D-04)
  - tests/phase5/test_strategies.py — 12 tests covering payload enrichment, fallback deviation, determinism, protocol conformance
affects: [05-04, 05-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy imports inside select() for UncertaintyReport + deviation() — mirrors src/judge.py pattern, avoids circular imports"
    - "Seeded random.Random per strategy instance for deterministic tie-breaking (Phase 4 D-19 carry-forward)"
    - "Positional pos-map: {c.id: j for j, c in enumerate(state.clusters)} for soft_probs indexing invariant"
    - "Turn-index-seeded sampler random.Random(state.turn_index) for capping ambiguous_items to MAX_SUBSET_SIZE"

key-files:
  created:
    - tests/phase5/__init__.py
    - tests/phase5/test_strategies.py
  modified:
    - src/strategy.py

key-decisions:
  - "AMBIGUOUS_P_MIN=0.3: items qualify as ambiguous when both P(A) and P(B) exceed 0.3 (from CONTEXT.md threshold spec)"
  - "MAX_SUBSET_SIZE=8: cognitive-load cap on show_subset item_ids list, deterministically sampled by state.turn_index"
  - "All degenerate fallbacks return ask_question with empty payload + deviation() — never crash, never random-fallback (ablation signal preserved)"
  - "BoundaryDriven validates cluster_a/cluster_b presence in pos-map before dereferencing — fires deviation() if unknown id"

patterns-established:
  - "Strategy degenerate-fallback pattern: check condition -> deviation() -> return Action(ask_question, {})"
  - "Ambiguous-zone scan: build pos map, iterate soft_probs, threshold both P values, assert top-2 == {ia, ib}"

requirements-completed: [ALAB-01]

# Metrics
duration: 15min
completed: 2026-05-17
---

# Phase 5 Plan 03: UncertaintyDrivenStrategy and BoundaryDrivenStrategy Summary

**Two deterministic Phase 5 strategies appended to src/strategy.py: UncertaintyDrivenStrategy (highest-entropy cluster targeting) and BoundaryDrivenStrategy (closest-pair ambiguous-zone show_subset), completing ALAB-01 with all 3 strategies under one Protocol.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-17T00:00:00Z
- **Completed:** 2026-05-17T00:15:00Z
- **Tasks:** 2
- **Files modified:** 3 (src/strategy.py, tests/phase5/__init__.py, tests/phase5/test_strategies.py)

## Accomplishments
- UncertaintyDrivenStrategy appended to src/strategy.py: targets split_candidates[0] (highest mean-entropy cluster), seeded tie-breaking, degenerate fallback fires deviation()
- BoundaryDrivenStrategy appended to src/strategy.py: scans merge_candidates[0] pair, collects ambiguous zone via positional soft_probs pos-map (AMBIGUOUS_P_MIN=0.3), caps at MAX_SUBSET_SIZE=8, degenerate fallbacks fire deviation()
- 12 tests in tests/phase5/test_strategies.py: all pass — happy paths, empty-split/merge/zone fallbacks, STRICT_MODE deviation raises, determinism, and structural StrategyProtocol conformance for all 3 strategy classes
- RandomStrategy left entirely unmodified (no existing code changed)

## Task Commits

1. **Task 1: Append UncertaintyDrivenStrategy + BoundaryDrivenStrategy to src/strategy.py** - `e363411` (feat)
2. **Task 2: Add strategy tests — enriched payloads, fallback deviation, determinism** - `88df52f` (test)

**Plan metadata:** see docs commit below

## Files Created/Modified
- `src/strategy.py` - Appended UncertaintyDrivenStrategy (lines ~89-147) and BoundaryDrivenStrategy (lines ~149-249) after RandomStrategy; no existing code modified
- `tests/phase5/__init__.py` - Empty package init created (phase5 directory did not exist from prior plans)
- `tests/phase5/test_strategies.py` - 12-test suite for both new strategies

## Decisions Made
- AMBIGUOUS_P_MIN=0.3 — threshold per CONTEXT.md spec; items qualify for ambiguous zone when both P(A) and P(B) are >= 0.3
- MAX_SUBSET_SIZE=8 — cognitive-load cap; deterministic sample via random.Random(state.turn_index) mirrors PairBag.sample (D-19)
- Lazy imports of UncertaintyReport and deviation() inside select() to avoid circular imports (same pattern as src/judge.py)
- BoundaryDrivenStrategy validates cluster IDs from merge_candidates against pos-map; unknown ID fires deviation() rather than crashing

## Deviations from Plan

None — plan executed exactly as written. The tests/phase5/__init__.py was documented as "may need to be created" and was created as specified.

## Issues Encountered

Pre-existing Phase 2 test failures (unrelated to this plan):
- `tests/phase2/test_agent_functions.py` — 10 tests fail with `AssertionError: Expected embedding dim 384, got 768` in EmbeddingStore.__init__. These failures exist on the commit prior to this plan's first commit (confirmed via git stash). Scope boundary: out of scope for this plan.

## Known Stubs

None.

## Threat Flags

None — new code is pure local computation (state + uncertainty_report -> Action), no I/O, no network.

## Next Phase Readiness
- ALAB-01 complete: all 3 strategies (RandomStrategy, UncertaintyDrivenStrategy, BoundaryDrivenStrategy) in src/strategy.py under StrategyProtocol
- Plan 05-04 (harness) can now import both via STRATEGY_REGISTRY dict and run ablation experiments
- Plan 05-05 (analysis) can compare convergence curves across all 3 strategies + no_dialogue baseline

---
*Phase: 05-ablation-harness-and-strategies*
*Completed: 2026-05-17*
