# Plan 04-06 Summary — CLI + UI + Docs

**Plan:** 04-06
**Phase:** 4 — Judge Agent
**Executed:** 2026-05-16
**Status:** Complete

## What Was Built

### Task 1 — examples/run_baseline.py (D-25, JUDG-03)
CLI entry point for the no-dialogue baseline. Accepts `--dataset`, `--persona`, `--seed`, `--config`, `--name` args. Loads JSONL dataset with flexible field detection, loads YAML config, opens DB, calls `run_baseline()`, prints `ExperimentRead` JSON to stdout.

### Task 2 — Web UI D-28 Panels
**web/templates/index.html**: Added 4 new divs inside `<aside id="metrics-sidebar">`:
- `contradiction-count` — running total of oracle contradictions
- `convergence-signal` — color-coded stop reason or "running"
- `pairwise-accuracy` — percentage display
- `sparkline-container` + `sparkline` — 10-bar mini chart

**web/static/main.js**: Added `_pairwiseHistory` buffer, `updateJudgeMetrics()` function, `renderSparkline()` function, updated `session_stopped` handler to set convergence signal. ~80 lines of vanilla JS, no new dependencies.

### Task 3 — Documentation (D-32, D-33)
**CLAUDE.md**: 5 new bullets appended to Key Constraints (`src/db/` SQL rule, WAL, `deviation()`, JSONL truth, `datetime.utcnow()` ban). `## Database` section added. "Fail Loudly" section untouched.

**README.md**: No-Dialogue Baseline quickstart, Environment Variables table (with STRICT_MODE), Experiments Database section.

### Schema Fix (discovered during 04-05 test execution)
The `turns` table was missing a `deleted_at` column required by `cascade_delete()`. Fixed:
- `src/db/connection.py`: Added `deleted_at TEXT` to turns CREATE TABLE
- `src/db/turns.py`: `query()` now filters `WHERE deleted_at IS NULL`
- `docs/MODEL.md`: Updated turns table spec + cascade order

## Verification

- `python -m examples.run_baseline --help` exits 0 ✓
- `python -c "from web.app import app"` exits 0 ✓
- CLAUDE.md contains "src/db/ is the ONLY" and "Fail Loudly" ✓
- README.md contains "STRICT_MODE" and "run_baseline" ✓

## Commits

- `feat(04-06): CLI baseline + D-28 UI panels + CLAUDE.md/README.md docs; fix turns schema`
- `docs(04-06): execution summary`
