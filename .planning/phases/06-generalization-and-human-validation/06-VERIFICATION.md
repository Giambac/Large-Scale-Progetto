---
phase: 06-generalization-and-human-validation
verified: 2026-05-17T00:00:00Z
status: gaps_found
score: 6/8 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Human study N>=10 completed with within-subject protocol"
    status: failed
    reason: "The human study UI infrastructure is built (POST /study/sessions, study.html, study.js, DB with oracle_type='human'), but no actual human participants have run sessions. The experiments.db contains no human oracle rows. Roadmap SC-2 requires the study to be executed, not merely enabled."
    artifacts:
      - path: "notebooks/llm_vs_human.ipynb"
        issue: "Key Finding cell contains only placeholder tokens [MEAN_LLM], [MEAN_HUMAN], [DIFF] — no real data has been collected yet"
      - path: "experiments.db"
        issue: "No human-oracle experiment rows exist (study not yet executed)"
    missing:
      - "Run at least 10 human participants through /study/sessions using the built UI"
      - "Replace [PLACEHOLDER] tokens in notebook Key Finding cell with measured values after data collection"

  - truth: "LLM-vs-human gap quantified with CI on at least one headline metric"
    status: failed
    reason: "The comparison infrastructure (compare_oracle_types.py, llm_vs_human.ipynb) is built but contains no measured output. The notebook Key Finding cell explicitly uses placeholder template tokens [MEAN_LLM], [LO_LLM-HI_LLM], [N_LLM], [MEAN_HUMAN], etc. Roadmap SC-3 requires a reported quantitative finding, not just the tooling to produce one."
    artifacts:
      - path: "notebooks/llm_vs_human.ipynb"
        issue: "Cell 3 grouping code will print data only after experiments exist; all output is currently empty. Cell 4 Key Finding is entirely placeholder text."
      - path: "examples/compare_oracle_types.py"
        issue: "Script is correct but produces no output against an empty or LLM-only DB. The LLM-vs-human comparison table requires both oracle_type='llm' and oracle_type='human' rows in experiments.db."
    missing:
      - "Execute LLM oracle runs (N>=some) and human study runs (N>=10) to populate experiments.db"
      - "Run compare_oracle_types.py and record the actual turns-to-convergence comparison with 95% CI"
      - "Fill in notebook Key Finding cell with actual measured values"

human_verification:
  - test: "Verify study UI end-to-end flow"
    expected: "POST /study/sessions creates a session; GET /study/{id} serves the HTML page; a real user can send feedback; cluster state updates; satisfaction detection fires; session ends with convergence_reason logged to DB"
    why_human: "Requires a running uvicorn server, a browser, and manual interaction — cannot be verified programmatically without live server"

  - test: "Verify faceted UMAP mini-plots render correctly"
    expected: "K canvas elements appear side-by-side with correct global coordinate bounds; items belonging to each cluster appear at full opacity on that cluster's canvas; other items appear at 0.15 opacity; hovering an item in the list highlights the corresponding dot"
    why_human: "Visual rendering and hover interaction require a browser — cannot verify canvas drawing output programmatically"
---

# Phase 6: Generalization and Human Validation — Verification Report

**Phase Goal:** Oracle preferences are codified into a mapping function evaluated on held-out data, and LLM oracles are quantitatively compared with real humans — direct answer to research question 2
**Verified:** 2026-05-17T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

The code infrastructure for all three requirements (GEN-01, GEN-02, EXP-V2-01) is complete, substantive, and wired. 14/14 tests pass. However, two of the four roadmap success criteria require research execution (running actual human sessions and recording the LLM-vs-human gap), which has not yet occurred. The phase goal as written — "LLM oracles are **quantitatively compared** with real humans — **direct answer** to research question 2" — is not yet achieved because no comparison data exists.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `src/mapping.py` exists with all 6 exports (OracleRuleSet, extract_oracle_rules, MappingProtocol, LLMMappingStrategy, CentroidMappingStrategy, MAPPING_REGISTRY) | VERIFIED | File read: all 6 exports present with full implementation; `python -c "from src.mapping import MAPPING_REGISTRY, OracleRuleSet, MappingProtocol; assert set(MAPPING_REGISTRY) == {'llm', 'centroid'}"` passes |
| 2 | Both mapping strategies satisfy MappingProtocol (isinstance check) | VERIFIED | `isinstance(LLMMappingStrategy(), MappingProtocol)` = True confirmed live |
| 3 | `examples/evaluate_mapping.py` evaluates held-out split with both strategies, bootstrap CI, and comparison table | VERIFIED | File read: all logic present; `--help` shows all required flags; `python -c "import examples.evaluate_mapping"` passes |
| 4 | Human study UI operational: POST /study/sessions, GET /study/{id}, satisfaction detection, 30-turn cap, DB writes with oracle_type='human' | VERIFIED | `web/app.py` contains all routes (grepped); `STUDY_MAX_TURNS=30` confirmed; `oracle_type="human"` in ExperimentCreate call at line 748; 7/7 test_study_ui.py tests pass |
| 5 | `examples/compare_oracle_types.py` groups by (oracle_type, strategy_id), computes bootstrap CI, prints satisfaction table | VERIFIED | File read: full implementation; `--help` shows all 4 flags; `import examples.compare_oracle_types` passes |
| 6 | `notebooks/llm_vs_human.ipynb` is valid nbformat 4 with Key Finding + Limitations sections | VERIFIED | `assert nb['nbformat'] == 4` passes; 8 cells confirmed; Key Finding cell (cell 4) and Limitations cell (cell 8) present |
| 7 | Human study N>=10 completed with within-subject protocol | FAILED | No human sessions have been run. Notebook Key Finding cell contains only placeholder tokens `[MEAN_LLM]`, `[MEAN_HUMAN]`, `[DIFF]`. No experiments.db rows with oracle_type='human'. |
| 8 | LLM-vs-human gap quantified with CI on at least one headline metric | FAILED | No measured comparison exists. compare_oracle_types.py will output nothing against an empty/LLM-only DB. The gap is a placeholder template, not a reported finding. |

**Score:** 6/8 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/mapping.py` | OracleRuleSet, extract_oracle_rules, MappingProtocol, LLMMappingStrategy, CentroidMappingStrategy, MAPPING_REGISTRY | VERIFIED | 384 lines; all 6 exports implemented; no stubs; no sqlite3; commit 8e98138 |
| `examples/evaluate_mapping.py` | 25-item held-out eval, bootstrap CI, comparison table | VERIFIED | 211 lines; full logic; all 9 acceptance criteria pass; commit ee84865 |
| `web/app.py` (study routes) | POST /study/sessions, GET /study/{id}, study_feedback event, STUDY_MAX_TURNS | VERIFIED | Lines 700-1073 in web/app.py; all 9 acceptance criteria pass; commit 579680a |
| `web/templates/study.html` | Three-panel study page with cluster cards, UMAP canvases, feedback input | VERIFIED | 193 lines; all required element IDs present; socket.io CDN + study.js included |
| `web/static/study.js` | SocketIO event handlers, canvas drawing, feedback emit | VERIFIED | All 5 events handled; `getContext('2d')` present; no Chart.js/D3 |
| `examples/compare_oracle_types.py` | Groups by (oracle_type, strategy_id), bootstrap CI, satisfaction table | VERIFIED | 175 lines; all 7 acceptance criteria pass; commit 50f1fd5 |
| `notebooks/llm_vs_human.ipynb` | nbformat 4, 8 cells, Key Finding + Limitations | VERIFIED | Valid JSON; nbformat=4; 8 cells (4 code, 4 markdown); commit e09c10d |
| `tests/test_mapping.py` | 7 tests: OracleRuleSet, MAPPING_REGISTRY, isinstance, empty audit log, centroid assignment, empty cluster error, invalid LLM ID | VERIFIED | 7/7 pass in 18.5s; commit 5404514 |
| `tests/test_study_ui.py` | 7 tests: POST valid/invalid-dataset/invalid-backend, GET known/unknown, STUDY_MAX_TURNS default/override | VERIFIED | 7/7 pass in 0.88s; commit ca37af7 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `examples/evaluate_mapping.py` | `src/mapping.py` | `from src.mapping import MAPPING_REGISTRY, extract_oracle_rules` | WIRED | Line 58 |
| `examples/evaluate_mapping.py` | `src/analysis.py` | `from src.analysis import compute_bootstrap_ci` | WIRED | Line 55 |
| `examples/evaluate_mapping.py` | `dataset/held_out.jsonl` | `open(args.held_out, encoding="utf-8")` read-only | WIRED | Lines 87-99; never opened with write mode |
| `web/app.py` | `src/db/experiments` | `ExperimentCreate(oracle_type="human")` at line 742 | WIRED | oracle_type='human' in DB write; src/db is the only SQL layer |
| `web/app.py` | `src/db/turns` | `_turns_db.create(db, _TurnCreate(...))` at line 975 | WIRED | Turn rows written per feedback turn |
| `examples/compare_oracle_types.py` | `src/db/experiments` | `exp_db.query(db, dataset=args.dataset)` at line 63 | WIRED | Groups by oracle_type + strategy_id |
| `examples/compare_oracle_types.py` | `src/analysis.py` | `compute_bootstrap_ci(values, ...)` | WIRED | Lines 79, 111 |
| `notebooks/llm_vs_human.ipynb` | `src/db/experiments` | `exp_db.query(db)` in cell 2 | WIRED | DB closed in finally block |
| `notebooks/llm_vs_human.ipynb` | `src/analysis.py` | `compute_bootstrap_ci(vals, seed=0)` in cell 3 | WIRED | Per-group CI |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `study.html` cluster cards | cluster names/descriptions via `study_state` SocketIO event | `_run_study_background` in `web/app.py` — actual clustering result from HDBSCAN/KMeans | Yes (when session is live) | FLOWING |
| `study.js` mini-plot canvases | `_allCoords`, `_globalBounds`, `_perCluster` via `study_projection` event | UMAP projection computed in worker thread from real embeddings | Yes (when session is live) | FLOWING |
| `compare_oracle_types.py` table | `rows` from `exp_db.query()` | `experiments.db` DB query | Not yet — DB has no human rows | STATIC (no data yet) |
| `llm_vs_human.ipynb` Key Finding | `[MEAN_LLM]`, `[MEAN_HUMAN]` | Researcher fills in after running cell 3 | Not yet — placeholder tokens only | STATIC (placeholder) |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `src.mapping` module imports cleanly | `python -c "from src.mapping import MAPPING_REGISTRY, OracleRuleSet, MappingProtocol"` | MAPPING_REGISTRY == {'llm', 'centroid'} | PASS |
| `LLMMappingStrategy` satisfies `MappingProtocol` | `python -c "from src.mapping import MappingProtocol, LLMMappingStrategy; print(isinstance(LLMMappingStrategy(), MappingProtocol))"` | True | PASS |
| `OracleRuleSet` constructs and stores fields | `python -c "from src.mapping import OracleRuleSet; r = OracleRuleSet(synonyms=[['a','b']], focus_areas=['x'], exclusions=[], cluster_rules=[]); assert r.synonyms == [['a','b']]"` | Exits 0 | PASS |
| `evaluate_mapping` --help | `python -m examples.evaluate_mapping --help` | All 6 flags shown | PASS |
| `compare_oracle_types` --help | `python -m examples.compare_oracle_types --help` | All 4 flags shown | PASS |
| notebook nbformat valid | `python -c "import json; nb = json.load(open('notebooks/llm_vs_human.ipynb')); assert nb['nbformat'] == 4"` | 8 cells confirmed | PASS |
| `test_mapping.py` suite | `python -m pytest tests/test_mapping.py -v` | 7 passed in 18.5s | PASS |
| `test_study_ui.py` suite | `python -m pytest tests/test_study_ui.py -v` | 7 passed in 0.88s | PASS |
| web/app.py has no sqlite3.execute | grep sqlite3 web/app.py | No matches | PASS |
| web/app.py has no eventlet/gevent | grep eventlet\|gevent web/app.py | No matches | PASS |
| src/mapping.py has no sqlite3/src.db | grep sqlite3\|src.db src/mapping.py | No matches | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| GEN-01 | 06-01-PLAN.md | Mapping function (rule set + 2 strategies) that assigns new items without oracle interaction | SATISFIED | `src/mapping.py` with all 6 exports; MAPPING_REGISTRY has both strategies; isinstance checks pass |
| GEN-02 | 06-02-PLAN.md | Mapping function evaluated on frozen held-out split; accuracy validated by oracle on 20-30 items | SATISFIED (code) / UNCERTAIN (execution) | `examples/evaluate_mapping.py` is complete and correct; actual accuracy numbers require a real session + running the script |
| EXP-V2-01 | 06-03-PLAN.md, 06-05-PLAN.md | Full N oracles × M tasks with LLM and human oracles, comparing on oracle satisfaction, turns-to-convergence, generalization accuracy | PARTIAL — code infrastructure complete; human study execution missing | Study UI + compare CLI + notebook all built; no human sessions run; no measured gap |

---

## Anti-Patterns Found

The REVIEW.md (committed as `97f8120`) documents 4 critical issues and 5 warnings found by code review. These are noted here for completeness — they were found post-implementation but are not yet fixed.

| File | Issue ID | Pattern | Severity | Impact |
|------|----------|---------|---------|--------|
| `web/app.py:934,952` | CR-01 | `feedback_queue.pop(0)` without length check — IndexError if `_end_study_session` fires concurrently | WARNING | Daemon thread crash during concurrent session end |
| `web/app.py:1024-1027` | CR-02 | `_detect_satisfaction` swallows `anthropic.APIError` via `deviation()` — silent wrong answer | WARNING | Violates CLAUDE.md "fail loudly"; participant stuck until turn budget |
| `examples/evaluate_mapping.py:178` | CR-03 | `int(prediction)` unguarded — raises ValueError on non-integer cluster ID strings | WARNING | Evaluation loop crash after spending API budget |
| `web/app.py:41` | CR-04 | `_make_session_timestamp()` uses naive local time; no sub-second suffix — same-second collision overwrites session state | WARNING | Silent data loss on concurrent session creation |
| `web/app.py:939-959` | WR-04 | Satisfaction-rejected path does not increment `turn_index` — budget silently undercounts | INFO | Turn counter diverges from actual oracle interactions |

None of these prevent the code from running correctly in the single-user, no-concurrent-call research scenario. They would need to be fixed before a real multi-participant study deployment.

---

## Human Verification Required

### 1. End-to-End Study Session Flow

**Test:** Start uvicorn (`uvicorn web.app:asgi_app --host 0.0.0.0 --port 5000`), POST to `/study/sessions` with a valid dataset_path (e.g. `dataset/train.jsonl`) and `backend=hdbscan`, navigate to the returned `study_url`, send 3-4 feedback messages, verify session state updates.

**Expected:** Cluster cards render in left panel; UMAP mini-plots appear in center panel; feedback input enables after `study_awaiting_feedback` event; after sending "this looks good to me", satisfaction banner appears asking for confirmation; clicking Yes logs `convergence_reason=oracle_satisfied` in experiments.db.

**Why human:** Requires running server, browser interaction, and SocketIO real-time event verification.

### 2. Faceted UMAP Mini-Plot Correctness

**Test:** In a live study session, hover over an item in the cluster card list and verify the corresponding dot is highlighted in the correct mini-plot canvas.

**Expected:** Dot at the item's UMAP coordinates becomes larger (radius 6) and brighter; other clusters' canvases are unaffected; global coordinate bounds are shared so spatial positions are consistent across all K mini-plots.

**Why human:** Canvas pixel-level output requires visual inspection; no programmatic assertion can verify the drawing output matches the expected scatter plot layout.

---

## Gaps Summary

The phase delivered complete, tested, wired code for all three requirements (GEN-01, GEN-02, EXP-V2-01). All 14 tests pass. The mapping layer, held-out evaluation script, human study UI, and LLM-vs-human analysis tools are fully implemented.

Two roadmap success criteria remain unmet because they require **research execution**, not code:

1. **SC-2 (Human study N>=10):** The study UI exists and works, but no human participants have used it. The experiments.db has no `oracle_type='human'` rows.

2. **SC-3 (Gap quantified with CI):** The comparison tooling exists but the notebook Key Finding cell contains only placeholder tokens (`[MEAN_LLM]`, `[MEAN_HUMAN]`). Without actual data from both LLM and human runs, the headline research finding — a direct answer to research question 2 — cannot be stated.

The code review (commit `97f8120`) also identified 4 critical correctness issues in `web/app.py` and `evaluate_mapping.py` that should be fixed before running the human study to avoid data integrity problems.

**To close these gaps:** Fix the 4 critical code issues (CR-01 through CR-04), then run the human study (N>=10 participants), then run `compare_oracle_types.py` and populate the notebook Key Finding cell with measured values.

---

_Verified: 2026-05-17T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
