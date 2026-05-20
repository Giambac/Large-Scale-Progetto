---
status: complete
phase: 06-generalization-and-human-validation
source:
  - 06-01-SUMMARY.md
  - 06-02-SUMMARY.md
  - 06-03-SUMMARY.md
  - 06-04-SUMMARY.md
  - 06-05-SUMMARY.md
started: 2026-05-19T00:00:00Z
updated: 2026-05-20T20:35:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Test Suite Passes
expected: Running `python -m pytest tests/test_mapping.py tests/test_study_ui.py -v` from repo root completes with "14 passed" in under 30 seconds. No live LLM or DB calls (all monkeypatched).
result: pass
note: "Fixed during UAT — test_llm_strategy_validates_cluster_id patched a nonexistent path (src.mapping.anthropic). Now patches the real LLM surface (src.mapping.resolve_llm_key/build_client/chat). 14/14 pass."
prior_result: issue (major) — ImportError: 'src.mapping' is not a package

### 2. evaluate_mapping CLI Help
expected: Running `python -m examples.evaluate_mapping --help` exits 0 and shows flags: --session-dir, --n-items, --held-out, --embeddings, --seed, --json.
result: pass

### 3. compare_oracle_types CLI Help
expected: Running `python -m examples.compare_oracle_types --help` exits 0 and shows flags: --dataset, --n-bootstrap, --ci, --json.
result: pass

### 4. Web Server Boots Cold
expected: Stop any running server. Run `uvicorn web.app:asgi_app --host 0.0.0.0 --port 5000`. Server boots without errors, no eventlet/gevent imports, and `http://localhost:5000/` returns a live page.
result: pass

### 5. Create Study Session via POST
expected: With ANTHROPIC_API_KEY set and a valid dataset path, `POST /study/sessions` with `{"dataset_path": "<path>", "backend": "hdbscan"}` returns 200 with `session_id` and `study_url` fields. A DB row with `oracle_type='human'` is created.
result: pass

### 6. Study Page Three-Panel Layout
expected: Navigate to `/study/<session_id>` in a browser. Page renders three panels: left (cluster cards ~300px), center (UMAP mini-plots area, flex), right (feedback box ~280px). No browser console errors.
result: pass

### 7. Initial Clustering Renders
expected: After page load, cluster cards appear in the left panel (collapsible, with bold cluster name + italic description + expandable item list of up to 8 items each). Faceted per-cluster UMAP mini-plots render as canvases in the center panel sharing common bounds.
result: pass

### 8. Submit Feedback Updates State
expected: Type a free-text instruction into the feedback box and click Send. After processing, cluster cards update with new state, turn count increments, and a new audit_log.jsonl row is appended in the session directory. The DB receives a turn row with `action_type='human_feedback'`.
result: pass
note: "Fixed during UAT — f_next_state now validates cluster-referencing deltas against live state and skips invalid ones via deviation() (warn in prod, raise under STRICT_MODE) instead of crashing the worker thread. Turn counter added (header #turn-counter + turn_index in study_state payload + study.js render). User confirmed: cards update, Turn counter increments, session survives bad input. User noted unspecified design issues (see Open Design Notes)."
prior_result: issue (blocker) — worker thread AssertionError at src/agent_functions.py:368 froze the session

### 9. Satisfaction Detection Flow
expected: Submit feedback like "I'm satisfied with these clusters". The LLM satisfaction detector flags YES; a satisfaction banner appears with Yes/No confirmation buttons. Clicking Yes seals the session with `convergence_reason='oracle_satisfied'`; clicking No continues feedback.
result: pass

### 10. Session Cap at 15 Turns
expected: At turn 15 (`STUDY_MAX_TURNS`), the session auto-ends with `convergence_reason='turn_budget'`, the feedback input is disabled, and a "session complete" banner appears.
result: pass
note: "Cap changed from 30 to 15 during UAT per user request (web/app.py STUDY_MAX_TURNS default, tests/test_study_ui.py assertions, notebook Limitations note). User confirmed live: session auto-ended at turn 15 with disabled input + complete banner."

### 11. Notebook Structure Valid
expected: `notebooks/llm_vs_human.ipynb` opens in Jupyter (or `jupyter nbconvert --to script` succeeds). It has 8 cells (4 code, 4 markdown), including a "## Key Finding" section (placeholder tokens until data is collected) and a "## Limitations" section.
result: pass
note: Verified via JSON parse — 8 cells (4 code, 4 markdown), both "## Key Finding" and "## Limitations" present.

## Summary

total: 11
passed: 11
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "All 14 Phase 6 tests pass: tests/test_mapping.py and tests/test_study_ui.py complete green under 30 seconds"
  status: failed
  reason: "User reported: FAILED tests/test_mapping.py::test_llm_strategy_validates_cluster_id - ImportError: import error in src.mapping.anthropic: No module named 'src.mapping.anthropic'; 'src.mapping' is not a package. 1 failed, 13 passed in 15.82s"
  severity: major
  test: 1
  artifacts: []
  missing: []
  status_update: "RESOLVED during UAT — patched test to mock src.mapping.resolve_llm_key/build_client/chat instead of nonexistent src.mapping.anthropic path. 14/14 pass."

- truth: "Submitting human feedback applies it to the clustering and the study UI reflects the new state; the session survives feedback that the LLM parser cannot map to a valid cluster operation"
  status: failed
  reason: "Worker thread crashes with AssertionError at src/agent_functions.py:368 (_apply_merge: cluster_b_id 13 not found in state) when parse_feedback emits a MergeFeedback referencing a cluster id not present in the live state (LLM hallucination or id retired by an earlier delta in the same batch). Thread death freezes the session permanently. Boundary validation of LLM-proposed deltas is missing in f_next_state / study worker. Secondary: study.js renders no turn counter."
  severity: blocker
  test: 8
  artifacts: ["src/agent_functions.py:368", "src/agent_functions.py:507-508", "web/app.py:1096-1126", "src/feedback_parser.py", "web/static/study.js"]
  missing: ["delta validation against live cluster ids before _apply_merge/_apply_split", "turn counter in study UI"]
  status_update: "RESOLVED during UAT — f_next_state validates cluster-referencing deltas against live state, skipping invalid ones via deviation() (warn in prod, raise under STRICT_MODE). Turn counter wired (study.html + app.py payload + study.js). User confirmed end-to-end."

## Open Design Notes

User reported (Test 8, 2026-05-20): the study feedback flow works but has unspecified design issues. Capture details before finalizing the phase / human study. Not blocking UAT.
