---
status: testing
phase: 06-generalization-and-human-validation
source:
  - 06-01-SUMMARY.md
  - 06-02-SUMMARY.md
  - 06-03-SUMMARY.md
  - 06-04-SUMMARY.md
  - 06-05-SUMMARY.md
started: 2026-05-19T00:00:00Z
updated: 2026-05-19T00:25:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 6
name: Study Page Three-Panel Layout
expected: |
  Navigate to `/study/<session_id>` in a browser. Page renders three panels: left (cluster cards ~300px), center (UMAP mini-plots area, flex), right (feedback box ~280px). No browser console errors.
awaiting: user response

## Tests

### 1. Test Suite Passes
expected: Running `python -m pytest tests/test_mapping.py tests/test_study_ui.py -v` from repo root completes with "14 passed" in under 30 seconds. No live LLM or DB calls (all monkeypatched).
result: issue
reported: "FAILED tests/test_mapping.py::test_llm_strategy_validates_cluster_id - ImportError: import error in src.mapping.anthropic: No module named 'src.mapping.anthropic'; 'src.mapping' is not a package. 1 failed, 13 passed in 15.82s"
severity: major

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
result: [pending]

### 7. Initial Clustering Renders
expected: After page load, cluster cards appear in the left panel (collapsible, with bold cluster name + italic description + expandable item list of up to 8 items each). Faceted per-cluster UMAP mini-plots render as canvases in the center panel sharing common bounds.
result: [pending]

### 8. Submit Feedback Updates State
expected: Type a free-text instruction into the feedback box and click Send. After processing, cluster cards update with new state, turn count increments, and a new audit_log.jsonl row is appended in the session directory. The DB receives a turn row with `action_type='human_feedback'`.
result: [pending]

### 9. Satisfaction Detection Flow
expected: Submit feedback like "I'm satisfied with these clusters". The LLM satisfaction detector flags YES; a satisfaction banner appears with Yes/No confirmation buttons. Clicking Yes seals the session with `convergence_reason='oracle_satisfied'`; clicking No continues feedback.
result: [pending]

### 10. Session Cap at 30 Turns
expected: At turn 30 (or `STUDY_MAX_TURNS` value), the session auto-ends with `convergence_reason='turn_budget'`, the feedback input is disabled, and a "session complete" banner appears.
result: [pending]

### 11. Notebook Structure Valid
expected: `notebooks/llm_vs_human.ipynb` opens in Jupyter (or `jupyter nbconvert --to script` succeeds). It has 8 cells (4 code, 4 markdown), including a "## Key Finding" section (placeholder tokens until data is collected) and a "## Limitations" section.
result: [pending]

## Summary

total: 11
passed: 4
issues: 1
pending: 6
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
