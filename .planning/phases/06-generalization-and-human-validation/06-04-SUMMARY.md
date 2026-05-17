---
phase: "06"
plan: "04"
subsystem: "tests"
tags: [tests, mapping, study-ui, pytest, monkeypatch, fastapi, test-client]
dependency_graph:
  requires:
    - src/mapping.py             # OracleRuleSet, MappingProtocol, LLMMappingStrategy, CentroidMappingStrategy, MAPPING_REGISTRY, extract_oracle_rules
    - src/state.py               # ClusteringState, Cluster
    - src/embedding_store.py     # EmbeddingStore
    - src/serialization.py       # serialize/deserialize_state (via load_audit_log in extract_oracle_rules)
    - web/app.py                 # FastAPI study routes, _study_sessions, STUDY_MAX_TURNS
    - src/db/connection.py       # connect(), init_schema() — monkeypatched in UI tests
    - src/db/experiments.py      # create() — monkeypatched in UI tests
  provides:
    - tests/test_mapping.py      # 7 unit tests for src/mapping.py
    - tests/test_study_ui.py     # 7 endpoint tests for study routes
  affects: []
tech_stack:
  added: []
  patterns:
    - pytest fixtures (tmp_path, monkeypatch, study_client, test_dataset)
    - fastapi.testclient.TestClient for endpoint tests (not asgi_app)
    - monkeypatch.setattr on module attributes for LLM and DB isolation
    - Structured JSONL fixtures (valid ClusteringState + extra "deltas" key) for audit log tests
    - Patch _run_study_background (not threading.Thread.start) to avoid breaking TestClient internals
key_files:
  created:
    - tests/test_mapping.py
    - tests/test_study_ui.py
  modified: []
decisions:
  - "Audit log test JSONL must include all ClusteringState fields (turn_index, timestamp, clusters, assignments, soft_probs) because load_audit_log() calls deserialize_state() on every line — extra keys like 'deltas' are ignored by deserialize_state"
  - "Do NOT patch threading.Thread.start globally in study UI tests — TestClient uses threading.Thread internally; global patch prevents the ASGI app worker from starting, causing test to hang indefinitely"
  - "Patch web.app._run_study_background to a no-op instead — the thread starts and immediately returns, keeping TestClient functional"
  - "monkeypatch src.db.connection.connect / src.db.connection.init_schema / src.db.experiments.create before the request so inline imports inside create_study_session pick up patched values"
metrics:
  duration: "~20 minutes"
  completed: "2026-05-17T12:30:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
---

# Phase 6 Plan 04: Tests — test_mapping.py and test_study_ui.py

**One-liner:** 14 pytest tests for the Phase 6 mapping layer and study UI routes, all isolated via monkeypatch with no live LLM or DB calls, completing in under 20 seconds.

## What Was Built

### tests/test_mapping.py (7 tests)

Unit tests for `src/mapping.py` covering the full export surface:

1. **test_oracle_rule_set_fields** — Constructs `OracleRuleSet` and verifies all four fields (`synonyms`, `focus_areas`, `exclusions`, `cluster_rules`) are stored correctly.

2. **test_mapping_registry_keys** — Asserts `set(MAPPING_REGISTRY) == {"llm", "centroid"}`.

3. **test_mapping_protocol_isinstance** — Verifies that both `LLMMappingStrategy()` and `CentroidMappingStrategy()` satisfy `isinstance(x, MappingProtocol)` (structural/runtime_checkable subtyping).

4. **test_extract_oracle_rules_empty_audit_log** — Writes a valid JSONL file with a global delta (type != "instructional"), patches `src.mapping.deviation`, calls `extract_oracle_rules`, and asserts: (a) all OracleRuleSet fields are empty lists, (b) `deviation("no_fb04_entries", ...)` was called exactly once.

5. **test_centroid_strategy_assigns_correctly** — Uses structured embeddings (cluster 0 items have high dim-0 values, cluster 1 items have high dim-1 values), monkeypatches `strategy._model.encode` to return a dim-0 vector, and asserts the result is `"0"`.

6. **test_centroid_strategy_empty_cluster_raises** — Builds state with an empty `cluster.item_ids`, asserts `ValueError` with message matching `"has no items"`.

7. **test_llm_strategy_validates_cluster_id** — Monkeypatches `anthropic.Anthropic` to return cluster ID `"99"`, asserts `ValueError` with message matching `"unknown cluster id"`.

### tests/test_study_ui.py (7 tests)

Endpoint tests for study routes in `web/app.py`:

1. **test_post_study_sessions_valid** — Monkeypatches `src.db.connection.connect` / `init_schema` / `src.db.experiments.create` and patches `web.app._run_study_background` to a no-op. Verifies 200 response with `session_id` and `study_url` fields.

2. **test_post_study_sessions_nonexistent_dataset** — Passes a non-existent path, asserts 400.

3. **test_post_study_sessions_invalid_backend** — Passes `backend="invalid"`, asserts 400.

4. **test_get_study_page_known_session** — Pre-populates `_study_sessions` with a fake entry, verifies 200 and `b"study"` in response body.

5. **test_get_study_page_unknown_session** — Requests unknown session ID, verifies 404.

6. **test_study_max_turns_env_default** — Verifies `STUDY_MAX_TURNS == 30` at module level (no env var set).

7. **test_study_max_turns_env_override** — Verifies `int(os.environ.get("STUDY_MAX_TURNS", "30"))` returns 5 when env var set to "5".

## Verification Results

1. `python -m pytest tests/test_mapping.py -v` — **7 passed** (20.6s)
2. `python -m pytest tests/test_study_ui.py -v` — **7 passed** (0.84s)
3. `python -m pytest tests/test_mapping.py tests/test_study_ui.py -v` — **14 passed** (19.5s)
4. No live Anthropic API calls — all LLM surfaces monkeypatched
5. `test_centroid_strategy_empty_cluster_raises` verifies the `ValueError("has no items")` from T-06-01-02

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] JSONL fixture required full ClusteringState schema**
- **Found during:** T-06-04-01 (test_extract_oracle_rules_empty_audit_log)
- **Issue:** The plan spec wrote a stub JSONL line with only `{"turn_index": 0, "deltas": [...]}`. But `load_audit_log()` (called first in `extract_oracle_rules`) deserializes every line via `deserialize_state()`, which asserts `"timestamp" in d` — causing `AssertionError` on malformed input.
- **Fix:** Wrote a valid minimal `ClusteringState` JSONL line (with all required fields) plus the extra `"deltas"` key. `deserialize_state` ignores unknown keys, so this is forward-compatible.
- **Files modified:** tests/test_mapping.py

**2. [Rule 3 - Blocking Issue] threading.Thread.start global patch breaks TestClient**
- **Found during:** T-06-04-02 (test_post_study_sessions_valid)
- **Issue:** The plan spec patches `threading.Thread.start` globally to prevent the background clustering worker from running. But `TestClient` (Starlette) itself runs the ASGI app in a worker thread — patching `Thread.start` to a no-op prevents the TestClient worker from starting, causing the test to hang indefinitely.
- **Fix:** Instead of patching `Thread.start`, patch `web.app._run_study_background` to a no-op lambda. The `threading.Thread` is created and started normally (so TestClient works), but the background function returns immediately.
- **Files modified:** tests/test_study_ui.py

## Known Stubs

None. Both test files are complete implementations with no placeholder content.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes. Test files only — no production code modified.

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| T-06-04-01 | tests/test_mapping.py — 7 unit tests for src/mapping.py | 5404514 |
| T-06-04-02 | tests/test_study_ui.py — 7 endpoint tests for study routes | ca37af7 |

## Self-Check

- [x] `tests/test_mapping.py` exists
- [x] `tests/test_study_ui.py` exists
- [x] Commit `5404514` exists in git log
- [x] Commit `ca37af7` exists in git log
- [x] 14/14 tests pass in combined run
- [x] No live LLM calls (all anthropic.Anthropic surfaces monkeypatched)
- [x] No live DB connections (src.db.connection.connect monkeypatched)
- [x] Run time: 19.5s (well under 30s limit)

## Self-Check: PASSED
