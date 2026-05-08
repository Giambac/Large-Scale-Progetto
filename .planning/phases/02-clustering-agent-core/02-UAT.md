---
status: testing
phase: 02-clustering-agent-core
source:
  - .planning/phases/02-clustering-agent-core/02-01-SUMMARY.md
  - .planning/phases/02-clustering-agent-core/02-02-SUMMARY.md
  - .planning/phases/02-clustering-agent-core/02-03-SUMMARY.md
  - .planning/phases/02-clustering-agent-core/02-04-SUMMARY.md
  - .planning/phases/02-clustering-agent-core/02-05-SUMMARY.md
started: 2026-05-08T00:00:00Z
updated: 2026-05-08T00:00:00Z
---

## Current Test

number: 1
name: Cold Start Smoke Test
expected: |
  Kill any running Flask server. From the repo root, run:
    python web/app.py
  The server should boot without errors and show a line like:
    * Running on http://127.0.0.1:5000
  Then hit http://127.0.0.1:5000/status in a browser or curl — it should return JSON with {"status": "idle", ...}.
  No import errors, no crashes on startup.
awaiting: user response

## Tests

### 1. Cold Start Smoke Test
expected: Kill any running Flask server. From the repo root, run `python web/app.py`. Server should boot without errors and show a running address line. Hitting `/status` returns JSON `{"status": "idle", ...}`.
result: [pending]

### 2. Phase 2 pytest suite green
expected: Running `pytest tests/phase2/ -m "not llm" -q` in the repo root completes with **65 passed** (1 deselected for @pytest.mark.llm). No failures, no errors.
result: [pending]

### 3. Feedback data model — create and inspect all five types
expected: Running the following in a Python REPL completes without error and prints the expected values:
  ```python
  from src.feedback import SplitFeedback, MergeFeedback, MoveItemFeedback, GlobalFeedback, InstructionalFeedback, ORACLE_MOVE_CONFIDENCE
  print(ORACLE_MOVE_CONFIDENCE)  # 0.95
  sf = SplitFeedback(cluster_id=0)
  mf = MergeFeedback(cluster_a_id=0, cluster_b_id=1)
  mv = MoveItemFeedback(item_id=3, target_cluster_id=1)
  gf = GlobalFeedback(instruction_text="keep animals together")
  inf = InstructionalFeedback(instruction_text="ignore size")
  print("ok")  # ok
  ```
result: [pending]

### 4. f_uncertainty ranks boundary items and candidates
expected: Running the following prints non-empty ranked lists without error:
  ```python
  from src.uncertainty import f_uncertainty
  from src.state import ClusteringState, Cluster
  import numpy as np
  clusters = [Cluster(id=0, name="A", description="", item_ids=[0,1,2]), Cluster(id=1, name="B", description="", item_ids=[3,4,5])]
  soft_probs = {i: [0.6,0.4] if i<3 else [0.2,0.8] for i in range(6)}
  assignments = {i: 0 if i<3 else 1 for i in range(6)}
  state = ClusteringState(turn_index=0, session_id="t", clusters=clusters, assignments=assignments, soft_probs=soft_probs)
  report = f_uncertainty(state)
  print(len(report.boundary_items), len(report.split_candidates), len(report.merge_candidates))
  # prints: 6 2 1
  ```
result: [pending]

### 5. f_next_state split — K increases by 1
expected: Starting from a 2-cluster state and applying SplitFeedback on cluster 0 produces a 3-cluster state:
  ```python
  from src.feedback import SplitFeedback
  from src.agent_functions import f_next_state
  # (set up a 2-cluster ClusteringState with at least 2 items in cluster 0)
  # apply f_next_state with [SplitFeedback(cluster_id=0)]
  # new_state.clusters should have 3 entries
  # len(new_state.clusters) == 3
  ```
  No assertion error, no crash. New state has K=3 clusters.
result: [pending]

### 6. f_next_state merge — K decreases by 1
expected: Starting from a 3-cluster state and applying MergeFeedback on clusters 0 and 1 produces a 2-cluster state:
  ```python
  from src.feedback import MergeFeedback
  from src.agent_functions import f_next_state
  # apply f_next_state with [MergeFeedback(cluster_a_id=0, cluster_b_id=1)]
  # len(new_state.clusters) == 2
  ```
  No crash. New state has K=2 clusters.
result: [pending]

### 7. f_next_state move-item — K unchanged, soft_probs updated
expected: Applying MoveItemFeedback(item_id=0, target_cluster_id=1) to a 2-cluster state leaves K=2 but boosts item 0's confidence for cluster 1 to ~0.95:
  ```python
  from src.feedback import MoveItemFeedback
  from src.agent_functions import f_next_state
  # after f_next_state with [MoveItemFeedback(item_id=0, target_cluster_id=1)]
  # len(new_state.clusters) == 2  (K unchanged)
  # new_state.soft_probs[0][cluster_1_index] ≈ 0.95
  ```
  No crash. ORACLE_MOVE_CONFIDENCE (0.95) is visible in soft_probs.
result: [pending]

### 8. Global instructions accumulate across turns
expected: After two turns each with a GlobalFeedback delta, the `global_instructions` list passed in has 2 entries:
  ```python
  from src.feedback import GlobalFeedback
  from src.agent_functions import f_next_state
  global_instructions = []
  # turn 1: f_next_state(..., deltas=[GlobalFeedback("keep animals together")], global_instructions=global_instructions)
  # turn 2: f_next_state(..., deltas=[GlobalFeedback("ignore size")], global_instructions=global_instructions)
  assert len(global_instructions) == 2  # ["keep animals together", "ignore size"]
  print(global_instructions)
  ```
  List grows in-place; no crash.
result: [pending]

### 9. Conversation loop runs with MockOracle
expected: Running run_conversation() with a MockOracle that has satisfied=True on turn 1 completes in a single turn without error:
  ```python
  from src.oracle_protocol import MockOracle, OracleReply
  from src.conversation_loop import run_conversation
  # build minimal ClusteringState, MockOracle([OracleReply(raw_text='', satisfied=True)])
  # run_conversation(...) returns an AuditLog with at least 1 turn
  # no crash, no infinite loop
  ```
  Returns within a second. AuditLog JSONL is written to the specified log_path.
result: [pending]

### 10. Flask debug UI homepage loads
expected: With the Flask server running (`python web/app.py`), opening http://127.0.0.1:5000 in a browser shows:
  - A page title / header
  - A file upload form
  - A status banner showing "idle" or similar
  - A cluster cards area (empty at start, showing placeholder text)
  - A metrics sidebar
  No 404, no 500, no blank page.
result: [pending]

### 11. Flask upload accepts a CSV dataset
expected: With the server running, uploading a small CSV file (columns: text) via the upload form (or curl) to POST /upload returns HTTP 200 immediately (before the background task finishes). The status banner updates to "running" via WebSocket push shortly after.
  ```bash
  echo 'text\nhello world\nfoo bar\nbaz qux' > /tmp/test.csv
  curl -s -o /dev/null -w "%{http_code}" -F file=@/tmp/test.csv http://127.0.0.1:5000/upload
  # prints: 200
  ```
result: [pending]

## Summary

total: 11
passed: 0
issues: 0
pending: 11
skipped: 0
blocked: 0

## Gaps

[none yet]
