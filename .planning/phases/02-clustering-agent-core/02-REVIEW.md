---
phase: 02-clustering-agent-core
reviewed: 2026-05-07T00:00:00Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - src/agent_functions.py
  - src/conversation_loop.py
  - src/feedback.py
  - src/feedback_parser.py
  - src/hierarchy.py
  - src/oracle_protocol.py
  - src/strategy.py
  - src/uncertainty.py
  - tests/conftest.py
  - tests/phase2/test_agent_functions.py
  - tests/phase2/test_app.py
  - tests/phase2/test_conversation_loop.py
  - tests/phase2/test_feedback.py
  - tests/phase2/test_feedback_parser.py
  - tests/phase2/test_hierarchy.py
  - tests/phase2/test_oracle_protocol.py
  - tests/phase2/test_uncertainty.py
  - web/__init__.py
  - web/app.py
  - web/static/main.js
  - web/static/style.css
findings:
  critical: 4
  warning: 4
  info: 3
  total: 11
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-07T00:00:00Z
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

The core agent pipeline — feedback types, feedback parser, strategy, hierarchy, oracle protocol, uncertainty computation, agent functions, and the conversation loop — is largely well-structured and follows the project's fail-loudly philosophy. The JSONL audit log is written every turn, the frozen dataclasses model oracle intent correctly, and the hierarchy store tracks lineage with proper monotonic IDs.

Four BLOCKER-level correctness bugs were found. Three are in `agent_functions.py` and affect `_apply_move_item`, `_apply_merge`, and `_apply_split`. The fourth affects `f_uncertainty` and is reachable through any of the three state-mutation paths. None of these are caught by existing assertions or tests. All four can produce silently corrupted state — which is precisely what the project's coding philosophy says to avoid.

---

## Critical Issues

### CR-01: `_apply_move_item` silently corrupts state when source and target cluster are the same

**File:** `src/agent_functions.py:134`

**Issue:** When `feedback.target_cluster_id == state.assignments[feedback.item_id]` (a no-op move), the loop's `if/elif` structure removes the item from the source cluster via the `if c.id == source_cluster_id` branch but never adds it back, because the `elif c.id == feedback.target_cluster_id` branch is unreachable once the `if` branch has matched. The resulting state has `assignments[item_id]` pointing to a cluster whose `item_ids` list no longer contains that item. This invariant violation is silent: none of the `f_next_state` post-condition assertions check that every `assignments[item_id]` value appears in the corresponding `Cluster.item_ids` list.

This path is reachable from the LLM output: `parse_feedback` validates that `target_cluster_id` is an active cluster ID, but does not check whether it equals the item's current cluster.

**Fix:** Add an assertion or early return at the top of `_apply_move_item` before the cluster loop:

```python
assert feedback.target_cluster_id != source_cluster_id, (
    f"_apply_move_item: item {feedback.item_id} is already in cluster "
    f"{feedback.target_cluster_id} — no-op moves must be filtered before calling"
)
```

Or, if a no-op should be a silent pass (not fail-loudly), return the original state immediately:

```python
if source_cluster_id == feedback.target_cluster_id:
    return state  # already in target cluster — nothing to do
```

---

### CR-02: `_apply_merge` does not guard against merging a cluster with itself

**File:** `src/agent_functions.py:356`

**Issue:** Neither `_apply_merge` nor `parse_feedback` asserts that `cluster_a_id != cluster_b_id`. A hallucinating LLM can emit `{"type": "merge", "cluster_a_id": 0, "cluster_b_id": 0}`, which passes `_build_delta`'s validation (both IDs are in `valid_cluster_ids`). Inside `_apply_merge`, the single cluster is removed from `old_clusters_kept`, and `merged_item_ids = list(cluster_a.item_ids) + list(cluster_b.item_ids)` doubles every item ID in the merged cluster list. The resulting `Cluster.item_ids` contains duplicates, which corrupts all future item-count-based computations (`f_uncertainty`, `f_next_best_step` payloads in Phase 5, per-turn metric logging).

**Fix:** Add an assertion in `_apply_merge` immediately after the two `id_to_idx` assertions:

```python
assert delta.cluster_a_id != delta.cluster_b_id, (
    f"_apply_merge: cluster_a_id and cluster_b_id are identical ({delta.cluster_a_id}) "
    "— cannot merge a cluster with itself"
)
```

Optionally, also add this check in `_build_delta` in `feedback_parser.py`:

```python
if feedback_type == "merge":
    assert item["cluster_a_id"] != item["cluster_b_id"], (
        f"parse_feedback: merge cluster_a_id == cluster_b_id == {item['cluster_a_id']}"
    )
```

---

### CR-03: `_apply_split` produces degenerate K-Means when exactly one seed item is provided

**File:** `src/agent_functions.py:217`

**Issue:** When `delta.seed_item_ids` has exactly one element, the code pads the centroid array by duplicating it:

```python
if len(seed_embeddings) == 1:
    seed_embeddings = np.vstack([seed_embeddings, seed_embeddings])
```

This passes two identical initial centroids to `KMeans(n_clusters=2, init=centroids, n_init=1, random_state=0)`. K-Means with identical initial centroids produces degenerate output: all items receive label `0` or the assignment is arbitrary depending on distance tie-breaking. This means `items_b` may be empty after `km.fit()`, creating a `Cluster` with `item_ids=[]`. An empty cluster is then registered in the `HierarchyStore` as active and passed forward to `f_uncertainty`, which produces a `nan` centroid for that cluster (see CR-04).

**Fix:** Either reject single-seed input with an assertion, or fall back to k-means++ when only one seed is provided:

```python
if delta.seed_item_ids and len(delta.seed_item_ids) >= 2:
    seed_embeddings = np.array([store.get(sid) for sid in delta.seed_item_ids[:2]])
    centroids = seed_embeddings[:2]
    km = KMeans(n_clusters=2, init=centroids, n_init=1, random_state=0)
else:
    # 0 or 1 seeds — fall back to k-means++ (k-means++ handles 0 and 1 seed uniformly)
    km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
```

---

### CR-04: `f_uncertainty` produces `nan` centroids and undefined sort order for empty clusters

**File:** `src/uncertainty.py:67`

**Issue:** `_apply_move_item` can create empty clusters (when the last item is moved out of a single-item cluster) and `_apply_split` with a degenerate seed can create an empty sub-cluster (CR-03). Both are reachable in normal operation. Once an empty cluster exists in the state, `f_uncertainty` reaches this code:

```python
vecs = np.array([state.soft_probs[i] for i in cluster.item_ids])
cluster_centroids[cluster.id] = vecs.mean(axis=0)
```

For an empty `item_ids`, `np.array([])` produces a 1-D array of shape `(0,)`. `mean(axis=0)` on a 1-D empty array returns `nan` (a scalar, not an array). When this `nan` scalar is subsequently used in `np.linalg.norm(centroid_a - centroid_b)`, it propagates as `nan`. `merge_candidates.sort(key=lambda x: x[2])` on a list containing `nan` values produces undefined ordering in CPython (nan comparisons are not transitive), meaning `merge_candidates` is silently not sorted correctly. Any strategy that consumes `UncertaintyReport.merge_candidates` receives a semantically incorrect ranking.

**Fix:** Guard against empty clusters at the top of the centroid-computation loop:

```python
for cluster in state.clusters:
    assert len(cluster.item_ids) > 0, (
        f"f_uncertainty: cluster {cluster.id} has no items — "
        "empty clusters must be removed before f_uncertainty is called"
    )
    vecs = np.array([state.soft_probs[i] for i in cluster.item_ids])
    cluster_centroids[cluster.id] = vecs.mean(axis=0)
```

The assertion surfaces the upstream bug (empty cluster created by move/split) immediately rather than producing a silent bad sort.

---

## Warnings

### WR-01: `SplitFeedback` and `MoveItemFeedback` frozen dataclasses contain mutable `list` fields

**File:** `src/feedback.py:28`

**Issue:** `SplitFeedback.seed_item_ids` is declared as `list[int]` inside a `frozen=True` dataclass. `frozen=True` prevents reassigning the field reference, but does not prevent mutating the list contents. Any caller holding a reference to the `seed_item_ids` list can call `.append()`, `.clear()`, or `.__setitem__()` on it and silently alter the feedback object. This violates the immutability contract implied by `frozen=True`.

**Fix:** Use `tuple[int, ...]` instead of `list[int]`, or convert to a tuple in `_build_delta` before constructing `SplitFeedback`:

```python
@dataclass(frozen=True)
class SplitFeedback:
    cluster_id: int
    seed_item_ids: tuple[int, ...]  # immutable; use tuple not list
```

In `_build_delta` (feedback_parser.py line 80):
```python
return SplitFeedback(
    cluster_id=item["cluster_id"],
    seed_item_ids=tuple(item["seed_item_ids"]),
)
```

---

### WR-02: `_apply_split` does not validate that `seed_item_ids` elements belong to the target cluster

**File:** `src/agent_functions.py:214`

**Issue:** `parse_feedback` validates that cluster IDs in `SplitFeedback` exist in the current state but does not validate that `seed_item_ids` items belong to the cluster being split. `_apply_split` then passes these seed item IDs directly to `store.get()` to build initial KMeans centroids. If the LLM hallucinates seed items from a different cluster, the centroids are initialized outside the target cluster's embedding subspace, producing a semantically incorrect split without any crash or warning. This violates the fail-loudly principle: wrong centroids produce wrong sub-clusters silently.

**Fix:** Add a membership assertion in `_apply_split` before building `seed_embeddings`:

```python
if delta.seed_item_ids:
    target_item_set = set(target.item_ids)
    for sid in delta.seed_item_ids[:2]:
        assert sid in target_item_set, (
            f"_apply_split: seed_item_id {sid} is not in cluster {delta.cluster_id} "
            f"(items: {target.item_ids})"
        )
    seed_embeddings = np.array([store.get(sid) for sid in delta.seed_item_ids[:2]])
```

---

### WR-03: Concurrent `/upload` requests corrupt shared session state without locking

**File:** `web/app.py:61`

**Issue:** `_session` is a module-level mutable dict. When two POST `/upload` requests arrive concurrently (possible with `async_mode='threading'`), both handlers reset `_session["state"]` and `_session["task"]` and each starts a background task. Both background tasks then write to `_session["state"]` (lines 109 and 129), to the same file `embeddings/session_embeddings.npy` (line 107), and to the same `audit_log.jsonl` path. These are unsynchronized concurrent writes. The second task's `_session["state"]` update can be overwritten by the first task, and the audit log can have interleaved entries from two separate sessions.

This is documented as "single session per server run, D-15" but there is no enforcement mechanism (no lock, no semaphore) to actually prevent concurrent sessions.

**Fix:** Add a threading lock around the session reset and task start in `upload_dataset`, and check whether a task is already running before starting a new one:

```python
import threading
_session_lock = threading.Lock()

@app.route("/upload", methods=["POST"])
def upload_dataset():
    ...
    with _session_lock:
        _session["state"] = None
        _session["task"] = None
        _session["task"] = socketio.start_background_task(...)
    ...
```

---

### WR-04: `try/except` with immediate `raise` in `parse_feedback` is redundant dead code

**File:** `src/feedback_parser.py:155`

**Issue:**

```python
try:
    raw_items = json.loads(cleaned_text)
except json.JSONDecodeError:
    raise  # fail loudly — do not swallow
```

A `try/except` that unconditionally re-raises the caught exception is semantically identical to no `try` block at all. This pattern violates the spirit of the project's "fail loudly" rule (which prohibits defensive `try/except` chains): it wraps an operation in exception handling machinery and then immediately undoes it. It also adds noise that could mislead a future maintainer into thinking the exception is being handled.

**Fix:** Remove the `try/except` entirely:

```python
raw_items = json.loads(cleaned_text)
```

---

## Info

### IN-01: `test_30_turn_loop_state_integrity_at_turn_10` may not test turn 10

**File:** `tests/phase2/test_conversation_loop.py:75`

**Issue:** The test uses `states[min(10, len(states) - 1)]` to access the state at turn 10. If the loop stops before producing 11 entries (which is possible if the oracle's `satisfied=True` reply fires early or if the turn budget is overridden), `min(10, len(states) - 1)` silently returns an index less than 10, and the test asserts the invariant against a different turn. The test name implies a specific turn is being checked, but the assertion may run against turn 3 or turn 7 without any signal to the developer.

**Fix:** Assert the expected index is available:

```python
assert len(states) > 10, f"Expected at least 11 turns in audit log, got {len(states)}"
state_at_10 = states[10]
assert len(state_at_10.assignments) == 6
```

---

### IN-02: `InstructionalFeedback` is silently discarded with `pass` — no list accumulator

**File:** `src/agent_functions.py:497`

**Issue:** `GlobalFeedback` appends to `global_instructions` (an explicit accumulator owned by the caller). `InstructionalFeedback` is handled with a bare `pass`, which discards the `instruction_text` entirely. The asymmetry between these two similar feedback types means any oracle utterance parsed as `InstructionalFeedback` is lost with no record in state, the audit log, or any accumulator. The comment says "Phase 3 storage", but unlike the `GlobalFeedback` design, there is no placeholder list to hold these instructions until Phase 3.

**Fix:** Create a parallel `instructional_instructions: list[str]` parameter (matching the `global_instructions` pattern) so the data is preserved even if unused:

```python
def f_next_state(
    ...
    instructional_instructions: list[str] | None = None,
) -> ClusteringState:
    if instructional_instructions is None:
        instructional_instructions = []
    ...
    elif isinstance(delta, InstructionalFeedback):
        instructional_instructions.append(delta.instruction_text)
```

---

### IN-03: Whitespace-only oracle reply triggers unnecessary LLM call in `run_conversation`

**File:** `src/conversation_loop.py:121`

**Issue:** The guard at line 121 is `reply.raw_text.strip()`, which prevents calling `parse_feedback` with whitespace-only text when `llm_client is None`. However, when `llm_client is not None` and `reply.raw_text` is whitespace-only (e.g. `" "`), `parse_feedback` is called. `parse_feedback`'s own fast-path guard is `if not raw_text` (line 132), which evaluates the original non-stripped string — so `" "` is truthy and the fast path is skipped. The LLM is then invoked with a whitespace-only oracle utterance, receiving a malformed prompt and likely returning `[]`. This is an unnecessary API call that incurs latency and token cost.

**Fix:** Move the strip check inside `parse_feedback` for consistency:

```python
def parse_feedback(raw_text: str, state: ClusteringState, client: object) -> list[FeedbackDelta]:
    if not raw_text or not raw_text.strip():
        return []
    ...
```

---

_Reviewed: 2026-05-07T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
