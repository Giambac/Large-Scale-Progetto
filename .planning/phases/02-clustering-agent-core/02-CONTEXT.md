# Phase 2: Clustering Agent Core - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the Clustering Agent's pure functions (`f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state`), the structured oracle feedback data model, the feedback parser, the plain-Python conversation loop, and the web debug UI. By the end of this phase a mocked oracle loop runs correctly for 30+ turns with state integrity verified, and a developer can observe the running session through a Flask web interface.

</domain>

<decisions>
## Implementation Decisions

### Orchestrator Loop

- **D-01:** Use a **plain Python `while` loop** — no LangGraph. Simple sequential loop: `f_next_best_step` → oracle → `parse_feedback` → `f_next_state` → `check_stopping` → JSONL write.
- **D-02:** The orchestrator lives in a **standalone module** (`agent.py` or `conversation_loop.py`). The `f_*` functions are imported and called from it; they are not methods on an agent class.
- **D-03:** Oracle connection via **`OracleProtocol`** (ABC or `typing.Protocol` with a `.reply(state, message) → OracleReply` method). Phase 2 ships a `MockOracle` that returns deterministic scripted replies. Phase 3 replaces it with the real Oracle Agent without touching the loop.
- **D-04:** The **loop owns the JSONL write**. After each `f_next_state` call the loop serializes the new `ClusteringState` to the AuditLog. The `f_*` functions are pure — no I/O side effects.

### Feedback Data Model

- **D-05:** Oracle feedback is parsed by a **separate `feedback_parser.py` module**: `parse_feedback(raw_text: str, state: ClusteringState) → list[FeedbackDelta]`. This is the only place that calls the LLM for parsing. `f_next_state` receives only structured objects.
- **D-06:** **Separate dataclass per feedback type** — type-safe by construction, impossible to create an invalid delta:
  ```python
  @dataclass class SplitFeedback:      cluster_id: int
  @dataclass class MergeFeedback:      cluster_a_id: int; cluster_b_id: int
  @dataclass class MoveItemFeedback:   item_id: int; target_cluster_id: int
  @dataclass class GlobalFeedback:     instruction_text: str
  @dataclass class InstructionalFeedback: instruction_text: str
  FeedbackDelta = SplitFeedback | MergeFeedback | MoveItemFeedback | GlobalFeedback | InstructionalFeedback
  ```
- **D-07:** Compound oracle messages produce multiple `FeedbackDelta` objects applied in **type-priority order**: global → cluster-level (split/merge) → point-level (move_item) → instructional. This matches D-10's weighting hierarchy from Phase 1.

### Re-clustering Mechanics in f_next_state

- **D-08:** **Split:** Oracle-seeded K-means. If the oracle provides representative item IDs (e.g., "item 42 in one group, item 87 in the other"), look up their embeddings from `EmbeddingStore` and pass as `init` to `sklearn.cluster.KMeans(n_clusters=2, init=centroids)`. If no seeds provided, fall back to k-means++ initialization (`init='k-means++'`). The original cluster's ID is retired; two new cluster IDs are assigned from the monotonic counter.
- **D-09:** **Merge:** Column pooling on `soft_probs`. For each item `i`: `soft_probs[i][new_id] = soft_probs[i][A] + soft_probs[i][B]`, drop columns A and B, re-normalize the row. O(N), no re-clustering. Both old IDs are retired; one new cluster ID is assigned from the monotonic counter.
- **D-10:** **Point move:** `soft_probs[item_id][target_cluster_id] = 0.95`. Remaining `0.05` redistributed **proportionally** to the original soft_probs values of all other clusters (excluding the target). Preserves relative uncertainty ordering among non-target clusters.
- **D-11:** **Cluster IDs:** Monotonic counter — old IDs are **never reused** after split or merge (D-11 from Phase 1). If current max cluster ID is 7 and a split fires, old cluster is retired and two new clusters get IDs 8 and 9.
- **D-12:** **Re-naming:** `f_next_state` triggers LLM cluster naming **for affected clusters only** — both sub-clusters after a split, the merged cluster after a merge, and source + target clusters after a point move. Unaffected clusters keep their existing names.

### Debug Web UI (UI-01, UI-02)

- **D-13:** Framework: **Flask + vanilla HTML**. WebSocket (via `flask-socketio` or similar) for live turn-by-turn updates as the conversation loop runs.
- **D-14:** Layout: **Cluster cards + metrics sidebar**. One card per cluster showing name, description, item count, top-5 items by confidence with soft probability bars. Sidebar shows conversation history (turn-by-turn oracle feedback and system replies) and per-turn metrics (cognitive-load score, contradiction count, convergence signal).
- **D-15:** Dataset upload (UI-02): **single session per server run**. Uploading a new dataset starts a fresh session (current ClusteringState cleared, AuditLog flushed to disk). Multi-session management (chat-history-style switching) is deferred to v2 alongside full persistence (UI-V2-01).

### Claude's Discretion

- Exact `OracleReply` schema (fields beyond `satisfied: bool` and `raw_text: str`) — planner decides based on Phase 3 Oracle Agent contract.
- `f_uncertainty` scoring formula — how to weight entropy, boundary proximity, split/merge candidacy into a ranked list. Researcher/planner decides.
- `f_next_best_step` `RandomStrategy` implementation — random uniform selection over valid actions is sufficient for Phase 2.
- Flask port and serving configuration for the debug UI.
- WebSocket library choice (`flask-socketio` vs `simple-websocket`) — planner decides.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning Artifacts
- `.planning/REQUIREMENTS.md` — CLUS-01 through CLUS-04, FB-01 through FB-03, HIER-01, HIER-02, UI-01, UI-02 define the 11 Phase 2 requirements in full
- `.planning/ROADMAP.md` §Phase 2 — 6 success criteria that define done
- `.planning/PROJECT.md` — fail-loudly coding philosophy, architecture constraints, dataset-agnostic constraint
- `.planning/phases/01-pre-code-obligations-and-foundation/01-CONTEXT.md` — D-11 (cluster ID never reused), D-12 (EmbeddingStore read-only), D-13 (ClusteringState schema frozen), D-10 (feedback weighting hierarchy) all carry forward into Phase 2

### Phase 1 Source Code (integration points)
- `src/state.py` — `ClusteringState` and `Cluster` dataclasses. Schema is FROZEN — no field additions.
- `src/clustering.py` — `run_hdbscan()`, `assign_noise_to_nearest()`, `build_initial_clustering_state()`. Phase 2 reads embeddings from `EmbeddingStore` but does NOT re-run the full pipeline per turn.
- `src/embedding_store.py` — Read-only store; Phase 2 reads embeddings for K-means centroid seeding (D-08).
- `src/stopping.py` — `check_stopping()`, `StopReason`, `StoppingCriteria`. Phase 2 calls this at the end of each loop iteration.
- `src/serialization.py` — Used by the loop to write JSONL AuditLog each turn (D-04).
- `src/cluster_naming.py` — `ClusterNamer` reused by `f_next_state` for re-naming affected clusters (D-12).

### Reference Scripts (review before implementing)
- `Conversational Clustering Script.txt` — prototype/reference; review for existing design patterns before writing the orchestrator loop and `f_*` functions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ClusteringState` / `Cluster` (src/state.py): The canonical state contract. Phase 2 reads and writes it; never mutates the schema.
- `ClusterNamer` (src/cluster_naming.py): Reused directly in `f_next_state` for post-split/merge/move cluster naming (D-12).
- `check_stopping()` (src/stopping.py): Called by the orchestrator loop each turn. Returns `StopReason` or `None`.
- `serialize_state()` / `deserialize_state()` (src/serialization.py): Used by the loop for JSONL AuditLog writes (D-04).
- `EmbeddingStore.get(item_id)` (src/embedding_store.py): Used in `f_next_state` split path to retrieve centroid vectors for oracle-seeded K-means (D-08).

### Established Patterns
- **Fail loudly:** No `try/except` outside CLI entry point and LLM API calls. Use `assert` freely to enforce state invariants (complete assignments, valid cluster IDs, soft_probs summing to 1.0). A crash is preferable to silent wrong state.
- **Pure functions:** `f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state` take state as input and return new state — no I/O, no global mutation. All I/O (JSONL write, LLM calls) lives in the orchestrator loop or dedicated modules.
- **Dataset-agnostic:** No hardcoded field names or dataset-specific logic in the agent or `f_*` functions.

### Integration Points
- `EmbeddingStore` → `f_next_state` (split path): read-only lookup of item embeddings for K-means initialization.
- `ClusterNamer` → `f_next_state`: called after split/merge/move to re-name affected clusters.
- `check_stopping()` → orchestrator loop: called each turn to detect loop termination.
- `serialization.py` → orchestrator loop: JSONL write after each `f_next_state` call.
- `OracleProtocol` → Phase 3 boundary: `MockOracle` in Phase 2; real Oracle Agent in Phase 3 implements the same interface.

</code_context>

<specifics>
## Specific Ideas

- **Oracle-seeded K-means:** The oracle can guide splits by naming representative items. `parse_feedback` should extract these item references from the oracle's raw text and include them in `SplitFeedback` (e.g., `SplitFeedback(cluster_id=3, seed_item_ids=[42, 87])`).
- **0.95 soft override for point moves:** The constant `0.95` should be a named module-level constant (`ORACLE_MOVE_CONFIDENCE = 0.95`) — not an inline magic number.
- **Type-priority ordering in `f_next_state`:** The ordering (global → cluster → point → instructional) should match the `FeedbackMagnitudeWeights` hierarchy from `stopping.py` exactly — same ordering, documented as intentional alignment.
- **WebSocket live updates:** Each turn the loop emits a WebSocket event with the serialized `ClusteringState` + latest turn metrics. The Flask UI re-renders the cluster cards and metrics sidebar on receipt.

</specifics>

<deferred>
## Deferred Ideas

- **Multi-session management (chat-history style):** Switching between session histories in the UI while the server runs. Deferred to v2 alongside full session persistence (UI-V2-01).
- **Oracle cognitive-load weight parameters:** Phase 3 decision — the Oracle Agent is the one that computes and receives the cognitive-load score.
- **Exact `ε` and `N_fallback` values for stopping criteria:** Phase 4 decision (Judge Agent).
- **`InformationGain` strategy for `f_next_best_step`:** Phase 5 ablation scope.

</deferred>

---

*Phase: 2-Clustering-Agent-Core*
*Context gathered: 2026-05-07*
