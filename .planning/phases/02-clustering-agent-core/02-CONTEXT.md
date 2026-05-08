# Phase 2: Clustering Agent Core - Context

**Gathered:** 2026-05-07 (v1) / 2026-05-08 (Trio update)
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the Clustering Agent's pure functions (`f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state`), the structured oracle feedback data model, the feedback parser, the plain-Python conversation loop, and the web debug UI. Support multiple clustering backends (HDBSCAN and k-means) selectable at startup. Add a 2D UMAP projection panel to the web UI with per-turn color updates. Make sessions persistent across server restarts with a session-list UI for switching between past runs.

</domain>

<decisions>
## Implementation Decisions

### Orchestrator Loop (v1)

- **D-01:** Use a **plain Python `while` loop** — no LangGraph. Simple sequential loop: `f_next_best_step` → oracle → `parse_feedback` → `f_next_state` → `check_stopping` → JSONL write.
- **D-02:** The orchestrator lives in a **standalone module** (`agent.py` or `conversation_loop.py`). The `f_*` functions are imported and called from it; they are not methods on an agent class.
- **D-03:** Oracle connection via **`OracleProtocol`** (ABC or `typing.Protocol` with a `.reply(state, message) → OracleReply` method). Phase 2 ships a `MockOracle` that returns deterministic scripted replies. Phase 3 replaces it with the real Oracle Agent without touching the loop.
- **D-04:** The **loop owns the JSONL write**. After each `f_next_state` call the loop serializes the new `ClusteringState` to the AuditLog. The `f_*` functions are pure — no I/O side effects.

### Feedback Data Model (v1)

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

### Re-clustering Mechanics in f_next_state (v1)

- **D-08:** **Split:** Oracle-seeded K-means. If the oracle provides representative item IDs (e.g., "item 42 in one group, item 87 in the other"), look up their embeddings from `EmbeddingStore` and pass as `init` to `sklearn.cluster.KMeans(n_clusters=2, init=centroids)`. If no seeds provided, fall back to k-means++ initialization (`init='k-means++'`). The original cluster's ID is retired; two new cluster IDs are assigned from the monotonic counter.
- **D-09:** **Merge:** Column pooling on `soft_probs`. For each item `i`: `soft_probs[i][new_id] = soft_probs[i][A] + soft_probs[i][B]`, drop columns A and B, re-normalize the row. O(N), no re-clustering. Both old IDs are retired; one new cluster ID is assigned from the monotonic counter.
- **D-10:** **Point move:** `soft_probs[item_id][target_cluster_id] = 0.95`. Remaining `0.05` redistributed **proportionally** to the original soft_probs values of all other clusters (excluding the target). Preserves relative uncertainty ordering among non-target clusters.
- **D-11:** **Cluster IDs:** Monotonic counter — old IDs are **never reused** after split or merge (D-11 from Phase 1). If current max cluster ID is 7 and a split fires, old cluster is retired and two new clusters get IDs 8 and 9.
- **D-12:** **Re-naming:** `f_next_state` triggers LLM cluster naming **for affected clusters only** — both sub-clusters after a split, the merged cluster after a merge, and source + target clusters after a point move. Unaffected clusters keep their existing names.

### Debug Web UI (v1)

- **D-13:** Framework: **Flask + vanilla HTML**. WebSocket (via `flask-socketio` or similar) for live turn-by-turn updates as the conversation loop runs.
- **D-14:** Layout: **Cluster cards + metrics sidebar**. One card per cluster showing name, description, item count, top-5 items by confidence with soft probability bars. Sidebar shows conversation history (turn-by-turn oracle feedback and system replies) and per-turn metrics (cognitive-load score, contradiction count, convergence signal).
- **D-15:** Dataset upload (UI-02): **single session per server run** for v1. Uploading a new dataset starts a fresh session (current ClusteringState cleared, AuditLog flushed to disk). Multi-session management superseded by D-25 (UI-V2-01).

---

### Clustering Backends — BACK-V2-01 (Trio)

- **D-16:** **`ClusteringBackend` Protocol** — define a `ClusteringBackend` Protocol (or ABC) with a `fit(embeddings: np.ndarray) → tuple[np.ndarray, np.ndarray]` method returning `(labels, soft_probs)`. `HDBSCANBackend` and `KMeansBackend` both implement it. `build_initial_clustering_state` accepts a `backend: ClusteringBackend` instance instead of calling `run_hdbscan` directly. Easy to add future backends without touching the orchestrator.
- **D-17:** **K initialization for k-means via BIC on GMM** — fit `sklearn.mixture.GaussianMixture` for K = 2…sqrt(N) (where N is dataset size), pick the K that minimizes BIC. This happens once at startup before the conversation loop begins. K is then fixed — changes only through oracle intent (no ongoing automatic optimization). The chosen K is logged to the AuditLog so runs are reproducible.
- **D-18:** **Backend selection via `--backend` CLI flag** — `python web/app.py --backend hdbscan|kmeans`. Default is `hdbscan`. Fails loudly (assertion) if an unknown value is passed.

### k-means Soft Probabilities — BACK-V2-01 (Trio)

- **D-19:** **Softmax of negative centroid distances** — `soft_probs[i][c] = softmax(-dist(item_i, centroid_c) / T)` for all clusters `c`. Produces a proper probability distribution that preserves uncertainty signal at cluster boundaries (equidistant items get ~0.5/0.5). Uses L2 distance to centroids.
- **D-20:** **Fixed temperature T = 1.0** (raw distances, no scaling). Named constant `KMEANS_SOFTMAX_TEMP = 1.0` in `clustering.py`. Not a CLI parameter.

### 2D Projection — VIZ-V2-01 (Trio)

- **D-21:** **UMAP (umap-learn)** for dimensionality reduction. Set `random_state` for reproducibility across runs. No t-SNE alternative — single library, single code path.
- **D-22:** **Projection recomputed on cluster-count change only** (split or merge events). Not recomputed on point moves (assignments change but embedding positions don't). Not recomputed every turn (too expensive and disorienting). Initial projection computed once after the first clustering.
- **D-23:** **Full-width panel above the two columns** — scatter plot sits above the cluster cards + metrics sidebar. Both the viz and existing panels remain visible at all times. No layout restructuring required.
- **D-24:** **Server-side UMAP, coords via SocketIO, client renders with `<canvas>`** — UMAP computes 2D coordinates server-side and emits them as JSON over SocketIO alongside the `state_update` event (or a dedicated `projection_update` event on cluster-count change). Vanilla JS draws points on a `<canvas>` element. No new JS dependencies — consistent with D-13's vanilla HTML stack.

### Persistent Sessions — UI-V2-01 (Trio)

- **D-25:** **Multiple named sessions** — each dataset upload creates a new session. Sessions are independent; the server can switch between them without losing state.
- **D-26:** **Timestamp-based session directories** — `sessions/2026-05-08T14-32-00/` containing `state.json` (latest ClusteringState snapshot) + `audit_log.jsonl` (full turn history) + `embeddings.npy` (embedding vectors). Human-readable, inspectable, no database required.
- **D-27:** **Persist ClusteringState + AuditLog + embeddings** per session. `state.json` is written after every turn (same timing as the AuditLog JSONL write, D-04). On resume, load `state.json` directly — no AuditLog replay needed.
- **D-28:** **Session list in the metrics sidebar** — the existing sidebar gains a "Sessions" section listing past sessions (timestamp + cluster count + turn count). Clicking a session loads it. Minimal layout change; the sidebar already exists (D-14).

### Claude's Discretion

- Exact `OracleReply` schema (fields beyond `satisfied: bool` and `raw_text: str`) — planner decides based on Phase 3 Oracle Agent contract.
- `f_uncertainty` scoring formula — how to weight entropy, boundary proximity, split/merge candidacy into a ranked list. Researcher/planner decides.
- `f_next_best_step` `RandomStrategy` implementation — random uniform selection over valid actions is sufficient for Phase 2.
- Flask port and serving configuration for the debug UI.
- WebSocket library choice (`flask-socketio` vs `simple-websocket`) — planner decides.
- UMAP hyperparameters (`n_neighbors`, `min_dist`) — planner decides reasonable defaults for 768-dim text embeddings.
- Canvas rendering details (point size, opacity encoding for soft-assignment confidence) — planner decides.
- `/resume` endpoint response shape (JSON payload sent to client on session load) — planner decides.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning Artifacts
- `.planning/REQUIREMENTS.md` — CLUS-01 through CLUS-04, FB-01 through FB-03, HIER-01, HIER-02, UI-01, UI-02 (v1 requirements); BACK-V2-01, VIZ-V2-01, UI-V2-01 (Trio requirements)
- `.planning/ROADMAP.md` §Phase 2 — success criteria including Trio additions (UMAP visualization, persistent sessions, k-means backend)
- `.planning/PROJECT.md` — fail-loudly coding philosophy, architecture constraints, dataset-agnostic constraint
- `.planning/phases/01-pre-code-obligations-and-foundation/01-CONTEXT.md` — D-11 (cluster ID never reused), D-12 (EmbeddingStore read-only), D-13 (ClusteringState schema frozen), D-10 (feedback weighting hierarchy) all carry forward into Phase 2

### Phase 2 v1 Source Code (integration points for Trio additions)
- `src/state.py` — `ClusteringState` and `Cluster` dataclasses. Schema is FROZEN.
- `src/clustering.py` — `run_hdbscan()`, `build_initial_clustering_state()`. BACK-V2-01 refactors this to use `ClusteringBackend` Protocol (D-16).
- `src/embedding_store.py` — Read-only store; used by k-means centroid seeding (D-08) and UMAP projection (D-21).
- `src/stopping.py` — `check_stopping()`, `StopReason`, `StoppingCriteria`.
- `src/serialization.py` — Used by the loop to write JSONL AuditLog each turn (D-04). UI-V2-01 adds `state.json` write using same pattern.
- `src/cluster_naming.py` — `ClusterNamer` reused by `f_next_state` for re-naming affected clusters (D-12).
- `web/app.py` — Flask + SocketIO server. VIZ-V2-01 adds projection emit; UI-V2-01 adds session save/load and `/resume` endpoint.
- `web/templates/index.html` — Existing two-column layout. VIZ-V2-01 adds full-width `<canvas>` panel above; UI-V2-01 adds session list to sidebar.
- `web/static/main.js` — Existing SocketIO client. Extend for `projection_update` event and session-switch clicks.

### Reference Scripts
- `Conversational Clustering Script.txt` — prototype/reference; review for existing design patterns before writing the orchestrator loop and `f_*` functions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ClusteringState` / `Cluster` (src/state.py): The canonical state contract. Phase 2 reads and writes it; never mutates the schema.
- `ClusterNamer` (src/cluster_naming.py): Reused directly in `f_next_state` for post-split/merge/move cluster naming (D-12).
- `check_stopping()` (src/stopping.py): Called by the orchestrator loop each turn. Returns `StopReason` or `None`.
- `serialize_state()` / `deserialize_state()` (src/serialization.py): Used by the loop for JSONL AuditLog writes (D-04). Also reuse for `state.json` snapshot writes (D-27).
- `EmbeddingStore.get(item_id)` / `EmbeddingStore.get_all()` (src/embedding_store.py): Used in split path (D-08) and for UMAP input (D-21).
- `socketio.emit()` instance method in `web/app.py`: Pattern for background-thread safe SocketIO events. Extend for `projection_update` events (D-24).

### Established Patterns
- **Fail loudly:** No `try/except` outside CLI entry point and LLM API calls. Use `assert` freely to enforce state invariants.
- **Pure functions:** `f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state` take state as input and return new state — no I/O, no global mutation.
- **Dataset-agnostic:** No hardcoded field names or dataset-specific logic.
- **`soft_probs` as dict-of-dicts** in SocketIO events: `{str(item_id): {str(cluster_id): prob}}` — positional lists break after split/merge.

### Integration Points
- `ClusteringBackend.fit()` → `build_initial_clustering_state`: new Protocol replaces direct `run_hdbscan()` call (D-16).
- `GaussianMixture` BIC loop → `KMeansBackend.__init__`: K selected before `fit()` is called (D-17).
- `EmbeddingStore.get_all()` → `umap.UMAP().fit_transform()`: projection computed from full embedding matrix (D-21).
- `projection_update` SocketIO event → `<canvas>` JS handler: emitted after initial clustering and after each split/merge (D-22, D-24).
- Session save → `sessions/<timestamp>/` directory: written after every `f_next_state` call alongside AuditLog (D-26, D-27).
- `/resume` Flask endpoint → session load: reads `state.json` + emits full state to reconnected client (D-27).

</code_context>

<specifics>
## Specific Ideas

- **BIC search range:** K = 2…sqrt(N). For 12K items that's up to ~110 GMM fits — potentially slow. Planner should consider caching or a tighter ceiling if startup time is a concern, but the user explicitly chose sqrt(N).
- **KMEANS_SOFTMAX_TEMP = 1.0:** Named module-level constant in `clustering.py`, not an inline magic number. Same convention as `ORACLE_MOVE_CONFIDENCE = 0.95`.
- **UMAP random_state:** Set a fixed `random_state` integer (e.g., 42) so projections are reproducible across runs on the same dataset.
- **Projection recompute trigger:** The `f_next_state` return value already signals what type of feedback was applied. The orchestrator loop can check `isinstance(delta, (SplitFeedback, MergeFeedback))` to decide whether to recompute UMAP.
- **Session timestamp format:** Use `datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")` — colons are replaced with hyphens for filesystem compatibility.
- **`state.json` write:** Use the existing `serialize_state()` from `serialization.py` to produce a JSON-serializable dict, then `json.dump` to `sessions/<ts>/state.json`.

</specifics>

<deferred>
## Deferred Ideas

- **Multi-session management (chat-history style):** Now implemented in UI-V2-01 (D-25–D-28). No longer deferred.
- **Oracle cognitive-load weight parameters:** Phase 3 decision — the Oracle Agent is the one that computes and receives the cognitive-load score.
- **Exact `ε` and `N_fallback` values for stopping criteria:** Phase 4 decision (Judge Agent).
- **`InformationGain` strategy for `f_next_best_step`:** Phase 5 ablation scope.
- **LLM-first clustering backend (BACK-V2-01 stretch):** BACK-V2-01 notes "LLM-first optional if time allows." k-means is the priority; LLM-based clustering is stretch and deferred if time is short.
- **BACK-V2-02 (representation choice):** Exposing raw features vs. sentence embeddings vs. fine-tuned embeddings as a config parameter. Deferred to later phase.

</deferred>

---

*Phase: 2-Clustering-Agent-Core*
*Context gathered: 2026-05-07 (v1), 2026-05-08 (Trio: BACK-V2-01, VIZ-V2-01, UI-V2-01)*
