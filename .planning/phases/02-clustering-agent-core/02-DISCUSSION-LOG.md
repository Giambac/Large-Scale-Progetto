# Phase 2: Clustering Agent Core - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 2-Clustering-Agent-Core
**Areas discussed:** Orchestrator loop, Feedback data model, Re-clustering mechanics, Debug UI stack

---

## Orchestrator Loop

| Option | Description | Selected |
|--------|-------------|----------|
| Plain Python while-loop | Simple sequential loop, no framework overhead | ✓ |
| LangGraph | HITL checkpointing built-in; benefit not unlocked until v2 persistence feature | |

**User's choice:** Plain Python

---

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone module | agent.py or conversation_loop.py; f_* functions imported and called | ✓ |
| Inside ClusteringAgent class | Agent object owns the loop; harder to test f_* in isolation | |

**User's choice:** Standalone module

---

| Option | Description | Selected |
|--------|-------------|----------|
| OracleProtocol interface | ABC/Protocol with .reply(); Phase 2 ships MockOracle | ✓ |
| Direct LLM call | Couples loop to LLM client; Phase 3 refactor needed | |

**User's choice:** OracleProtocol interface

---

| Option | Description | Selected |
|--------|-------------|----------|
| Loop owns JSONL write | f_* functions stay pure; write in one predictable place | ✓ |
| f_next_state writes to JSONL | Side effect inside pure function; hard to unit-test | |

**User's choice:** Loop owns the write

---

## Feedback Data Model

| Option | Description | Selected |
|--------|-------------|----------|
| Separate parser module | parse_feedback(raw_text, state) → list[FeedbackDelta]; one LLM call per turn | ✓ |
| f_next_state parses internally | f_next_state makes LLM calls; no longer pure | |

**User's choice:** Separate parser (feedback_parser.py)

---

| Option | Description | Selected |
|--------|-------------|----------|
| Tagged union / single class | One FeedbackDelta with feedback_type enum; None fields for inapplicable data | |
| Separate class per type | SplitFeedback, MergeFeedback, MoveItemFeedback etc.; type-safe by construction | ✓ |

**User's choice:** Separate class per type
**Notes:** User wanted verbose explanation with code examples before deciding. The type-safety argument (impossible to create an invalid delta) resonated, aligning with fail-loudly philosophy.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Parser-order, sequential | Applied in oracle's statement order | |
| Type-priority order | global → cluster → point → instructional | ✓ |

**User's choice:** Type-priority order

---

| Option | Description | Selected |
|--------|-------------|----------|
| feedback_parser.py standalone module | parse_feedback(raw_text, state) → list[FeedbackDelta] | ✓ |
| Inside orchestrator loop | Mixes I/O and orchestration logic | |

**User's choice:** feedback_parser.py module

---

## Re-clustering Mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| Oracle-seeded K-means | Oracle-provided item embeddings as centroids; fallback k-means++ | ✓ |
| K-means K=2, no seeding | Ignores oracle's representative items | |
| Re-run HDBSCAN on sub-cluster | Slow, may produce 0 clusters, soft_probs recomputation needed | |

**User's choice:** Oracle-seeded K-means
**Notes:** User proactively asked "is it possible to perform K-means with centroids defined by the oracle?" — this became the recommended approach. sklearn KMeans `init` parameter accepts explicit centroid arrays.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Column pooling | P(A∪B) = P(A) + P(B), re-normalize; O(N), no re-clustering | ✓ |
| Re-run full HDBSCAN | Exact but O(N²) per merge, latency per turn | |

**User's choice:** Column pooling
**Notes:** User asked "is there a way to recalculate the probability without redoing HDBSCAN, which is reliable?" — column pooling was explained as mathematically grounded (marginalizing out the A/B distinction). Slight overestimate at old boundary is acceptable.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Hard override, soft_probs stays | assignments[item] = target; soft_probs unchanged (preserves genuine uncertainty) | |
| Force soft_probs to 1-hot | Destroys uncertainty signal for oracle-moved items | |
| 0.95 at target, uniform residual | 0.95 at target; 0.05 spread uniformly across other clusters | |
| 0.95 at target, proportional residual | 0.95 at target; 0.05 redistributed proportionally to original values of other clusters | ✓ |

**User's choice:** 0.95 proportional residual
**Notes:** User proposed the 0.95 idea themselves — "what about setting at 0.95?" — as a middle ground between hard override and 1-hot. Proportional redistribution chosen to preserve relative uncertainty ordering among non-target clusters.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Re-name affected clusters only | One LLM call per changed cluster | ✓ |
| Re-name all clusters every turn | Up to K LLM calls per turn for unchanged clusters | |
| No re-naming in f_next_state | Naming becomes a coordination problem | |

**User's choice:** Re-name affected clusters only

---

| Option | Description | Selected |
|--------|-------------|----------|
| Monotonic counter, old ID retired | IDs 8 and 9 after max was 7; D-11 preserved | ✓ |
| Old ID recycled for one sub-cluster | Violates D-11; AuditLog history becomes ambiguous | |

**User's choice:** Monotonic counter, old ID retired

---

## Debug UI Stack

| Option | Description | Selected |
|--------|-------------|----------|
| Gradio | Python-native, built-in file upload, zero HTML/CSS | |
| Flask + vanilla HTML | Maximum control; requires HTML templates from scratch | ✓ |
| Streamlit | Page-refresh driven; awkward for real-time updates | |

**User's choice:** Flask + vanilla HTML

---

| Option | Description | Selected |
|--------|-------------|----------|
| Polling | /api/state endpoint every N seconds; simple | |
| Server-Sent Events | Flask pushes updates; lower latency than polling | |
| WebSocket | Full bidirectional streaming; most responsive | ✓ |

**User's choice:** WebSocket

---

| Option | Description | Selected |
|--------|-------------|----------|
| Cluster cards + metrics panel | One card per cluster; sidebar for history + metrics | ✓ |
| Table view only | Flat table per item; loses cluster-centric view | |

**User's choice:** Cluster cards + metrics panel

---

| Option | Description | Selected |
|--------|-------------|----------|
| New session, current discarded | AuditLog flushed; fresh pipeline start | |
| Upload queued, explicit start | Staged upload with 'Start new session' button | |
| Defer to v2 | Single session per server run; multi-session is v2 scope | ✓ |

**User's choice:** Defer to v2
**Notes:** User asked about a "chat histories" model — switching between sessions like a chat app. After clarifying the distinction between in-memory multi-session and fully persistent sessions (UI-V2-01), user chose to defer the feature entirely to v2.

---

## Claude's Discretion (v1)

- Exact `OracleReply` schema (fields beyond `satisfied: bool` and `raw_text: str`)
- `f_uncertainty` scoring formula (entropy weighting, boundary proximity, split/merge candidacy)
- `f_next_best_step` `RandomStrategy` implementation details
- Flask port and serving configuration
- WebSocket library choice (`flask-socketio` vs `simple-websocket`)

## Deferred Ideas (v1)

- Multi-session management (chat-history style switching between sessions) → now implemented in UI-V2-01
- Oracle cognitive-load weight parameters → Phase 3
- Exact ε and N_fallback for stopping criteria → Phase 4
- InformationGain strategy for f_next_best_step → Phase 5

---

# Trio Discussion — 2026-05-08 (BACK-V2-01, VIZ-V2-01, UI-V2-01)

**Areas discussed:** k-means backend interface, k-means soft probs, viz scatter plot placement, session persistence

---

## k-means Backend Interface (BACK-V2-01)

### Backend API structure

| Option | Description | Selected |
|--------|-------------|----------|
| ClusteringBackend Protocol | Protocol with fit(embeddings) → (labels, soft_probs). HDBSCANBackend + KMeansBackend implement it. | ✓ |
| Flat functions + backend param | Keep run_hdbscan() + add run_kmeans(). build_initial_clustering_state gets a backend: str param. | |
| You decide | Let the planner choose. | |

**User's choice:** ClusteringBackend Protocol

### K initialization

| Option | Description | Selected |
|--------|-------------|----------|
| HDBSCAN-derived K | Run HDBSCAN once, use cluster count as K for k-means. | |
| Silhouette analysis | K = 2..sqrt(N), pick highest silhouette score. | |
| BIC on GMM | Fit GMM for K = 2..sqrt(N), pick minimum BIC. | ✓ |
| User-specified --k flag | Explicit K at startup, no automatic selection. | |

**User's choice:** BIC on GMM, search range K = 2..sqrt(N)
**Notes:** User asked about BIC/AIC options. Clarified that automatic K optimization *during* the oracle conversation is out of scope (REQUIREMENTS.md "Out of Scope" section). K *initialization* before the oracle starts is in scope. User chose BIC on GMM with sqrt(N) upper bound.

### Backend selection mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| CLI flag --backend | python web/app.py --backend hdbscan\|kmeans. Fails loudly on unknown value. | ✓ |
| Config file parameter | backend: kmeans in config.json or .env. | |

**User's choice:** CLI flag `--backend`

---

## k-means Soft Probabilities (BACK-V2-01)

### Soft prob derivation

| Option | Description | Selected |
|--------|-------------|----------|
| Softmax of negative centroid distances | soft_probs[i][c] = softmax(-dist(item_i, centroid_c)). Boundary items get ~0.5/0.5. | ✓ |
| One-hot hard assignment | soft_probs[i][assigned_cluster] = 1.0. Breaks f_uncertainty. | |
| You decide | Let the planner choose. | |

**User's choice:** Softmax of negative centroid distances

### Temperature scaling

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed T=1.0 | Raw distances, named constant KMEANS_SOFTMAX_TEMP = 1.0. | ✓ |
| Configurable --softmax-temp | CLI parameter. More flexibility, more audit overhead. | |
| You decide | Let the planner pick. | |

**User's choice:** Fixed T=1.0

---

## Viz: Scatter Plot Placement (VIZ-V2-01)

### Layout position

| Option | Description | Selected |
|--------|-------------|----------|
| Full-width panel above two columns | Scatter above cluster cards + sidebar. Both panels stay visible. | ✓ |
| New third column | Three-column layout: scatter \| cluster cards \| sidebar. | |
| Replace cluster cards | Scatter replaces cards section entirely. | |

**User's choice:** Full-width panel above the two columns

### Recompute timing

| Option | Description | Selected |
|--------|-------------|----------|
| Only on initial clustering | Computed once at upload. Colors update each turn. | |
| Every turn | Recomputed after each oracle turn. Expensive and disorienting. | |
| On cluster-count change only | Recompute on split or merge. Points stable within same K. | ✓ |

**User's choice:** On cluster-count change only

### Library

| Option | Description | Selected |
|--------|-------------|----------|
| UMAP (umap-learn) | Faster, preserves global structure, set random_state for reproducibility. | ✓ |
| t-SNE (sklearn) | No new dependency, slower for large N. | |
| Both via CLI flag | --projection umap\|tsne. More parameters. | |

**User's choice:** UMAP

### Rendering

| Option | Description | Selected |
|--------|-------------|----------|
| Server JSON + <canvas> | UMAP server-side, coords via SocketIO, vanilla JS canvas. No new JS deps. | ✓ |
| Plotly.js | Richer interactivity, adds JS CDN dependency. | |

**User's choice:** Server-side UMAP + SocketIO JSON + `<canvas>`

---

## Session Persistence (UI-V2-01)

### Session scope

| Option | Description | Selected |
|--------|-------------|----------|
| Single session with resume | One session survives restarts. Simpler. | |
| Multiple named sessions | Each upload creates a named session. UI lists and switches. | ✓ |

**User's choice:** Multiple named sessions

### Session identification

| Option | Description | Selected |
|--------|-------------|----------|
| Timestamp-based directories | sessions/2026-05-08T14-32-00/ — human-readable, no DB. | ✓ |
| UUID per session | sessions/<uuid>/ — collision-proof but opaque. | |
| User-provided label | User types name at upload. Friendly but adds UI + collision handling. | |

**User's choice:** Timestamp-based directories

### What to persist

| Option | Description | Selected |
|--------|-------------|----------|
| ClusteringState + AuditLog + embeddings | state.json + audit_log.jsonl + embeddings.npy. Full restore. | ✓ |
| AuditLog only | Replay to reconstruct state. Slower, error-prone. | |
| ClusteringState + AuditLog (no embeddings) | User must re-upload dataset to resume. | |

**User's choice:** ClusteringState + AuditLog + embeddings

### Session switch UI

| Option | Description | Selected |
|--------|-------------|----------|
| Session list in sidebar | Sessions section in existing metrics sidebar. Minimal layout change. | ✓ |
| Dropdown in header | Always visible but clutters header. | |
| You decide | Let the planner design the session-switch UI. | |

**User's choice:** Session list in the metrics sidebar

---

## Claude's Discretion (Trio)

- UMAP hyperparameters (`n_neighbors`, `min_dist`) — planner decides defaults for 768-dim text embeddings
- Canvas rendering details (point size, opacity for soft-assignment confidence) — planner decides
- `/resume` endpoint response shape — planner decides
- `OracleReply` schema beyond `satisfied` and `raw_text` — Phase 3 decision

## Deferred Ideas (Trio)

- LLM-first clustering backend (BACK-V2-01 stretch) — deferred if time short
- BACK-V2-02 (representation choice: raw features vs. embeddings vs. fine-tuned) — later phase
