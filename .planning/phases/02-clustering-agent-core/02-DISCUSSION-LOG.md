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

## Claude's Discretion

- Exact `OracleReply` schema (fields beyond `satisfied: bool` and `raw_text: str`)
- `f_uncertainty` scoring formula (entropy weighting, boundary proximity, split/merge candidacy)
- `f_next_best_step` `RandomStrategy` implementation details
- Flask port and serving configuration
- WebSocket library choice (`flask-socketio` vs `simple-websocket`)

## Deferred Ideas

- Multi-session management (chat-history style switching between sessions) → v2, UI-V2-01
- Oracle cognitive-load weight parameters → Phase 3
- Exact ε and N_fallback for stopping criteria → Phase 4
- InformationGain strategy for f_next_best_step → Phase 5
