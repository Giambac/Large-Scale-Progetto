# Phase 2: Clustering Agent Core - Research

**Researched:** 2026-05-07
**Domain:** Python pure-function agent, feedback data model, LLM-based parsing, Flask+WebSocket debug UI
**Confidence:** HIGH (all core decisions locked; research fills implementation-level gaps)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Plain Python `while` loop orchestrator — no LangGraph
- **D-02:** Standalone module (`agent.py` or `conversation_loop.py`) — `f_*` functions imported, not methods on a class
- **D-03:** `OracleProtocol` (ABC or `typing.Protocol`) with `.reply(state, message) → OracleReply`; `MockOracle` ships in Phase 2; Phase 3 replaces it without touching the loop
- **D-04:** Loop owns JSONL write; `f_*` functions are pure (no I/O)
- **D-05:** `feedback_parser.py` standalone module; `parse_feedback(raw_text, state) → list[FeedbackDelta]`; only place that calls LLM for parsing
- **D-06:** Separate dataclass per feedback type: `SplitFeedback`, `MergeFeedback`, `MoveItemFeedback`, `GlobalFeedback`, `InstructionalFeedback`; union type alias `FeedbackDelta`
- **D-07:** Compound oracle messages produce multiple `FeedbackDelta` objects applied in type-priority order: global → cluster (split/merge) → point (move_item) → instructional
- **D-08:** Split: oracle-seeded K-means (`sklearn.cluster.KMeans`); seed items → embeddings from `EmbeddingStore`; fallback to `k-means++` if no seeds; old ID retired, two new IDs assigned
- **D-09:** Merge: column pooling on `soft_probs`; `soft_probs[i][new_id] = soft_probs[i][A] + soft_probs[i][B]`, drop A and B columns, re-normalize; O(N), no re-clustering
- **D-10:** Point move: `soft_probs[item_id][target] = ORACLE_MOVE_CONFIDENCE (0.95)`; remaining 0.05 distributed proportionally to original soft_probs of non-target clusters
- **D-11:** Cluster IDs: monotonic counter, old IDs never reused (carries forward from Phase 1 D-11)
- **D-12:** Re-naming: `ClusterNamer` called for affected clusters only after split/merge/move
- **D-13:** Framework: Flask + vanilla HTML + WebSocket via `flask-socketio`
- **D-14:** Layout: cluster cards + metrics sidebar; one card per cluster showing top-5 items by confidence; sidebar shows conversation history and per-turn metrics
- **D-15:** Single session per server run; uploading new dataset clears state and flushes AuditLog

### Claude's Discretion

- Exact `OracleReply` schema (fields beyond `satisfied: bool` and `raw_text: str`)
- `f_uncertainty` scoring formula — how to weight entropy, boundary proximity, split/merge candidacy
- `f_next_best_step` `RandomStrategy` implementation
- Flask port and serving configuration
- WebSocket library choice (`flask-socketio` vs `simple-websocket`)

### Deferred Ideas (OUT OF SCOPE)

- Multi-session management (chat-history style) — UI-V2-01
- Oracle cognitive-load weight parameters — Phase 3 decision
- Exact `ε` and `N_fallback` values for stopping criteria — Phase 4 decision
- `InformationGain` strategy for `f_next_best_step` — Phase 5 ablation scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLUS-01 | `f_output` always returns a complete clustering assignment (anytime behavior) | Pure function returning ClusteringState; assert completeness invariant on every output |
| CLUS-02 | `f_uncertainty` identifies boundary points, ambiguous assignments, low-confidence clusters | Shannon entropy on soft_probs rows; per-item and per-cluster aggregation; ranked list |
| CLUS-03 | `f_next_best_step` selects next action via pluggable Strategy interface (RandomStrategy) | `typing.Protocol` Strategy ABC; RandomStrategy uses `random.choice` over valid actions |
| CLUS-04 | `f_next_state` applies oracle feedback; latest oracle intent wins on contradictions | Type-priority dispatch; K-means for split; column pooling for merge; 0.95 override for move |
| FB-01 | System accepts and acts on global oracle feedback | `GlobalFeedback` dataclass; handled first in type-priority ordering |
| FB-02 | System accepts and acts on cluster-level oracle feedback (split, merge) | `SplitFeedback`, `MergeFeedback` dataclasses; oracle-seeded K-means and column pooling |
| FB-03 | System accepts and acts on point-level oracle feedback | `MoveItemFeedback` dataclass; 0.95 soft override mechanic |
| HIER-01 | System maintains navigable cluster hierarchy | `ClusterNode` dataclass with `parent_id: int | None`, `children_ids: list[int]`; stored in separate `HierarchyStore` |
| HIER-02 | Hierarchy grown incrementally as oracle refines (not one-shot upfront) | Split adds children to retiring parent node; merge creates new parent of both retired nodes |
| UI-01 | Web debug UI: cluster state, conversation history, per-turn metrics | Flask + flask-socketio; `socketio.emit('state_update', ...)` each turn from background task |
| UI-02 | Web UI allows dataset upload to start fresh session | Flask `/upload` POST endpoint; clears ClusteringState and flushes AuditLog on receipt |
</phase_requirements>

---

## Summary

Phase 2 builds the Clustering Agent's core logic on top of the Phase 1 foundation (EmbeddingStore, ClusteringState, ClusterNamer, check_stopping, serialization). All implementation decisions are locked in CONTEXT.md. This research resolves the five open questions not decided there: the `f_uncertainty` scoring formula, the cluster hierarchy data structure for HIER-01/HIER-02, the `RandomStrategy` pattern for CLUS-03, the `feedback_parser.py` LLM prompt design, and the Flask-SocketIO threading model for running the conversation loop concurrently with the web server.

The Phase 1 codebase is real and verified. `ClusteringState` schema is frozen at `{turn_index, timestamp, clusters, assignments, soft_probs}`. The `hdbscan` package (0.8.42) is installed and provides `all_points_membership_vectors()` for soft probability vectors. `sklearn` 1.8.0 is installed and its `KMeans(init=array, n_init=1)` API covers oracle-seeded split (D-08). Flask 3.1.3 and flask-socketio 5.6.1 are available via pip; simple-websocket 1.1.0 is also available.

The main architectural risk is the Flask-SocketIO threading model: running the blocking conversation loop concurrently with a responsive Flask server requires starting the loop as a background task via `socketio.start_background_task()` and emitting with `socketio.emit()` (not the context-bound `emit()`) from that task. The `threading` async mode is the safest choice for this project — no eventlet/gevent monkey-patching, straightforward Python threads, compatible with numpy/sklearn.

**Primary recommendation:** Use `async_mode='threading'` in SocketIO, start the conversation loop via `socketio.start_background_task()` on first dataset upload, and emit state updates with `socketio.emit('state_update', payload)` at the end of each turn.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| f_output, f_uncertainty, f_next_best_step, f_next_state | Python module (src/agent/) | — | Pure functions; no network or I/O |
| Feedback parsing (LLM call) | Python module (feedback_parser.py) | — | Only place that calls LLM for parsing; isolated from f_* |
| Cluster naming after split/merge/move | Python module (cluster_naming.py reused) | — | Phase 1 asset reused directly |
| Conversation loop (orchestration, JSONL write) | Python module (conversation_loop.py) | — | Owns all I/O; calls f_* as pure functions |
| Cluster hierarchy (HIER-01, HIER-02) | In-process data structure | — | Simple dict of ClusterNode; no external storage needed in Phase 2 |
| WebSocket live state push | Flask-SocketIO server | Background thread | Loop thread calls socketio.emit(); Flask thread serves HTTP/WS |
| Dataset upload & session reset (UI-02) | Flask HTTP endpoint | — | POST /upload triggers state reset and background task start |
| Client-side rendering | Vanilla HTML/JS (static files served by Flask) | — | No build step; cluster cards re-rendered on each WS event |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.4.4 | Entropy computation, soft_probs manipulation, K-means centroid arrays | Already installed; Phase 1 uses it throughout |
| scikit-learn | 1.8.0 | `KMeans` for oracle-seeded split (D-08) | Already installed; clean `init=array` API verified |
| hdbscan | 0.8.42 | Not re-run per turn; `all_points_membership_vectors` used at init | Already installed; Phase 1 decision (D-05) |
| flask | 3.1.3 | Web server for debug UI | Locked by D-13 |
| flask-socketio | 5.6.1 | WebSocket live state push | Locked by D-13; latest stable verified |
| simple-websocket | 1.1.0 | WebSocket transport layer used by flask-socketio in threading mode | Available; pairs with threading async_mode without gevent/eventlet |
| anthropic | (installed) | LLM API for feedback_parser.py and ClusterNamer | Phase 1 pattern; AnthropicClusterNamer already exists |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats.entropy | (via scipy) | Reference implementation of H = -sum(p*log(p)) | Optional; can compute inline with numpy — see Code Examples |
| dataclasses (stdlib) | Python 3.13 | FeedbackDelta dataclasses | Always; no external dep needed |
| typing (stdlib) | Python 3.13 | `Protocol`, `runtime_checkable`, union type alias for FeedbackDelta | Always |
| abc (stdlib) | Python 3.13 | Alternative to typing.Protocol for OracleProtocol if isinstance checks needed | Use if Phase 3 Oracle needs to be isinstance-checked |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| flask-socketio (threading mode) | eventlet / gevent | eventlet/gevent require monkey-patching which breaks numpy/sklearn; threading mode is safe for scientific Python |
| sklearn KMeans for split | hdbscan re-fit on sub-cluster | hdbscan re-fit is much slower and produces variable K; K-means with n_clusters=2 and oracle seeds is predictable |
| typing.Protocol for OracleProtocol | abc.ABC | Protocol enables structural subtyping (Phase 3 Oracle Agent does not need to inherit); Phase 1 ClusterNamer uses the same Protocol pattern |

**Installation (what is NOT yet installed):**
```bash
pip install flask flask-socketio simple-websocket
```

**Version verification:** [VERIFIED: pip registry 2026-05-07]
- flask: 3.1.3
- flask-socketio: 5.6.1
- simple-websocket: 1.1.0
- numpy: 2.4.4 (already installed)
- scikit-learn: 1.8.0 (already installed)
- hdbscan: 0.8.42 (already installed)

---

## Architecture Patterns

### System Architecture Diagram

```
dataset file (upload)
        │
        ▼
  Flask /upload endpoint
        │  clears ClusteringState, flushes AuditLog
        │  calls socketio.start_background_task(run_conversation)
        ▼
  run_conversation() [background thread]
  ┌──────────────────────────────────────────────────────┐
  │  while True:                                          │
  │    action = f_next_best_step(state, strategy)         │
  │       └─ Strategy.select(state, UncertaintyReport)    │
  │    message = format_message(action, state)            │
  │    reply = oracle.reply(state, message)   ← OracleProtocol│
  │    deltas = parse_feedback(reply.raw_text, state)     │
  │       └─ LLM call (Anthropic) → list[FeedbackDelta]  │
  │    new_state = f_next_state(state, deltas, store, namer) │
  │    append_to_audit_log(new_state, log_path)           │
  │    socketio.emit('state_update', serialize_payload(new_state)) │
  │    stop = check_stopping(turn_index, reply.satisfied, ...) │
  │    if stop: break                                     │
  └──────────────────────────────────────────────────────┘
        │
        ▼                               ▲
  Browser WebSocket client              │ Flask HTTP server (main thread)
  re-renders cluster cards on each      │ serves static HTML/JS
  'state_update' event                  │ handles /upload, /status routes
```

### Recommended Project Structure
```
src/
├── state.py              # ClusteringState, Cluster (Phase 1, frozen)
├── clustering.py         # run_hdbscan, build_initial_clustering_state (Phase 1)
├── embedding_store.py    # EmbeddingStore (Phase 1, read-only)
├── stopping.py           # check_stopping, StoppingCriteria (Phase 1)
├── serialization.py      # serialize_state, append_to_audit_log (Phase 1)
├── cluster_naming.py     # ClusterNamer protocol + concrete implementations (Phase 1)
├── feedback.py           # FeedbackDelta union + dataclasses (NEW Phase 2)
├── feedback_parser.py    # parse_feedback() LLM call (NEW Phase 2)
├── hierarchy.py          # ClusterNode, HierarchyStore (NEW Phase 2)
├── uncertainty.py        # f_uncertainty, UncertaintyReport (NEW Phase 2)
├── agent_functions.py    # f_output, f_next_best_step, f_next_state (NEW Phase 2)
├── strategy.py           # StrategyProtocol, RandomStrategy (NEW Phase 2)
├── oracle_protocol.py    # OracleProtocol, OracleReply, MockOracle (NEW Phase 2)
└── conversation_loop.py  # run_conversation() orchestrator (NEW Phase 2)
web/
├── app.py                # Flask app + SocketIO init + routes (NEW Phase 2)
├── static/
│   ├── main.js           # WebSocket client, card renderer
│   └── style.css
└── templates/
    └── index.html        # Cluster cards + metrics sidebar layout
tests/
├── conftest.py           # Phase 1 shared fixtures (extend for Phase 2)
├── phase1/               # Phase 1 test files (existing)
└── phase2/               # NEW Phase 2 test files
    ├── test_feedback.py
    ├── test_feedback_parser.py
    ├── test_uncertainty.py
    ├── test_agent_functions.py
    ├── test_hierarchy.py
    ├── test_oracle_protocol.py
    └── test_conversation_loop.py
```

### Pattern 1: FeedbackDelta Union Type (D-06)
**What:** Separate frozen dataclasses per feedback type, combined into a union alias
**When to use:** Any code that receives or produces structured feedback

```python
# Source: CONTEXT.md D-06 (locked decision)
from __future__ import annotations
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class SplitFeedback:
    cluster_id: int
    seed_item_ids: list[int]  # may be empty; empty = k-means++ fallback (D-08)

@dataclass(frozen=True)
class MergeFeedback:
    cluster_a_id: int
    cluster_b_id: int

@dataclass(frozen=True)
class MoveItemFeedback:
    item_id: int
    target_cluster_id: int

@dataclass(frozen=True)
class GlobalFeedback:
    instruction_text: str

@dataclass(frozen=True)
class InstructionalFeedback:
    instruction_text: str

FeedbackDelta = Union[
    GlobalFeedback, SplitFeedback, MergeFeedback,
    MoveItemFeedback, InstructionalFeedback
]
```

### Pattern 2: OracleProtocol (D-03)
**What:** `typing.Protocol` so Phase 3's real Oracle Agent requires no inheritance
**When to use:** Any code calling the oracle; all code passes `OracleProtocol` typed parameter

```python
# Source: CONTEXT.md D-03; typing.Protocol pattern consistent with Phase 1 ClusterNamer
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from src.state import ClusteringState

@dataclass
class OracleReply:
    raw_text: str
    satisfied: bool
    # Additional fields at Claude's discretion (see OracleReply Schema section)

@runtime_checkable
class OracleProtocol(Protocol):
    def reply(self, state: ClusteringState, message: str) -> OracleReply:
        ...
```

### Pattern 3: StrategyProtocol + RandomStrategy (CLUS-03, Claude's discretion)
**What:** `typing.Protocol` for pluggable strategy; RandomStrategy is a uniform random selector
**When to use:** `f_next_best_step` delegates selection to the injected strategy

```python
# Source: CONTEXT.md Claude's Discretion + REQUIREMENTS.md ALAB-01 (future strategies)
from __future__ import annotations
import random
from typing import Protocol
from src.state import ClusteringState

class Action:
    """Represents one possible next action."""
    # action_type: Literal["show_full", "show_subset", "ask_question", "stop"]
    # payload: dict  (cluster_id, item_ids, question_text, etc.)

class StrategyProtocol(Protocol):
    def select(self, state: ClusteringState, uncertainty_report: "UncertaintyReport") -> Action:
        ...

class RandomStrategy:
    """Phase 2 implementation: uniform random over valid actions (D-Claude's discretion)."""
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, state: ClusteringState, uncertainty_report: "UncertaintyReport") -> Action:
        valid_actions = _enumerate_valid_actions(state, uncertainty_report)
        assert len(valid_actions) > 0, "BUG: no valid actions to select from"
        return self._rng.choice(valid_actions)
```

### Pattern 4: f_uncertainty Formula (CLUS-02, Claude's discretion)
**What:** Shannon entropy per item, aggregated to cluster-level and global scores, producing a ranked report
**When to use:** Called once per turn before `f_next_best_step`

The formula for per-item entropy from a soft probability vector `p` of length K:

```
H(i) = -sum(p[k] * log(p[k]) for k in range(K) if p[k] > 0)
```

Normalized to [0, 1] by dividing by `log(K)` (max entropy = uniform distribution).

```python
# Source: Shannon entropy formula [VERIFIED: scipy.stats.entropy docs + Wikipedia]
# numpy implementation verified for correctness with soft_probs from hdbscan
import numpy as np
from dataclasses import dataclass

@dataclass
class UncertaintyReport:
    # Items ranked by entropy (highest first) — boundary candidates
    boundary_items: list[tuple[int, float]]    # (item_id, entropy)
    # Clusters ranked by mean item entropy (highest first) — split candidates
    split_candidates: list[tuple[int, float]]  # (cluster_id, mean_entropy)
    # Cluster pairs ranked by centroid proximity — merge candidates
    merge_candidates: list[tuple[int, int, float]]  # (cluster_a, cluster_b, distance)

def f_uncertainty(state: ClusteringState) -> UncertaintyReport:
    """Pure function. No I/O."""
    K = len(state.clusters)
    assert K > 0, "f_uncertainty called on empty state"
    log_K = np.log(K) if K > 1 else 1.0  # avoid div-by-zero for K=1

    item_entropy: dict[int, float] = {}
    for item_id, probs in state.soft_probs.items():
        p = np.array(probs, dtype=np.float64)
        # Mask zeros to avoid log(0)
        mask = p > 0
        h = -np.sum(p[mask] * np.log(p[mask]))
        item_entropy[item_id] = float(h / log_K)  # normalized to [0, 1]

    # Boundary items: all items, sorted descending by entropy
    boundary_items = sorted(item_entropy.items(), key=lambda x: x[1], reverse=True)

    # Split candidates: clusters sorted by mean item entropy
    cluster_entropy: dict[int, float] = {}
    for cluster in state.clusters:
        if cluster.item_ids:
            cluster_entropy[cluster.id] = float(
                np.mean([item_entropy[i] for i in cluster.item_ids])
            )
    split_candidates = sorted(cluster_entropy.items(), key=lambda x: x[1], reverse=True)

    # Merge candidates: cluster pairs ranked by soft_probs cosine similarity
    # Compute mean soft_probs vector per cluster as centroid proxy
    cluster_centroids: dict[int, np.ndarray] = {}
    for cluster in state.clusters:
        vecs = np.array([state.soft_probs[i] for i in cluster.item_ids])
        cluster_centroids[cluster.id] = vecs.mean(axis=0)

    merge_candidates = []
    cluster_ids = [c.id for c in state.clusters]
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            a, b = cluster_ids[i], cluster_ids[j]
            ca, cb = cluster_centroids[a], cluster_centroids[b]
            # Euclidean distance in soft_probs space — lower = more merge-worthy
            dist = float(np.linalg.norm(ca - cb))
            merge_candidates.append((a, b, dist))
    merge_candidates.sort(key=lambda x: x[2])  # ascending: closest pair first

    return UncertaintyReport(
        boundary_items=boundary_items,
        split_candidates=split_candidates,
        merge_candidates=merge_candidates,
    )
```

### Pattern 5: f_next_state Split Mechanic (D-08)
**What:** Oracle-seeded K-means on the sub-cluster, returns two new clusters with new IDs

```python
# Source: CONTEXT.md D-08; sklearn KMeans API [VERIFIED: scikit-learn 1.8.0 docs]
from sklearn.cluster import KMeans
import numpy as np

def _apply_split(
    feedback: SplitFeedback,
    state: ClusteringState,
    store: EmbeddingStore,
    next_id: int,  # next available cluster ID from monotonic counter
) -> tuple[ClusteringState, int]:
    """Returns (new_state, updated_next_id)."""
    target = next(c for c in state.clusters if c.id == feedback.cluster_id)
    assert target is not None, f"Split target cluster {feedback.cluster_id} not found"

    item_ids = target.item_ids
    assert len(item_ids) >= 2, f"Cannot split cluster {feedback.cluster_id}: fewer than 2 items"

    sub_embeddings = np.array([store.get(i) for i in item_ids])  # shape (M, 768)

    if feedback.seed_item_ids:
        # Oracle-seeded: look up centroid vectors
        assert all(s in item_ids for s in feedback.seed_item_ids), (
            f"Seed items {feedback.seed_item_ids} not all in cluster {feedback.cluster_id}"
        )
        centroids = np.array([store.get(s) for s in feedback.seed_item_ids[:2]])
        km = KMeans(n_clusters=2, init=centroids, n_init=1, random_state=0)
    else:
        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)

    km.fit(sub_embeddings)
    sub_labels = km.labels_  # shape (M,) — values 0 or 1

    # ... assemble new ClusteringState with two new clusters (IDs: next_id, next_id+1)
    # retire feedback.cluster_id; soft_probs for sub-cluster items updated proportionally
```

### Pattern 6: f_next_state Merge Mechanic (D-09)
**What:** Column pooling on soft_probs matrix — O(N), no re-clustering

```python
# Source: CONTEXT.md D-09 (locked decision)
# Column pooling: new_prob[i] = soft_probs[i][A] + soft_probs[i][B]
# Then re-normalize row to sum to 1.0

def _apply_merge(feedback: MergeFeedback, state: ClusteringState, next_id: int):
    a_idx = _cluster_index(state, feedback.cluster_a_id)
    b_idx = _cluster_index(state, feedback.cluster_b_id)
    new_soft_probs = {}
    for item_id, probs in state.soft_probs.items():
        p = list(probs)
        pooled = p[a_idx] + p[b_idx]
        # Build new probs vector: drop a_idx and b_idx, insert pooled at new position
        new_p = [v for k, v in enumerate(p) if k not in (a_idx, b_idx)]
        new_p.append(pooled)
        row_sum = sum(new_p)
        assert row_sum > 0, f"Zero-sum soft_probs after merge for item {item_id}"
        new_soft_probs[item_id] = [v / row_sum for v in new_p]
    # ... assemble new ClusteringState
```

### Pattern 7: Flask-SocketIO Threading Model (D-13)
**What:** `async_mode='threading'` + `socketio.start_background_task()` + `socketio.emit()`
**When to use:** Single-session debug UI where the conversation loop runs as a background thread

```python
# Source: Flask-SocketIO docs [CITED: flask-socketio.readthedocs.io/en/latest/api.html]
# async_mode='threading' is safe with numpy/sklearn (no monkey-patching)
# simple-websocket is the transport layer for threading mode

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

_session = {"state": None, "task": None}

@app.route('/upload', methods=['POST'])
def upload():
    # ... parse dataset, build initial state
    # Start conversation loop as background task
    _session["task"] = socketio.start_background_task(
        run_conversation, initial_state, store, namer, socketio
    )
    return {"status": "started"}

def run_conversation(state, store, namer, socketio):
    """Runs in background thread. Emits 'state_update' each turn."""
    while True:
        # ... f_next_best_step, oracle.reply, parse_feedback, f_next_state
        socketio.emit('state_update', serialize_payload(new_state))
        # DO NOT use flask_socketio.emit() — that is context-bound
        # USE socketio.emit() — the SocketIO instance method, context-free
        stop = check_stopping(...)
        if stop:
            socketio.emit('session_stopped', {'reason': stop.value})
            break
```

### Pattern 8: Cluster Hierarchy (HIER-01, HIER-02)
**What:** Parent-child dict of `ClusterNode`; grows only when oracle splits or merges
**When to use:** All split/merge operations in `f_next_state`

```python
# Source: CONTEXT.md HIER-01, HIER-02 requirements; [ASSUMED] incremental tree pattern
from dataclasses import dataclass, field

@dataclass
class ClusterNode:
    cluster_id: int
    parent_id: int | None   # None = root-level cluster
    children_ids: list[int] = field(default_factory=list)
    is_active: bool = True   # False = cluster was retired (split/merge)

@dataclass
class HierarchyStore:
    """
    Tracks the full history of cluster splits and merges.
    Oracle "drill-in": navigate to a child cluster_id.
    Oracle "zoom-out": navigate to parent_id.
    """
    nodes: dict[int, ClusterNode] = field(default_factory=dict)

    def register(self, cluster_id: int, parent_id: int | None = None) -> None:
        assert cluster_id not in self.nodes, f"cluster_id {cluster_id} already registered"
        self.nodes[cluster_id] = ClusterNode(cluster_id=cluster_id, parent_id=parent_id)

    def record_split(self, parent_id: int, child_a_id: int, child_b_id: int) -> None:
        assert parent_id in self.nodes
        self.nodes[parent_id].is_active = False
        self.nodes[parent_id].children_ids = [child_a_id, child_b_id]
        self.register(child_a_id, parent_id=parent_id)
        self.register(child_b_id, parent_id=parent_id)

    def record_merge(self, parent_a_id: int, parent_b_id: int, merged_id: int) -> None:
        assert parent_a_id in self.nodes and parent_b_id in self.nodes
        self.nodes[parent_a_id].is_active = False
        self.nodes[parent_b_id].is_active = False
        # merged cluster is a child of both parents conceptually; use parent_a by convention
        self.register(merged_id, parent_id=parent_a_id)
```

**Drill-in / zoom-out:** Phase 2 stores the hierarchy but oracle navigation is a future UI feature. The `HierarchyStore` provides the structure; Phase 3/UI can add navigation commands.

### Pattern 9: feedback_parser.py LLM Prompt Design (D-05)
**What:** Prompt that asks the LLM to extract all `FeedbackDelta` objects from one oracle utterance
**When to use:** Only in `feedback_parser.py`; called once per oracle reply

The prompt must handle compound messages ("split cluster 2 and move item 45 to cluster 3").

```python
# Source: [ASSUMED] — based on Phase 1 ClusterNamer pattern + LLM structured output best practices
PARSE_FEEDBACK_PROMPT_TEMPLATE = """
You are parsing oracle feedback in a clustering conversation.
Current clusters: {cluster_summary}
Oracle said: "{raw_text}"

Extract ALL feedback intents as a JSON array. Each item has exactly one of these schemas:
- {{"type": "global", "instruction_text": "..."}}
- {{"type": "split", "cluster_id": <int>, "seed_item_ids": [<int>, ...]}}
- {{"type": "merge", "cluster_a_id": <int>, "cluster_b_id": <int>}}
- {{"type": "move_item", "item_id": <int>, "target_cluster_id": <int>}}
- {{"type": "instructional", "instruction_text": "..."}}

Rules:
- Return ONLY a JSON array. No markdown. No explanation.
- If no feedback: return []
- Compound messages produce multiple array items.
- seed_item_ids may be [] if oracle names no specific items.
- cluster_id values must exist in the current cluster list.
"""
```

Key design choices for this prompt:
1. Pass the current cluster summary (IDs + names) so the LLM can resolve cluster references
2. Enumerate exact JSON schemas so the LLM cannot invent new keys
3. Return empty array for "no feedback" — never crash on silence
4. Handle compound messages explicitly in the prompt ("Compound messages produce multiple array items")

### Pattern 10: OracleReply Schema (Claude's discretion)
**What:** Fields beyond `satisfied` and `raw_text` that support the Phase 3 Oracle Agent contract

Recommended schema:

```python
@dataclass
class OracleReply:
    raw_text: str           # The oracle's natural-language response
    satisfied: bool         # Explicit satisfaction token (D-08 primary stop condition)
    turn_cognitive_load: float = 0.0  # Phase 3 will compute this; Phase 2 stub = 0.0
    # Not included: drift_history, contradiction_flag — Phase 3 Oracle Agent owns these
```

**Rationale for `turn_cognitive_load`:** Phase 3's Oracle Agent computes cognitive load per turn (ORC-03). Including the field in `OracleReply` from Phase 2 means Phase 3 just fills it in — no schema change at the boundary. `MockOracle` returns 0.0 always.

### Anti-Patterns to Avoid

- **Calling `emit()` inside the background thread:** The Flask-SocketIO context-bound `emit()` from `flask_socketio` requires a request context. From a background thread, always call `socketio.emit()` on the SocketIO instance. [CITED: flask-socketio.readthedocs.io/en/latest/getting_started.html]
- **Mutating ClusteringState in place:** `f_*` functions are pure — they must return a new `ClusteringState` object. Never mutate the input state. The loop holds the current state reference and replaces it with the return value.
- **Re-running HDBSCAN per turn:** HDBSCAN is only run once at session startup (`build_initial_clustering_state`). Per-turn updates use K-means (split), column pooling (merge), or direct probability override (move).
- **Reusing retired cluster IDs:** After split or merge, the old cluster ID is retired forever (D-11). The monotonic counter only increments. Reusing IDs breaks AuditLog replay and hierarchy consistency.
- **Using `random.random()` inline in Strategy:** Strategy must be deterministic from a seed. Use `random.Random(seed)` instance, not the global `random.choice`.
- **Swallowing parse_feedback errors:** If the LLM returns malformed JSON, `json.loads` will raise. Let it propagate — fail loudly. This is the one place with an LLM API call, so try/except is permitted at the outer boundary only.
- **Hardcoding 0.95 inline:** Must be `ORACLE_MOVE_CONFIDENCE = 0.95` at module level (CONTEXT.md Specifics).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| K-means with seeded centroids | Custom centroid init code | `sklearn.cluster.KMeans(init=array, n_init=1)` | KMeans handles convergence, empty cluster recovery, etc. |
| Shannon entropy | Manual `-sum(p*log(p))` loops | numpy vectorized ops (see Code Examples) | Numpy handles zeros, float precision, vectorization |
| WebSocket server | Raw socket server | `flask-socketio` 5.6.1 | Handles reconnection, rooms, async modes |
| Soft-prob row normalization | Custom loop | `probs / probs.sum()` with numpy | Vectorized; handles near-zero sums |
| JSON structured output validation | Manual key checks | `assert "type" in item and item["type"] in VALID_TYPES` | Fail loudly per project philosophy; no schema library needed |

**Key insight:** All the "complex" mechanics (split, merge, move) have O(N) numpy implementations. The only real complexity is the LLM call in `feedback_parser.py`, which is isolated in one module.

---

## Runtime State Inventory

> Omitted — this is a greenfield phase (Phase 2 adds new modules; no rename/migration involved).

---

## Common Pitfalls

### Pitfall 1: flask-socketio emit from background thread uses wrong emit function
**What goes wrong:** Calling `from flask_socketio import emit` then using `emit(...)` in the background thread raises a RuntimeError about missing request context.
**Why it happens:** The module-level `emit()` from flask_socketio is request-context-bound. Background threads have no request context.
**How to avoid:** Always call `socketio.emit(event, data)` on the SocketIO *instance*, not the imported module function. Pass `socketio` as a parameter to the background task function. [CITED: flask-socketio.readthedocs.io/en/latest/api.html]
**Warning signs:** `RuntimeError: Working outside of request context` in the background thread log.

### Pitfall 2: soft_probs vector length mismatch after split/merge
**What goes wrong:** After a split, the soft_probs vector for each item changes length (K+1). If any code still uses the old K-length index mapping, it will index wrongly.
**Why it happens:** `ClusteringState.soft_probs[i]` is a plain `list[float]` — there is no named-column index. The position of a cluster in the vector is its position in the ordered cluster list, not its cluster ID.
**How to avoid:** Maintain a cluster-ID-to-index mapping inside `f_next_state` operations. Build this mapping fresh from `state.clusters` at the start of each operation. Assert that every item's soft_probs length equals `len(state.clusters)` after each mutation.

### Pitfall 3: Proportional residual redistribution in point-move produces sum > 1.0
**What goes wrong:** D-10's formula sets `target = 0.95`, then redistributes `0.05` proportionally to other clusters based on original probs. If original other-cluster probs sum to 0 (point was already at 1.0 in another cluster), division by zero occurs.
**Why it happens:** Edge case where `sum(original_probs) - original_target_prob == 0`.
**How to avoid:** Assert `original_non_target_sum > 0` before redistribution. If zero, distribute the residual uniformly across non-target clusters. Add this as a named constant: `UNIFORM_FALLBACK_THRESHOLD = 1e-9`.

### Pitfall 4: parse_feedback returns cluster IDs that no longer exist
**What goes wrong:** The LLM in feedback_parser.py hallucinates a cluster_id that is not in the current state (e.g., refers to a retired cluster).
**Why it happens:** The cluster summary in the prompt may be truncated or the LLM confabulates.
**How to avoid:** After `parse_feedback` returns, validate every cluster ID in every `FeedbackDelta` against `state.clusters`. Assert or raise on any invalid reference before passing to `f_next_state`.

### Pitfall 5: MockOracle determinism breaks 30-turn loop test
**What goes wrong:** MockOracle that always returns `satisfied=False` and the same feedback causes `f_next_state` to apply the same delta repeatedly, creating degenerate state (all items in one cluster after repeated merges, or infinite cluster proliferation after repeated splits).
**Why it happens:** The loop test needs to run 30 turns with valid state at each turn.
**How to avoid:** Design MockOracle as a scripted sequence (list of `OracleReply` objects indexed by turn). After the script ends, default to neutral `GlobalFeedback` or `satisfied=True`. This ensures the 30-turn test exercises multiple feedback types and reaches natural termination.

### Pitfall 6: KMeans on a sub-cluster with N=1 or N=2
**What goes wrong:** `KMeans(n_clusters=2).fit(X)` on X with shape (1, 768) raises a ValueError.
**Why it happens:** The oracle can issue a split on a cluster with very few items.
**How to avoid:** Assert `len(target.item_ids) >= 2` before running K-means. If the assertion fails (1-item cluster), raise with a clear message: "Cannot split cluster {id}: has only {N} item(s)."

---

## Code Examples

### Entropy computation (numpy, no scipy)
```python
# Source: Shannon entropy formula [VERIFIED: scipy.stats.entropy docs + Wikipedia]
import numpy as np

def item_normalized_entropy(probs: list[float]) -> float:
    """Normalized Shannon entropy in [0, 1]. 0 = certain, 1 = maximally uncertain."""
    p = np.array(probs, dtype=np.float64)
    K = len(p)
    if K <= 1:
        return 0.0
    mask = p > 0
    h = -np.sum(p[mask] * np.log(p[mask]))
    return float(h / np.log(K))
```

### Soft-prob row normalization after merge
```python
# Source: CONTEXT.md D-09
import numpy as np

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a 2D array to sum to 1. Asserts no all-zero rows."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    assert np.all(row_sums > 0), "BUG: zero-sum row in soft_probs"
    return matrix / row_sums
```

### KMeans oracle-seeded split (verified API)
```python
# Source: scikit-learn 1.8.0 [VERIFIED: sklearn KMeans docs]
from sklearn.cluster import KMeans
import numpy as np

def split_cluster_embeddings(
    sub_embeddings: np.ndarray,   # shape (M, 768)
    seed_embeddings: np.ndarray | None,  # shape (2, 768) or None
) -> np.ndarray:
    """Returns labels array of shape (M,) with values 0 or 1."""
    assert sub_embeddings.shape[0] >= 2, "Need at least 2 items to split"
    if seed_embeddings is not None:
        assert seed_embeddings.shape == (2, sub_embeddings.shape[1])
        km = KMeans(n_clusters=2, init=seed_embeddings, n_init=1, random_state=0)
    else:
        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
    km.fit(sub_embeddings)
    return km.labels_
```

### Flask-SocketIO server skeleton (threading mode)
```python
# Source: Flask-SocketIO docs [CITED: flask-socketio.readthedocs.io/en/latest/api.html]
from flask import Flask, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
# async_mode='threading': safe with numpy/sklearn; simple-websocket as transport
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

@socketio.on('connect')
def on_connect():
    pass  # single session — no per-client state

@app.route('/upload', methods=['POST'])
def upload_dataset():
    # ... load dataset, build initial state
    socketio.start_background_task(run_conversation_loop, initial_state)
    return jsonify({"status": "session_started"})

# Run with: socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ABC for interfaces | `typing.Protocol` (structural subtyping) | Python 3.8+ | Phase 1 already uses Protocol for ClusterNamer — consistent |
| Polling for live updates | WebSocket (flask-socketio) | Flask-SocketIO 1.0+ | Full-duplex; server pushes per-turn without client polling |
| Eventlet/gevent for async | threading mode + simple-websocket | flask-socketio 5.x | Safe for scientific Python (numpy/sklearn) without monkey-patching |
| sklearn.cluster.HDBSCAN | standalone hdbscan package | Phase 1 D-05 | `all_points_membership_vectors` not available in sklearn HDBSCAN |

**Deprecated/outdated:**
- `eventlet` and `gevent` for this project: compatible with flask-socketio but require monkey-patching that breaks numpy. Use `async_mode='threading'` instead. [CITED: Flask-SocketIO deployment docs]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Cluster hierarchy is best represented as a flat `dict[int, ClusterNode]` with parent_id pointers | Pattern 8 (Hierarchy) | If deep recursive navigation is needed, a tree library (anytree) might be better — but flat dict is simpler and sufficient for Phase 2 |
| A2 | `f_uncertainty` merge candidates should use soft_probs centroid distance rather than embedding space distance | Pattern 4 (f_uncertainty) | Soft_probs space is lower-dimensional (K dims) vs embedding space (768 dims); soft_probs distance may not capture semantic proximity. However, it is consistent with the information available in ClusteringState (EmbeddingStore is not passed to f_uncertainty) |
| A3 | `OracleReply.turn_cognitive_load: float = 0.0` stub is the right Phase 3 boundary | Pattern 10 (OracleReply) | If Phase 3 changes the cognitive load API significantly, the field might need to be richer — but adding a stub float causes no harm |
| A4 | The feedback_parser.py prompt structure (JSON array with type discriminant) will reliably produce parseable output from claude-haiku-4-5 | Pattern 9 (feedback_parser) | LLMs can hallucinate cluster IDs or malform JSON; mitigation is post-parse validation with asserts |
| A5 | `MockOracle` as a scripted turn-indexed sequence is sufficient for the 30-turn loop integrity test | Pitfall 5 | If tests need dynamic behavior (e.g., MockOracle that responds to state), scripted sequence may be too rigid — but CONTEXT.md says "deterministic scripted replies" |

---

## Open Questions

1. **f_uncertainty: should merge candidates use soft_probs centroid distance or embedding space distance?**
   - What we know: `f_uncertainty` is a pure function that takes only `ClusteringState` — it does not receive `EmbeddingStore`
   - What's unclear: CONTEXT.md CLUS-02 says "derived from calibrated soft assignments" — soft_probs space seems intended
   - Recommendation: Use soft_probs centroid distance in Phase 2. If this produces poor merge suggestions in practice, Phase 5 can add `EmbeddingStore` as an optional parameter

2. **OracleReply: does Phase 3 need a `contradiction_flag: bool` field?**
   - What we know: ORC-04 says "contradictions with prior intents are detected, logged, and optionally flagged to the Clustering Agent"
   - What's unclear: "optionally flagged to the Clustering Agent" could mean a field in OracleReply or a separate mechanism
   - Recommendation: Do not add `contradiction_flag` in Phase 2; Phase 3 adds it if needed. MockOracle does not detect contradictions.

3. **Hierarchy: does "zoom-out" mean the oracle can request to view a parent cluster's full membership?**
   - What we know: HIER-01 says "navigable cluster hierarchy"; HIER-02 says incremental growth
   - What's unclear: "Drill-in / zoom-out" as an action type is not defined in CLUS-03's action space
   - Recommendation: Phase 2 builds the `HierarchyStore` data structure. Phase 3/UI adds navigation commands as a new action type in `f_next_best_step`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.13.12 | — |
| numpy | f_uncertainty, soft_probs ops | ✓ | 2.4.4 | — |
| scikit-learn | KMeans split (D-08) | ✓ | 1.8.0 | — |
| hdbscan | Initial clustering (Phase 1) | ✓ | 0.8.42 | — |
| flask | Web debug UI (D-13) | ✗ not installed | 3.1.3 available | — must install |
| flask-socketio | WebSocket live updates (D-13) | ✗ not installed | 5.6.1 available | — must install |
| simple-websocket | Flask-SocketIO threading transport | ✓ | 1.1.0 | — |
| anthropic | LLM API for feedback_parser + ClusterNamer | ✗ not confirmed | check ANTHROPIC_API_KEY env | No fallback for Phase 2 LLM calls |

**Missing dependencies — must install before Phase 2 begins:**
```bash
pip install flask flask-socketio
```

**Missing dependencies with fallback:**
- None — all missing deps have a clear install path.

**Note:** `simple-websocket` 1.1.0 is already installed. This is the transport layer used by flask-socketio in threading mode, so no additional install is required beyond flask and flask-socketio.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in pyproject.toml) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/phase2/ -q -x` |
| Full suite command | `pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLUS-01 | f_output returns complete assignment (all N items) | unit | `pytest tests/phase2/test_agent_functions.py::test_f_output_complete -x` | ❌ Wave 0 |
| CLUS-02 | f_uncertainty returns ranked boundary/split/merge lists | unit | `pytest tests/phase2/test_uncertainty.py -x` | ❌ Wave 0 |
| CLUS-03 | f_next_best_step RandomStrategy selects valid action | unit | `pytest tests/phase2/test_agent_functions.py::test_random_strategy -x` | ❌ Wave 0 |
| CLUS-03 | 30-turn MockOracle loop with state integrity at turns 10, 20, 30 | integration | `pytest tests/phase2/test_conversation_loop.py::test_30_turn_loop -x` | ❌ Wave 0 |
| CLUS-04 | f_next_state applies split/merge/move correctly | unit | `pytest tests/phase2/test_agent_functions.py::test_f_next_state_split -x` | ❌ Wave 0 |
| FB-01 | GlobalFeedback is applied first in type-priority order | unit | `pytest tests/phase2/test_agent_functions.py::test_feedback_priority_order -x` | ❌ Wave 0 |
| FB-02 | Split produces 2 new clusters, retires old ID | unit | `pytest tests/phase2/test_agent_functions.py::test_split_retires_id -x` | ❌ Wave 0 |
| FB-02 | Merge produces 1 new cluster, soft_probs rows sum to 1 | unit | `pytest tests/phase2/test_agent_functions.py::test_merge_soft_probs_normalized -x` | ❌ Wave 0 |
| FB-03 | MoveItemFeedback sets target to 0.95, residual distributed | unit | `pytest tests/phase2/test_agent_functions.py::test_move_item_0_95 -x` | ❌ Wave 0 |
| HIER-01 | HierarchyStore records split and merge with parent-child links | unit | `pytest tests/phase2/test_hierarchy.py -x` | ❌ Wave 0 |
| HIER-02 | Hierarchy grows only on split/merge, not at session start | unit | `pytest tests/phase2/test_hierarchy.py::test_no_upfront_hierarchy -x` | ❌ Wave 0 |
| UI-01 | Flask server starts and serves index.html | smoke | `pytest tests/phase2/test_app.py::test_index_route -x` | ❌ Wave 0 |
| UI-02 | /upload POST resets session state | smoke | `pytest tests/phase2/test_app.py::test_upload_resets_session -x` | ❌ Wave 0 |
| FB-01..03 | parse_feedback extracts correct FeedbackDelta from known oracle utterances | unit | `pytest tests/phase2/test_feedback_parser.py -x` | ❌ Wave 0 |

**Note:** `test_feedback_parser.py` tests that call the real Anthropic API should be gated with a pytest mark (e.g., `@pytest.mark.llm`) and excluded from the default fast run. Use a `MockLLMClient` for unit tests.

### Sampling Rate
- **Per task commit:** `pytest tests/phase2/ -q -x`
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/phase2/__init__.py` — empty init for test discovery
- [ ] `tests/phase2/test_feedback.py` — FeedbackDelta dataclass construction and type checks
- [ ] `tests/phase2/test_feedback_parser.py` — parse_feedback with MockLLMClient returning scripted JSON
- [ ] `tests/phase2/test_uncertainty.py` — f_uncertainty entropy computation on known soft_probs
- [ ] `tests/phase2/test_agent_functions.py` — f_output, f_next_state split/merge/move, f_next_best_step
- [ ] `tests/phase2/test_hierarchy.py` — HierarchyStore register, record_split, record_merge
- [ ] `tests/phase2/test_oracle_protocol.py` — MockOracle scripted reply sequence
- [ ] `tests/phase2/test_conversation_loop.py` — 30-turn loop with MockOracle, state integrity checks
- [ ] `tests/phase2/test_app.py` — Flask routes smoke test (no real WebSocket needed)
- [ ] `tests/conftest.py` extension — add Phase 2 shared fixtures (tiny ClusteringState with 3 clusters, MockOracle factory)

---

## Security Domain

> `security_enforcement` not set to false in config.json — section included per policy.

This is an academic/research system with no user authentication, public data, or sensitive storage. Security review is proportional.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth in debug UI (single-session, localhost) |
| V3 Session Management | no | Single-session, no persistent sessions until UI-V2-01 |
| V4 Access Control | no | No access control; debug UI is developer-only |
| V5 Input Validation | yes (low risk) | Validate cluster IDs from parse_feedback output before passing to f_next_state; assert rather than sanitize |
| V6 Cryptography | no | No secrets stored; API keys via env vars (existing pattern) |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| LLM prompt injection via oracle raw_text | Tampering | feedback_parser validates output schema with assert before any ID lookup |
| Uploaded CSV/JSONL with malformed records | Tampering | data_loader.py (Phase 1) already asserts "text" field presence — reuse |
| Flask debug mode in production | Elevation of privilege | Never use `debug=True` in socketio.run(); set `debug=False` explicitly |

---

## Sources

### Primary (HIGH confidence)
- Phase 1 source code (src/state.py, src/clustering.py, src/embedding_store.py, src/stopping.py, src/serialization.py, src/cluster_naming.py) — API contracts verified by reading actual code
- scikit-learn 1.8.0 KMeans docs [VERIFIED: scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html] — `init` parameter array API
- hdbscan 0.8.42 installed — `all_points_membership_vectors` verified in clustering.py
- pip registry [VERIFIED: 2026-05-07] — flask 3.1.3, flask-socketio 5.6.1, simple-websocket 1.1.0 versions
- Python 3.13 stdlib `typing.Protocol` — consistent with Phase 1 ClusterNamer pattern

### Secondary (MEDIUM confidence)
- Flask-SocketIO 5.6.1 docs [CITED: flask-socketio.readthedocs.io/en/latest/api.html] — `socketio.emit()` context-free usage, `start_background_task()`, async_mode parameter
- Flask-SocketIO deployment docs [CITED: flask-socketio.readthedocs.io/en/latest/deployment.html] — threading mode with simple-websocket
- scikit-learn KMeans docs [VERIFIED: scikit-learn.org] — `init=array`, `n_init=1` pattern for seeded initialization
- Shannon entropy formula [VERIFIED: scipy.stats.entropy docs, Wikipedia Entropy (information theory)] — `H = -sum(p * log(p))`, normalized by `log(K)`

### Tertiary (LOW confidence)
- A1-A5 in Assumptions Log — flagged for planner review

---

## Project Constraints (from CLAUDE.md)

The following CLAUDE.md directives are binding on all Phase 2 implementation:

| Directive | Scope | Impact on Phase 2 |
|-----------|-------|------------------|
| No `try/except` outside CLI entry point and LLM API calls | All modules | `feedback_parser.py` may have try/except around `json.loads` (LLM output); all other code must let exceptions propagate |
| No silent failures, no `except: pass` | All modules | Validation failures in parse_feedback and f_next_state must raise, not log-and-continue |
| Use `assert` freely to document invariants | All modules | Assert soft_probs sum to 1.0 after every mutation; assert complete assignments; assert no retired IDs |
| `f_output` ALWAYS returns complete assignment | f_output | No partial states; assert `len(state.assignments) == N` before return |
| K changes ONLY through oracle intent | f_next_state | No automatic K optimization in f_uncertainty or anywhere else |
| AuditLog serialized to JSONL every turn from Phase 1 | conversation_loop.py | `append_to_audit_log` called in the loop body, after every `f_next_state` call, before checking stopping |
| Held-out split locked — never modify | dataset/ | No Phase 2 code should read or write dataset/held_out.jsonl |
| All headline quantitative claims must include bootstrap 95% CIs | Phase 5 scope | Not Phase 2 concern; noted for future |

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified via pip registry and Phase 1 source code
- Architecture: HIGH — all structural decisions are locked in CONTEXT.md; open questions are implementation-level details
- Pitfalls: HIGH — derived from the actual Phase 1 codebase + Flask-SocketIO documented threading behavior
- f_uncertainty formula: MEDIUM — Shannon entropy formula is well-established; merge candidate metric (A2) is assumed
- feedback_parser prompt: MEDIUM — pattern consistent with Phase 1 ClusterNamer; specific prompt tuning needs empirical testing

**Research date:** 2026-05-07
**Valid until:** 2026-06-07 (flask-socketio and sklearn are stable; formula choices are stable indefinitely)
