# Phase 2: Clustering Agent Core - Pattern Map

**Mapped:** 2026-05-07
**Files analyzed:** 13 new/modified files (9 src modules, 1 web app, 2 web static, 1 template, 9 test files)
**Analogs found:** 11 / 13

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/feedback.py` | model | transform | `src/stopping.py` (frozen dataclasses + union enum pattern) | role-match |
| `src/feedback_parser.py` | service | request-response (LLM) | `src/cluster_naming.py` (LLM call → JSON parse → assert schema) | exact |
| `src/hierarchy.py` | model | CRUD | `src/state.py` (dataclass + field default_factory) | role-match |
| `src/uncertainty.py` | utility | transform | `src/clustering.py` (numpy ops on ClusteringState fields) | role-match |
| `src/agent_functions.py` | service | transform | `src/clustering.py` (pure functions returning new state) | role-match |
| `src/strategy.py` | utility | transform | `src/cluster_naming.py` (typing.Protocol + concrete class) | exact |
| `src/oracle_protocol.py` | model + utility | request-response | `src/cluster_naming.py` (Protocol + dataclass + concrete impl) | exact |
| `src/conversation_loop.py` | service | event-driven | `src/clustering.py` + `src/serialization.py` (orchestration + JSONL write) | role-match |
| `web/app.py` | controller | request-response | none — no Flask files exist in Phase 1 | no analog |
| `web/static/main.js` | utility | event-driven | none — no JS files exist | no analog |
| `web/static/style.css` | config | n/a | none | no analog |
| `web/templates/index.html` | config | n/a | none | no analog |
| `tests/phase2/*.py` | test | CRUD | `tests/phase1/test_stopping_criteria.py`, `tests/phase1/test_cluster_naming.py` | exact |

---

## Pattern Assignments

### `src/feedback.py` (model, transform)

**Analog:** `src/stopping.py` (frozen dataclasses, Enum, explicit field ordering)

**Imports pattern** (stopping.py lines 1-8):
```python
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional
```

**Core frozen-dataclass pattern** (stopping.py lines 35-52):
```python
@dataclass(frozen=True)
class StoppingCriteria:
    turn_budget: int = 15
    magnitude_threshold_epsilon: float = float("nan")   # Phase 4 placeholder
    magnitude_fallback_turns: int = -1                  # Phase 4 placeholder
```

**Apply to `src/feedback.py`:** Use `@dataclass(frozen=True)` for every FeedbackDelta subtype. Define the union alias after all classes. Follow the same `from __future__ import annotations` header and explicit field typing conventions.

```python
# feedback.py target pattern (derived from stopping.py frozen-dataclass + Union pattern)
from __future__ import annotations
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class SplitFeedback:
    cluster_id: int
    seed_item_ids: list[int]   # may be empty; empty = k-means++ fallback (D-08)

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

# Module-level constant required by CONTEXT.md Specifics
ORACLE_MOVE_CONFIDENCE: float = 0.95
UNIFORM_FALLBACK_THRESHOLD: float = 1e-9  # for point-move zero-sum edge case
```

---

### `src/feedback_parser.py` (service, request-response / LLM)

**Analog:** `src/cluster_naming.py` — the only Phase 1 module that makes an LLM call, parses JSON, and asserts schema correctness.

**Imports pattern** (cluster_naming.py lines 1-17):
```python
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable
```

**LLM call + JSON parse + assert schema pattern** (cluster_naming.py lines 68-106):
```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt}],
)

raw_text = response.content[0].text.strip()
# Strip markdown code fences if the model wraps its JSON response
if raw_text.startswith("```"):
    raw_text = raw_text.split("```")[1]
    if raw_text.startswith("json"):
        raw_text = raw_text[4:]
    raw_text = raw_text.strip()
result = json.loads(raw_text)

assert "name" in result and "description" in result, (
    f"LLM returned bad schema for cluster {cluster_id}: {result}"
)
```

**Apply to `src/feedback_parser.py`:**
- Same `client.messages.create` call pattern
- Same markdown fence stripping before `json.loads`
- `json.loads` raises on malformed JSON — let it propagate (fail loudly; LLM call is the permitted try/except boundary)
- After parsing, validate every cluster_id in every delta against `state.clusters` with `assert`
- Prompt template must include cluster summary (IDs + names) so the LLM can resolve references

**Post-parse validation pattern** (copy from cluster_naming.py assert style):
```python
# After json.loads, validate each item in the returned array
VALID_TYPES = {"global", "split", "merge", "move_item", "instructional"}
for item in raw_items:
    assert "type" in item and item["type"] in VALID_TYPES, (
        f"parse_feedback: unknown feedback type in LLM response: {item}"
    )
```

---

### `src/hierarchy.py` (model, CRUD)

**Analog:** `src/state.py` (dataclass composition with `field(default_factory=...)`) + `src/embedding_store.py` (class with mutation methods and assert invariants).

**Imports pattern** (state.py lines 17-21):
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
```

**Dataclass with field default_factory pattern** (state.py lines 23-53):
```python
@dataclass
class Cluster:
    id: int
    name: str
    description: str
    item_ids: list[int]

@dataclass
class ClusteringState:
    turn_index: int
    timestamp: str
    clusters: list[Cluster]
    assignments: dict[int, int]
    soft_probs: dict[int, list[float]]
```

**Assert-on-mutation pattern** (embedding_store.py lines 42-50):
```python
assert embeddings.ndim == 2, (
    f"Embeddings must be 2D (n_items, dim), got shape {embeddings.shape}"
)
```

**Apply to `src/hierarchy.py`:** Use `@dataclass` with `field(default_factory=list)` for `children_ids`. Use `assert` before every mutation to enforce invariants (e.g., `assert cluster_id not in self.nodes`). No try/except anywhere.

---

### `src/uncertainty.py` (utility, transform)

**Analog:** `src/clustering.py` — pure functions that take numpy arrays / state fields and return new data structures; heavy use of `assert` to document invariants.

**Imports pattern** (clustering.py lines 12-23):
```python
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import hdbscan
import numpy as np

from src.state import Cluster, ClusteringState

if TYPE_CHECKING:
    from src.cluster_naming import ClusterNamer
```

**Pure function with assert-invariants pattern** (clustering.py lines 33-110, key excerpts):
```python
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    assert embeddings.ndim == 2, f"Expected 2D array, got shape {embeddings.shape}"
    assert embeddings.shape[0] > 0, "Cannot cluster empty embedding set"
    # ... computation ...
    assert soft_probs.shape[0] == len(embeddings), (
        f"soft_probs row count {soft_probs.shape[0]} != N={len(embeddings)}"
    )
    return clusterer.labels_, soft_probs
```

**Row normalization pattern** (clustering.py lines 103-108):
```python
row_sums = raw_soft_probs.sum(axis=1, keepdims=True)
row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
soft_probs = raw_soft_probs / row_sums_safe
```

**Apply to `src/uncertainty.py`:** Pure function `f_uncertainty(state: ClusteringState) -> UncertaintyReport`. Begin with `assert K > 0`. Use numpy vectorized entropy computation (no scipy). Return a named `UncertaintyReport` dataclass (not a tuple). No I/O.

---

### `src/agent_functions.py` (service, transform)

**Analog:** `src/clustering.py` — pure pipeline functions that assemble new state objects from existing state + computed data; assert completeness invariants at every return site.

**Completeness-assert-before-return pattern** (clustering.py lines 213-218):
```python
# Verify completeness invariants before returning
assert len(state.assignments) == len(embeddings), "assignments incomplete"
assert len(state.soft_probs) == len(embeddings), "soft_probs incomplete"
assert len(state.clusters) > 0, "No clusters in initial state"

return state
```

**State assembly pattern** (clustering.py lines 205-218):
```python
state = ClusteringState(
    turn_index=0,
    timestamp=timestamp,
    clusters=clusters,
    assignments=assignments,
    soft_probs=soft_probs,
)
assert len(state.assignments) == len(embeddings), "assignments incomplete"
```

**Apply to `src/agent_functions.py`:**
- `f_output(state: ClusteringState) -> ClusteringState` — asserts `len(state.assignments) == N` before return; returns state unmodified (anytime behavior = current state is already complete)
- `f_next_state(state, deltas, store, namer) -> ClusteringState` — applies deltas in type-priority order (global → split/merge → move_item → instructional); calls `ClusterNamer` for affected clusters; asserts soft_probs rows sum to 1.0 after every mutation; assigns new cluster IDs from monotonic counter (never reuse retired IDs)
- `f_next_best_step(state, strategy, uncertainty_report) -> Action` — delegates to `strategy.select()`; no I/O
- All four functions: no try/except, no global mutation, return new state objects

**Import pattern for agent_functions.py:**
```python
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans

from src.state import Cluster, ClusteringState
from src.feedback import (
    FeedbackDelta, SplitFeedback, MergeFeedback,
    MoveItemFeedback, GlobalFeedback, InstructionalFeedback,
    ORACLE_MOVE_CONFIDENCE, UNIFORM_FALLBACK_THRESHOLD,
)
from src.uncertainty import UncertaintyReport

if TYPE_CHECKING:
    from src.embedding_store import EmbeddingStore
    from src.cluster_naming import ClusterNamer
    from src.strategy import StrategyProtocol, Action
```

---

### `src/strategy.py` (utility, transform)

**Analog:** `src/cluster_naming.py` — `typing.Protocol` with `@runtime_checkable` + concrete implementation class that wraps state and delegates.

**Protocol pattern** (cluster_naming.py lines 19-43):
```python
@runtime_checkable
class ClusterNamer(Protocol):
    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        ...
```

**Concrete class wrapping client pattern** (cluster_naming.py lines 109-123):
```python
class AnthropicClusterNamer:
    def __init__(self, client: object) -> None:
        self._client = client

    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        return name_cluster(self._client, sample_texts, cluster_id)
```

**Apply to `src/strategy.py`:** Declare `StrategyProtocol` with `@runtime_checkable` using `typing.Protocol`. `RandomStrategy` wraps a `random.Random(seed)` instance (not `random.random()` global — see anti-patterns in RESEARCH.md). Assert `len(valid_actions) > 0` before `self._rng.choice(valid_actions)`.

```python
from __future__ import annotations
import random
from typing import Protocol, runtime_checkable
from src.state import ClusteringState

@runtime_checkable
class StrategyProtocol(Protocol):
    def select(self, state: ClusteringState, uncertainty_report: "UncertaintyReport") -> "Action":
        ...

class RandomStrategy:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, state: ClusteringState, uncertainty_report: "UncertaintyReport") -> "Action":
        valid_actions = _enumerate_valid_actions(state, uncertainty_report)
        assert len(valid_actions) > 0, "BUG: no valid actions to select from"
        return self._rng.choice(valid_actions)
```

---

### `src/oracle_protocol.py` (model + utility, request-response)

**Analog:** `src/cluster_naming.py` — Protocol definition + dataclass for the reply object + concrete implementation (MockOracle mirrors AnthropicClusterNamer pattern).

**Protocol + dataclass pairing pattern** (cluster_naming.py lines 19-44 for Protocol, lines 109-123 for concrete):
```python
@runtime_checkable
class ClusterNamer(Protocol):
    def name_cluster(self, sample_texts: list[str], cluster_id: int) -> dict[str, str]:
        ...

class AnthropicClusterNamer:
    def __init__(self, client: object) -> None:
        self._client = client
    def name_cluster(self, sample_texts: list[str], cluster_id: int) -> dict[str, str]:
        return name_cluster(self._client, sample_texts, cluster_id)
```

**Apply to `src/oracle_protocol.py`:**
- `OracleReply` is a plain `@dataclass` (not frozen — Phase 3 may add fields without breaking callers)
- `OracleProtocol` uses `@runtime_checkable` + `typing.Protocol`
- `MockOracle` is a scripted sequence: `replies: list[OracleReply]` indexed by `turn_index`; after script exhausted, default to neutral `OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)`
- Assert `isinstance(oracle, OracleProtocol)` pattern at loop entry to catch mis-wiring early

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from src.state import ClusteringState

@dataclass
class OracleReply:
    raw_text: str
    satisfied: bool
    turn_cognitive_load: float = 0.0   # Phase 3 fills in; Phase 2 stub = 0.0

@runtime_checkable
class OracleProtocol(Protocol):
    def reply(self, state: ClusteringState, message: str) -> OracleReply:
        ...

class MockOracle:
    """Scripted turn-indexed reply sequence. Deterministic for 30-turn loop test."""
    def __init__(self, script: list[OracleReply]) -> None:
        assert len(script) > 0, "MockOracle script must be non-empty"
        self._script = script
        self._turn = 0

    def reply(self, state: ClusteringState, message: str) -> OracleReply:
        if self._turn < len(self._script):
            r = self._script[self._turn]
        else:
            r = OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)
        self._turn += 1
        return r
```

---

### `src/conversation_loop.py` (service, event-driven)

**Analog:** `src/clustering.py` (pipeline orchestration) + `src/serialization.py` (`append_to_audit_log` write pattern).

**JSONL write pattern** (serialization.py lines 115-132):
```python
def append_to_audit_log(state: ClusteringState, log_path: str) -> None:
    assert isinstance(log_path, str) and log_path, "log_path must be a non-empty string"
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = serialize_state(state)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
```

**Sequential pipeline pattern** (clustering.py lines 145-219, annotated structure):
```python
def build_initial_clustering_state(embeddings, records, namer, min_cluster_size=None):
    # 1. run HDBSCAN
    # 2. assign noise
    # 3. group item_ids
    # 4. LLM-name each cluster
    # 5. assemble ClusteringState
    # 6. assert completeness invariants
    return state
```

**Apply to `src/conversation_loop.py`:** The loop runs as a background thread (via `socketio.start_background_task()`). Loop body sequence:
1. `action = f_next_best_step(state, strategy, uncertainty_report)`
2. `message = format_message(action, state)`
3. `reply = oracle.reply(state, message)` — OracleProtocol boundary
4. `deltas = parse_feedback(reply.raw_text, state)` — LLM call, only try/except permitted here
5. `new_state = f_next_state(state, deltas, store, namer)`
6. `append_to_audit_log(new_state, log_path)` — JSONL write (D-04: loop owns this)
7. `socketio.emit('state_update', ...)` — use instance method, not module-level emit
8. `stop = check_stopping(new_state.turn_index, reply.satisfied, recent_magnitudes, criteria)`
9. `if stop: socketio.emit('session_stopped', ...); break`
10. `state = new_state`

No try/except inside the loop body except around `parse_feedback`'s `json.loads`.

---

### `web/app.py` (controller, request-response + WebSocket)

**Analog:** None in Phase 1. Use RESEARCH.md Pattern 7 (Flask-SocketIO threading model) as primary reference.

**Reference pattern from RESEARCH.md** (Pattern 7):
```python
from flask import Flask, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

_session = {"state": None, "task": None}

@app.route('/upload', methods=['POST'])
def upload_dataset():
    # ... parse dataset, build initial state
    _session["task"] = socketio.start_background_task(
        run_conversation_loop, initial_state
    )
    return jsonify({"status": "session_started"})

# Run with: socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

**Key conventions from project (apply from cluster_naming.py pattern):**
- Module-level `app` and `socketio` objects (analogous to module-level `EMBEDDING_MODEL` constant)
- `assert` on uploaded data fields (analogous to `assert "text" in record` in embedding_store.py)
- Never `debug=True` in production; always `debug=False` in `socketio.run()`
- `socketio.emit()` (instance method) from background thread, NOT `from flask_socketio import emit`

---

### `web/static/main.js` (utility, event-driven)

**Analog:** None in Phase 1.

**Key pattern from RESEARCH.md:** WebSocket client connects to Flask-SocketIO server. On `'state_update'` event, re-render all cluster cards and metrics sidebar. On `'session_stopped'` event, display stop reason.

```javascript
// Derived from RESEARCH.md Pattern 7 — Flask-SocketIO client side
const socket = io();
socket.on('state_update', function(data) {
    renderClusterCards(data.clusters, data.soft_probs);
    renderMetricsSidebar(data.turn_index, data.conversation_history);
});
socket.on('session_stopped', function(data) {
    showStopBanner(data.reason);
});
```

---

### `web/static/style.css` and `web/templates/index.html` (config)

**Analog:** None in Phase 1. Use CONTEXT.md D-14 as spec: cluster cards grid + metrics sidebar. Vanilla HTML, no build step.

---

### `tests/phase2/*.py` (test)

**Analog:** `tests/phase1/test_stopping_criteria.py` (dataclass + enum tests), `tests/phase1/test_cluster_naming.py` (mock LLM client pattern), `tests/phase1/test_serialization.py` (round-trip tests with `tmp_path`).

**Test file header pattern** (test_stopping_criteria.py lines 1-5):
```python
"""Tests for stopping.py — PRE-02: stopping criteria spec."""
import pytest

from src.stopping import StoppingCriteria, StopReason, check_stopping, FeedbackMagnitudeWeights
```

**Mock LLM client pattern** (test_cluster_naming.py lines 8-15):
```python
def _make_mock_client(name: str, description: str):
    """Build a mock Anthropic client that returns a fixed JSON response."""
    import json
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({"name": name, "description": description}))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client
```

**AssertionError crash test pattern** (test_cluster_naming.py lines 42-50):
```python
def test_name_cluster_crashes_on_bad_llm_schema():
    with pytest.raises(AssertionError, match="bad schema"):
        name_cluster(mock_client, ["Some text."], cluster_id=0)
```

**Fixture-based state construction pattern** (conftest.py lines 19-39):
```python
@pytest.fixture
def tiny_clustering_state_dict():
    return {
        "turn_index": 0,
        "timestamp": "2026-05-04T12:00:00",
        "clusters": [
            {"id": 0, "name": "Positive Reviews", "description": "Happy customers.", "item_ids": [0, 2, 4]},
            {"id": 1, "name": "Negative Reviews", "description": "Unhappy customers.", "item_ids": [1, 3]},
        ],
        "assignments": {0: 0, 1: 1, 2: 0, 3: 1, 4: 0},
        "soft_probs": {
            0: [0.9, 0.1],
            1: [0.1, 0.9],
            2: [0.8, 0.2],
            3: [0.2, 0.8],
            4: [0.85, 0.15],
        },
    }
```

**tmp_path fixture pattern** (test_serialization.py lines 94-100):
```python
def test_append_to_audit_log_creates_file(tmp_path):
    log_path = str(tmp_path / "audit_log.jsonl")
    state = _make_tiny_state()
    append_to_audit_log(state, log_path)
    assert os.path.exists(log_path)
```

**Apply to `tests/phase2/`:**
- Each test file imports only from the module under test
- Shared Phase 2 fixtures go in `tests/conftest.py` extension (tiny 3-cluster ClusteringState, MockOracle factory, mock embeddings with shape (N, 768))
- `@pytest.mark.llm` gate on any test that calls real Anthropic API
- `MockOracle` in tests uses scripted `OracleReply` list indexed by turn (see oracle_protocol.py pattern)
- Assert-crash tests use `pytest.raises(AssertionError)` matching specific message substrings

---

## Shared Patterns

### Fail-Loudly Assert Pattern
**Source:** `src/clustering.py` (throughout), `src/serialization.py` (lines 58-62, 86-90)
**Apply to:** All Phase 2 modules (no try/except except `feedback_parser.py` around `json.loads`)

```python
# From serialization.py lines 58-62
assert isinstance(state, ClusteringState), (
    f"serialize_state expects ClusteringState, got {type(state)}"
)
line = json.dumps(dataclasses.asdict(state), cls=_StateEncoder)
assert "\n" not in line, "BUG: serialized state contains newline (would break JSONL)"
```

```python
# From clustering.py lines 57-58
assert embeddings.ndim == 2, f"Expected 2D array, got shape {embeddings.shape}"
assert embeddings.shape[0] > 0, "Cannot cluster empty embedding set"
```

### typing.Protocol Pattern
**Source:** `src/cluster_naming.py` lines 19-43
**Apply to:** `src/strategy.py` (StrategyProtocol), `src/oracle_protocol.py` (OracleProtocol)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ClusterNamer(Protocol):
    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        ...
```

### `from __future__ import annotations` Header
**Source:** All Phase 1 src files (state.py line 17, clustering.py line 12, stopping.py line 20, serialization.py line 14)
**Apply to:** Every new Phase 2 Python module — always the first non-docstring line.

### LLM Call + JSON Parse + Schema Assert Pattern
**Source:** `src/cluster_naming.py` lines 68-106
**Apply to:** `src/feedback_parser.py` (only Phase 2 file that calls the LLM)

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt}],
)
raw_text = response.content[0].text.strip()
if raw_text.startswith("```"):
    raw_text = raw_text.split("```")[1]
    if raw_text.startswith("json"):
        raw_text = raw_text[4:]
    raw_text = raw_text.strip()
result = json.loads(raw_text)
assert "name" in result and "description" in result, (
    f"LLM returned bad schema for cluster {cluster_id}: {result}"
)
```

### Soft-Probs Completeness Invariant
**Source:** `src/clustering.py` lines 213-218
**Apply to:** `f_output`, `f_next_state` return sites in `src/agent_functions.py`

```python
assert len(state.assignments) == len(embeddings), "assignments incomplete"
assert len(state.soft_probs) == len(embeddings), "soft_probs incomplete"
assert len(state.clusters) > 0, "No clusters in initial state"
```

### Named Module-Level Constants (not magic numbers)
**Source:** `src/clustering.py` lines 29-30, `src/embedding_store.py` lines 20-22
**Apply to:** `src/feedback.py` (ORACLE_MOVE_CONFIDENCE, UNIFORM_FALLBACK_THRESHOLD), `src/uncertainty.py`

```python
# From clustering.py
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES = 10

# From embedding_store.py
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
BATCH_SIZE = 32
```

### `TYPE_CHECKING` Guard for Circular Imports
**Source:** `src/clustering.py` lines 14-18
**Apply to:** `src/agent_functions.py`, `src/conversation_loop.py` (import EmbeddingStore, ClusterNamer, StrategyProtocol only at type-check time)

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.cluster_naming import ClusterNamer
```

### Test Mock Client Pattern
**Source:** `tests/phase1/test_cluster_naming.py` lines 8-15
**Apply to:** `tests/phase2/test_feedback_parser.py` (MockLLMClient for parse_feedback unit tests)

```python
from unittest.mock import MagicMock
import json

def _make_mock_client(response_json: list) -> MagicMock:
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(response_json))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client
```

---

## No Analog Found

Files with no close match in the codebase (planner uses RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `web/app.py` | controller | request-response + WebSocket | No Flask or web framework files exist in Phase 1 |
| `web/static/main.js` | utility | event-driven (WebSocket client) | No JavaScript files exist anywhere in the project |
| `web/static/style.css` | config | n/a | No CSS files exist |
| `web/templates/index.html` | config | n/a | No HTML template files exist |

**For these files:** Use RESEARCH.md Patterns 7 (Flask-SocketIO threading model) and CONTEXT.md D-13/D-14/D-15 as sole references. Apply project-wide conventions: no debug=True, no magic numbers inline, assert on uploaded data fields.

---

## Metadata

**Analog search scope:** `src/` (6 Phase 1 files), `tests/conftest.py`, `tests/phase1/` (7 test files)
**Files scanned:** 13 source files read in full
**Pattern extraction date:** 2026-05-07
