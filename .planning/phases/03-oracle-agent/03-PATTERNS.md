# Phase 3: Oracle Agent - Pattern Map

**Mapped:** 2026-05-11
**Files analyzed:** 9 (6 new, 3 modified)
**Analogs found:** 9 / 9

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/oracle_agent.py` | service/agent | request-response (LLM) | `src/feedback_parser.py` | role-match (LLM call pattern) + `src/oracle_protocol.py` (class structure) |
| `src/cognitive_load.py` | utility (pure function) | transform | `src/uncertainty.py` | exact (pure function, assert-first, named constants, returns scalar/dataclass) |
| `src/oracle_protocol.py` (modify) | interface/dataclass | — | itself (extend OracleReply) | self-match |
| `src/conversation_loop.py` (modify) | orchestrator | request-response | itself (add two new steps) | self-match |
| `tests/phase3/__init__.py` | test init | — | `tests/phase2/__init__.py` | exact (empty file) |
| `tests/phase3/test_oracle_agent.py` | test | unit | `tests/phase2/test_feedback_parser.py` | exact (mock LLM client, MagicMock, pytest.raises) |
| `tests/phase3/test_cognitive_load.py` | test | unit | `tests/phase2/test_uncertainty.py` | exact (pure function test structure, assert range, edge cases) |
| `tests/phase3/test_oracle_loop_integration.py` | test | integration | `tests/phase2/test_conversation_loop.py` | exact (run_conversation + MagicMock namer + tmp_path) |
| `tests/conftest.py` (modify) | test fixture | — | itself (extend with new fixture) | self-match |

---

## Pattern Assignments

---

### `src/oracle_agent.py` (service/agent, request-response)

**Primary analog:** `src/feedback_parser.py` (LLM call pattern, assert-loudly, no try/except outside API boundary)
**Secondary analog:** `src/oracle_protocol.py` (OracleReply, MockOracle class structure)

**Imports pattern** — copy from `src/feedback_parser.py` lines 1-23 and `src/oracle_protocol.py` lines 1-13:

```python
"""
oracle_agent.py — OracleAgent, OracleSpec, NoiseParams (ORC-01..ORC-04).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.cognitive_load import COG_LOAD_THRESHOLD, f_cognitive_load
from src.feedback import (
    FeedbackDelta,
    InstructionalFeedback,
    MergeFeedback,
    MoveItemFeedback,
    SplitFeedback,
)
from src.oracle_protocol import OracleReply

if TYPE_CHECKING:
    from src.state import ClusteringState
```

**Dataclass-first pattern** — copy from `src/feedback.py` lines 28-77 (frozen=True for value objects; NOT frozen for OracleSpec/NoiseParams which are mutable holders):

```python
# src/feedback.py lines 28-65 — frozen dataclass pattern for value types
@dataclass(frozen=True)
class SplitFeedback:
    cluster_id: int
    seed_item_ids: list[int]
```

For OracleSpec and NoiseParams use plain `@dataclass` (NOT frozen — they are
configuration holders injected at construction, never compared as dict keys):

```python
@dataclass
class OracleSpec:
    preferred_k: int
    semantic_axes: list[str]
    persona_description: str

@dataclass
class NoiseParams:
    consistency_rate: float       # 0.0-1.0
    drift_probability: float      # 0.0-1.0
    sycophancy_resistance: float  # 0.0-1.0
```

**Assert-loudly invariant pattern** — copy from `src/uncertainty.py` lines 42-43 and `src/serialization.py` lines 58-62:

```python
# src/uncertainty.py lines 42-43 — assert preconditions first, before any computation
assert K > 0, "f_uncertainty called on empty ClusteringState (no clusters)"
log_K = np.log(K) if K > 1 else 1.0  # avoid div-by-zero for K=1

# src/serialization.py lines 58-62 — type assertion before work
assert isinstance(state, ClusteringState), (
    f"serialize_state expects ClusteringState, got {type(state)}"
)
```

Apply to OracleAgent.__init__:
```python
def __init__(self, spec: OracleSpec, noise_params: NoiseParams,
             client: object, model: str = "claude-haiku-4-5",
             window_size: int = 10) -> None:
    assert 0.0 <= noise_params.consistency_rate <= 1.0
    assert 0.0 <= noise_params.drift_probability <= 1.0
    assert 0.0 <= noise_params.sycophancy_resistance <= 1.0
    self._spec = spec
    self._noise = noise_params
    self._client = client
    self._model = model
    self._delta_window: deque = deque(maxlen=window_size)
```

**LLM call + ONLY try/except pattern** — copy from `src/feedback_parser.py` lines 144-151:

```python
# src/feedback_parser.py lines 144-151 — only permitted try/except is at the API boundary
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)
cleaned_text = response.content[0].text.strip()
```

For OracleAgent the call uses the same duck-type interface. Because `_OpenAIAdapter`
and `_GoogleAdapter` (see `src/llm_provider.py` lines 54-69, 84-104) do NOT accept a
`system=` kwarg, the system prompt must be prepended to the user message. Detect the
provider by checking for `anthropic` in the module of `self._client`:

```python
# Pattern: provider-aware system prompt delivery
import anthropic as _anthropic_module  # noqa: F401 (import for isinstance)

try:
    import anthropic as _anthropic_module
    _is_anthropic = isinstance(self._client, _anthropic_module.Anthropic)
except ImportError:
    _is_anthropic = False

if _is_anthropic:
    # Anthropic SDK: system= is a top-level kwarg (NOT a message role)
    try:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
        )
    except Exception as exc:
        raise RuntimeError(
            f"OracleAgent LLM call failed at turn {state.turn_index}: {exc}"
        ) from exc
else:
    # OpenAI/Google adapters: prepend system prompt as leading user content
    full_message = system_prompt + "\n\n" + message
    try:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": full_message}],
        )
    except Exception as exc:
        raise RuntimeError(
            f"OracleAgent LLM call failed at turn {state.turn_index}: {exc}"
        ) from exc

raw_text = response.content[0].text
```

**Stateful deque pattern** — from `src/oracle_protocol.py` (MockOracle maintains `self._turn` counter), extended to `deque(maxlen=N)`:

```python
# src/oracle_protocol.py lines 52-54 — stateful private field pattern
def __init__(self, script: list[OracleReply]) -> None:
    assert len(script) > 0, "MockOracle script must be non-empty"
    self._script = script
    self._turn = 0
```

For OracleAgent the rolling window follows the same private-field pattern:
```python
self._delta_window: deque = deque(maxlen=window_size)
# After each reply, extend the window:
for delta in new_deltas:
    self._delta_window.append((state.turn_index, delta))
```

**JSONL event write pattern** — copy from `src/serialization.py` lines 126-132 (append_to_audit_log):

```python
# src/serialization.py lines 126-132 — open in append mode, write one JSON line
parent = os.path.dirname(log_path)
if parent:
    os.makedirs(parent, exist_ok=True)
line = serialize_state(state)
with open(log_path, "a", encoding="utf-8") as f:
    f.write(line + "\n")
```

For oracle_init and drift_event sidecar records (write to `events.jsonl`, NOT to `audit_log.jsonl` — see Pitfall 5 in RESEARCH.md):
```python
# Same open-append pattern, different file path
import json, os

def _write_event(record: dict, events_path: str) -> None:
    parent = os.path.dirname(events_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

---

### `src/cognitive_load.py` (utility, transform)

**Analog:** `src/uncertainty.py` — exact match (pure function, no I/O, no global state, named constants, assert preconditions, returns scalar)

**Module structure pattern** — copy from `src/uncertainty.py` lines 1-11 (module docstring, imports, no global state):

```python
# src/uncertainty.py lines 1-11
"""
uncertainty.py — f_uncertainty pure function + UncertaintyReport dataclass (CLUS-02).

Computes normalized Shannon entropy per item...
No I/O. No global state. Pure function only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.state import ClusteringState
```

**Named constants pattern** — copy from `src/feedback.py` lines 24-25:

```python
# src/feedback.py lines 24-25 — module-level named constants, not magic numbers inline
ORACLE_MOVE_CONFIDENCE: float = 0.95
UNIFORM_FALLBACK_THRESHOLD: float = 1e-9
```

For cognitive_load.py:
```python
MAX_K: int = 20
MAX_MSG_LEN: int = 500
TOP_K_ITEMS_PER_CLUSTER: int = 5
COG_LOAD_THRESHOLD: float = 0.7
```

**Pure function with assert-first pattern** — copy from `src/uncertainty.py` lines 31-43:

```python
# src/uncertainty.py lines 31-43 — assert preconditions, then pure computation
def f_uncertainty(state: ClusteringState) -> UncertaintyReport:
    """
    Pure function. No I/O. No global state.
    ...
    """
    K = len(state.clusters)
    assert K > 0, "f_uncertainty called on empty ClusteringState (no clusters)"
    log_K = np.log(K) if K > 1 else 1.0
```

For cognitive_load.py:
```python
def f_cognitive_load(state: "ClusteringState", message: str) -> float:
    """
    Pure function. No I/O. No global state.
    Compute normalized cognitive load score in [0, 1].
    """
    assert len(state.clusters) > 0, "f_cognitive_load: no clusters in state"
    total_items = len(state.assignments)
    assert total_items > 0, "f_cognitive_load: no items in state"
    # ... formula using named constants
```

**min() clamping pattern** — from `src/uncertainty.py` lines 47-51 (normalize to [0,1]):

```python
# src/uncertainty.py lines 47-51 — normalization guard
p = np.array(probs, dtype=np.float64)
mask = p > 0
h = -np.sum(p[mask] * np.log(p[mask]))
item_entropy[item_id] = float(h / log_K)  # normalized to [0, 1]
```

For cognitive_load.py use `min(..., 1.0)` clamping so no term ever exceeds 1.0:
```python
w = 1.0 / 3.0
cluster_term = min(len(state.clusters) / MAX_K, 1.0) * w
items_shown = len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER
items_term = min(items_shown / total_items, 1.0) * w
msg_term = min(len(message) / MAX_MSG_LEN, 1.0) * w
return cluster_term + items_term + msg_term
```

---

### `src/oracle_protocol.py` (modify — extend OracleReply)

**Self-match.** The only change is adding two fields to the existing `OracleReply` dataclass.

**Existing dataclass field addition pattern** — `src/oracle_protocol.py` lines 16-29. OracleReply is NOT frozen (note in docstring line 26: "NOT frozen — Phase 3 may add fields"):

```python
# src/oracle_protocol.py lines 16-29 — existing OracleReply (current state)
@dataclass
class OracleReply:
    raw_text: str
    satisfied: bool
    turn_cognitive_load: float = 0.0  # Phase 3 fills this in; Phase 2 stub = 0.0
```

Add Phase 3 fields with defaults so existing callers (MockOracle, tests) do not break:
```python
@dataclass
class OracleReply:
    raw_text: str
    satisfied: bool
    turn_cognitive_load: float = 0.0
    contradiction_detected: bool = False        # Phase 3: drift detection result
    contradicted_turn: int | None = None        # Phase 3: which prior turn was contradicted
```

---

### `src/conversation_loop.py` (modify — wire f_cognitive_load, drift logging, oracle_init)

**Self-match.** Three surgical insertions into `run_conversation()`.

**Existing step insertion pattern** — `src/conversation_loop.py` lines 111-134. Steps are
numbered comments; new steps follow the same numbering style.

```python
# src/conversation_loop.py lines 119-121 — CURRENT Step 3 (before oracle.reply)
message = _format_message(action, state)
reply = oracle.reply(state, message)
```

Phase 3 modification — insert cognitive load before oracle.reply(), pass global_instructions to OracleAgent only:

```python
# Step 3a: Compute cognitive load (Phase 3 — pure function, no I/O)
from src.cognitive_load import f_cognitive_load
cognitive_load = f_cognitive_load(state, message)

# Step 3b: Get oracle reply (pass global_instructions to OracleAgent if supported)
from src.oracle_agent import OracleAgent
if isinstance(oracle, OracleAgent):
    reply = oracle.reply(state, message, global_instructions=global_instructions)
else:
    reply = oracle.reply(state, message)
```

After receiving reply, add drift event logging (between current Step 6 AuditLog write and Step 7 SocketIO emit):

```python
# Step 6c: Log drift event if oracle detected a contradiction (Phase 3 — D-10)
if reply.contradiction_detected:
    drift_record = {
        "event": "drift_event",
        "turn": new_state.turn_index,
        "contradicted_turn": reply.contradicted_turn,
        "timestamp": new_state.timestamp,
    }
    _write_event(drift_record, events_path)   # sidecar events.jsonl
```

**import pattern** — `src/conversation_loop.py` lines 21-34 (TYPE_CHECKING guard for heavy imports):

```python
# src/conversation_loop.py lines 19-34 — TYPE_CHECKING for circular-safe imports
from typing import TYPE_CHECKING, Optional, Callable
if TYPE_CHECKING:
    from src.oracle_protocol import OracleProtocol
    from src.oracle_agent import OracleAgent  # Phase 3 — add here
```

**oracle_init event** — write at the top of `run_conversation()`, just after hierarchy and global_instructions are initialized (lines 100-107), before the while loop:

```python
# src/conversation_loop.py lines 100-107 — existing initialization block (location for oracle_init)
hierarchy = HierarchyStore()
for cluster in initial_state.clusters:
    hierarchy.register(cluster.id)
global_instructions: list[str] = []

# Phase 3 addition (after global_instructions init):
from src.oracle_agent import OracleAgent
if isinstance(oracle, OracleAgent):
    _log_oracle_init(oracle, events_path, initial_state)
```

---

### `tests/phase3/__init__.py` (empty init)

**Analog:** `tests/phase2/__init__.py` — exact match (empty file, zero bytes).

```python
# empty — same as tests/phase2/__init__.py
```

---

### `tests/phase3/test_oracle_agent.py` (unit test)

**Analog:** `tests/phase2/test_feedback_parser.py` — exact match (mock LLM client via MagicMock, fixture-based state, pytest.raises for fail-loudly assertions)

**Mock LLM client builder pattern** — copy from `tests/phase2/test_feedback_parser.py` lines 7-13:

```python
# tests/phase2/test_feedback_parser.py lines 7-13 — mock Anthropic-compatible client
def _make_mock_client(response_payload):
    """Build a mock Anthropic client returning response_payload as JSON text."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(response_payload))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client
```

For OracleAgent tests, the mock client returns free-text oracle replies:
```python
def _make_oracle_client(reply_text: str):
    """Build a mock client returning free-text oracle reply."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=reply_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client
```

**pytest.raises pattern** — copy from `tests/phase2/test_feedback_parser.py` lines 78-84:

```python
# tests/phase2/test_feedback_parser.py lines 78-84
def test_parse_feedback_crashes_on_invalid_cluster_id(tiny_state_3cluster):
    from src.feedback_parser import parse_feedback
    payload = [{"type": "split", "cluster_id": 999, "seed_item_ids": []}]
    client = _make_mock_client(payload)
    with pytest.raises(AssertionError):
        parse_feedback("Split cluster 999.", tiny_state_3cluster, client)
```

Apply to OracleAgent noise param validation:
```python
def test_oracle_agent_crashes_on_invalid_noise_params(tiny_state_3cluster):
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="test")
    bad_noise = NoiseParams(consistency_rate=1.5, drift_probability=0.1,
                            sycophancy_resistance=0.9)  # rate > 1.0 — should crash
    with pytest.raises(AssertionError):
        OracleAgent(spec=spec, noise_params=bad_noise, client=MagicMock())
```

**Protocol structural subtyping test pattern** — copy from `tests/phase2/test_oracle_protocol.py` lines 57-62:

```python
# tests/phase2/test_oracle_protocol.py lines 57-62 — isinstance check for Protocol
def test_mock_oracle_satisfies_oracle_protocol(tiny_state_3cluster):
    from src.oracle_protocol import OracleProtocol, MockOracle
    oracle = MockOracle(script=[_make_reply()])
    assert isinstance(oracle, OracleProtocol)
```

Apply to OracleAgent:
```python
def test_oracle_agent_satisfies_protocol(tiny_state_3cluster):
    from src.oracle_protocol import OracleProtocol
    from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="x")
    noise = NoiseParams(consistency_rate=0.8, drift_probability=0.1,
                        sycophancy_resistance=0.9)
    agent = OracleAgent(spec=spec, noise_params=noise, client=MagicMock())
    assert isinstance(agent, OracleProtocol)
```

**@pytest.mark.llm pattern** — copy from `tests/phase2/test_feedback_parser.py` lines 96-107:

```python
# tests/phase2/test_feedback_parser.py lines 96-107 — real LLM test, skipped in CI
@pytest.mark.llm
def test_parse_feedback_real_llm(tiny_state_3cluster):
    import anthropic
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)
    ...
```

---

### `tests/phase3/test_cognitive_load.py` (unit test)

**Analog:** `tests/phase2/test_uncertainty.py` — exact match (pure function tests, range assertions, edge-case crash test, no mocking needed)

**Range assertion pattern** — copy from `tests/phase2/test_uncertainty.py` lines 29-34:

```python
# tests/phase2/test_uncertainty.py lines 29-34 — value-in-range check
def test_entropy_values_in_zero_one(tiny_state_3cluster):
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    for _, entropy in report.boundary_items:
        assert 0.0 <= entropy <= 1.0, f"Entropy {entropy} outside [0, 1]"
```

Apply to cognitive_load:
```python
def test_load_in_range(tiny_state_3cluster):
    from src.cognitive_load import f_cognitive_load
    load = f_cognitive_load(tiny_state_3cluster, "show clusters")
    assert 0.0 <= load <= 1.0, f"load {load} outside [0, 1]"
```

**Crash-on-empty-state pattern** — copy from `tests/phase2/test_uncertainty.py` lines 77-89:

```python
# tests/phase2/test_uncertainty.py lines 77-89 — empty state crash test
def test_f_uncertainty_crashes_on_empty_state():
    from src.uncertainty import f_uncertainty
    from src.state import ClusteringState
    empty = ClusteringState(
        turn_index=0, timestamp="2026-05-07T00:00:00",
        clusters=[], assignments={}, soft_probs={},
    )
    with pytest.raises(AssertionError):
        f_uncertainty(empty)
```

Apply to cognitive_load:
```python
def test_f_cognitive_load_crashes_on_empty_state():
    from src.cognitive_load import f_cognitive_load
    from src.state import ClusteringState
    empty = ClusteringState(
        turn_index=0, timestamp="2026-05-11T00:00:00",
        clusters=[], assignments={}, soft_probs={},
    )
    with pytest.raises(AssertionError):
        f_cognitive_load(empty, "any message")
```

**Pure function idempotency pattern** — copy from `tests/phase2/test_uncertainty.py` lines 68-74:

```python
# tests/phase2/test_uncertainty.py lines 68-74 — same input → same output (pure)
def test_f_uncertainty_pure_no_side_effects(tiny_state_3cluster):
    r1 = f_uncertainty(tiny_state_3cluster)
    r2 = f_uncertainty(tiny_state_3cluster)
    assert r1.boundary_items == r2.boundary_items
```

---

### `tests/phase3/test_oracle_loop_integration.py` (integration test)

**Analog:** `tests/phase2/test_conversation_loop.py` — exact match (MagicMock namer, EmbeddingStore, tmp_path, run_conversation, load_audit_log)

**Integration test scaffold** — copy from `tests/phase2/test_conversation_loop.py` lines 25-47:

```python
# tests/phase2/test_conversation_loop.py lines 25-47 — full integration loop scaffold
def test_30_turn_loop_completes(tiny_state_3cluster, mock_embeddings_3cluster, tmp_path):
    from src.oracle_protocol import MockOracle
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    oracle = MockOracle(script=_build_30_turn_script())
    criteria = StoppingCriteria(turn_budget=50)
    log_path = str(tmp_path / "audit.jsonl")
    final_state = run_conversation(
        initial_state=tiny_state_3cluster,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=criteria,
        socketio=None,
    )
    assert final_state is not None
```

For OracleAgent integration test, replace MockOracle with OracleAgent + mock client:
```python
def test_oracle_agent_loop_5_turns(tiny_state_3cluster, mock_embeddings_3cluster,
                                    oracle_agent_factory, tmp_path):
    from src.conversation_loop import run_conversation
    from src.stopping import StoppingCriteria
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock

    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    # oracle_agent_factory fixture (see conftest.py pattern below)
    oracle = oracle_agent_factory(reply_text="looks good to me")
    criteria = StoppingCriteria(turn_budget=5)
    log_path = str(tmp_path / "audit.jsonl")
    final_state = run_conversation(
        initial_state=tiny_state_3cluster,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=criteria,
        socketio=None,
    )
    assert final_state is not None
```

**tmp_path + events sidecar pattern** — copy from `tests/phase2/test_conversation_loop.py` lines 56-76:

```python
# tests/phase2/test_conversation_loop.py lines 56-76 — tmp_path for log files
log_path = str(tmp_path / "audit.jsonl")
# For phase3 add sidecar:
events_path = str(tmp_path / "events.jsonl")
```

---

### `tests/conftest.py` (modify — add oracle_agent_factory fixture)

**Self-match.** Add new fixture following the existing `mock_oracle_factory` pattern.

**Factory fixture pattern** — copy from `tests/conftest.py` lines 114-125:

```python
# tests/conftest.py lines 114-125 — factory fixture pattern (returns a callable)
@pytest.fixture
def mock_oracle_factory():
    """
    Factory for MockOracle with a scripted OracleReply sequence.
    Usage: mock_oracle_factory([reply1, reply2, ...])
    """
    def _factory(replies):
        from src.oracle_protocol import MockOracle
        return MockOracle(script=replies)
    return _factory
```

New fixture to add immediately after mock_oracle_factory:

```python
@pytest.fixture
def oracle_agent_factory():
    """
    Factory for OracleAgent with a MagicMock LLM client.
    Usage: oracle_agent_factory(reply_text="some oracle reply")
    The mock client returns reply_text for every call to client.messages.create().

    Requires src/oracle_agent.py to exist (will ImportError otherwise).
    """
    def _factory(reply_text: str = "looks good to me",
                 consistency_rate: float = 0.8,
                 drift_probability: float = 0.1,
                 sycophancy_resistance: float = 0.9,
                 preferred_k: int = 3):
        from unittest.mock import MagicMock
        from src.oracle_agent import OracleAgent, OracleSpec, NoiseParams
        spec = OracleSpec(
            preferred_k=preferred_k,
            semantic_axes=["topic"],
            persona_description="A neutral test analyst.",
        )
        noise = NoiseParams(
            consistency_rate=consistency_rate,
            drift_probability=drift_probability,
            sycophancy_resistance=sycophancy_resistance,
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=reply_text)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        return OracleAgent(spec=spec, noise_params=noise, client=mock_client)
    return _factory
```

---

## Shared Patterns

### 1. Fail-Loudly (assert-first, no bare except)

**Source:** `src/uncertainty.py` lines 42-43, `src/serialization.py` lines 58-62, `src/feedback_parser.py` lines 68-70
**Apply to:** All new functions in `src/oracle_agent.py` and `src/cognitive_load.py`

```python
# Pattern: assert preconditions at entry, let unexpected state crash immediately
assert len(state.clusters) > 0, "descriptive message about what invariant failed"
assert isinstance(thing, ExpectedType), f"got {type(thing)}"
# No try/except except at LLM API call boundary
```

### 2. LLM Client Duck-Type Interface

**Source:** `src/feedback_parser.py` lines 144-151, `src/llm_provider.py` lines 40-104
**Apply to:** `OracleAgent.reply()` LLM call

```python
# Pattern: always use duck-type interface client.messages.create(model, max_tokens, messages)
# The adapter shims in llm_provider.py guarantee this works across Anthropic/OpenAI/Google.
# EXCEPTION: Anthropic-specific system= kwarg requires provider detection first.
response = client.messages.create(
    model=self._model,
    max_tokens=512,
    messages=[{"role": "user", "content": content}],
)
text = response.content[0].text
```

### 3. JSONL Append Pattern

**Source:** `src/serialization.py` lines 126-132
**Apply to:** `_write_event()` helper in `src/conversation_loop.py` for oracle_init and drift_event records

```python
# Pattern: open(..., "a") + json.dumps + newline — same for all JSONL writes
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
```

### 4. TYPE_CHECKING Import Guard

**Source:** `src/conversation_loop.py` lines 19-34, `src/oracle_protocol.py` line 8
**Apply to:** `src/oracle_agent.py` (imports ClusteringState only for type hints)

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.state import ClusteringState
```

### 5. MagicMock LLM Client in Tests

**Source:** `tests/phase2/test_feedback_parser.py` lines 7-13
**Apply to:** All test functions in `tests/phase3/` that test OracleAgent without hitting real API

```python
from unittest.mock import MagicMock
mock_response = MagicMock()
mock_response.content = [MagicMock(text="<reply text here>")]
mock_client = MagicMock()
mock_client.messages.create.return_value = mock_response
```

### 6. Named Constants at Module Level

**Source:** `src/feedback.py` lines 24-25, `src/uncertainty.py` (all constants inline in function)
**Apply to:** `src/cognitive_load.py` module-level constants

```python
# Pattern: UPPER_CASE module-level constants, never magic numbers inline
ORACLE_MOVE_CONFIDENCE: float = 0.95   # from feedback.py line 24
# For cognitive_load.py:
MAX_K: int = 20
MAX_MSG_LEN: int = 500
TOP_K_ITEMS_PER_CLUSTER: int = 5
COG_LOAD_THRESHOLD: float = 0.7
```

---

## No Analog Found

All files have analogs. No new patterns need to be invented from scratch.

---

## Critical Integration Notes for Planner

1. **events.jsonl sidecar (Pitfall 5):** oracle_init and drift_event records MUST NOT be written to `audit_log.jsonl`. Write to a separate `events.jsonl` in the same session directory. `load_audit_log()` (serialization.py lines 135-150) calls `deserialize_state()` on every line and will crash on non-state records. This is a Wave 0 blocker.

2. **llm_provider adapter incompatibility (Pitfall 2):** `_OpenAIAdapter.messages.create` (llm_provider.py lines 58-69) and `_GoogleMessages.create` (lines 88-104) do NOT accept `system=` kwarg. OracleAgent must detect the provider type and handle both paths. See `OracleAgent.reply()` LLM call pattern above.

3. **Merged cluster ID (Pitfall 3 / Assumption A1):** Before implementing `_contradicts()`, read `src/agent_functions.py` merge branch to verify whether merged cluster gets `cluster_a_id`, `cluster_b_id`, or a new monotonic ID (D-11). The contradiction rule depends on this.

4. **run_conversation signature:** `run_conversation()` (conversation_loop.py lines 60-72) must receive `events_path` to write oracle_init and drift_event records. Either add it as a new optional parameter (default `None`) or derive it from `log_path` by replacing the filename suffix.

---

## Metadata

**Analog search scope:** `src/`, `tests/phase2/`, `tests/conftest.py`
**Files scanned:** 12 source files read in full
**Pattern extraction date:** 2026-05-11
