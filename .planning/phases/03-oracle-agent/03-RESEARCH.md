# Phase 3: Oracle Agent - Research

**Researched:** 2026-05-11
**Domain:** LLM oracle simulation — prompt engineering, cognitive load modeling, structural drift detection
**Confidence:** HIGH (all critical implementation details verified against codebase and SDK docs)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** `OracleSpec` dataclass — `preferred_k: int`, `semantic_axes: list[str]`, `persona_description: str`. Fixed for agent lifetime.
- **D-02:** `OracleAgent.__init__(spec: OracleSpec, noise_params: NoiseParams, ...)`. Spec and noise injected at construction, never mutated mid-run.
- **D-03:** `OracleSpec` and `NoiseParams` are separate dataclasses.
- **D-04:** Noise params as prompt-injected natural-language behavioral rules — no post-processing, no temperature manipulation, no second LLM call.
  - `consistency_rate` → "You agree with the proposed clustering X% of the time. The other Y% you request minor adjustments."
  - `drift_probability` → "With probability X you introduce a new preference per turn that may contradict a prior one."
  - `sycophancy_resistance` → "You maintain your stated position even when the system pushes back, at rate X."
- **D-05:** `NoiseParams(consistency_rate: float, drift_probability: float, sycophancy_resistance: float)`. All three logged as `oracle_init` JSONL event at run start.
- **D-06:** Standalone `f_cognitive_load(state, message) -> float` in `src/cognitive_load.py`. Loop calls it BEFORE `oracle.reply()`.
- **D-07:** Formula: `load = (len(state.clusters)/MAX_K)*w1 + (items_shown/total_items)*w2 + (len(message)/MAX_MSG_LEN)*w3`. Default `w1=w2=w3=1/3`. `items_shown = len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER`.
- **D-08:** `COG_LOAD_THRESHOLD = 0.7`. Above: inject "OVERLOAD: Focus on one thing only." into system prompt END. Below: no constraint.
- **D-09:** Structural `FeedbackDelta` comparison using a rolling `deque(maxlen=N)` of `(turn_index, delta)` pairs. Zero LLM cost. Contradiction rules defined.
- **D-10:** Drift logged to AuditLog only (`drift_event` JSONL records). No `drift_history` field on `OracleAgent`.
- **D-11:** `OracleReply` extended with `contradiction_detected: bool = False` and `contradicted_turn: int | None = None`.
- **D-12:** `InstructionalFeedback` wired via existing `global_instructions` accumulator in `conversation_loop.py`.

### Claude's Discretion

- Which Anthropic model to use (Haiku for speed in ablations, Sonnet for quality in human-comparison runs).
- Exact system prompt structure (ordering of spec, noise params, cognitive load, global instructions, state summary).
- Rolling window size N for structural drift comparison (default suggested: 10 turns).
- Named constants: `MAX_K`, `MAX_MSG_LEN`, `TOP_K_ITEMS_PER_CLUSTER`.

### Deferred Ideas (OUT OF SCOPE)

- Mutable noise params (fatigue simulation, turn-by-turn `consistency_rate` degradation) — Phase 5.
- LLM-based semantic contradiction detection — Phase 5 ablation.
- Per-persona `cog_load_threshold` in `NoiseParams` — deferred.
- BACK-V2-02 (representation choice) — out of scope for all phases so far.
- 4th synthetic data agent — deferred.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ORC-01 | Oracle Agent is an LLM with explicit preference spec and persona — not a bare LLM call | D-01/D-02: `OracleSpec` + `OracleAgent` class; system prompt assembles spec + persona + noise rules; `reply()` returns structured `OracleReply` |
| ORC-02 | Explicit noise parameters: `consistency_rate`, `drift_probability`, `sycophancy_resistance` — measurably different behavior across runs | D-04/D-05: prompt-injected behavioral rules; `oracle_init` JSONL logging enables cross-run correlation |
| ORC-03 | Per-turn cognitive-load score received by oracle; simulates fatigue/overload above threshold | D-06/D-07/D-08: `f_cognitive_load()` pure function; threshold gate; OVERLOAD instruction injected into system prompt |
| ORC-04 | Drift detection — contradictions with prior intent detected, logged, flagged to Clustering Agent | D-09/D-10/D-11: structural `FeedbackDelta` deque; `drift_event` JSONL; `OracleReply.contradiction_detected` |
| FB-04 | Instructional feedback parsed into structured constraints applied on next turn | D-12: `InstructionalFeedback.instruction_text` appended to `global_instructions`; injected into oracle system prompt next turn |
</phase_requirements>

---

## Summary

Phase 3 adds the real LLM oracle that replaces `MockOracle` everywhere. The implementation spans three new files (`src/oracle_agent.py`, `src/cognitive_load.py`, and the new test module) plus small modifications to `src/oracle_protocol.py` (extend `OracleReply`) and `src/conversation_loop.py` (wire cognitive load and drift logging).

The LLM backend is already abstracted by `src/llm_provider.py`. The Anthropic Python SDK (v0.101.0, available via pip) uses a synchronous `client.messages.create(model, max_tokens, system=..., messages=[...])` call — the `system` parameter is a top-level string, separate from the `messages` list. The only permitted `try/except` is at this API call boundary. Everything else asserts and crashes loudly per CLAUDE.md.

Noise simulation is entirely prompt-engineering: three behavioral rules injected into the system prompt. No post-processing, no temperature manipulation. Cognitive load is a pure mathematical function computed before each call. Drift detection compares typed Python dataclasses in a rolling deque — zero LLM cost.

**Primary recommendation:** implement `OracleAgent` as a single class in `src/oracle_agent.py` that assembles a multi-section system prompt and calls the existing `llm_provider` client. The loop changes are minimal (two new lines before and after `oracle.reply()`). All new behavior is testable without hitting the real API by using `unittest.mock.patch` on the LLM client.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Oracle preference spec + persona | OracleAgent (src/oracle_agent.py) | — | Spec is injected at construction; oracle owns its own prompt |
| Noise rule injection | OracleAgent (prompt assembly) | — | D-04: rules become natural-language sentences in system prompt |
| Cognitive load computation | cognitive_load.py (pure function) | conversation_loop.py (caller) | D-06: pure function, called by loop, not by oracle |
| Cognitive load → prompt gate | OracleAgent (reply method) | — | Oracle receives load score and conditionally appends overload instruction |
| Drift detection | OracleAgent (delta window) | — | D-09: rolling deque inside OracleAgent; structural comparison is oracle-local |
| Drift logging | conversation_loop.py | serialization.py | D-10: loop reads `reply.contradiction_detected` and writes `drift_event` JSONL |
| FB-04 constraint propagation | conversation_loop.py (global_instructions) | OracleAgent (prompt reader) | D-12: loop accumulates; oracle reads on next turn |
| LLM API call | OracleAgent.reply() | llm_provider.py | Only try/except boundary (CLAUDE.md) |
| oracle_init event logging | conversation_loop.py (run_conversation start) | serialization.py | D-05: one event per run start |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| anthropic | 0.101.0 (available via pip) | Anthropic LLM client | Project already uses this duck-type interface in feedback_parser.py and cluster_naming.py [VERIFIED: pip install --dry-run] |
| openai | 2.34.0 (installed) | OpenAI LLM client (via _OpenAIAdapter) | Already in llm_provider.py; OracleAgent calls the abstracted client, not the SDK directly [VERIFIED: pip list] |
| google-genai | 1.75.0 (installed) | Google AI Studio client (via _GoogleAdapter) | Already in llm_provider.py [VERIFIED: pip list] |
| collections.deque | stdlib | Rolling window for drift detection | Bounded FIFO with O(1) append and pop; `deque(maxlen=N)` auto-evicts oldest [VERIFIED: Python stdlib] |
| dataclasses | stdlib | OracleSpec, NoiseParams dataclasses | Established pattern in this codebase (see feedback.py, state.py) [VERIFIED: codebase] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | stdlib | Mock LLM client in tests | Every test for OracleAgent.reply() that must not hit the real API |
| pytest | 9.0.2 (installed) | Test framework | All tests use this; `@pytest.mark.llm` marker already configured in pyproject.toml [VERIFIED: pyproject.toml] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Direct anthropic client | llm_provider.make_llm_client() | Project uses the abstracted client — OracleAgent should accept a client object, matching the pattern in parse_feedback(). No reason to bypass the abstraction. |
| deque-based rolling window | list with manual slicing | deque(maxlen=N) is simpler and O(1); list slicing requires copying. Use deque. |
| Inline cognitive load formula | Separate module | D-06 locks cognitive_load.py as a separate module. |

**Installation:**
```bash
pip install anthropic
```

**Version verification:**
```
anthropic: 0.101.0 [VERIFIED: pip install anthropic --dry-run, 2026-05-11]
```

---

## Architecture Patterns

### System Architecture Diagram

```
run_conversation() loop
        |
        |---> f_cognitive_load(state, message) --> load: float
        |                                              |
        |                                        [load > 0.7?]
        |                                              |
        v                                              v
oracle.reply(state, message) <--- OracleAgent receives load score
        |                                              |
        |   [assemble system prompt]                   |
        |     1. persona + preferred_k + semantic_axes |
        |     2. noise behavioral rules                |
        |     3. global_instructions (FB-04)            |
        |     4. state summary (clusters, turn)         |
        |     5. if load > threshold: OVERLOAD instr.  |
        |                                              |
        |   [LLM call -- only try/except boundary]     |
        |   client.messages.create(                    |
        |     system=assembled_prompt,                 |
        |     messages=[{"role": "user", "content": message}]
        |   )                                          |
        |                                              |
        |   [parse reply for satisfaction token]       |
        |   [structural drift check against deque]     |
        |   [return OracleReply(                       |
        |     raw_text=..., satisfied=...,             |
        |     turn_cognitive_load=load,                |
        |     contradiction_detected=...,              |
        |     contradicted_turn=...                    |
        |   )]                                         |
        |                                              |
        v                                              v
loop receives OracleReply
        |
        |---> if reply.contradiction_detected:
        |         append_to_audit_log(drift_event, log_path)
        |         socketio.emit("state_update", {...})
        |
        |---> parse_feedback(reply.raw_text, state, llm_client)
        |         --> [InstructionalFeedback] appended to global_instructions
        |
        v
f_next_state(state, deltas, ..., global_instructions)
```

### Recommended Project Structure

```
src/
├── oracle_agent.py      # OracleAgent, OracleSpec, NoiseParams
├── cognitive_load.py    # f_cognitive_load(), MAX_K, MAX_MSG_LEN, COG_LOAD_THRESHOLD, TOP_K_ITEMS_PER_CLUSTER
├── oracle_protocol.py   # OracleReply extended (contradiction_detected, contradicted_turn), OracleProtocol, MockOracle
└── conversation_loop.py # run_conversation() — add oracle_init log, f_cognitive_load call, drift_event log

tests/phase3/
├── __init__.py
├── test_oracle_agent.py         # OracleAgent construction, reply(), drift detection
├── test_cognitive_load.py       # f_cognitive_load() formula, threshold gate, edge cases
└── test_oracle_loop_integration.py  # OracleAgent wired into run_conversation()
```

### Pattern 1: System Prompt Assembly

**What:** Assemble a multi-section system prompt string from OracleSpec, NoiseParams, global_instructions, state summary, and optionally the OVERLOAD instruction.
**When to use:** Inside `OracleAgent.reply()` before every LLM call.

```python
# Source: verified against Anthropic SDK docs (system= top-level string parameter)
def _build_system_prompt(
    self,
    state: ClusteringState,
    cognitive_load: float,
    global_instructions: list[str],
) -> str:
    parts = []

    # Section 1: Preference spec and persona
    parts.append(
        f"You are a human data analyst with the following preferences:\n"
        f"- Target number of clusters: {self._spec.preferred_k}\n"
        f"- You group data by: {', '.join(self._spec.semantic_axes)}\n"
        f"- Persona: {self._spec.persona_description}"
    )

    # Section 2: Noise behavioral rules (D-04)
    cr = self._noise.consistency_rate
    dp = self._noise.drift_probability
    sr = self._noise.sycophancy_resistance
    parts.append(
        f"Behavioral rules:\n"
        f"- You agree with the proposed clustering {cr*100:.0f}% of the time. "
        f"The other {(1-cr)*100:.0f}% you request minor adjustments.\n"
        f"- With probability {dp:.2f} you introduce a new preference per turn "
        f"that may contradict a prior one.\n"
        f"- You maintain your stated position even when the system pushes back, "
        f"at rate {sr:.2f}."
    )

    # Section 3: Global instructions from FB-04 accumulator
    if global_instructions:
        parts.append("Standing instructions from prior turns:\n" +
                     "\n".join(f"- {i}" for i in global_instructions))

    # Section 4: Current state summary
    cluster_summary = "; ".join(
        f"Cluster {c.id} '{c.name}' ({len(c.item_ids)} items)"
        for c in state.clusters
    )
    parts.append(f"Current clustering (turn {state.turn_index}): {cluster_summary}")

    # Section 5: Cognitive load gate (D-08) — appended last
    if cognitive_load > COG_LOAD_THRESHOLD:
        parts.append("OVERLOAD: Focus on one thing only.")

    return "\n\n".join(parts)
```

### Pattern 2: LLM Call with try/except at Boundary

**What:** The ONLY permitted try/except in OracleAgent. Catches Anthropic API errors and re-raises with context.
**When to use:** Inside `OracleAgent.reply()` wrapping `client.messages.create()`.

```python
# Source: Anthropic SDK docs — system= is top-level string param, not a message role
# [VERIFIED: Context7 /anthropics/anthropic-sdk-python]
from anthropic import APIError, RateLimitError, AuthenticationError

try:
    response = self._client.messages.create(
        model=self._model,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": message}],
    )
except (RateLimitError, AuthenticationError, APIError) as exc:
    raise RuntimeError(
        f"OracleAgent LLM call failed at turn {state.turn_index}: {exc}"
    ) from exc

raw_text = response.content[0].text
```

**Important:** The Anthropic SDK uses `system=` as a top-level keyword argument to `messages.create()`, NOT as a `{"role": "system", ...}` message. This is different from the OpenAI convention. [VERIFIED: Context7 MessageCreateParams]

**Provider compatibility:** The existing `_OpenAIAdapter` and `_GoogleAdapter` in `llm_provider.py` currently do NOT support a `system=` kwarg — they only handle the `messages` list. The OracleAgent call pattern needs to either: (a) pass the system prompt as the first item in the messages list with appropriate formatting, or (b) extend the adapters. See Pitfall 2 below.

### Pattern 3: Structural Drift Detection with deque

**What:** After parse_feedback produces deltas, compare each structural delta against a rolling window of prior turn deltas.
**When to use:** Inside `OracleAgent.reply()` after the LLM call, before returning `OracleReply`.

```python
# Source: Python stdlib collections.deque — [VERIFIED: Python docs]
from collections import deque

class OracleAgent:
    def __init__(self, spec, noise_params, client, model="claude-haiku-4-5"):
        self._spec = spec
        self._noise = noise_params
        self._client = client
        self._model = model
        # Rolling window: (turn_index, FeedbackDelta) pairs
        self._delta_window: deque = deque(maxlen=10)

    def _check_contradiction(
        self, delta, current_turn: int
    ) -> tuple[bool, int | None]:
        """Compare delta against rolling window. Returns (contradicted, prior_turn)."""
        for prior_turn, prior_delta in self._delta_window:
            if _contradicts(delta, prior_delta):
                return True, prior_turn
        return False, None
```

**Contradiction rules (D-09):**
```python
# [VERIFIED: 03-CONTEXT.md D-09]
from src.feedback import MergeFeedback, SplitFeedback, MoveItemFeedback

def _contradicts(new_delta, prior_delta) -> bool:
    # MergeFeedback(A, B) contradicts SplitFeedback(cluster_id=A) or SplitFeedback(cluster_id=B)
    if isinstance(new_delta, MergeFeedback) and isinstance(prior_delta, SplitFeedback):
        return prior_delta.cluster_id in (new_delta.cluster_a_id, new_delta.cluster_b_id)
    # SplitFeedback(A) contradicts a prior MergeFeedback that produced A
    # (MergeFeedback result cluster ID is the lower of the two inputs, by Phase 2 convention)
    if isinstance(new_delta, SplitFeedback) and isinstance(prior_delta, MergeFeedback):
        # Merged cluster gets ID = cluster_a_id (Phase 2 convention from f_next_state)
        return new_delta.cluster_id == prior_delta.cluster_a_id
    # MoveItemFeedback(item, target=B) contradicts MoveItemFeedback(item, target=C) where C != B
    if (isinstance(new_delta, MoveItemFeedback) and
            isinstance(prior_delta, MoveItemFeedback)):
        return (new_delta.item_id == prior_delta.item_id and
                new_delta.target_cluster_id != prior_delta.target_cluster_id)
    return False
    # GlobalFeedback and InstructionalFeedback: ignored (D-09)
```

### Pattern 4: oracle_init JSONL Event

**What:** One event written at the start of `run_conversation()` when an OracleAgent is passed.
**When to use:** Immediately after `hierarchy` and `global_instructions` are initialized in `run_conversation()`.

```python
# [VERIFIED: 03-CONTEXT.md Specifics — mirrors backend_init event format in web/app.py]
# Written directly to log_path using open(..., "a") — same style as append_to_audit_log
import json

def _log_oracle_init(oracle, log_path: str, initial_state: ClusteringState) -> None:
    """Write oracle_init JSONL event if oracle is an OracleAgent (has spec/noise)."""
    from src.oracle_agent import OracleAgent  # local import to avoid circular
    if not isinstance(oracle, OracleAgent):
        return
    record = {
        "event": "oracle_init",
        "turn": 0,
        "timestamp": initial_state.timestamp,
        "preferred_k": oracle.spec.preferred_k,
        "semantic_axes": oracle.spec.semantic_axes,
        "consistency_rate": oracle.noise_params.consistency_rate,
        "drift_probability": oracle.noise_params.drift_probability,
        "sycophancy_resistance": oracle.noise_params.sycophancy_resistance,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

### Pattern 5: drift_event JSONL Record

**What:** One record per detected contradiction, written by the loop after receiving a reply with `contradiction_detected=True`.
**When to use:** In `run_conversation()` after `oracle.reply()` returns.

```python
# [VERIFIED: 03-CONTEXT.md D-10]
if reply.contradiction_detected:
    drift_record = {
        "event": "drift_event",
        "turn": new_state.turn_index,
        "contradicted_turn": reply.contradicted_turn,
        "timestamp": new_state.timestamp,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(drift_record) + "\n")
```

### Anti-Patterns to Avoid

- **Swallowing LLM exceptions:** Never catch `Exception` broadly in `OracleAgent.reply()`. Catch only specific SDK errors (`RateLimitError`, `AuthenticationError`, `APIError`) and re-raise as `RuntimeError` with context.
- **Post-processing for noise:** Do not adjust `raw_text` after the LLM call to inject or remove noise. Noise is prompt-only (D-04).
- **Temperature manipulation for noise:** Do not change temperature per-turn. The LLM follows the behavioral rules verbatim — temperature changes would require per-turn client reconfiguration, not justified.
- **Drift history in memory:** Do not add `drift_history: list[...]` to `OracleAgent` beyond the rolling deque. D-10: AuditLog is the persistence layer.
- **Oracle computing its own load:** `f_cognitive_load()` is a loop concern. The oracle receives the score; it does not compute it.
- **Passing system prompt as a message role:** Never do `messages=[{"role": "system", "content": ...}]`. Use the top-level `system=` parameter.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLM client abstraction | Custom HTTP client to Anthropic API | `src/llm_provider.make_llm_client()` | Already exists, supports 3 providers, already tested |
| Rolling bounded window | Manual list with `del lst[0]` | `collections.deque(maxlen=N)` | O(1) append/pop, auto-evicts, stdlib |
| JSON serialization of drift events | Custom format | `json.dumps()` + same `open(..., "a")` pattern as `append_to_audit_log` | Consistent with existing AuditLog format |
| Feedback parsing for oracle replies | Second LLM call inside OracleAgent | `parse_feedback()` from `feedback_parser.py` called in `run_conversation()` loop | D-05: parse_feedback is the only LLM parsing point |

**Key insight:** The oracle generates natural language. Parsing that language back into structured FeedbackDelta objects is the loop's job (via `parse_feedback`), not the oracle's job. The oracle has no parsing responsibility.

---

## Common Pitfalls

### Pitfall 1: system= vs message role for system prompt
**What goes wrong:** OracleAgent uses `messages=[{"role": "system", ...}]` instead of `system=...`. The Anthropic API rejects "system" as a message role with a validation error.
**Why it happens:** OpenAI convention uses a system message role; Anthropic uses a top-level `system=` parameter.
**How to avoid:** Always pass the assembled system prompt as `system=system_prompt` in `client.messages.create()`.
**Warning signs:** `BadRequestError: messages: roles must alternate between "user" and "assistant"` or `invalid role: system`.
**Note for non-Anthropic providers:** `_OpenAIAdapter` and `_GoogleAdapter` in `llm_provider.py` only handle `messages=` (no `system=` kwarg). OracleAgent must either prepend the system prompt as the first user message, or patch the adapters to accept a `system=` kwarg. [VERIFIED: llm_provider.py source]

### Pitfall 2: llm_provider adapter incompatibility
**What goes wrong:** `_OpenAIAdapter.messages.create()` and `_GoogleMessages.create()` accept `(model, max_tokens, messages)` — no `system=` kwarg. Calling with `system=...` raises `TypeError: create() got an unexpected keyword argument 'system'`.
**Why it happens:** The adapters were written to match the feedback_parser.py call pattern, which does not use a system prompt.
**How to avoid:** Two options — (a) prepend system content as a user turn: `messages=[{"role": "user", "content": system_prompt + "\n\n" + message}]`, keeping `system=` only for the real Anthropic client (detect via `isinstance(client, anthropic.Anthropic)`); or (b) extend both adapter `create()` methods to accept and discard or incorporate `system=` kwargs. Option (b) is cleaner for Phase 6 cross-provider ablations. The planner should choose and be consistent.
**Warning signs:** `TypeError` on first `OracleAgent.reply()` call with non-Anthropic provider.

### Pitfall 3: MergeFeedback result cluster ID assumption
**What goes wrong:** Drift detection assumes the merged cluster gets `cluster_a_id`, but Phase 2's `f_next_state` may use a different convention.
**Why it happens:** The contradiction rule `SplitFeedback(A) contradicts prior MergeFeedback that produced A` requires knowing what ID the merged cluster received.
**How to avoid:** Read `src/agent_functions.py` `f_next_state` merge branch before implementing `_contradicts()`. The Phase 2 monotonic counter (D-11) means merged clusters get new IDs, not recycled ones. Adjust the rule to compare against both `cluster_a_id` and `cluster_b_id` inputs if the merged cluster ID is not recoverable from the `MergeFeedback` delta alone.
**Warning signs:** Zero contradictions detected in a 20-turn loop with `drift_probability=0.5`.

### Pitfall 4: deque stores live FeedbackDelta references
**What goes wrong:** `FeedbackDelta` objects are frozen dataclasses — they are safely hashable and immutable. The deque holds direct references, which is fine. No copies needed.
**Why it happens:** Non-issue, but worth stating explicitly since later phases may mutate state.
**How to avoid:** No action needed. `SplitFeedback`, `MergeFeedback`, `MoveItemFeedback` are all `frozen=True` dataclasses.

### Pitfall 5: oracle_init event breaks load_audit_log()
**What goes wrong:** `load_audit_log()` calls `deserialize_state()` on every line. The `oracle_init` line is not a `ClusteringState` — it will fail with `AssertionError: Missing 'turn_index'`.
**Why it happens:** The AuditLog currently stores only `ClusteringState` JSONL lines. Phase 3 adds non-state event records.
**How to avoid:** Either (a) write oracle_init and drift_event records to a separate sidecar file (e.g., `events.jsonl`), or (b) add an `event` field check to `load_audit_log()` to skip non-state lines (`if "event" in d: continue`). Option (a) is cleaner and avoids touching `serialization.py`. Option (b) is simpler but requires modifying a Phase 1 function. The planner must choose. This is a **blocking integration issue** that must be resolved in Wave 0.
**Warning signs:** `AssertionError: Missing 'turn_index' in deserialized state` when `load_audit_log()` is called after a run with `OracleAgent`.

### Pitfall 6: Cognitive load formula edge cases
**What goes wrong:** `total_items` is zero (empty ClusteringState), or `MAX_K`/`MAX_MSG_LEN` constants are too small, causing load to always be 1.0.
**Why it happens:** Formula has no guard for zero denominators.
**How to avoid:** Assert `total_items > 0` and `len(state.clusters) > 0` at the start of `f_cognitive_load()`. Choose `MAX_K` >= the maximum realistic K (suggested: 20), `MAX_MSG_LEN` >= the longest message `_format_message` can produce (suggested: 500 chars), `TOP_K_ITEMS_PER_CLUSTER = 5` (matching `_format_message` default).

### Pitfall 7: Satisfaction token detection
**What goes wrong:** Oracle LLM reply contains "I'm satisfied" but the code never sets `satisfied=True` on `OracleReply`, so the loop never terminates via oracle satisfaction.
**Why it happens:** The oracle returns free text. The code must detect the satisfaction signal from raw_text.
**How to avoid:** Define a convention in the system prompt: "When you are fully satisfied with the clustering, end your reply with the exact token: [SATISFIED]". Then check `"[SATISFIED]" in raw_text` in `reply()`. This is deterministic and token-exact, no LLM parsing needed.

---

## Code Examples

### OracleAgent skeleton
```python
# Source: derived from codebase patterns in oracle_protocol.py, feedback_parser.py, llm_provider.py
# [VERIFIED: codebase reading 2026-05-11]
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.cognitive_load import f_cognitive_load, COG_LOAD_THRESHOLD
from src.oracle_protocol import OracleReply

if TYPE_CHECKING:
    from src.state import ClusteringState

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

class OracleAgent:
    def __init__(
        self,
        spec: OracleSpec,
        noise_params: NoiseParams,
        client: object,
        model: str = "claude-haiku-4-5",
        window_size: int = 10,
    ) -> None:
        assert 0.0 <= noise_params.consistency_rate <= 1.0
        assert 0.0 <= noise_params.drift_probability <= 1.0
        assert 0.0 <= noise_params.sycophancy_resistance <= 1.0
        self._spec = spec
        self._noise = noise_params
        self._client = client
        self._model = model
        self._delta_window: deque = deque(maxlen=window_size)

    @property
    def spec(self) -> OracleSpec:
        return self._spec

    @property
    def noise_params(self) -> NoiseParams:
        return self._noise

    def reply(
        self,
        state: "ClusteringState",
        message: str,
        global_instructions: list[str] | None = None,
    ) -> OracleReply:
        ...  # assemble prompt, call LLM, check drift, return OracleReply
```

### f_cognitive_load skeleton
```python
# Source: D-07 formula from 03-CONTEXT.md [VERIFIED: context file]
# src/cognitive_load.py

MAX_K: int = 20
MAX_MSG_LEN: int = 500
TOP_K_ITEMS_PER_CLUSTER: int = 5
COG_LOAD_THRESHOLD: float = 0.7

def f_cognitive_load(state: "ClusteringState", message: str) -> float:
    """
    Compute a normalized cognitive load score in [0, 1].
    Formula: (clusters/MAX_K)*w1 + (items_shown/total_items)*w2 + (msg_len/MAX_MSG_LEN)*w3
    Default weights: w1=w2=w3=1/3.
    """
    assert len(state.clusters) > 0, "f_cognitive_load: no clusters in state"
    total_items = len(state.assignments)
    assert total_items > 0, "f_cognitive_load: no items in state"

    w = 1.0 / 3.0
    cluster_term = min(len(state.clusters) / MAX_K, 1.0) * w
    items_shown = len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER
    items_term = min(items_shown / total_items, 1.0) * w
    msg_term = min(len(message) / MAX_MSG_LEN, 1.0) * w
    return cluster_term + items_term + msg_term
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MockOracle (scripted) | LLM OracleAgent (prompt-driven) | Phase 3 | Real behavioral variation; enables Phase 6 human comparison |
| No cognitive load tracking | f_cognitive_load() per turn | Phase 3 | Fills `reply.turn_cognitive_load` for Phase 4 per-turn metrics |
| No drift detection | Structural deque comparison | Phase 3 | `contradiction_detected` in OracleReply; `drift_event` in AuditLog |
| global_instructions unused by oracle | Injected into oracle system prompt | Phase 3 | FB-04: instructional constraints propagate to oracle behavior |

**Note on models:** The project currently uses `"claude-haiku-4-5"` for feedback parsing (fast/cheap). For OracleAgent, the same model is appropriate for ablations where many runs are needed. Use `"claude-sonnet-4-5"` or similar for human-comparison runs where reply quality matters. The model string is a constructor parameter (Claude's discretion). [ASSUMED — exact model name strings should be verified against Anthropic's current model list before use]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Merged cluster receives `cluster_a_id` as its ID in Phase 2's f_next_state | Common Pitfalls §3, Pattern 3 | Drift detection never fires for split-after-merge contradictions; need to read agent_functions.py to verify |
| A2 | `_format_message` in conversation_loop.py produces messages at most ~500 chars under normal operation | Standard Stack (MAX_MSG_LEN constant) | If messages are longer, cognitive load is always clamped at 1.0 for the message term; adjust constant |
| A3 | The "[SATISFIED]" token approach is reliable enough for test purposes | Pitfall 7 | If oracle LLM paraphrases or omits token, loop never terminates via oracle satisfaction; alternative: parse for positive sentiment words |
| A4 | Extending `_OpenAIAdapter` and `_GoogleAdapter` to accept `system=` kwarg is the right approach (vs. prepend to messages) | Pitfall 2 | If planner chooses prepend approach, system prompt is visible in conversation history; may confuse multi-turn context |

---

## Open Questions (RESOLVED)

1. **Which file gets oracle_init and drift_event records?**
   - What we know: `load_audit_log()` in serialization.py calls `deserialize_state()` on every line; non-state records cause AssertionError (Pitfall 5).
   - What's unclear: Does Phase 3 write events to the same `audit_log.jsonl` or a sidecar `events.jsonl`?
   - Recommendation: Write to a separate `events.jsonl` sidecar in the same session directory. This keeps `load_audit_log()` unchanged (Phase 1 function) and avoids a breaking change to serialization.py. The planner should pick one approach and enforce it across Wave 0 (test setup) and all oracle waves.
   - **RESOLVED:** Use `events.jsonl` sidecar. `_write_event()` helper in `conversation_loop.py` writes all oracle_init and drift_event records to the sidecar. `audit_log.jsonl` remains ClusteringState-only. Enforced by Plan 04 Task 1 and Pitfall 5 documentation.

2. **How does OracleAgent receive global_instructions?**
   - What we know: `global_instructions` is a local variable in `run_conversation()`. `oracle.reply(state, message)` matches the existing `OracleProtocol` signature (2 args). D-12 says FB-04 is wired via this accumulator.
   - What's unclear: Does Phase 3 change the `OracleProtocol.reply()` signature to add `global_instructions=` kwarg, or does the loop pass it via a different mechanism?
   - Recommendation: Add an optional `global_instructions: list[str] | None = None` kwarg to `OracleAgent.reply()`. The protocol signature `reply(state, message) -> OracleReply` is structural (duck typing), so adding an optional kwarg to the concrete class does not break structural subtyping. The loop passes it explicitly only when calling an `OracleAgent`.
   - **RESOLVED:** Optional `global_instructions: list[str] | None = None` kwarg added to `OracleAgent.reply()` (Plan 01 Task 2). Loop calls `oracle.reply(state, message, global_instructions=global_instructions)` inside an `isinstance(oracle, _OracleAgent)` branch (Plan 04 Task 2 Modification A). Protocol structural subtyping preserved.

3. **Merged cluster ID after MergeFeedback**
   - What we know: Phase 2 uses a monotonic counter (D-11). The merge branch in `f_next_state` creates a new cluster.
   - What's unclear: Does the merged cluster get a brand-new ID (next counter value) or does it reuse `cluster_a_id`?
   - Recommendation: Read `src/agent_functions.py` merge branch before implementing `_contradicts()`. If merged cluster gets a new ID, the drift detection rule needs to track (old_a, old_b) -> new_id mapping in the deque, not just the MergeFeedback delta.
   - **RESOLVED:** Verified via `src/agent_functions.py` `_apply_merge()`: merged cluster receives a brand-new ID via `_next_cluster_id(state) = max(cluster.id) + 1` — it does NOT reuse `cluster_a_id` or `cluster_b_id`. Contradiction rule in `_contradicts()` checks both `cluster_a_id` and `cluster_b_id` of a prior `MergeFeedback` against the `cluster_id` of a new `SplitFeedback`. Implemented in Plan 03 per the interfaces block in 03-00-PLAN.md.
---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| anthropic SDK | OracleAgent.reply() (Anthropic path) | Not installed | — (0.101.0 available via pip) | openai (2.34.0) or google-genai (1.75.0) already installed |
| openai SDK | OracleAgent (OpenAI path via _OpenAIAdapter) | Yes | 2.34.0 | anthropic or google |
| google-genai SDK | OracleAgent (Google path via _GoogleAdapter) | Yes | 1.75.0 | anthropic or openai |
| Python deque | Drift detection | Yes | stdlib | — |
| pytest | Tests | Yes | 9.0.2 | — |
| unittest.mock | LLM mock in tests | Yes | stdlib | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- anthropic SDK: not installed but available via `pip install anthropic`. At least one of openai or google-genai is available as fallback. OracleAgent should work with any of the three via `llm_provider.make_llm_client()`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | pyproject.toml (`[tool.pytest.ini_options]`) |
| Quick run command | `pytest tests/phase3/ -q -m "not llm"` |
| Full suite command | `pytest tests/ -q -m "not llm"` |
| LLM tests (real API) | `pytest tests/phase3/ -q -m llm` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ORC-01 | OracleAgent satisfies OracleProtocol (structural subtyping) | unit | `pytest tests/phase3/test_oracle_agent.py::test_oracle_agent_satisfies_protocol -x` | Wave 0 |
| ORC-01 | OracleAgent.reply() returns OracleReply with non-empty raw_text (mocked LLM) | unit | `pytest tests/phase3/test_oracle_agent.py::test_reply_returns_oracle_reply -x` | Wave 0 |
| ORC-01 | OracleSpec fields accessible from OracleAgent (preferred_k, semantic_axes, persona_description) | unit | `pytest tests/phase3/test_oracle_agent.py::test_oracle_spec_fields -x` | Wave 0 |
| ORC-02 | oracle_init JSONL record written to events log at run start | unit | `pytest tests/phase3/test_oracle_agent.py::test_oracle_init_logged -x` | Wave 0 |
| ORC-02 | NoiseParams values appear in assembled system prompt | unit | `pytest tests/phase3/test_oracle_agent.py::test_noise_params_in_prompt -x` | Wave 0 |
| ORC-03 | f_cognitive_load returns float in [0, 1] for all valid states | unit | `pytest tests/phase3/test_cognitive_load.py::test_load_in_range -x` | Wave 0 |
| ORC-03 | f_cognitive_load returns > COG_LOAD_THRESHOLD for max-stress state | unit | `pytest tests/phase3/test_cognitive_load.py::test_load_above_threshold -x` | Wave 0 |
| ORC-03 | OVERLOAD instruction appears in prompt when load > 0.7 | unit | `pytest tests/phase3/test_oracle_agent.py::test_overload_prompt_injected -x` | Wave 0 |
| ORC-03 | OVERLOAD instruction absent when load <= 0.7 | unit | `pytest tests/phase3/test_oracle_agent.py::test_no_overload_below_threshold -x` | Wave 0 |
| ORC-04 | Contradiction detected: MergeFeedback(A,B) after SplitFeedback(A) | unit | `pytest tests/phase3/test_oracle_agent.py::test_contradiction_merge_after_split -x` | Wave 0 |
| ORC-04 | Contradiction detected: MoveItemFeedback(item, B) after MoveItemFeedback(item, C) | unit | `pytest tests/phase3/test_oracle_agent.py::test_contradiction_move_item -x` | Wave 0 |
| ORC-04 | No contradiction when window is empty | unit | `pytest tests/phase3/test_oracle_agent.py::test_no_contradiction_empty_window -x` | Wave 0 |
| ORC-04 | drift_event JSONL record written when contradiction_detected=True | unit | `pytest tests/phase3/test_oracle_loop_integration.py::test_drift_event_logged -x` | Wave 0 |
| ORC-04 | contradiction_detected=False when no prior conflicting delta | unit | `pytest tests/phase3/test_oracle_agent.py::test_no_false_positive_contradiction -x` | Wave 0 |
| FB-04 | global_instructions content appears in oracle system prompt | unit | `pytest tests/phase3/test_oracle_agent.py::test_global_instructions_in_prompt -x` | Wave 0 |
| FB-04 | InstructionalFeedback from reply is appended to global_instructions by loop | integration | `pytest tests/phase3/test_oracle_loop_integration.py::test_instructional_feedback_accumulates -x` | Wave 0 |
| ALL | OracleAgent wired into run_conversation() runs 5 turns without error (mocked LLM) | integration | `pytest tests/phase3/test_oracle_loop_integration.py::test_oracle_agent_loop_5_turns -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/phase3/ -q -m "not llm"` (all mocked unit tests, < 5 seconds)
- **Per wave merge:** `pytest tests/ -q -m "not llm"` (full suite, < 30 seconds)
- **Phase gate:** Full suite green before `/gsd-verify-work`; LLM tests (`-m llm`) optional smoke test if API key available

### Wave 0 Gaps

- [ ] `tests/phase3/__init__.py` — empty init file
- [ ] `tests/phase3/test_oracle_agent.py` — OracleSpec, NoiseParams, OracleAgent, prompt assembly, drift detection (all mocked)
- [ ] `tests/phase3/test_cognitive_load.py` — f_cognitive_load formula, threshold, edge cases
- [ ] `tests/phase3/test_oracle_loop_integration.py` — OracleAgent wired into run_conversation() with mocked LLM; oracle_init and drift_event logging

**Shared fixture needed:** `conftest.py` will need an `oracle_agent_factory` fixture (creates OracleAgent with a MagicMock client) — add to existing `tests/conftest.py`.

---

## Project Constraints (from CLAUDE.md)

| Directive | Applies To | Enforcement |
|-----------|-----------|-------------|
| No silent failures — let code crash | OracleAgent, cognitive_load.py | `assert` preconditions; no bare `except: pass` |
| No defensive if/else chains for unexpected state | OracleAgent._check_contradiction(), _build_system_prompt() | Assert expected types; unexpected delta type = crash |
| `try/except` ONLY at LLM API call and CLI entry point | OracleAgent.reply() | Wrap only `client.messages.create()`. All other code: no try/except |
| `assert` statements freely for invariants | NoiseParams values, state fields, prompt assembly | Assert 0.0 <= consistency_rate <= 1.0, etc. |
| Dataclass-first | OracleSpec, NoiseParams | Both as `@dataclass`, not dict |
| Named constants, not magic numbers | cognitive_load.py | MAX_K, MAX_MSG_LEN, TOP_K_ITEMS_PER_CLUSTER, COG_LOAD_THRESHOLD as module-level constants |
| JSONL AuditLog from Phase 1 — never skip | oracle_init, drift_event | Must write both event types; neither is optional once OracleAgent is used |

---

## Sources

### Primary (HIGH confidence)
- Codebase: `src/oracle_protocol.py`, `src/conversation_loop.py`, `src/feedback.py`, `src/feedback_parser.py`, `src/serialization.py`, `src/state.py`, `src/llm_provider.py` — [VERIFIED: read 2026-05-11]
- `.planning/phases/03-oracle-agent/03-CONTEXT.md` — all 12 decisions D-01..D-12 [VERIFIED: read 2026-05-11]
- Context7 `/anthropics/anthropic-sdk-python` — `system=` top-level parameter, `messages.create()` signature, error types [VERIFIED: Context7 CLI 2026-05-11]
- `pyproject.toml` — pytest 9.0.2, `llm` marker already configured [VERIFIED: read 2026-05-11]

### Secondary (MEDIUM confidence)
- `pip install anthropic --dry-run` — version 0.101.0 available [VERIFIED: shell 2026-05-11]
- `pip list` output — openai 2.34.0, google-genai 1.75.0 confirmed installed [VERIFIED: shell 2026-05-11]
- Python stdlib docs — `collections.deque(maxlen=N)` auto-eviction behavior [VERIFIED: Python stdlib knowledge, HIGH confidence]

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified via pip; SDK call pattern verified via Context7
- Architecture: HIGH — derived directly from CONTEXT.md locked decisions and verified codebase
- Pitfalls: HIGH for Pitfalls 1, 2, 5 (directly verifiable); MEDIUM for Pitfalls 3, 7 (depend on implementation choices)

**Research date:** 2026-05-11
**Valid until:** 2026-06-11 (model names may change; verify before use)
