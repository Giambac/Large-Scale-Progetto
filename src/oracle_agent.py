"""
oracle_agent.py — OracleAgent, OracleSpec, NoiseParams (ORC-01, ORC-02, ORC-03, ORC-04).

OracleAgent is an LLM-backed oracle that satisfies OracleProtocol via structural subtyping.
It assembles a multi-section system prompt from OracleSpec, NoiseParams, global_instructions,
current state summary, and optional cognitive load gate, then calls the LLM client.

Key behaviors:
- OracleSpec/NoiseParams injected at construction; fixed for agent lifetime (D-02).
- Noise parameters are prompt-injected behavioral rules (D-04).
- oracle_init JSONL event written to events_path at construction time if provided (D-05 / ORC-02).
- Provider-aware LLM call: Anthropic uses system= kwarg; OpenAI/Google adapters prepend to message.
- contradiction_detected always False in Wave 1; Wave 2 (Plan 03) adds _check_contradiction() call.
- try/except ONLY at the two LLM client.messages.create() call sites (CLAUDE.md fail-loudly rule).
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.oracle_protocol import OracleReply

if TYPE_CHECKING:
    from src.state import ClusteringState


@dataclass
class OracleSpec:
    """Preference specification for the oracle agent (ORC-01, D-01).

    preferred_k: target number of clusters the oracle aims for.
    semantic_axes: dimensions the oracle groups data by (e.g. ['topic', 'sentiment']).
    persona_description: free-text flavor injected into the LLM system prompt.

    Fixed for the agent lifetime — different runs use different OracleAgent instances (D-02).
    """
    preferred_k: int
    semantic_axes: list[str]
    persona_description: str


@dataclass
class NoiseParams:
    """Noise parameters controlling oracle behavioral variation (ORC-02, D-03, D-04).

    consistency_rate: 0.0-1.0 — fraction of turns oracle agrees with proposed clustering.
    drift_probability: 0.0-1.0 — per-turn probability of introducing a new contradicting preference.
    sycophancy_resistance: 0.0-1.0 — rate at which oracle maintains position under pushback.

    All three become prompt-injected behavioral rules (D-04). No post-processing or temperature
    manipulation. Logged as oracle_init JSONL event at run start (D-05).
    """
    consistency_rate: float       # 0.0-1.0
    drift_probability: float      # 0.0-1.0
    sycophancy_resistance: float  # 0.0-1.0


def _contradicts(new_delta, prior_delta) -> bool:
    """Structural contradiction check between two FeedbackDelta objects (D-09).

    Rules:
    - MergeFeedback(A, B) contradicts SplitFeedback(cluster_id=A) or SplitFeedback(cluster_id=B)
    - SplitFeedback(X) contradicts a prior MergeFeedback if X was one of the input cluster IDs
      (cluster_a_id or cluster_b_id) — those clusters were retired by the merge.
    - MoveItemFeedback(item, target=B) contradicts MoveItemFeedback(item, target=C) where C != B.
    GlobalFeedback and InstructionalFeedback are ignored (too semantic for structural comparison).
    """
    from src.feedback import MergeFeedback, SplitFeedback, MoveItemFeedback

    # MergeFeedback(A, B) contradicts prior SplitFeedback(cluster_id=A) or SplitFeedback(cluster_id=B)
    if isinstance(new_delta, MergeFeedback) and isinstance(prior_delta, SplitFeedback):
        return prior_delta.cluster_id in (new_delta.cluster_a_id, new_delta.cluster_b_id)

    # SplitFeedback(X) contradicts prior MergeFeedback if X was one of the merged cluster inputs
    # (Phase 2 convention: merged cluster gets new monotonic ID — D-11 / Pitfall 3 resolved)
    if isinstance(new_delta, SplitFeedback) and isinstance(prior_delta, MergeFeedback):
        return new_delta.cluster_id in (prior_delta.cluster_a_id, prior_delta.cluster_b_id)

    # MoveItemFeedback(item, target=B) contradicts prior MoveItemFeedback(item, target=C) where C != B
    if isinstance(new_delta, MoveItemFeedback) and isinstance(prior_delta, MoveItemFeedback):
        return (new_delta.item_id == prior_delta.item_id and
                new_delta.target_cluster_id != prior_delta.target_cluster_id)

    return False
    # GlobalFeedback and InstructionalFeedback: ignored (D-09)


class OracleAgent:
    """LLM-backed oracle satisfying OracleProtocol via structural subtyping (ORC-01).

    Assembles a multi-section system prompt from spec, noise params, global instructions,
    current state summary, and cognitive load gate. Calls the injected LLM client.

    Wave 2 (Plan 03) adds update_delta_window() for structural drift detection (ORC-04).
    The conversation loop calls oracle.update_delta_window(deltas, turn_index) after
    parse_feedback() to check new deltas against the rolling window and append them.
    """

    def __init__(
        self,
        spec: OracleSpec,
        noise_params: NoiseParams,
        client: object,
        model: str = "claude-haiku-4-5",
        window_size: int = 10,
        events_path: Path | None = None,
    ) -> None:
        assert 0.0 <= noise_params.consistency_rate <= 1.0
        assert 0.0 <= noise_params.drift_probability <= 1.0
        assert 0.0 <= noise_params.sycophancy_resistance <= 1.0
        self._spec = spec
        self._noise = noise_params
        self._client = client
        self._model = model
        self._delta_window: deque = deque(maxlen=window_size)

        # ORC-02 / D-05: write oracle_init event at construction time if events_path provided.
        # This enables unit testing ORC-02 without requiring the conversation loop.
        if events_path is not None:
            _events_path_str = str(events_path)
            _parent = os.path.dirname(_events_path_str)
            if _parent:
                os.makedirs(_parent, exist_ok=True)
            _record = {
                "event": "oracle_init",
                "turn": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "preferred_k": self._spec.preferred_k,
                "semantic_axes": self._spec.semantic_axes,
                "consistency_rate": self._noise.consistency_rate,
                "drift_probability": self._noise.drift_probability,
                "sycophancy_resistance": self._noise.sycophancy_resistance,
            }
            with open(_events_path_str, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_record) + "\n")

    @property
    def spec(self) -> OracleSpec:
        """Return the preference specification."""
        return self._spec

    @property
    def noise_params(self) -> NoiseParams:
        """Return the noise parameters."""
        return self._noise

    def _build_system_prompt(
        self,
        state: "ClusteringState",
        cognitive_load: float,
        global_instructions: list[str],
    ) -> str:
        """Assemble the five-section system prompt for the oracle LLM call.

        Sections (joined by double newline):
        1. Persona + preference spec + satisfaction token instruction
        2. Noise behavioral rules (D-04)
        3. Standing instructions from FB-04 accumulator (if any)
        4. Current state summary (clusters, turn index)
        5. Cognitive load gate (D-08) — appended LAST, only if load > threshold
        """
        from src.cognitive_load import COG_LOAD_THRESHOLD

        parts = []

        # Section 1: Persona + preference spec
        cr = self._noise.consistency_rate
        parts.append(
            f"You are a human data analyst with the following preferences:\n"
            f"- Target number of clusters: {self._spec.preferred_k}\n"
            f"- You group data by: {', '.join(self._spec.semantic_axes)}\n"
            f"- Persona: {self._spec.persona_description}\n"
            f"When you are fully satisfied with the current clustering, end your reply with the exact token: [SATISFIED]"
        )

        # Section 2: Noise behavioral rules (D-04)
        dp = self._noise.drift_probability
        sr = self._noise.sycophancy_resistance
        parts.append(
            f"Behavioral rules:\n"
            f"- You agree with the proposed clustering {int(cr * 100)}% of the time. "
            f"The other {int((1 - cr) * 100)}% you request minor adjustments.\n"
            f"- With probability {dp:.2f} you introduce a new preference per turn "
            f"that may contradict a prior one.\n"
            f"- You maintain your stated position even when the system pushes back, "
            f"at rate {sr:.2f}."
        )

        # Section 3: Global instructions from FB-04 accumulator (only if non-empty)
        if global_instructions:
            parts.append(
                "Standing instructions from prior turns:\n" +
                "\n".join(f"- {i}" for i in global_instructions)
            )

        # Section 4: Current state summary
        cluster_summary = "; ".join(
            f"Cluster {c.id} '{c.name}' ({len(c.item_ids)} items)"
            for c in state.clusters
        )
        parts.append(f"Current clustering (turn {state.turn_index}): {cluster_summary}")

        # Section 5: Cognitive load gate (D-08) — appended LAST
        if cognitive_load > COG_LOAD_THRESHOLD:
            parts.append("OVERLOAD: Focus on one thing only.")

        return "\n\n".join(parts)

    def _check_contradiction(
        self, delta, current_turn: int
    ) -> tuple[bool, int | None]:
        """Compare delta against the rolling window of prior structural deltas.

        Returns (True, prior_turn_index) if a contradiction is found;
        (False, None) if no contradiction.

        Does NOT mutate the window — call update_delta_window() to add deltas.
        """
        for prior_turn, prior_delta in self._delta_window:
            if _contradicts(delta, prior_delta):
                return True, prior_turn
        return False, None

    def update_delta_window(
        self, deltas: list, turn_index: int
    ) -> tuple[bool, int | None]:
        """Check new deltas for contradictions against the rolling window, then append
        structural deltas to the window.

        Called by run_conversation() after parse_feedback() returns deltas (D-09 / ORC-04).

        The check runs BEFORE appending so that deltas within the same turn do not
        contradict each other (same-turn deltas are all new; no prior context for them).

        GlobalFeedback and InstructionalFeedback are skipped for both checking and
        appending — they are structurally opaque (D-09).

        Args:
            deltas: List of FeedbackDelta objects parsed from the current oracle reply.
            turn_index: The turn index of the current oracle reply.

        Returns:
            (contradiction_detected, contradicted_turn) — first contradiction found,
            or (False, None) if none.
        """
        from src.feedback import GlobalFeedback, InstructionalFeedback

        first_contradiction: bool = False
        first_contradicted_turn: int | None = None

        for delta in deltas:
            if isinstance(delta, (GlobalFeedback, InstructionalFeedback)):
                continue  # ignored for structural comparison (D-09)

            detected, prior_turn = self._check_contradiction(delta, turn_index)
            if detected and not first_contradiction:
                first_contradiction = True
                first_contradicted_turn = prior_turn

        # Append structural deltas to window AFTER checking (so this turn's deltas
        # don't contradict each other within the same turn)
        for delta in deltas:
            if not isinstance(delta, (GlobalFeedback, InstructionalFeedback)):
                self._delta_window.append((turn_index, delta))

        return first_contradiction, first_contradicted_turn

    def reply(
        self,
        state: "ClusteringState",
        message: str,
        global_instructions: list[str] | None = None,
    ) -> OracleReply:
        """Generate an oracle reply by calling the LLM with an assembled system prompt.

        Args:
            state: Current ClusteringState for prompt assembly.
            message: The clustering system's message to the oracle.
            global_instructions: Accumulated FB-04 instructions. None treated as [].

        Returns:
            OracleReply with raw_text, satisfied flag, turn_cognitive_load, and
            contradiction_detected=False (Wave 1; Wave 2 adds contradiction check).
        """
        from src.cognitive_load import f_cognitive_load, COG_LOAD_THRESHOLD

        if global_instructions is None:
            global_instructions = []

        cognitive_load = f_cognitive_load(state, message)
        system_prompt = self._build_system_prompt(state, cognitive_load, global_instructions)

        # Provider-aware LLM call (Pitfall 2):
        # Anthropic SDK uses system= as a top-level kwarg.
        # OpenAI/Google adapters only accept messages= (no system= kwarg).
        try:
            import anthropic as _anthropic_mod
            _is_anthropic = isinstance(self._client, _anthropic_mod.Anthropic)
        except ImportError:
            _is_anthropic = False

        if _is_anthropic:
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
        satisfied = "[SATISFIED]" in raw_text

        # Drift detection (ORC-04): the conversation loop calls
        # oracle.update_delta_window(deltas, turn_index) AFTER parse_feedback() returns
        # the parsed FeedbackDelta objects. OracleReply.contradiction_detected is set
        # by the loop from the return value of update_delta_window() (Wave 3 wiring).
        # reply() itself always returns contradiction_detected=False here; the loop
        # overwrites the field after calling update_delta_window().

        return OracleReply(
            raw_text=raw_text,
            satisfied=satisfied,
            turn_cognitive_load=cognitive_load,
            contradiction_detected=False,
            contradicted_turn=None,
        )
