"""
strategy.py — StrategyProtocol, Action, RandomStrategy.

Phase 5 adds UncertaintyDrivenStrategy and BoundaryDrivenStrategy without changing this file.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from src.state import ClusteringState


@dataclass
class Action:
    """
    Represents one possible next action for the agent to take.

    action_type: One of the four valid action types for Phase 2.
    payload:     Optional metadata (cluster_id, item_ids, question_text, etc.).
                 Phase 2 uses empty dict; Phase 5 enriches payloads.
    """
    action_type: Literal["show_full", "show_subset", "ask_question", "stop"]
    payload: dict = field(default_factory=dict)


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Interface for action-selection strategies.

    Phase 2: RandomStrategy (uniform random over valid actions).
    Phase 5: UncertaintyDrivenStrategy, BoundaryDrivenStrategy.

    Structural subtyping: implementations do NOT need to inherit from this class.
    """
    def select(self, state: ClusteringState, uncertainty_report: "object") -> Action:
        ...


def _enumerate_valid_actions(
    state: ClusteringState,
    uncertainty_report: object,
) -> list[Action]:
    """
    Build a non-empty list of valid Action objects for the current state.

    Phase 2 payload is always {} — Phase 5 enriches payloads with specific cluster_ids
    and item_ids drawn from the uncertainty_report.

    Rules:
      - "show_full"    always valid
      - "ask_question" always valid if there are clusters (any question makes sense)
      - "show_subset"  valid if there are >= 2 clusters (subset requires at least 2 to show)
      - "stop"         always valid (loop checks oracle satisfaction separately)
    """
    actions: list[Action] = []
    actions.append(Action(action_type="show_full"))
    if len(state.clusters) > 0:
        actions.append(Action(action_type="ask_question"))
    if len(state.clusters) >= 2:
        actions.append(Action(action_type="show_subset"))
    actions.append(Action(action_type="stop"))
    assert len(actions) > 0, (
        "BUG: _enumerate_valid_actions returned empty list — state has no clusters at all"
    )
    return actions


class RandomStrategy:
    """
    Phase 2 implementation: uniform random selection over valid actions.

    Uses a seeded random.Random instance (NOT the global random module) to ensure
    determinism when a seed is provided. Tests always pass a seed (T-02-07).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, state: ClusteringState, uncertainty_report: object) -> Action:
        """Select a uniformly random valid action for the current state."""
        valid_actions = _enumerate_valid_actions(state, uncertainty_report)
        assert len(valid_actions) > 0, "BUG: no valid actions to select from"
        return self._rng.choice(valid_actions)
