"""
feedback.py — FeedbackDelta union type and frozen dataclasses.

Five feedback types represent all oracle intent categories:
  - GlobalFeedback:       high-level re-clustering instruction (type-priority: 1st)
  - SplitFeedback:        split one cluster into two (type-priority: 2nd)
  - MergeFeedback:        merge two clusters into one (type-priority: 3rd)
  - MoveItemFeedback:     move a single item to a target cluster (type-priority: 4th)
  - InstructionalFeedback: soft instructional hint with no direct structural effect (type-priority: 5th)

FeedbackDelta = Union alias over all five types. Compound oracle messages
produce multiple FeedbackDelta objects, applied in type-priority order per D-07.

Module constants:
  ORACLE_MOVE_CONFIDENCE      — soft_probs override value for point-move (D-10, 0.95)
  UNIFORM_FALLBACK_THRESHOLD  — zero-sum guard for point-move redistribution edge case
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# Named constants — not magic numbers inline (per CONTEXT.md Specifics and D-10)
ORACLE_MOVE_CONFIDENCE: float = 0.95
UNIFORM_FALLBACK_THRESHOLD: float = 1e-9


@dataclass(frozen=True)
class SplitFeedback:
    """Oracle requests splitting cluster_id into two sub-clusters.

    seed_item_ids: representative items the oracle names for each sub-cluster.
    May be empty — empty list triggers k-means++ fallback initialization (D-08).
    Uses list[int] to match the JSON array type produced by the feedback parser.
    """
    cluster_id: int
    seed_item_ids: list[int]


@dataclass(frozen=True)
class MergeFeedback:
    """Oracle requests merging cluster_a_id and cluster_b_id into one cluster."""
    cluster_a_id: int
    cluster_b_id: int


@dataclass(frozen=True)
class MoveItemFeedback:
    """Oracle requests moving item_id into target_cluster_id."""
    item_id: int
    target_cluster_id: int


@dataclass(frozen=True)
class GlobalFeedback:
    """High-level instruction that may trigger a full re-clustering.

    Carries highest type-priority weight (global → cluster → point → instructional).
    """
    instruction_text: str


@dataclass(frozen=True)
class InstructionalFeedback:
    """Soft instructional hint — no direct structural cluster operation."""
    instruction_text: str


# Union alias — ordering matches type-priority dispatch in f_next_state (D-07):
# global → cluster-level (split/merge) → point-level (move_item) → instructional
FeedbackDelta = Union[
    GlobalFeedback,
    SplitFeedback,
    MergeFeedback,
    MoveItemFeedback,
    InstructionalFeedback,
]
