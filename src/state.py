"""
state.py — Core data types for the clustering system (D-13).

ClusteringState is the single source of truth for one turn's clustering.
It is serialized to JSONL after every turn (FOUND-04).

SCHEMA IS FROZEN: any field change after Phase 1 is a breaking change.
All subsequent phases (2-6) depend on this schema.

D-13 field spec:
    turn_index: int              — 0-based turn counter
    timestamp: str               — ISO 8601 string (set at state creation time)
    clusters: list[Cluster]      — all current clusters
    assignments: dict[int, int]  — item_id -> cluster_id (complete: all N items)
    soft_probs: dict[int, list[float]]  — item_id -> prob vector of length K
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Cluster:
    """
    One cluster in the current ClusteringState.

    id: Assigned at cluster creation. NEVER reused after deletion (D-11).
    name: LLM-generated, 2-5 words.
    description: LLM-generated, 1-2 sentences.
    item_ids: Item IDs (integers) belonging to this cluster. Not sorted; order is insertion order.
    """
    id: int
    name: str
    description: str
    item_ids: list[int]


@dataclass
class ClusteringState:
    """
    Complete system state at one turn. Serializable to JSONL (FOUND-04).

    assignments must be complete — every item_id from 0 to N-1 must appear.
    soft_probs must be complete — every item_id must have a probability vector.
    Both dicts use int keys; JSON serialization converts these to strings,
    so deserialization must cast back with int(k) (see serialization.py).
    """
    turn_index: int
    timestamp: str                          # ISO 8601 string
    clusters: list[Cluster]
    assignments: dict[int, int]             # item_id -> cluster_id
    soft_probs: dict[int, list[float]]      # item_id -> probability vector [float × K]
