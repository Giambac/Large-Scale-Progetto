"""
hierarchy.py — cluster lineage tracking (HIER-01, HIER-02).
Grows only on oracle split/merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterNode:
    """One node in the cluster hierarchy tree."""
    cluster_id: int
    parent_id: int | None  # None = root-level cluster
    children_ids: list[int] = field(default_factory=list)
    is_active: bool = True


@dataclass
class HierarchyStore:
    """
    Tracks the full history of cluster splits and merges.

    Oracle "drill-in": navigate to a child cluster_id.
    Oracle "zoom-out": navigate to parent_id.

    Starts empty. Grows only when record_split or record_merge is called (HIER-02).
    """
    nodes: dict[int, ClusterNode] = field(default_factory=dict)

    def register(self, cluster_id: int, parent_id: int | None = None) -> None:
        """Register a new cluster node. Crashes loudly if already registered."""
        assert cluster_id not in self.nodes, (
            f"cluster_id {cluster_id} already registered in HierarchyStore"
        )
        self.nodes[cluster_id] = ClusterNode(
            cluster_id=cluster_id,
            parent_id=parent_id,
        )

    def record_split(self, parent_id: int, child_a_id: int, child_b_id: int) -> None:
        """
        Record an oracle-requested split of parent into two children.

        Marks parent as inactive, sets children_ids, registers both children.
        """
        assert parent_id in self.nodes, (
            f"record_split: parent_id {parent_id} not found in HierarchyStore"
        )
        self.nodes[parent_id].is_active = False
        self.nodes[parent_id].children_ids = [child_a_id, child_b_id]
        self.register(child_a_id, parent_id=parent_id)
        self.register(child_b_id, parent_id=parent_id)

    def record_merge(self, parent_a_id: int, parent_b_id: int, merged_id: int) -> None:
        """
        Record an oracle-requested merge of two clusters into one.

        Marks both parents as inactive. Registers merged cluster as a child of parent_a
        by convention (both conceptually parent it; parent_a chosen for determinism).
        """
        assert parent_a_id in self.nodes and parent_b_id in self.nodes, (
            f"record_merge: parent_a_id {parent_a_id} or parent_b_id {parent_b_id} "
            "not found in HierarchyStore"
        )
        self.nodes[parent_a_id].is_active = False
        self.nodes[parent_b_id].is_active = False
        # By convention, use parent_a as the canonical parent of the merged cluster.
        self.register(merged_id, parent_id=parent_a_id)
