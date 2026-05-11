"""Tests for hierarchy.py — HIER-01 (navigable hierarchy), HIER-02 (incremental growth)."""
import pytest


def test_hierarchy_store_register():
    """register() adds a root ClusterNode with parent_id=None."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    store.register(cluster_id=0, parent_id=None)
    assert 0 in store.nodes
    assert store.nodes[0].parent_id is None
    assert store.nodes[0].is_active is True


def test_hierarchy_store_register_crashes_on_duplicate():
    """register() raises AssertionError if cluster_id already exists (fail loudly)."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    store.register(cluster_id=0)
    with pytest.raises(AssertionError):
        store.register(cluster_id=0)


def test_record_split_marks_parent_inactive():
    """record_split() marks parent as is_active=False and sets children_ids."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    store.register(0)
    store.record_split(parent_id=0, child_a_id=1, child_b_id=2)
    assert store.nodes[0].is_active is False
    assert set(store.nodes[0].children_ids) == {1, 2}


def test_record_split_registers_children():
    """record_split() registers both child nodes with parent_id pointing to retired parent."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    store.register(0)
    store.record_split(parent_id=0, child_a_id=1, child_b_id=2)
    assert 1 in store.nodes
    assert 2 in store.nodes
    assert store.nodes[1].parent_id == 0
    assert store.nodes[2].parent_id == 0
    assert store.nodes[1].is_active is True
    assert store.nodes[2].is_active is True


def test_record_merge_marks_both_parents_inactive():
    """record_merge() marks both merged clusters as is_active=False."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    store.register(0)
    store.register(1)
    store.record_merge(parent_a_id=0, parent_b_id=1, merged_id=2)
    assert store.nodes[0].is_active is False
    assert store.nodes[1].is_active is False


def test_record_merge_registers_merged_node():
    """record_merge() registers merged cluster as new active node."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    store.register(0)
    store.register(1)
    store.record_merge(parent_a_id=0, parent_b_id=1, merged_id=2)
    assert 2 in store.nodes
    assert store.nodes[2].is_active is True


def test_no_hierarchy_at_init():
    """HIER-02: empty HierarchyStore starts with no nodes (not pre-computed)."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    assert len(store.nodes) == 0


def test_hierarchy_grows_incrementally():
    """HIER-02: hierarchy grows only when record_split or record_merge is called."""
    from src.hierarchy import HierarchyStore
    store = HierarchyStore()
    store.register(0)
    assert len(store.nodes) == 1
    store.record_split(0, 1, 2)
    assert len(store.nodes) == 3   # parent + 2 children
