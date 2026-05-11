"""
Tests for agent_functions.py — f_output, f_next_best_step, f_next_state.
Covers CLUS-01, CLUS-03, CLUS-04, FB-01, FB-02, FB-03.
"""
import pytest


# ── f_output (CLUS-01) ───────────────────────────────────────────────

def test_f_output_returns_complete_assignment(tiny_state_3cluster):
    """CLUS-01: f_output returns a ClusteringState with all 6 items in assignments."""
    from src.agent_functions import f_output
    from src.state import ClusteringState
    result = f_output(tiny_state_3cluster)
    assert isinstance(result, ClusteringState)
    assert len(result.assignments) == 6
    assert set(result.assignments.keys()) == {0, 1, 2, 3, 4, 5}


def test_f_output_complete_soft_probs(tiny_state_3cluster):
    """CLUS-01: f_output result has soft_probs for all N items."""
    from src.agent_functions import f_output
    result = f_output(tiny_state_3cluster)
    assert len(result.soft_probs) == 6


def test_f_output_is_pure(tiny_state_3cluster):
    """f_output does not mutate the input state."""
    from src.agent_functions import f_output
    original_turn = tiny_state_3cluster.turn_index
    f_output(tiny_state_3cluster)
    assert tiny_state_3cluster.turn_index == original_turn


# ── f_next_best_step (CLUS-03) ────────────────────────────────────────

def test_f_next_best_step_returns_action(tiny_state_3cluster):
    """CLUS-03: f_next_best_step returns an Action instance."""
    from src.agent_functions import f_next_best_step
    from src.strategy import RandomStrategy, Action
    from src.uncertainty import f_uncertainty
    strategy = RandomStrategy(seed=42)
    report = f_uncertainty(tiny_state_3cluster)
    action = f_next_best_step(tiny_state_3cluster, strategy, report)
    assert isinstance(action, Action)


def test_random_strategy_deterministic(tiny_state_3cluster):
    """RandomStrategy with same seed returns same action on same state."""
    from src.agent_functions import f_next_best_step
    from src.strategy import RandomStrategy
    from src.uncertainty import f_uncertainty
    report = f_uncertainty(tiny_state_3cluster)
    a1 = f_next_best_step(tiny_state_3cluster, RandomStrategy(seed=0), report)
    a2 = f_next_best_step(tiny_state_3cluster, RandomStrategy(seed=0), report)
    assert a1.action_type == a2.action_type


# ── f_next_state — split (FB-02) ──────────────────────────────────────

def test_split_produces_two_new_clusters(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-02: split produces 2 new clusters and retires the old cluster ID."""
    from src.agent_functions import f_next_state
    from src.feedback import SplitFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "Sub", "description": "Sub cluster."}
    delta = SplitFeedback(cluster_id=0, seed_item_ids=[])
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    assert len(new_state.clusters) == 4  # 3 - 1 + 2


def test_split_retires_old_id(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-02: split retires the original cluster ID (D-11: never reused)."""
    from src.agent_functions import f_next_state
    from src.feedback import SplitFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "Sub", "description": "Sub."}
    delta = SplitFeedback(cluster_id=0, seed_item_ids=[])
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    active_ids = {c.id for c in new_state.clusters}
    assert 0 not in active_ids, "Retired cluster ID 0 must not appear in new state"


def test_split_soft_probs_rows_sum_to_one(tiny_state_3cluster, mock_embeddings_3cluster):
    """After split, all soft_probs rows sum to 1.0 (within tolerance)."""
    from src.agent_functions import f_next_state
    from src.feedback import SplitFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "Sub", "description": "Sub."}
    delta = SplitFeedback(cluster_id=0, seed_item_ids=[])
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    for item_id, probs in new_state.soft_probs.items():
        total = sum(probs)
        assert abs(total - 1.0) < 1e-5, f"item {item_id} soft_probs sum={total}"


# ── f_next_state — merge (FB-02) ─────────────────────────────────────

def test_merge_produces_one_new_cluster(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-02: merge reduces cluster count by 1 (3 clusters -> 2 clusters)."""
    from src.agent_functions import f_next_state
    from src.feedback import MergeFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "Merged", "description": "Merged cluster."}
    delta = MergeFeedback(cluster_a_id=0, cluster_b_id=1)
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    assert len(new_state.clusters) == 2


def test_merge_soft_probs_normalized(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-02: after merge, all soft_probs rows sum to 1.0."""
    from src.agent_functions import f_next_state
    from src.feedback import MergeFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "Merged", "description": "Merged."}
    delta = MergeFeedback(cluster_a_id=0, cluster_b_id=1)
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    for item_id, probs in new_state.soft_probs.items():
        assert abs(sum(probs) - 1.0) < 1e-5, f"item {item_id} not normalized after merge"


# ── f_next_state — move_item (FB-03) ─────────────────────────────────

def test_move_item_sets_0_95(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-03: MoveItemFeedback sets target cluster probability to ORACLE_MOVE_CONFIDENCE (0.95)."""
    from src.agent_functions import f_next_state
    from src.feedback import MoveItemFeedback, ORACLE_MOVE_CONFIDENCE
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    delta = MoveItemFeedback(item_id=0, target_cluster_id=2)
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    # item 0 was in cluster 0 with soft_probs [0.80, 0.10, 0.10]
    # after move to cluster 2, soft_probs[0][index_of_cluster_2] should be 0.95
    probs = new_state.soft_probs[0]
    cluster_ids = [c.id for c in new_state.clusters]
    target_idx = cluster_ids.index(2)
    assert abs(probs[target_idx] - ORACLE_MOVE_CONFIDENCE) < 1e-6


def test_move_item_soft_probs_sum_to_one(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-03: after point move, item's soft_probs row sums to 1.0."""
    from src.agent_functions import f_next_state
    from src.feedback import MoveItemFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    delta = MoveItemFeedback(item_id=0, target_cluster_id=2)
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    assert abs(sum(new_state.soft_probs[0]) - 1.0) < 1e-5


# ── f_next_state — GlobalFeedback accumulation (FB-01) ───────────────

def test_global_feedback_accumulates(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-01: two GlobalFeedback deltas → global_instructions list grows to len 2."""
    from src.agent_functions import f_next_state
    from src.feedback import GlobalFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    global_instructions: list[str] = []
    deltas = [
        GlobalFeedback(instruction_text="Focus on sentiment."),
        GlobalFeedback(instruction_text="Separate by topic."),
    ]
    f_next_state(tiny_state_3cluster, deltas, store, namer, global_instructions=global_instructions)
    assert len(global_instructions) == 2
    assert global_instructions[0] == "Focus on sentiment."
    assert global_instructions[1] == "Separate by topic."


# ── f_next_state — type priority order (FB-01) ────────────────────────

def test_feedback_priority_order_global_first(tiny_state_3cluster, mock_embeddings_3cluster):
    """FB-01: GlobalFeedback is processed before cluster-level deltas (D-07)."""
    from src.agent_functions import f_next_state
    from src.feedback import SplitFeedback, GlobalFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    # Compound: global + split. Both should be applied; global first.
    deltas = [
        SplitFeedback(cluster_id=0, seed_item_ids=[]),
        GlobalFeedback(instruction_text="Focus on topic."),
    ]
    # Should not raise; global processed before split regardless of list order
    new_state = f_next_state(tiny_state_3cluster, deltas, store, namer)
    assert len(new_state.clusters) == 4  # split still fires


# ── completeness invariant ────────────────────────────────────────────

def test_f_next_state_all_items_assigned(tiny_state_3cluster, mock_embeddings_3cluster):
    """After any f_next_state call, all N items have an assignment."""
    from src.agent_functions import f_next_state
    from src.feedback import MergeFeedback
    from src.embedding_store import EmbeddingStore
    from unittest.mock import MagicMock
    store = EmbeddingStore(mock_embeddings_3cluster)
    namer = MagicMock()
    namer.name_cluster.return_value = {"name": "X", "description": "X."}
    delta = MergeFeedback(cluster_a_id=1, cluster_b_id=2)
    new_state = f_next_state(tiny_state_3cluster, [delta], store, namer)
    assert len(new_state.assignments) == 6
