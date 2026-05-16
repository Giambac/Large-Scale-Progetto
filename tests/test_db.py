"""
tests/test_db.py — Unit tests for the src/db/ layer (DB-01, DB-02, DB-03).

All tests use the :memory: db fixture from conftest.py (recipe §9, D-01).
No file I/O, no network, no LLM calls.
"""
import sqlite3

import pytest

from src.db import NotFound
from src.db.experiments import ExperimentCreate, ExperimentUpdate, create, get, get_by_slug, update, delete, query
from src.db.turns import TurnCreate, create as turn_create, get as turn_get, query as turn_query
from src.db.oracle_feedback import OracleFeedbackCreate, create as fb_create, get as fb_get, query as fb_query
from src.db.deletion import cascade_delete, hard_delete
from src.db.slugs import slugify, make_slug


# ── Slugs ─────────────────────────────────────────────────────────────────────

def test_slugify_basic():
    assert slugify("Random Trump seed42") == "random-trump-seed42"


def test_slugify_underscores():
    assert slugify("no_dialogue-curie-seed7") == "no-dialogue-curie-seed7"


def test_make_slug_unique(db):
    """Second slug for same name gets -2 suffix."""
    exp_data = ExperimentCreate(
        name="random-p1-seed0",
        strategy_id="random",
        persona_id="p1",
        seed=0,
        dataset="ds",
        start_timestamp="2026-01-01T00:00:00+00:00",
    )
    e1 = create(db, exp_data)
    e2 = create(db, exp_data)
    assert e1.slug == "random-p1-seed0"
    assert e2.slug == "random-p1-seed0-2"


# ── Experiments CRUD ──────────────────────────────────────────────────────────

def _make_exp(db, strategy_id="random", persona_id="p1", seed=42):
    return create(db, ExperimentCreate(
        name=f"{strategy_id}-{persona_id}-seed{seed}",
        strategy_id=strategy_id,
        persona_id=persona_id,
        seed=seed,
        dataset="test_dataset",
        start_timestamp="2026-01-01T00:00:00+00:00",
    ))


def test_create_and_get(db):
    exp = _make_exp(db)
    fetched = get(db, exp.id)
    assert fetched is not None
    assert fetched.strategy_id == "random"
    assert fetched.slug == "random-p1-seed42"


def test_get_missing_returns_none(db):
    assert get(db, 9999) is None


def test_get_by_slug(db):
    exp = _make_exp(db)
    found = get_by_slug(db, exp.slug)
    assert found is not None
    assert found.id == exp.id


def test_get_by_slug_missing_returns_none(db):
    assert get_by_slug(db, "no-such-slug") is None


def test_update_end_of_run(db):
    exp = _make_exp(db)
    updated = update(db, exp.id, ExperimentUpdate(
        total_turns=10,
        convergence_reason="oracle_satisfied",
        end_timestamp="2026-01-01T00:05:00+00:00",
        details={"turns_to_convergence": 10},
    ))
    assert updated.total_turns == 10
    assert updated.convergence_reason == "oracle_satisfied"
    assert updated.details["turns_to_convergence"] == 10


def test_update_missing_raises(db):
    with pytest.raises(NotFound):
        update(db, 9999, ExperimentUpdate(total_turns=5))


def test_soft_delete(db):
    exp = _make_exp(db)
    delete(db, exp.id)
    assert get(db, exp.id) is None  # soft-deleted returns None


def test_delete_missing_raises(db):
    with pytest.raises(NotFound):
        delete(db, 9999)


def test_query_filters(db):
    _make_exp(db, strategy_id="random")
    _make_exp(db, strategy_id="uncertainty")
    results = query(db, strategy_id="random")
    assert len(results) == 1
    assert results[0].strategy_id == "random"


# ── Turns CRUD ────────────────────────────────────────────────────────────────

# V-4-02 fix: TurnCreate field is cumulative_contradiction_count (not contradiction_count).
# Using the wrong field name causes Pydantic v2 ValidationError and all turn CRUD tests fail.
def _make_turn(db, exp_id, turn_index=0):
    return turn_create(db, TurnCreate(
        experiment_id=exp_id,
        turn_index=turn_index,
        action_type="ask_question",
        cognitive_load_score=0.3,
        cumulative_contradiction_count=0,   # correct field name per TurnCreate Pydantic model
        convergence_signal=None,
        details={"pairwise_accuracy": 0.8, "pairwise_sample_size": 50},
    ))


def test_turn_create_and_get(db):
    exp = _make_exp(db)
    turn = _make_turn(db, exp.id)
    fetched = turn_get(db, turn.id)
    assert fetched is not None
    assert fetched.turn_index == 0
    assert fetched.details["pairwise_accuracy"] == 0.8


def test_turn_fk_violation_raises(db):
    """Creating a turn with invalid experiment_id should raise IntegrityError."""
    with pytest.raises(sqlite3.IntegrityError):
        _make_turn(db, exp_id=9999)


def test_turn_query(db):
    exp = _make_exp(db)
    _make_turn(db, exp.id, turn_index=0)
    _make_turn(db, exp.id, turn_index=1)
    turns = turn_query(db, exp.id)
    assert len(turns) == 2
    assert turns[0].turn_index == 0
    assert turns[1].turn_index == 1


# ── OracleFeedback CRUD ───────────────────────────────────────────────────────

def test_oracle_feedback_create(db):
    """Compound oracle feedback = multiple rows for one turn (DB-03)."""
    exp = _make_exp(db)
    turn = _make_turn(db, exp.id)
    fb1 = fb_create(db, OracleFeedbackCreate(
        turn_id=turn.id,
        feedback_type="SplitFeedback",
        raw_text="split cluster 2",
        parsed_delta={"cluster_id": 2, "seed_item_ids": [5, 10]},
        is_contradiction=False,
    ))
    fb2 = fb_create(db, OracleFeedbackCreate(
        turn_id=turn.id,
        feedback_type="MergeFeedback",
        raw_text="merge 3 and 4",
        parsed_delta={"cluster_a_id": 3, "cluster_b_id": 4},
        is_contradiction=False,
    ))
    rows = fb_query(db, turn.id)
    assert len(rows) == 2
    assert rows[0].feedback_type == "SplitFeedback"
    assert rows[1].feedback_type == "MergeFeedback"


def test_oracle_feedback_fk_violation(db):
    with pytest.raises(sqlite3.IntegrityError):
        fb_create(db, OracleFeedbackCreate(
            turn_id=9999,
            feedback_type="GlobalFeedback",
            raw_text="ignore",
            is_contradiction=False,
        ))


# ── Cascade delete ────────────────────────────────────────────────────────────

def test_cascade_delete_soft(db):
    exp = _make_exp(db)
    turn = _make_turn(db, exp.id)
    fb_create(db, OracleFeedbackCreate(
        turn_id=turn.id,
        feedback_type="GlobalFeedback",
        raw_text="test",
        is_contradiction=False,
    ))
    cascade_delete(db, exp.id)
    # Experiment is soft-deleted
    assert get(db, exp.id) is None
    # Turns also soft-deleted — query() filters deleted_at IS NULL so returns []
    turns = turn_query(db, exp.id)
    assert turns == []


def test_cascade_delete_missing_raises(db):
    with pytest.raises(NotFound):
        cascade_delete(db, 9999)


def test_hard_delete_cascades(db):
    exp = _make_exp(db)
    turn = _make_turn(db, exp.id)
    fb_create(db, OracleFeedbackCreate(
        turn_id=turn.id, feedback_type="GlobalFeedback",
        raw_text="test", is_contradiction=False,
    ))
    hard_delete(db, exp.id)
    # Hard delete removes the row entirely — FK cascade removes turns + feedback
    row = db.execute("SELECT id FROM experiments WHERE id = ?", (exp.id,)).fetchone()
    assert row is None
    # Turns gone via ON DELETE CASCADE
    turn_rows = db.execute("SELECT id FROM turns WHERE experiment_id = ?", (exp.id,)).fetchall()
    assert turn_rows == []
