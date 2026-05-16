"""
tests/test_baseline.py — Integration test for run_baseline() (JUDG-03).

Uses :memory: DB fixture. Does NOT call LLM APIs — uses MockOracle with a scripted reply.
Skips embedding computation by using a tiny synthetic dataset (3 items, pre-set short texts).
"""
import pytest

from src.judge import run_baseline
from src.oracle_protocol import MockOracle, OracleReply


@pytest.mark.slow  # mark as slow due to embedding computation (tiny dataset, but still)
def test_run_baseline_returns_experiment_read(db):
    """run_baseline() returns ExperimentRead with strategy_id='no_dialogue' (D-23)."""
    # Use a tiny 3-item dataset — embeddings will be computed but text is short
    records = [
        {"item_id": 0, "text": "cats and dogs"},
        {"item_id": 1, "text": "weather forecast"},
        {"item_id": 2, "text": "machine learning"},
    ]
    oracle = MockOracle(script=[OracleReply(raw_text="", satisfied=True, turn_cognitive_load=0.0)])

    result = run_baseline(
        records=records,
        persona_id="test_persona",
        seed=42,
        db=db,
        oracle=oracle,
        dataset_name="test_dataset",
    )

    assert result.strategy_id == "no_dialogue"
    assert result.persona_id == "test_persona"
    assert result.seed == 42
    assert result.total_turns == 1
    assert "mean_pairwise_accuracy" in result.details
    assert result.convergence_reason in ("oracle_satisfied", "turn_budget", "diminishing_returns", None)


@pytest.mark.slow
def test_run_baseline_writes_turns_table(db):
    """run_baseline() writes exactly 1 turn to the turns table."""
    from src.db.turns import query as turn_query

    records = [
        {"item_id": 0, "text": "apples and oranges"},
        {"item_id": 1, "text": "cars and trucks"},
        {"item_id": 2, "text": "neural networks"},
    ]
    oracle = MockOracle(script=[OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)])

    result = run_baseline(records=records, persona_id="p1", seed=0, db=db, oracle=oracle)
    turns = turn_query(db, result.id)
    assert len(turns) == 1
    assert turns[0].experiment_id == result.id
