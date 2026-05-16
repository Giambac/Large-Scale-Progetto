"""Round-trip test for oracle_type column (EXP-V2-01)."""
import sqlite3
from datetime import datetime, timezone

import pytest

from src.db.connection import init_schema
from src.db.experiments import ExperimentCreate, create, query, get


def _ts():
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    init_schema(c)
    yield c
    c.close()


def test_default_oracle_type_is_llm(conn):
    created = create(conn, ExperimentCreate(
        name="default-llm",
        strategy_id="random",
        persona_id="p1",
        seed=1,
        dataset="dataset/train.jsonl",
        start_timestamp=_ts(),
    ))
    assert created.oracle_type == "llm"
    refetched = get(conn, created.id)
    assert refetched is not None and refetched.oracle_type == "llm"


def test_explicit_oracle_type_human(conn):
    created = create(conn, ExperimentCreate(
        name="explicit-human",
        strategy_id="random",
        persona_id="p1",
        seed=2,
        dataset="dataset/train.jsonl",
        start_timestamp=_ts(),
        oracle_type="human",
    ))
    assert created.oracle_type == "human"
    rows = query(conn)
    assert any(r.oracle_type == "human" for r in rows)


def test_index_exists_on_oracle_type(conn):
    idx = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()]
    assert "idx_experiments_oracle_type" in idx
