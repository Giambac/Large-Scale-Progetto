"""
src/db/turns.py — Turn entity CRUD (recipe §7, D-05, D-10).

Pydantic triples: TurnCreate / TurnRead.
Error conventions (D-11): create() raises on FK violation; get() returns None; query() returns [].
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TurnCreate(BaseModel):
    experiment_id: int
    turn_index: int
    action_type: str
    cognitive_load_score: float
    cumulative_contradiction_count: int
    convergence_signal: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TurnRead(BaseModel):
    id: int
    experiment_id: int
    created_at: str
    turn_index: int
    action_type: str
    cognitive_load_score: float
    cumulative_contradiction_count: int
    convergence_signal: str | None
    details: dict[str, Any]


def create(conn: sqlite3.Connection, data: TurnCreate) -> TurnRead:
    """Insert a turn row. Raises sqlite3.IntegrityError if experiment_id FK is invalid."""
    now = _now()
    details_json = json.dumps(data.details)
    cursor = conn.execute(
        """
        INSERT INTO turns
            (experiment_id, created_at, turn_index, action_type,
             cognitive_load_score, cumulative_contradiction_count, convergence_signal, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (data.experiment_id, now, data.turn_index, data.action_type,
         data.cognitive_load_score, data.cumulative_contradiction_count,
         data.convergence_signal, details_json),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM turns WHERE id = ?", (cursor.lastrowid,)).fetchone()
    assert row is not None
    return _read(row)


def get(conn: sqlite3.Connection, turn_id: int) -> TurnRead | None:
    """Return TurnRead for id, or None."""
    row = conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
    return _read(row) if row else None


def query(conn: sqlite3.Connection, experiment_id: int) -> list[TurnRead]:
    """Return all turns for an experiment ordered by turn_index. Returns []."""
    rows = conn.execute(
        "SELECT * FROM turns WHERE experiment_id = ? ORDER BY turn_index",
        (experiment_id,),
    ).fetchall()
    return [_read(r) for r in rows]


def _read(row: sqlite3.Row) -> TurnRead:
    d = dict(row)
    d["details"] = json.loads(d["details"] or "{}")
    return TurnRead(**d)
