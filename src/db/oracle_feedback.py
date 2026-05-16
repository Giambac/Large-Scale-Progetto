"""
src/db/oracle_feedback.py — OracleFeedback entity CRUD (recipe §7, D-05, D-10).

Pydantic triples: OracleFeedbackCreate / OracleFeedbackRead.
Compound oracle messages produce multiple rows per turn (DB-03 requirement).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OracleFeedbackCreate(BaseModel):
    turn_id: int
    feedback_type: str
    raw_text: str
    parsed_delta: dict[str, Any] | None = None
    is_contradiction: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class OracleFeedbackRead(BaseModel):
    id: int
    turn_id: int
    created_at: str
    feedback_type: str
    raw_text: str
    parsed_delta: dict[str, Any] | None
    is_contradiction: bool
    details: dict[str, Any]


def create(conn: sqlite3.Connection, data: OracleFeedbackCreate) -> OracleFeedbackRead:
    """Insert an oracle_feedback row. Raises sqlite3.IntegrityError if turn_id FK is invalid."""
    now = _now()
    parsed_json = json.dumps(data.parsed_delta) if data.parsed_delta is not None else None
    details_json = json.dumps(data.details)
    cursor = conn.execute(
        """
        INSERT INTO oracle_feedback
            (turn_id, created_at, feedback_type, raw_text,
             parsed_delta, is_contradiction, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (data.turn_id, now, data.feedback_type, data.raw_text,
         parsed_json, int(data.is_contradiction), details_json),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM oracle_feedback WHERE id = ?", (cursor.lastrowid,)).fetchone()
    assert row is not None
    return _read(row)


def get(conn: sqlite3.Connection, feedback_id: int) -> OracleFeedbackRead | None:
    """Return OracleFeedbackRead for id, or None."""
    row = conn.execute("SELECT * FROM oracle_feedback WHERE id = ?", (feedback_id,)).fetchone()
    return _read(row) if row else None


def query(conn: sqlite3.Connection, turn_id: int) -> list[OracleFeedbackRead]:
    """Return all feedback rows for a turn. Returns []."""
    rows = conn.execute(
        "SELECT * FROM oracle_feedback WHERE turn_id = ? ORDER BY id",
        (turn_id,),
    ).fetchall()
    return [_read(r) for r in rows]


def _read(row: sqlite3.Row) -> OracleFeedbackRead:
    d = dict(row)
    d["details"] = json.loads(d["details"] or "{}")
    d["parsed_delta"] = json.loads(d["parsed_delta"]) if d["parsed_delta"] else None
    d["is_contradiction"] = bool(d["is_contradiction"])
    return OracleFeedbackRead(**d)
