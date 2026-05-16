"""
src/db/experiments.py — Experiment entity CRUD (recipe §7, D-05, D-08, D-10).

Pydantic triples: ExperimentCreate / ExperimentRead / ExperimentUpdate.
Pydantic is used ONLY inside src/db/ — rest of codebase uses dataclasses (D-10).

Error conventions (D-11):
  create()        — raises sqlite3.IntegrityError on slug collision or FK violation
  get()           — returns None on miss
  get_by_slug()   — returns None on miss
  query()         — returns [] when no rows match
  update()        — raises NotFound when id doesn't exist
  delete()        — soft-delete; raises NotFound when id doesn't exist
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.db import NotFound
from src.db.slugs import make_slug


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


class ExperimentCreate(BaseModel):
    name: str
    strategy_id: str
    persona_id: str
    seed: int
    dataset: str
    oracle_type: str = "llm"
    start_timestamp: str
    details: dict[str, Any] = Field(default_factory=dict)
    slug_override: str | None = None


class ExperimentRead(BaseModel):
    id: int
    name: str
    slug: str
    created_at: str
    updated_at: str
    deleted_at: str | None
    strategy_id: str
    persona_id: str
    seed: int
    dataset: str
    oracle_type: str
    total_turns: int | None
    convergence_reason: str | None
    start_timestamp: str
    end_timestamp: str | None
    details: dict[str, Any]


class ExperimentUpdate(BaseModel):
    total_turns: int | None = None
    convergence_reason: str | None = None
    end_timestamp: str | None = None
    details: dict[str, Any] | None = None


def create(conn: sqlite3.Connection, data: ExperimentCreate) -> ExperimentRead:
    """Insert a new experiment row. Returns the created ExperimentRead."""
    now = _now()
    slug = make_slug(conn, "experiments", data.name, override=data.slug_override)
    details_json = json.dumps(data.details)
    cursor = conn.execute(
        """
        INSERT INTO experiments
            (name, slug, created_at, updated_at, strategy_id, persona_id, seed,
             dataset, oracle_type, start_timestamp, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (data.name, slug, now, now, data.strategy_id, data.persona_id, data.seed,
         data.dataset, data.oracle_type, data.start_timestamp, details_json),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (cursor.lastrowid,)).fetchone()
    assert row is not None
    return _read(row)


def get(conn: sqlite3.Connection, experiment_id: int) -> ExperimentRead | None:
    """Return ExperimentRead for id, or None if not found / soft-deleted."""
    row = conn.execute(
        "SELECT * FROM experiments WHERE id = ? AND deleted_at IS NULL",
        (experiment_id,),
    ).fetchone()
    return _read(row) if row else None


def get_by_slug(conn: sqlite3.Connection, slug: str) -> ExperimentRead | None:
    """Return ExperimentRead for slug (live rows only), or None."""
    row = conn.execute(
        "SELECT * FROM experiments WHERE slug = ? AND deleted_at IS NULL",
        (slug,),
    ).fetchone()
    return _read(row) if row else None


def update(conn: sqlite3.Connection, experiment_id: int, data: ExperimentUpdate) -> ExperimentRead:
    """
    Partial update — only non-None fields are written.
    Raises NotFound if experiment_id doesn't exist or is soft-deleted.
    """
    existing = get(conn, experiment_id)
    if existing is None:
        raise NotFound(f"Experiment id={experiment_id} not found")

    updates: dict[str, Any] = {"updated_at": _now()}
    if data.total_turns is not None:
        updates["total_turns"] = data.total_turns
    if data.convergence_reason is not None:
        updates["convergence_reason"] = data.convergence_reason
    if data.end_timestamp is not None:
        updates["end_timestamp"] = data.end_timestamp
    if data.details is not None:
        merged = {**existing.details, **data.details}
        updates["details"] = json.dumps(merged)

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [experiment_id]
    conn.execute(f"UPDATE experiments SET {set_clause} WHERE id = ?", values)  # noqa: S608
    conn.commit()
    result = get(conn, experiment_id)
    assert result is not None
    return result


def delete(conn: sqlite3.Connection, experiment_id: int) -> None:
    """
    Soft-delete an experiment (sets deleted_at). Raises NotFound if not live.
    Does NOT cascade — call deletion.cascade_delete() instead (D-07).
    """
    existing = get(conn, experiment_id)
    if existing is None:
        raise NotFound(f"Experiment id={experiment_id} not found")
    now = _now()
    conn.execute(
        "UPDATE experiments SET deleted_at = ?, updated_at = ? WHERE id = ?",
        (now, now, experiment_id),
    )
    conn.commit()


def query(
    conn: sqlite3.Connection,
    *,
    strategy_id: str | None = None,
    persona_id: str | None = None,
    dataset: str | None = None,
    include_deleted: bool = False,
) -> list[ExperimentRead]:
    """Filter experiments. Returns [] when nothing matches."""
    clauses = []
    params: list[Any] = []
    if not include_deleted:
        clauses.append("deleted_at IS NULL")
    if strategy_id is not None:
        clauses.append("strategy_id = ?")
        params.append(strategy_id)
    if persona_id is not None:
        clauses.append("persona_id = ?")
        params.append(persona_id)
    if dataset is not None:
        clauses.append("dataset = ?")
        params.append(dataset)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(f"SELECT * FROM experiments {where} ORDER BY id DESC", params).fetchall()  # noqa: S608
    return [_read(r) for r in rows]


def _read(row: sqlite3.Row) -> ExperimentRead:
    d = dict(row)
    d["details"] = json.loads(d["details"] or "{}")
    return ExperimentRead(**d)
