"""
src/db/deletion.py — Cascade delete orchestrator (recipe §4.2, D-07).

Soft-delete cascade: sets deleted_at on experiment + all its turns.
oracle_feedback has no deleted_at column (D-05 minimal shape) — child rows are excluded
from live reads by filtering parent turns (WHERE deleted_at IS NULL). Hard delete via
ON DELETE CASCADE FK removes oracle_feedback rows when their parent turn is deleted.

Hard delete: remove experiment row; ON DELETE CASCADE FK handles turns and oracle_feedback.

Rule: Never call individual entity delete() functions for cascade — always go through here
so the cascade is atomic and auditable.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src.db import NotFound
from src.db import experiments as exp_module


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cascade_delete(conn: sqlite3.Connection, experiment_id: int) -> None:
    """
    Soft-delete an experiment and all its dependent rows.

    Sets deleted_at on:
      1. turns rows for this experiment (WHERE deleted_at IS NULL)
      2. the experiment row itself

    oracle_feedback has no deleted_at per D-05 minimal shape — child rows are hard-deleted
    via ON DELETE CASCADE FK when their parent turn is hard-deleted. Filtering turns by
    deleted_at IS NULL is sufficient for live reads.

    Raises NotFound if the experiment id does not exist or is already soft-deleted.
    """
    existing = exp_module.get(conn, experiment_id)
    if existing is None:
        raise NotFound(f"Experiment id={experiment_id} not found for cascade delete")

    now = _now()

    # 1. Soft-delete turns
    conn.execute(
        "UPDATE turns SET deleted_at = ? WHERE experiment_id = ? AND deleted_at IS NULL",
        (now, experiment_id),
    )

    # 2. Soft-delete experiment
    conn.execute(
        "UPDATE experiments SET deleted_at = ?, updated_at = ? WHERE id = ?",
        (now, now, experiment_id),
    )
    conn.commit()


def hard_delete(conn: sqlite3.Connection, experiment_id: int) -> None:
    """
    Hard-delete an experiment row. ON DELETE CASCADE removes turns + oracle_feedback.

    Use only for test teardown or explicit data purge. Production code uses cascade_delete().
    Raises NotFound if the id doesn't exist (live or soft-deleted check).
    """
    row = conn.execute("SELECT id FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise NotFound(f"Experiment id={experiment_id} not found for hard delete")
    conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
    conn.commit()
