"""
src/db/connection.py — SQLite connection factory (D-01, D-04).

connect() opens (or creates) experiments.db with:
  - check_same_thread=False  (MANDATORY — run_conversation runs in a background thread)
  - PRAGMA journal_mode=WAL  (D-04: concurrent FastAPI reads during writes)
  - PRAGMA foreign_keys=ON   (D-06: FK integrity enforced at DB level)
  - PRAGMA synchronous=NORMAL (acceptable perf/safety tradeoff for non-financial workloads)

Tests use sqlite3.connect(":memory:", check_same_thread=False) — production code
never knows which connection it received (recipe §9).
"""
from __future__ import annotations

import sqlite3


DB_PATH = "experiments.db"


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at path.

    Returns a connection with WAL mode, foreign keys, and check_same_thread=False.
    Caller is responsible for closing the connection (use as context manager or explicit .close()).
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create all tables and indexes if they do not exist.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    Call once at run start before any inserts.

    Schema spec: docs/MODEL.md (always edit that file first when schema changes).
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id                  INTEGER PRIMARY KEY,
            name                TEXT    NOT NULL,
            slug                TEXT    NOT NULL,
            created_at          TEXT    NOT NULL,
            updated_at          TEXT    NOT NULL,
            deleted_at          TEXT,
            strategy_id         TEXT    NOT NULL,
            persona_id          TEXT    NOT NULL,
            seed                INTEGER NOT NULL,
            dataset             TEXT    NOT NULL,
            oracle_type         TEXT    NOT NULL DEFAULT 'llm',
            total_turns         INTEGER,
            convergence_reason  TEXT,
            start_timestamp     TEXT    NOT NULL,
            end_timestamp       TEXT,
            details             TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS experiment_slug_live
            ON experiments(slug) WHERE deleted_at IS NULL;

        CREATE INDEX IF NOT EXISTS idx_experiments_strategy
            ON experiments(strategy_id, persona_id, dataset);

        CREATE INDEX IF NOT EXISTS idx_experiments_oracle_type
            ON experiments(oracle_type);

        CREATE TABLE IF NOT EXISTS turns (
            id                  INTEGER PRIMARY KEY,
            experiment_id       INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            created_at          TEXT    NOT NULL,
            turn_index          INTEGER NOT NULL,
            action_type         TEXT    NOT NULL,
            cognitive_load_score REAL   NOT NULL,
            cumulative_contradiction_count INTEGER NOT NULL,
            convergence_signal  TEXT,
            deleted_at          TEXT,
            details             TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_turns_experiment
            ON turns(experiment_id, turn_index);

        CREATE TABLE IF NOT EXISTS oracle_feedback (
            id              INTEGER PRIMARY KEY,
            turn_id         INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
            created_at      TEXT    NOT NULL,
            feedback_type   TEXT    NOT NULL,
            raw_text        TEXT    NOT NULL,
            parsed_delta    TEXT,
            is_contradiction INTEGER NOT NULL DEFAULT 0,
            details         TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_oracle_feedback_turn
            ON oracle_feedback(turn_id);
    """)
    conn.commit()
