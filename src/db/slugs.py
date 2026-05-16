"""
src/db/slugs.py — Slug generation and uniqueness enforcement (D-12).

Slugs are immutable once written.
Collision suffix: -2, -3, ... backed by partial unique index WHERE deleted_at IS NULL.
Pattern for experiments: "{strategy_id}-{persona_id}-seed{seed}"
"""
from __future__ import annotations

import re
import sqlite3


def slugify(name: str) -> str:
    """
    Convert a name to a URL-safe slug.

    Lowercase, spaces and underscores to hyphens, strip non-alphanumeric (except hyphens),
    collapse multiple hyphens, strip leading/trailing hyphens.
    """
    s = name.lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def make_slug(
    conn: sqlite3.Connection,
    table: str,
    name: str,
    override: str | None = None,
) -> str:
    """
    Generate a unique slug for a new row in table.

    If override is provided, use it directly (caller is responsible for uniqueness).
    Otherwise slugify(name) and append -2, -3, ... until unique among live rows
    (WHERE deleted_at IS NULL).

    Args:
        conn:     Open DB connection.
        table:    Table name (e.g. "experiments").
        name:     Human-readable name to slugify.
        override: If provided, return this slug directly.

    Returns:
        A slug not currently used by any live row in table.
    """
    if override is not None:
        return override

    base = slugify(name)
    assert base, f"slugify({name!r}) produced an empty string — provide a valid name"

    candidate = base
    suffix = 2
    while True:
        row = conn.execute(
            f"SELECT id FROM {table} WHERE slug = ? AND deleted_at IS NULL",  # noqa: S608
            (candidate,),
        ).fetchone()
        if row is None:
            return candidate
        candidate = f"{base}-{suffix}"
        suffix += 1
