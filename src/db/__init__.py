"""
src/db/__init__.py — DB subpackage public surface.

Rule (recipe §6): src/db/ is the ONLY layer that touches SQL.
sqlite3.execute anywhere outside src/db/ is a bug — fix immediately.
"""
from __future__ import annotations


class NotFound(Exception):
    """Raised by update() and delete() when the requested id does not exist."""
