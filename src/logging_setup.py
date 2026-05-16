"""
src/logging_setup.py — deviation() helper + STRICT_MODE (D-29, recipe §11).

Usage in Phase 4 new code (src/db/, src/judge.py, examples/run_baseline.py):
    from src.logging_setup import deviation

    if something_unexpected:
        deviation("config fell back to default", key="epsilon", value=0.05)

STRICT_MODE behavior:
    STRICT_MODE=1 env var  → raises UnexpectedDeviation (for test assertions + CI later)
    STRICT_MODE unset/0    → logs a WARNING via standard logging

Phase 1-3 code uses `assert` for must-be-true invariants — do NOT replace those.
deviation() is complementary: use it for "this can happen but normally shouldn't" branches.

Testing deviation paths (recipe §11):
    import os
    os.environ["STRICT_MODE"] = "1"
    with pytest.raises(UnexpectedDeviation):
        deviation("something unexpected")
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


class UnexpectedDeviation(RuntimeError):
    """
    Raised by deviation() when STRICT_MODE=1.

    Inherits from RuntimeError so callers can catch it specifically
    without accidentally catching all exceptions.
    """


def deviation(msg: str, **kwargs: Any) -> None:
    """
    Mark an unexpected-but-possible branch.

    Args:
        msg:    Human-readable description of what happened.
        **kwargs: Key-value pairs to include in the log record / exception message.

    Behavior:
        STRICT_MODE=1  → raises UnexpectedDeviation(f"{msg} | {kwargs}")
        otherwise      → logs WARNING with extra={kwargs}

    Never use for must-be-true invariants — use assert for those.
    """
    if os.environ.get("STRICT_MODE", "0") == "1":
        extra_str = " | " + ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) if kwargs else ""
        raise UnexpectedDeviation(f"{msg}{extra_str}")
    log.warning(msg, extra=kwargs)
