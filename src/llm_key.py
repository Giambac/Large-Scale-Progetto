"""
src/llm_key.py — Resolve the active LLM API key from .env or environment.

Priority order (first non-empty key wins):
  1. ANTHROPIC_API_KEY
  2. OPENAI_API_KEY
  3. GOOGLE_API_KEY

.env is loaded once at import time without overwriting variables already set
in the process environment (so CI/shell exports always take precedence).
"""
from __future__ import annotations

import os
from pathlib import Path


_ENV_PATH = Path(__file__).parent.parent / ".env"

_PRIORITY: list[tuple[str, str]] = [
    ("openai",    "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("google",    "GOOGLE_API_KEY"),
]


def _load_dotenv(path: Path) -> None:
    """Parse KEY=VALUE lines from *path* into os.environ (no-overwrite)."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(_ENV_PATH)


def resolve_llm_key() -> tuple[str, str]:
    """Return *(provider, api_key)* for the first key found in priority order.

    Raises AssertionError (fail-loudly) if no key is set.
    """
    for provider, var in _PRIORITY:
        key = os.environ.get(var, "")
        if key:
            return provider, key

    raise AssertionError(
        "No LLM API key found. Add one to .env (copy .env.example):\n"
        "  ANTHROPIC_API_KEY=sk-ant-...   (Claude — checked first)\n"
        "  OPENAI_API_KEY=sk-...          (GPT — checked second)\n"
        "  GOOGLE_API_KEY=...             (Gemini — checked third)"
    )
