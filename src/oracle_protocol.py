"""
oracle_protocol.py — OracleProtocol interface, OracleReply, MockOracle scripted oracle (D-03).

Phase 3 replaces MockOracle without changing the loop.
The OracleProtocol is a typing.Protocol so Phase 3's real Oracle Agent requires no inheritance;
structural subtyping via isinstance check works because OracleProtocol is @runtime_checkable.

Phase 3 OracleReply extension: OracleReply gains contradiction_detected and contradicted_turn
fields for drift detection (ORC-04). Both fields have defaults so existing MockOracle callers
require no changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.state import ClusteringState


@dataclass
class OracleReply:
    """
    Reply from the oracle agent.

    raw_text:          Natural-language response from the oracle.
    satisfied:         Explicit satisfaction token — the primary stop condition (D-08).
    turn_cognitive_load: Phase 3 fills this in; Phase 2 MockOracle always returns 0.0.
    contradiction_detected / contradicted_turn: Phase 3 drift detection fields.
        Both have defaults so MockOracle callers require no changes.

    NOT frozen — Phase 3 may add fields without breaking existing callers.
    """
    raw_text: str
    satisfied: bool
    turn_cognitive_load: float = 0.0  # Phase 3 fills this in; Phase 2 stub = 0.0
    contradiction_detected: bool = False     # Phase 3: True if structural contradiction detected
    contradicted_turn: int | None = None     # Phase 3: turn_index of the conflicting prior delta


@runtime_checkable
class OracleProtocol(Protocol):
    """
    Interface for the oracle. Phase 2: MockOracle. Phase 3: LLM-backed Oracle Agent.

    Structural subtyping: implementations do NOT need to inherit from this class.
    """
    def reply(self, state: ClusteringState, message: str) -> OracleReply:
        ...


class MockOracle:
    """
    Scripted turn-indexed reply sequence. Deterministic for 30-turn loop test (Pitfall 5).

    Implements OracleProtocol via structural subtyping (does not inherit from it).
    After script is exhausted, returns a neutral default OracleReply to prevent
    degenerate state accumulation in long loop tests.
    """

    def __init__(self, script: list[OracleReply]) -> None:
        assert len(script) > 0, "MockOracle script must be non-empty"
        self._script = script
        self._turn = 0

    def reply(self, state: ClusteringState, message: str) -> OracleReply:
        """Return the next scripted reply, or neutral default after script is exhausted."""
        if self._turn < len(self._script):
            r = self._script[self._turn]
        else:
            r = OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)
        self._turn += 1
        return r
