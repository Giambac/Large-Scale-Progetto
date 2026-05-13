"""
strategy.py — Decide cosa fare al prossimo turno della conversazione.

Ad ogni turno il sistema deve scegliere un'azione: mostrare tutti i cluster, mostrarne solo un sottoinsieme, fare una domanda all'oracle, o fermarsi.

Questo file definisce le possibili azioni, l'interfaccia comune per le strategie, e la strategia della Phase 2 (RandomStrategy) che sceglie semplicemente a caso.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from src.state import ClusteringState

"""
class Action:
    Un'azione che il sistema può compiere al prossimo turno.

    action_type     — il tipo di azione. Quattro valori possibili:
                        "show_full" → mostra tutti i cluster all'oracle
                        "show_subset" → mostra solo alcuni cluster (richiede >= 2 cluster)
                        "ask_question" → fai una domanda mirata all'oracle
                        "stop" → ferma la conversazione

    payload         — informazioni aggiuntive sull'azione, es. quale cluster mostrare o quale domanda fare.
"""
@dataclass
class Action:
    action_type: Literal["show_full", "show_subset", "ask_question", "stop"]
    payload: dict = field(default_factory=dict)

"""
class StrategyProtocol(Protocol):
    L'interfaccia che qualsiasi strategia deve rispettare.

    Basta avere un metodo select(state, uncertainty_report) che restituisce un'Action. Non serve ereditare da questa classe.
"""
@runtime_checkable
class StrategyProtocol(Protocol):
    def select(self, state: ClusteringState, uncertainty_report: "object") -> Action:
        ...

"""
def _enumerate_valid_actions(
    Costruisce la lista di azioni disponibili in base allo stato corrente.

    Regole:
        - "show_full" è sempre disponibile.
        - "ask_question" è disponibile se c'è almeno un cluster.
        - "show_subset" è disponibile solo se ci sono almeno 2 cluster (non ha senso mostrare un "sottoinsieme" se ce n'è solo uno).
        - "stop" è sempre disponibile.

    La lista non può mai essere vuota — se lo fosse sarebbe un bug nel sistema.
"""
def _enumerate_valid_actions(
    state: ClusteringState,
    uncertainty_report: object,
) -> list[Action]:
    actions: list[Action] = []
    actions.append(Action(action_type="show_full"))
    if len(state.clusters) > 0:
        actions.append(Action(action_type="ask_question"))
    if len(state.clusters) >= 2:
        actions.append(Action(action_type="show_subset"))
    actions.append(Action(action_type="stop"))
    assert len(actions) > 0, (
        "BUG: _enumerate_valid_actions returned empty list — state has no clusters at all"
    )
    return actions

"""
class RandomStrategy:
    La strategia della Phase 2: sceglie un'azione a caso tra quelle disponibili.

    Usa random.Random(seed) — un generatore casuale privato con seed fisso, non il generatore globale di Python. Questo garantisce che con lo stesso
    seed la strategia faccia sempre le stesse scelte, rendendo i test completamente deterministici e riproducibili.
"""
class RandomStrategy:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, state: ClusteringState, uncertainty_report: object) -> Action:
        """Select a uniformly random valid action for the current state."""
        valid_actions = _enumerate_valid_actions(state, uncertainty_report)
        assert len(valid_actions) > 0, "BUG: no valid actions to select from"
        return self._rng.choice(valid_actions)
