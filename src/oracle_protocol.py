"""
oracle_protocol.py — Il contratto dell'oracle e l'oracle finto per i test.

Questo file definisce come il sistema parla con l'oracle — qualsiasi oracle, reale o simulato. 
Stabilisce la "forma" della risposta che il sistema si aspetta e fornisce un oracle finto (MockOracle) usato nei test e nel loop della Phase 2.

Nella Fase 3 il MockOracle viene sostituito dall'OracleAgent (un oracle LLM reale) senza dover cambiare nient'altro nel sistema — basta che il nuovo oggetto abbia
un metodo reply() con la stessa firma.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.state import ClusteringState

"""
class OracleReply:
    La risposta dell'oracle per un singolo turno.

    raw_text                    — testo in linguaggio naturale scritto dall'oracle.
    satisfied                   — True se l'oracle dice esplicitamente che va bene così. 
                                  È la condizione di stop principale: quando questo campo è True, il loop si ferma immediatamente.
    turn_cognitive_load         — quanto è stato "pesante" il turno per l'oracle.
    contradiction_detected      — True se il sistema ha rilevato che questo feedback contraddice qualcosa detto in un turno precedente.
    contradicted_turn           — il numero del turno che viene contraddetto. None se non c'è contraddizione.
"""
@dataclass
class OracleReply:
    raw_text: str
    satisfied: bool
    turn_cognitive_load: float = 0.0 
    contradiction_detected: bool = False
    contradicted_turn: int | None = None 

"""
class OracleProtocol(Protocol):
    L'interfaccia che qualsiasi oracle deve rispettare.

    Basta avere un metodo reply(state, message) che restituisce un OracleReply.
    Non serve ereditare da questa classe — Python controlla strutturalmente se l'oggetto ha il metodo giusto. 
    Questo permette di scrivere nuovi oracle senza toccare il codice esistente.
"""
@runtime_checkable
class OracleProtocol(Protocol):
    def reply(self, state: ClusteringState, message: str) -> OracleReply:
        ...

"""
class MockOracle:
    L'oracle finto usato nei test e nel loop della Phase 2.

    Funziona come un attore con un copione: riceve una lista di risposte in anticipo e le restituisce una per turno, nell'ordine dato. 
    Quando lo script finisce, restituisce una risposta neutra vuota invece di crashare — questo permette ai test del loop da 30 turni di girare anche con script più corti.

    Non eredita da OracleProtocol ma ha il metodo reply() con la stessa firma, quindi Python lo riconosce come oracle valido grazie al @runtime_checkable.
"""
class MockOracle:
    def __init__(self, script: list[OracleReply]) -> None:
        # Uno script vuoto non ha senso — l'oracle deve avere almeno una risposta.
        assert len(script) > 0, "MockOracle script must be non-empty"
        self._script = script
        self._turn = 0 # tiene traccia di quale risposta dello script è la prossima

    # Restituisce la prossima risposta dello script. Se lo script è esaurito, restituisce una risposta neutra vuota.
    def reply(self, state: ClusteringState, message: str) -> OracleReply:
        if self._turn < len(self._script):
            r = self._script[self._turn]
        else:
            # Script finito — risposta neutra per non bloccare il loop
            r = OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)
        self._turn += 1
        return r
