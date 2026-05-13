"""
stopping.py — Quando si ferma la conversazione.

Tre condizioni che possono mettere fine al loop conversazionale. Basta che una sola si verifichi per fermare tutto.

Le tre condizioni, in ordine di priorità:
  1. Oracle soddisfatto — l'oracle dice esplicitamente che va bene così.

  2. Budget turni — se si arriva al turno 15 senza che nessuno sia soddisfatto, il sistema si ferma comunque. 

  3. Rendimenti decrescenti — se il feedback dell'oracle diventa sempre più piccolo (piccoli aggiustamenti invece di grandi cambiamenti), 
                              il sistema capisce che si sta convergendo e si ferma. 
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

"""
class StopReason(Enum):

    Il motivo per cui la conversazione si è fermata. Tre valori possibili.

    Viene restituito da check_stopping() quando una condizione scatta.
    Se nessuna condizione scatta, check_stopping() restituisce None.
"""
class StopReason(Enum):
    ORACLE_SATISFIED = "oracle_satisfied"
    TURN_BUDGET = "turn_budget"
    DIMINISHING_RETURNS = "diminishing_returns"

"""
class StoppingCriteria:
   
    I parametri di configurazione delle tre condizioni di stop.

    frozen=True : una volta creato l'oggetto non si può modificare.

    turn_budget                     — numero massimo di turni (default 15). Quando urn_index raggiunge questo valore, il sistema si ferma comunque, indipendentemente da tutto.

    magnitude_threshold_epsilon     — soglia per la condizione dei rendimenti decrescenti. (Phase 4)

    magnitude_fallback_turns        — quanti turni consecutivi con feedback piccolo servono per attivare la condizione. (Phase 4)
"""
@dataclass(frozen=True)
class StoppingCriteria:
    turn_budget: int = 15
    magnitude_threshold_epsilon: float = float("nan")   # Phase 4 placeholder
    magnitude_fallback_turns: int = -1                  # Phase 4 placeholder

"""
class FeedbackMagnitudeWeights:
    I pesi per misurare quanto è "grande" un feedback.

    Non tutti i feedback hanno lo stesso peso: un feedback globale come "troppi cluster" è più significativo di "sposta questa recensione".
    Questi pesi riflettono quella differenza.

    Tutti i valori sono float("nan") per ora — la Phase 4 li sostituirà con valori reali. La formula finale sarà:
        magnitude = (
            global_feedback   * numero_di_feedback_globali
            + cluster_level   * numero_di_feedback_sui_cluster
            + point_level     * numero_di_feedback_sui_singoli_item
            + instructional   * numero_di_feedback_istruzionali
        )
"""
@dataclass
class FeedbackMagnitudeWeights:
    global_feedback: float = float("nan")   # Phase 4 — highest weight
    cluster_level: float = float("nan")      # Phase 4
    point_level: float = float("nan")        # Phase 4
    instructional: float = float("nan")      # Phase 4 — lowest weight

"""
def check_stopping( )
    Controlla le tre condizioni di stop e restituisce la prima che scatta.

    Viene chiamata alla fine di ogni turno del loop conversazionale.
    Se nessuna condizione scatta, restituisce None e il loop continua.

    Parametri:
        turn_index              — numero del turno corrente.
        oracle_satisfied        — True se l'oracle ha detto che è soddisfatto.
        recent_magnitudes       — lista delle magnitude dei feedback degli ultimi turni, usata dalla condizione dei rendimenti decrescenti.
        criteria                — oggetto con i parametri di configurazione.
"""
def check_stopping(
    turn_index: int,
    oracle_satisfied: bool,
    recent_magnitudes: list[float],
    criteria: StoppingCriteria,
) -> Optional[StopReason]:
    # Condizione 1: l'oracle è soddisfatto — ci si ferma subito. (D-08)
    if oracle_satisfied:
        return StopReason.ORACLE_SATISFIED

    # Condizione 2: raggiunto il numero massimo di turni — ci si ferma comunque. (D-09)
    if turn_index >= criteria.turn_budget:
        return StopReason.TURN_BUDGET

    # Condizione 3: rendimenti decrescenti — non ancora implementata. (D-10) (Phase 4)

    return None
