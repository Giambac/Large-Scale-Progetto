"""
cognitive_load.py — Misura quanto è "pesante" ogni turno per l'oracle.

Questo file calcola un numero tra 0 e 1 che rappresenta quanto è difficile per l'oracle elaborare le informazioni di un dato turno.

Se il punteggio supera la soglia 0.7, il loop inietta nel system prompt dell'oracle la frase "OVERLOAD: Focus on one thing only." — l'oracle risponde
con un feedback più semplice e focalizzato invece di fare più richieste insieme.

Il calcolo combina tre fattori con lo stesso peso (1/3 ciascuno):
    - quanti cluster ci sono rispetto al massimo possibile (20)
    - quante recensioni vengono mostrate rispetto al totale
    - quanto è lungo il messaggio rispetto alla lunghezza massima (500 caratteri)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.state import ClusteringState

# Costanti — non numeri magici nel codice
MAX_K: int = 20                         # numero massimo di cluster per la normalizzazione
MAX_MSG_LEN: int = 500                  # lunghezza massima del messaggio per la normalizzazione
TOP_K_ITEMS_PER_CLUSTER: int = 5        # quante recensioni vengono mostrate per cluster (deve corrispondere al valore usato in _format_message)
COG_LOAD_THRESHOLD: float = 0.7         # sopra questa soglia viene iniettata l'istruzione OVERLOAD

"""
def f_cognitive_load( )
    Calcola il carico cognitivo del turno corrente. Restituisce un numero in [0, 1].

    Formula (tre termini con peso uguale 1/3):
        - termine cluster : min(numero_cluster / 20, 1.0) × 1/3
        - termine recensioni : min((cluster × 5) / totale_recensioni, 1.0) × 1/3
         - termine messaggio : min(lunghezza_messaggio / 500, 1.0) × 1/3

    I min(..., 1.0) evitano che un caso estremo mandi il risultato sopra 1.

    Crasha subito se lo stato non ha cluster o non ha recensioni — questi casi non dovrebbero mai arrivare qui.
"""
def f_cognitive_load(state: "ClusteringState", message: str) -> float:
    assert len(state.clusters) > 0, "f_cognitive_load: no clusters in state"
    total_items = len(state.assignments)
    assert total_items > 0, "f_cognitive_load: no items in state"

    w = 1.0 / 3.0

    # Quanti cluster ci sono rispetto al massimo
    cluster_term = min(len(state.clusters) / MAX_K, 1.0) * w
    
    # Quante recensioni vengono mostrate rispetto al totale
    # _format_message mostra le prime 5 recensioni per cluster
    items_shown = len(state.clusters) * TOP_K_ITEMS_PER_CLUSTER
    items_term = min(items_shown / total_items, 1.0) * w

    # Quanto è lungo il messaggio
    msg_term = min(len(message) / MAX_MSG_LEN, 1.0) * w

    return cluster_term + items_term + msg_term
