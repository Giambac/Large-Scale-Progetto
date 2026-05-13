"""
state.py — Le strutture dati base del sistema.

Due "contenitori" fondamentali che vengono usati ovunque nel progetto:
  - Cluster: rappresenta un singolo gruppo di recensioni.
  - ClusteringState: rappresenta la fotografia completa del sistema in un dato momento della conversazione — quanti cluster ci sono, a quale cluster appartiene 
    ogni recensione, quanto il sistema è sicuro di ogni assegnazione.

Questi due oggetti vengono creati, modificati e salvati su file ad ogni turno della conversazione. Tutto il resto del codice li usa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

"""
class Cluster:
    Un singolo gruppo di recensioni.

    id              — numero identificativo del cluster. Una volta che un cluster viene eliminato (per uno split o merge), 
                      il suo ID non viene mai riassegnato ad un altro cluster.
    name            — nome breve generato dall'LLM, es. "Knitting Supplies".
    description     — frase descrittiva generata dall'LLM sul tema del cluster.
    item_ids        — lista degli ID delle recensioni che appartengono a questo cluster.
    """
@dataclass
class Cluster:
    id: int     
    name: str
    description: str
    item_ids: list[int]

"""
class ClusteringState:

    La fotografia completa del sistema in un dato turno della conversazione.

    È l'oggetto centrale del progetto: viene prodotto ad ogni turno e passato tra tutte le funzioni del sistema. Ogni turno parte da uno stato, 
    applica il feedback dell'oracle, e produce il nuovo stato.

    turn_index      — numero del turno corrente, parte da 0.
    timestamp       — data e ora in cui lo stato è stato creato.
    clusters        — lista di tutti i cluster attivi in questo turno.
    assignments     — dizionario che dice a quale cluster appartiene ogni recensione.
                    Esempio: {0: 2, 1: 0, 2: 2} significa che la recensione 0 è nel cluster 2, la recensione 1 nel cluster 0, ecc.
    soft_probs      — dizionario che dice quanto il sistema è sicuro di ogni assegnazione. Per ogni recensione c'è una lista di probabilità, una per cluster. 
                    Esempio: {0: [0.9, 0.1]} significa che la recensione 0 appartiene al cluster 0 con il 90% di certezza.
                    Se i valori fossero [0.5, 0.5] la recensione sarebbe un caso borderline — il sistema non sa dove metterla.

    Nota tecnica: JSON converte le chiavi dei dizionari in stringhe quando salva su file, quindi {0: 2} diventa {"0": 2}. Al momento della lettura bisogna
    riconvertire le chiavi in interi — questo viene gestito in serialization.py.
"""
@dataclass
class ClusteringState:
    turn_index: int
    timestamp: str                          # ISO 8601 string
    clusters: list[Cluster]
    assignments: dict[int, int]             # item_id -> cluster_id
    soft_probs: dict[int, list[float]]      # item_id -> probability vector [float × K]
