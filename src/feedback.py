"""
feedback.py — I tipi di feedback che l'oracle può dare al sistema.

Questo file definisce il linguaggio strutturato con cui l'oracle comunica cosa vuole cambiare nel clustering. 
Invece di testo libero, ogni richiesta dell'oracle diventa un oggetto Python preciso con campi ben definiti.

Cinque tipi di feedback, ognuno per un tipo diverso di richiesta:
  1. GlobalFeedback             — istruzione generale, es. "troppi cluster"
  2. SplitFeedback              — dividere un cluster in due
  3. MergeFeedback              — unire due cluster in uno
  4. MoveItemFeedback           — spostare una singola recensione in un altro cluster
  5. InstructionalFeedback      — suggerimento soft, es. "tratta X e Y come sinonimi"

Quando l'oracle scrive un messaggio con più richieste, il parser produce una lista di questi oggetti. Vengono poi applicati nell'ordine indicato sopra 
— prima i feedback globali, poi split e merge, poi spostamenti singoli, poi i suggerimenti soft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# Quando l'oracle sposta una recensione, il sistema imposta la sua probabilità per il cluster di destinazione a 0.95 — quasi certezza.
ORACLE_MOVE_CONFIDENCE: float = 0.95

# Guardia per un caso limite: se dopo uno spostamento tutte le probabilità rimanenti fossero zero, la distribuzione diventa uniforme invece di crashare.
UNIFORM_FALLBACK_THRESHOLD: float = 1e-9

"""
class SplitFeedback:
    L'oracle vuole dividere un cluster in due.

    cluster_id          — quale cluster dividere.
    seed_item_ids       — recensioni che l'oracle indica come rappresentative dei due nuovi gruppi. 
                          Può essere una lista vuota: in quel caso il sistema decide da solo come dividere usando KMeans.

    frozen=True : l'oggetto non può essere modificato dopo la creazione.
    Se si prova a cambiare un campo il programma crasha immediatamente.
    Questo evita che il feedback venga alterato accidentalmente mentre viene processato dal sistema.
"""
@dataclass(frozen=True)
class SplitFeedback:
    cluster_id: int
    seed_item_ids: list[int]

"""
class MergeFeedback:
    L'oracle vuole unire due cluster in uno solo.

    cluster_a_id e cluster_b_id vengono eliminati e ne viene creato uno nuovo.
    Le loro probabilità vengono sommate e rinormalizzate.
"""
@dataclass(frozen=True)
class MergeFeedback:
    cluster_a_id: int
    cluster_b_id: int

"""
class MoveItemFeedback:
    L'oracle vuole spostare una singola recensione in un altro cluster.

    item_id                 — quale recensione spostare.
    target_cluster_id       — in quale cluster spostarla.
"""
@dataclass(frozen=True)
class MoveItemFeedback:
    item_id: int
    target_cluster_id: int

"""
class GlobalFeedback:
    Istruzione generale sul clustering, senza riferimento a un cluster specifico.

    Esempi: "troppi cluster", "concentrati sui reclami di fatturazione".
    È il feedback con il peso più alto perché indica una preferenza strutturale ampia. Il testo viene salvato e usato nei prompt dei turni successivi.
"""
@dataclass(frozen=True)
class GlobalFeedback:
    instruction_text: str

"""
class InstructionalFeedback:
    Suggerimento soft che non cambia direttamente la struttura dei cluster.

    Esempi: "tratta 'errore' e 'fail' come sinonimi", "dai più peso alla qualità".
    È il feedback con il peso più basso. Come GlobalFeedback, il testo viene salvato e iniettato nei prompt successivi per influenzare il naming.
"""
@dataclass(frozen=True)
class InstructionalFeedback:
    instruction_text: str


# Alias che raggruppa tutti e cinque i tipi in uno solo.
# Usato come tipo nei parametri delle funzioni che accettano qualsiasi feedback.
# L'ordine riflette la priorità di applicazione in f_next_state: prima i globali, poi split/merge, poi spostamenti, poi istruzioni soft.
FeedbackDelta = Union[
    GlobalFeedback,
    SplitFeedback,
    MergeFeedback,
    MoveItemFeedback,
    InstructionalFeedback,
]
