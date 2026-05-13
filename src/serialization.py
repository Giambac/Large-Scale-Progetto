"""
serialization.py — Salva e rilegge i ClusteringState su file JSONL.

Questo file risolve un problema pratico: come si salva un ClusteringState su file e lo si rilegge identico al turno successivo o alla sessione successiva.

Due problemi tecnici da gestire:
    1. Numpy usa float32 per i numeri — JSON non lo conosce e crasha. 
    Il _StateEncoder converte automaticamente tutti i tipi numpy in tipi Python standard prima di scrivere.
    2. JSON converte sempre le chiavi dei dizionari in stringhe. Quindi {0: "cluster_0"} diventa {"0": "cluster_0"} sul file. 
    Quando si rilegge, state.assignments[0] darebbe KeyError perché la chiave è la stringa "0". 
    deserialize_state risolve questo riconvertendo ogni chiave in int con int(k).

Il file di log (audit_log.jsonl) è append-only: una riga per turno, una riga = un ClusteringState completo. Non si sovrascrive mai.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import numpy as np

from src.state import Cluster, ClusteringState

"""
class _StateEncoder(json.JSONEncoder):
    Encoder JSON personalizzato per gestire i tipi Python/numpy non standard.

    Gestisce tre casi che json.dumps non sa gestire di default:
        - dataclass : convertito in dizionario con dataclasses.asdict()
        - np.integer : convertito in int Python normale
        - np.floating : convertito in float Python normale
        - np.ndarray : convertito in lista Python con .tolist()
"""
class _StateEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

"""
def serialize_state(state: ClusteringState) -> str:
    Converte un ClusteringState in una singola riga JSON senza newline.

    La riga può essere scritta direttamente nel file JSONL. Non contiene spazi extra o indentazione — una riga, un oggetto.

    Crasha se lo stato contiene tipi non serializzabili non gestiti dall'encoder.
"""
def serialize_state(state: ClusteringState) -> str:
    assert isinstance(state, ClusteringState), (
        f"serialize_state expects ClusteringState, got {type(state)}"
    )
    line = json.dumps(dataclasses.asdict(state), cls=_StateEncoder)
    assert "\n" not in line, "BUG: serialized state contains newline (would break JSONL)"
    return line

"""
def deserialize_state(line: str) -> ClusteringState:
    Ricostruisce un ClusteringState da una riga JSONL.

    Punto critico: JSON converte sempre le chiavi dei dizionari in stringhe.
    Questa funzione riconverte le chiavi di assignments e soft_probs da stringa a int con int(k). 
    Senza questa conversione, state.assignments[0] darebbe KeyError perché la chiave è diventata "0".

    Crasha se la riga non è JSON valido o se mancano campi obbligatori.
"""
def deserialize_state(line: str) -> ClusteringState:
    d = json.loads(line)

    assert "turn_index" in d, f"Missing 'turn_index' in deserialized state: {list(d.keys())}"
    assert "timestamp" in d, f"Missing 'timestamp' in deserialized state"
    assert "clusters" in d, f"Missing 'clusters' in deserialized state"
    assert "assignments" in d, f"Missing 'assignments' in deserialized state"
    assert "soft_probs" in d, f"Missing 'soft_probs' in deserialized state"

    clusters = [
        Cluster(
            id=int(c["id"]),
            name=c["name"],
            description=c["description"],
            item_ids=[int(i) for i in c["item_ids"]],
        )
        for c in d["clusters"]
    ]

    # Cast dict keys from str back to int — JSON key type loss invariant
    assignments: dict[int, int] = {int(k): int(v) for k, v in d["assignments"].items()}
    soft_probs: dict[int, list[float]] = {int(k): [float(p) for p in v] for k, v in d["soft_probs"].items()}

    return ClusteringState(
        turn_index=d["turn_index"],
        timestamp=d["timestamp"],
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )

"""
def append_to_audit_log(state: ClusteringState, log_path: str) -> None:
    Aggiunge un ClusteringState come nuova riga al file di log.

    Il file viene creato se non esiste (modalità append). Ogni chiamata aggiunge esattamente una riga. 
    Il file non viene mai sovrascritto — cresce di una riga per turno per tutta la durata della sessione.
"""
def append_to_audit_log(state: ClusteringState, log_path: str) -> None:
    assert isinstance(log_path, str) and log_path, "log_path must be a non-empty string"
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = serialize_state(state)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

"""
    Rilegge tutti i turni dal file di log.

    Ogni riga non vuota viene deserializzata in un ClusteringState.
    Crasha se il file non esiste o se è completamente vuoto — un log vuoto indica che qualcosa è andato storto durante la sessione.
"""
def load_audit_log(log_path: str) -> list[ClusteringState]:
    assert os.path.exists(log_path), f"AuditLog not found: {log_path}"
    states = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(deserialize_state(line))
    assert len(states) > 0, f"AuditLog at {log_path} is empty — no turns recorded"
    return states
