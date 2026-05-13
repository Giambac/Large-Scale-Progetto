"""
conversation_loop.py — Il loop principale che fa girare la conversazione.

Questo file è il direttore d'orchestra del sistema. Ad ogni turno esegue
in sequenza questi passi:
    1. Calcola l'incertezza del clustering corrente
    2. Sceglie l'azione da fare (mostrare, chiedere, fermarsi)
    3. Formatta il messaggio per l'oracle e calcola il carico cognitivo
    4. Chiede la risposta all'oracle
    5. Trasforma la risposta in oggetti feedback strutturati
    6. Applica il feedback e produce il nuovo stato
    7. Controlla se c'è una contraddizione con feedback precedenti (Fase 3)
    8. Scrive il nuovo stato sul file di log
    9. Chiama il callback post-turno (per UMAP e sessioni persistenti)
    10. Emette l'aggiornamento al browser via SocketIO
    11. Controlla le condizioni di stop

È l'unico file che fa I/O: scrive il log, emette eventi. Le funzioni f_* sono pure e non scrivono nulla.

Note pratiche:
    - Gira in un thread separato quando avviato dal server Flask.
    - Se socketio=None salta tutti gli emit — utile nei test senza server.
    - Se llm_client=None salta il parsing — i delta sono sempre lista vuota.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional, Callable

from src.agent_functions import f_output, f_next_best_step, f_next_state
from src.feedback_parser import parse_feedback
from src.hierarchy import HierarchyStore
from src.serialization import append_to_audit_log
from src.state import ClusteringState
from src.stopping import StoppingCriteria, check_stopping
from src.strategy import RandomStrategy
from src.uncertainty import f_uncertainty

if TYPE_CHECKING:
    from src.embedding_store import EmbeddingStore
    from src.cluster_naming import ClusterNamer
    from src.oracle_protocol import OracleProtocol
    from src.strategy import StrategyProtocol

"""
def _format_message( )
    Trasforma un'Action in un messaggio leggibile da mandare all'oracle.

    Ogni tipo di azione produce un messaggio diverso:
        - show_full : elenca tutti i cluster con il loro nome e numero di recensioni
        - show_subset : dice al turno corrente che si mostrerà un sottoinsieme
        - ask_question : chiede se ci sono cluster da dividere, unire o spostare
        - stop : chiede conferma che il clustering va bene
"""
def _format_message(action: object, state: ClusteringState) -> str:
    from src.strategy import Action
    assert isinstance(action, Action), f"Expected Action, got {type(action)}"
    if action.action_type == "show_full":
        cluster_summary = "; ".join(
            f"Cluster {c.id} '{c.name}' ({len(c.item_ids)} items)"
            for c in state.clusters
        )
        return f"Current clustering: {cluster_summary}"
    elif action.action_type == "show_subset":
        return f"Showing a subset of clusters at turn {state.turn_index}."
    elif action.action_type == "ask_question":
        return "Do any clusters need to be split, merged, or items moved?"
    elif action.action_type == "stop":
        return "I believe the clustering is satisfactory. Are you happy with it?"
    else:
        return f"Action: {action.action_type}"

"""
def _write_event( )
    Scrive un evento JSON nel file sidecar events.jsonl.

    Usato per due tipi di eventi della Fase 3:
        - oracle_init: i parametri dell'oracle all'inizio della sessione
        - drift_event: quando viene rilevata una contraddizione

    IMPORTANTE: questo file è separato da audit_log.jsonl. L'audit log contiene solo ClusteringState — se ci finisse un record diverso, load_audit_log() crasherebbe.
"""
def _write_event(record: dict, events_path: str) -> None:
    parent = os.path.dirname(events_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

"""
def run_conversation( )
    Il loop principale della conversazione. Gira finché una condizione di stop non scatta, poi restituisce lo stato finale.

    Parametri:
        initial_state               - stato iniziale prodotto da build_initial_clustering_state.
        oracle                      - l'oracle (MockOracle nella Fase 2, OracleAgent nella Fase 3).
        store                       - gli embeddings, usati da f_next_state per gli split.
        namer                       - usato per rinominare i cluster dopo split/merge/move.
        strategy                    - la strategia di selezione dell'azione. None = RandomStrategy.
        log_path                    - percorso del file audit_log.jsonl.
        criteria                    - criteri di stop. None = default (15 turni).
        socketio                    - istanza SocketIO per aggiornare il browser. None = nessun emit.
        id_to_text                  - dizionario item_id -> testo della recensione.
        llm_client                  - client Anthropic per il parsing del feedback. None = no parsing.
        post_turn_callback          - funzione chiamata dopo ogni turno con (new_state, deltas). Usata per ricalcolare UMAP e salvare state.json nelle sessioni.
        events_path                 - percorso del sidecar events.jsonl per oracle_init e drift_event. None = stessa cartella di log_path.
"""
def run_conversation(
    initial_state: ClusteringState,
    oracle: "OracleProtocol",
    store: "EmbeddingStore",
    namer: "ClusterNamer",
    strategy: "StrategyProtocol | None",
    log_path: str,
    criteria: StoppingCriteria | None = None,
    socketio: object | None = None,
    id_to_text: dict[int, str] | None = None,
    llm_client: object | None = None,
    post_turn_callback: Optional[Callable] = None,  # (new_state, deltas) -> None
    events_path: str | None = None,  # sidecar file for oracle_init + drift_event records (Phase 3)
) -> ClusteringState:

    # f_cognitive_load viene importato qui, una volta sola fuori dal loop
    # Non va importato dentro il while — sarebbe reimportato ad ogni turno
    from src.cognitive_load import f_cognitive_load  # ORC-03: computed before oracle.reply()

    if criteria is None:
        criteria = StoppingCriteria()
    if strategy is None:
        strategy = RandomStrategy(seed=0)
    if id_to_text is None:
        id_to_text = {}

    # Percorso del sidecar events.jsonl — cartella diversa da audit_log.jsonl
    if events_path is None:
        _events_dir = os.path.dirname(log_path) or "."
        events_path = os.path.join(_events_dir, "events.jsonl")

    # Inizializza la gerarchia con tutti i cluster iniziali
    hierarchy = HierarchyStore()
    for cluster in initial_state.clusters:
        hierarchy.register(cluster.id)

    # Lista che accumula le istruzioni GlobalFeedback per tutta la sessione
    # Vive qui fuori dallo stato perché lo schema di ClusteringState è congelato
    global_instructions: list[str] = []

    # Phase 3: scrive oracle_init su events.jsonl se l'oracle è un OracleAgent
    from src.oracle_agent import OracleAgent as _OracleAgent  # local import — avoid circular
    if isinstance(oracle, _OracleAgent):
        _write_event({
            "event": "oracle_init",
            "turn": 0,
            "timestamp": initial_state.timestamp,
            "preferred_k": oracle.spec.preferred_k,
            "semantic_axes": oracle.spec.semantic_axes,
            "consistency_rate": oracle.noise_params.consistency_rate,
            "drift_probability": oracle.noise_params.drift_probability,
            "sycophancy_resistance": oracle.noise_params.sycophancy_resistance,
        }, events_path)

    state = initial_state
    recent_magnitudes: list[float] = []

    while True:
        # Passo 1: calcola cosa è incerto nel clustering corrente
        uncertainty_report = f_uncertainty(state)

        # Passo 2: scegli l'azione da fare questo turno
        action = f_next_best_step(state, strategy, uncertainty_report)

        # Passo 3: formatta il messaggio e calcola il carico cognitivo
        # Il carico viene calcolato qui e passato all'oracle — non ricalcolato dentro
        message = _format_message(action, state)
        cognitive_load = f_cognitive_load(state, message)

        # Passo 4: chiedi la risposta all'oracle.
        # Se è un OracleAgent passa anche global_instructions e cognitive_load
        if isinstance(oracle, _OracleAgent):
            reply = oracle.reply(
                state, message,
                global_instructions=global_instructions,
                cognitive_load=cognitive_load,
            )
        else:
            reply = oracle.reply(state, message)

        # Passo 5: trasforma la risposta dell'oracle in oggetti feedback strutturati.
        # Se llm_client è None oppure la risposta è vuota, delta = []
        if llm_client is not None and reply.raw_text.strip():
            deltas = parse_feedback(reply.raw_text, state, llm_client)
        else:
            deltas = []

        # Passo 6: applica i feedback e produce il nuovo stato.
        # global_instructions viene aggiornata in-place da f_next_state quando trova GlobalFeedback o InstructionalFeedback
        new_state = f_next_state(state, deltas, store, namer, hierarchy, id_to_text, global_instructions)

        # Passo 7 (Fase 3): aggiorna la finestra scorrevole dei delta e controlla se il nuovo feedback contraddice qualcosa detto nei turni precedenti.
        # Usa new_state.turn_index (dopo f_next_state), non state.turn_index — altrimenti il turno registrato nella finestra sarebbe sfasato di uno
        if isinstance(oracle, _OracleAgent) and deltas:
            contradiction_detected, contradicted_turn = oracle.update_delta_window(
                deltas, new_state.turn_index
            )
            reply.contradiction_detected = contradiction_detected
            reply.contradicted_turn = contradicted_turn

        # Passo 8: scrive il nuovo stato su audit_log.jsonl
        append_to_audit_log(new_state, log_path)

        # Passo 8b (Fase 3): se è stata rilevata una contraddizione, scrive drift_event su events.jsonl (file separato dall'audit log)
        if reply.contradiction_detected:
            _write_event({
                "event": "drift_event",
                "turn": new_state.turn_index,
                "contradicted_turn": reply.contradicted_turn,
                "timestamp": new_state.timestamp,
            }, events_path)

        # Passo 9: chiama il callback post-turno se definito.
        # Usato per ricalcolare la proiezione UMAP e salvare state.json nelle sessioni
        if post_turn_callback is not None:
            post_turn_callback(new_state, deltas)

        # Passo 10: aggiorna il browser via SocketIO.
        # Se socketio è None (test senza server) salta tutto
        if socketio is not None:
            import dataclasses
            socketio.emit("state_update", {
                "turn_index": new_state.turn_index,
                "clusters": [dataclasses.asdict(c) for c in new_state.clusters],
                "soft_probs": {
                    str(item_id): {
                        str(c.id): prob
                        for c, prob in zip(new_state.clusters, probs)
                    }
                    for item_id, probs in new_state.soft_probs.items()
                },
                "cognitive_load": reply.turn_cognitive_load,
            })

        # Registra quanti feedback ci sono stati questo turno. Usato dalla condizione dei rendimenti decrescenti (Phase 4)
        recent_magnitudes.append(float(len(deltas)))

        # Passo 11: controlla le condizioni di stop
        stop = check_stopping(
            turn_index=new_state.turn_index,
            oracle_satisfied=reply.satisfied,
            recent_magnitudes=recent_magnitudes,
            criteria=criteria,
        )

        # Avanza allo stato successivo o esce dal loop
        state = new_state
        if stop is not None:
            if socketio is not None:
                socketio.emit("session_stopped", {"reason": stop.value})
            break

    return state
