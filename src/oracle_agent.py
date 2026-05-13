"""
oracle_agent.py — L'oracle LLM reale con preferenze configurabili (Phase 3).

Questo file implementa l'oracle che sostituisce il MockOracle nella Phase 3. Invece di seguire uno script fisso, questo oracle chiama Claude con un sistema
di prompt costruito in cinque sezioni, che codifica le preferenze dell'oracle, il suo comportamento "umano" (rumore, deriva, sycofancy), e lo stato corrente.

Tre strutture dati:
    - OracleSpec : le preferenze fisse dell'oracle (quanti cluster vuole, su quali dimensioni raggruppa, che persona simula)
    - NoiseParams : i parametri che rendono l'oracle "imperfetto" come un umano (quanto spesso è d'accordo, quanto spesso cambia idea, quanto mantiene la sua posizione)
    - OracleAgent : la classe principale che assembla il prompt e chiama l'LLM

Funzione separata a livello di modulo:
    - _contradicts : controlla se due feedback si contraddicono strutturalmente
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.oracle_protocol import OracleReply

if TYPE_CHECKING:
    from src.state import ClusteringState

"""
class OracleSpec:
    Le preferenze fisse dell'oracle per tutta la durata della sessione.

    preferred_k             — quanti cluster vuole l'oracle.
    semantic_axes           — su quali dimensioni raggruppa i dati, es. ["topic", "sentiment"].
    persona_description     — descrizione della persona che l'oracle simula, iniettata nel system prompt.

    Una volta costruito l'OracleAgent, la spec non cambia. Per testare un oracle con preferenze diverse si crea una nuova istanza.
"""
@dataclass
class OracleSpec:
    preferred_k: int
    semantic_axes: list[str]
    persona_description: str

"""
    I parametri che controllano quanto l'oracle si comporta come un umano imperfetto.

    Tutti e tre sono numeri tra 0.0 e 1.0 e diventano istruzioni in linguaggio naturale nel system prompt — non c'è nessuna manipolazione tecnica della
    temperatura o post-processing della risposta.

    consistency_rate            — quanto spesso l'oracle accetta il clustering proposto. 0.8 = accetta l'80% delle volte.
    drift_probability           — con che probabilità introduce una nuova preferenza ad ogni turno, anche se contraddice qualcosa detto prima.
    sycophancy_resistance       — quanto l'oracle mantiene la sua posizione quando il sistema non è d'accordo. 0.9 = cede solo il 10% delle volte.
"""
@dataclass
class NoiseParams:
    consistency_rate: float       # 0.0-1.0
    drift_probability: float      # 0.0-1.0
    sycophancy_resistance: float  # 0.0-1.0

"""
def _contradicts( )
    Controlla se due feedback si contraddicono strutturalmente.

    Tre regole:
        - MergeFeedback(A, B) contraddice un precedente SplitFeedback sullo stesso cluster A o B — hai prima diviso, ora vuoi unire.
        - SplitFeedback(X) contraddice un precedente MergeFeedback che aveva X come input — hai prima unito, ora vuoi dividere.
        - MoveItemFeedback(item, target=B) contraddice un precedente MoveItemFeedback(item, target=C) sullo stesso item con destinazione diversa.

    GlobalFeedback e InstructionalFeedback vengono ignorati — sono troppo semantici per essere confrontati strutturalmente.
"""
def _contradicts(new_delta, prior_delta) -> bool:
    from src.feedback import MergeFeedback, SplitFeedback, MoveItemFeedback

    # Merge contraddice uno split precedente sugli stessi cluster
    if isinstance(new_delta, MergeFeedback) and isinstance(prior_delta, SplitFeedback):
        return prior_delta.cluster_id in (new_delta.cluster_a_id, new_delta.cluster_b_id)

    # Split contraddice un merge precedente che aveva usato gli stessi cluster come input
    if isinstance(new_delta, SplitFeedback) and isinstance(prior_delta, MergeFeedback):
        return new_delta.cluster_id in (prior_delta.cluster_a_id, prior_delta.cluster_b_id)

    # Move sullo stesso item ma con destinazione diversa
    if isinstance(new_delta, MoveItemFeedback) and isinstance(prior_delta, MoveItemFeedback):
        return (new_delta.item_id == prior_delta.item_id and
                new_delta.target_cluster_id != prior_delta.target_cluster_id)

    return False

"""
class OracleAgent:
    L'oracle LLM reale. Soddisfa OracleProtocol tramite duck typing — non eredita da nessuna classe, ma ha un metodo reply() con la stessa firma.

    Costruzione:
        - Valida i tre parametri di rumore (devono essere tra 0 e 1).
        - Crea la finestra scorrevole degli ultimi 10 feedback per il drift detection.
        - Se viene passato events_path, scrive subito oracle_init su events.jsonl.

    Metodi principali:
        - reply() : assembla il prompt e chiama l'LLM.
        - update_delta_window() : controlla le contraddizioni e aggiorna la finestra.
"""
class OracleAgent:
    def __init__(
        self,
        spec: OracleSpec,
        noise_params: NoiseParams,
        client: object,
        model: str = "claude-haiku-4-5",
        window_size: int = 10,
        events_path: Path | None = None,
    ) -> None:
        # Valida i parametri di rumore — se fuori range crasha subito
        assert 0.0 <= noise_params.consistency_rate <= 1.0
        assert 0.0 <= noise_params.drift_probability <= 1.0
        assert 0.0 <= noise_params.sycophancy_resistance <= 1.0
        self._spec = spec
        self._noise = noise_params
        self._client = client
        self._model = model

        # Finestra scorrevole degli ultimi N feedback strutturati.
        # maxlen=10 significa che i feedback più vecchi vengono scartati automaticamente
        self._delta_window: deque = deque(maxlen=window_size)

        # Scrive oracle_init su events.jsonl se viene passato il percorso.
        # Permette di testare ORC-02 anche senza passare dal loop conversazionale
        if events_path is not None:
            _events_path_str = str(events_path)
            _parent = os.path.dirname(_events_path_str)
            if _parent:
                os.makedirs(_parent, exist_ok=True)
            _record = {
                "event": "oracle_init",
                "turn": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "preferred_k": self._spec.preferred_k,
                "semantic_axes": self._spec.semantic_axes,
                "consistency_rate": self._noise.consistency_rate,
                "drift_probability": self._noise.drift_probability,
                "sycophancy_resistance": self._noise.sycophancy_resistance,
            }
            with open(_events_path_str, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_record) + "\n")

    @property
    def spec(self) -> OracleSpec:
        """Return the preference specification."""
        return self._spec

    @property
    def noise_params(self) -> NoiseParams:
        """Return the noise parameters."""
        return self._noise

    """
    def _build_system_prompt( )
        Assembla il system prompt in cinque sezioni.

        Sezione 1: persona e preferenze — chi è l'oracle e cosa vuole.
        Sezione 2: regole comportamentali — quanto spesso è d'accordo, quanto spesso cambia idea, quanto mantiene la posizione.
        Sezione 3: istruzioni accumulate dai turni precedenti (solo se ci sono).
        Sezione 4: riassunto dello stato corrente — quanti cluster, quante recensioni.
        Sezione 5: istruzione OVERLOAD — aggiunta solo se il carico cognitivo supera la soglia, per far rispondere con un feedback più semplice.
    """
    def _build_system_prompt(
        self,
        state: "ClusteringState",
        cognitive_load: float,
        global_instructions: list[str],
    ) -> str:
        from src.cognitive_load import COG_LOAD_THRESHOLD

        parts = []

        # Sezione 1: persona e preferenze
        cr = self._noise.consistency_rate
        parts.append(
            f"You are a human data analyst with the following preferences:\n"
            f"- Target number of clusters: {self._spec.preferred_k}\n"
            f"- You group data by: {', '.join(self._spec.semantic_axes)}\n"
            f"- Persona: {self._spec.persona_description}\n"
            f"When you are fully satisfied with the current clustering, end your reply with the exact token: [SATISFIED]"
        )

        # Sezione 2: regole comportamentali dai NoiseParams
        dp = self._noise.drift_probability
        sr = self._noise.sycophancy_resistance
        parts.append(
            f"Behavioral rules:\n"
            f"- You agree with the proposed clustering {int(cr * 100)}% of the time. "
            f"The other {int((1 - cr) * 100)}% you request minor adjustments.\n"
            f"- With probability {dp:.2f} you introduce a new preference per turn "
            f"that may contradict a prior one.\n"
            f"- You maintain your stated position even when the system pushes back, "
            f"at rate {sr:.2f}."
        )

        # Sezione 3: istruzioni accumulate (solo se la lista non è vuota)
        if global_instructions:
            parts.append(
                "Standing instructions from prior turns:\n" +
                "\n".join(f"- {i}" for i in global_instructions)
            )

        # Sezione 4: riassunto dello stato corrente
        cluster_summary = "; ".join(
            f"Cluster {c.id} '{c.name}' ({len(c.item_ids)} items)"
            for c in state.clusters
        )
        parts.append(f"Current clustering (turn {state.turn_index}): {cluster_summary}")

        # Sezione 5: istruzione OVERLOAD — aggiunta per ultima, solo se necessario
        if cognitive_load > COG_LOAD_THRESHOLD:
            parts.append("OVERLOAD: Focus on one thing only.")

        return "\n\n".join(parts)

    """
    def _check_contradiction( )
        Confronta un singolo delta con tutti i feedback nella finestra scorrevole.
        Non modifica la finestra — usa update_delta_window() per aggiungere delta.
        Restituisce (True, turno_precedente) alla prima contraddizione trovata.
    """
    def _check_contradiction(
        self, delta, current_turn: int
    ) -> tuple[bool, int | None]:
        for prior_turn, prior_delta in self._delta_window:
            if _contradicts(delta, prior_delta):
                return True, prior_turn
        return False, None

    """
    def update_delta_window( )
        Controlla le contraddizioni e poi aggiunge i nuovi delta alla finestra.

        L'ordine è importante: controlla PRIMA di aggiungere, così i feedback dello stesso turno non si contraddicono tra loro.

        Viene chiamato dal loop dopo che f_next_state ha applicato i feedback.
        Usa new_state.turn_index (dopo l'aggiornamento), non state.turn_index.

        Restituisce (True, turno_precedente) alla prima contraddizione trovata, oppure (False, None) se tutto è coerente.
    """
    def update_delta_window(
        self, deltas: list, turn_index: int
    ) -> tuple[bool, int | None]:
        from src.feedback import GlobalFeedback, InstructionalFeedback

        first_contradiction: bool = False
        first_contradicted_turn: int | None = None

        # Prima controlla le contraddizioni — senza ancora aggiungere alla finestra
        for delta in deltas:
            if isinstance(delta, (GlobalFeedback, InstructionalFeedback)):
                continue  # questi tipi non vengono confrontati strutturalmente

            detected, prior_turn = self._check_contradiction(delta, turn_index)
            if detected and not first_contradiction:
                first_contradiction = True
                first_contradicted_turn = prior_turn

        # Poi aggiunge i delta strutturali alla finestra
        for delta in deltas:
            if not isinstance(delta, (GlobalFeedback, InstructionalFeedback)):
                self._delta_window.append((turn_index, delta))

        return first_contradiction, first_contradicted_turn

    """
        Genera una risposta chiamando l'LLM con il system prompt assemblato.

        Se cognitive_load non viene passato, lo calcola internamente — ma il loop lo passa sempre esplicitamente per evitare di calcolarlo due volte.

        Gestione del provider: l'SDK Anthropic accetta il system prompt come kwarg separato (system=). Gli altri provider lo vogliono concatenato al messaggio.
        Il codice rileva il tipo di client e adatta la chiamata.

        Il token [SATISFIED] nella risposta imposta satisfied=True nell'OracleReply — è il segnale di stop principale del loop.

        contradiction_detected e contradicted_turn vengono impostati dal loop dopo la chiamata a update_delta_window(), non qui — perché i delta 
        non sono ancora disponibili quando reply() viene chiamato.
    """
    def reply(
        self,
        state: "ClusteringState",
        message: str,
        global_instructions: list[str] | None = None,
        cognitive_load: float | None = None,
    ) -> OracleReply:
        from src.cognitive_load import f_cognitive_load, COG_LOAD_THRESHOLD

        if global_instructions is None:
            global_instructions = []

        if cognitive_load is None:
            cognitive_load = f_cognitive_load(state, message)
        system_prompt = self._build_system_prompt(state, cognitive_load, global_instructions)

        # Rileva se il client è Anthropic o un altro provider.
        try:
            import anthropic as _anthropic_mod
            _is_anthropic = isinstance(self._client, _anthropic_mod.Anthropic)
        except ImportError:
            _is_anthropic = False

        if _is_anthropic:
            # Anthropic accetta system= come kwarg separato.
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message}],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"OracleAgent LLM call failed at turn {state.turn_index}: {exc}"
                ) from exc
        else:
            # Altri provider: system prompt concatenato al messaggio.
            full_message = system_prompt + "\n\n" + message
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": full_message}],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"OracleAgent LLM call failed at turn {state.turn_index}: {exc}"
                ) from exc

        raw_text = response.content[0].text
        # Il token [SATISFIED] nella risposta = l'oracle è soddisfatto → stop.
        satisfied = "[SATISFIED]" in raw_text

        return OracleReply(
            raw_text=raw_text,
            satisfied=satisfied,
            turn_cognitive_load=cognitive_load,
            contradiction_detected=False,
            contradicted_turn=None,
        )
