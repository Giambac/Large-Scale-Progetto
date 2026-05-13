"""
feedback_parser.py — Traduce il testo dell'oracle in oggetti feedback strutturati.

Questo file risolve un problema pratico: l'oracle scrive testo libero in linguaggio naturale, ma il sistema ha bisogno di oggetti Python precisi per sapere cosa fare.

Il flusso è semplice:
    1. Arriva il testo dell'oracle
    2. Viene mandato a Claude con un prompt che dice "trova i feedback in questo testo e restituiscili come array JSON"
    3. Claude risponde con un array JSON
    4. Il parser costruisce i dataclass corrispondenti
    5. Ogni cluster_id nella risposta viene verificato — se non esiste nello stato corrente, il programma crasha immediatamente

Tre cose fanno crashare il parser subito:
    - Un tipo di feedback sconosciuto nella risposta dell'LLM
    - Un cluster_id che non esiste nello stato corrente (allucinazione dell'LLM)
    - Un merge di un cluster con se stesso
"""
from __future__ import annotations

import json

from src.state import ClusteringState
from src.feedback import (
    FeedbackDelta,
    SplitFeedback,
    MergeFeedback,
    MoveItemFeedback,
    GlobalFeedback,
    InstructionalFeedback,
)

# I cinque tipi validi che l'LLM può restituire
VALID_FEEDBACK_TYPES = frozenset({"global", "split", "merge", "move_item", "instructional"})

# Il prompt mandato a Claude per parsare il feedback
# {cluster_summary} e {raw_text} vengono sostituiti prima della chiamata
_PARSE_FEEDBACK_PROMPT = """
You are parsing oracle feedback in a clustering conversation.
Current clusters: {cluster_summary}
Oracle said: "{raw_text}"

Extract ALL feedback intents as a JSON array. Each item has exactly one of these schemas:
- {{"type": "global", "instruction_text": "..."}}
- {{"type": "split", "cluster_id": <int>, "seed_item_ids": [<int>, ...]}}
- {{"type": "merge", "cluster_a_id": <int>, "cluster_b_id": <int>}}
- {{"type": "move_item", "item_id": <int>, "target_cluster_id": <int>}}
- {{"type": "instructional", "instruction_text": "..."}}

Rules:
- Return ONLY a JSON array. No markdown. No explanation.
- If no feedback: return []
- Compound messages produce multiple array items.
- seed_item_ids may be [] if oracle names no specific items.
- cluster_id values must exist in the current cluster list.
"""

"""
def _build_cluster_summary( )
    Costruisce una stringa compatta con tutti i cluster attivi.
    Viene iniettata nel prompt per aiutare l'LLM a capire quali cluster esistono.
    Esempio: "0: Alpha, 1: Beta, 2: Gamma"
"""
def _build_cluster_summary(state: ClusteringState) -> str:
    return ", ".join(f"{c.id}: {c.name}" for c in state.clusters)

"""
def _build_delta( )
    Trasforma un singolo elemento JSON in un oggetto FeedbackDelta.

    Tre guardie di sicurezza:
        - Se il tipo non è uno dei cinque validi → crash immediato
        - Se un cluster_id non esiste nello stato corrente → crash immediato (impedisce che un'allucinazione dell'LLM causi danni silenziosi)
        - Se si tenta di fare merge di un cluster con se stesso → crash immediato

    Se l'LLM omette un campo obbligatorio, la KeyError si propaga — non viene intercettata. Il sistema fallisce rumorosamente invece che in silenzio.
"""
def _build_delta(item: dict, valid_cluster_ids: set[int]) -> FeedbackDelta:
    assert "type" in item and item["type"] in VALID_FEEDBACK_TYPES, (
        f"parse_feedback: unknown feedback type in LLM response: {item}"
    )

    feedback_type = item["type"]

    if feedback_type == "global":
        return GlobalFeedback(instruction_text=item["instruction_text"])

    if feedback_type == "split":
        assert item["cluster_id"] in valid_cluster_ids, (
            f"parse_feedback: split cluster_id={item['cluster_id']} not in current clusters {valid_cluster_ids}"
        )
        return SplitFeedback(
            cluster_id=item["cluster_id"],
            seed_item_ids=list(item["seed_item_ids"]),
        )

    if feedback_type == "merge":
        assert item["cluster_a_id"] in valid_cluster_ids, (
            f"parse_feedback: merge cluster_a_id={item['cluster_a_id']} not in current clusters {valid_cluster_ids}"
        )
        assert item["cluster_b_id"] in valid_cluster_ids, (
            f"parse_feedback: merge cluster_b_id={item['cluster_b_id']} not in current clusters {valid_cluster_ids}"
        )
        assert item["cluster_a_id"] != item["cluster_b_id"], (
            f"parse_feedback: merge cluster_a_id == cluster_b_id == {item['cluster_a_id']}"
        )
        return MergeFeedback(
            cluster_a_id=item["cluster_a_id"],
            cluster_b_id=item["cluster_b_id"],
        )

    if feedback_type == "move_item":
        assert item["target_cluster_id"] in valid_cluster_ids, (
            f"parse_feedback: move_item target_cluster_id={item['target_cluster_id']} not in current clusters {valid_cluster_ids}"
        )
        return MoveItemFeedback(
            item_id=item["item_id"],
            target_cluster_id=item["target_cluster_id"],
        )

    # L'unico tipo rimasto valido è "instructional"
    return InstructionalFeedback(instruction_text=item["instruction_text"])

"""
    Converte il testo grezzo dell'oracle in una lista di oggetti FeedbackDelta.

    Flusso:
        1. Se raw_text è vuoto, restituisce [] senza chiamare l'LLM.
        2. Costruisce il riassunto dei cluster correnti e lo inietta nel prompt.
        3. Chiama Claude con il prompt.
        4. Pulisce la risposta rimuovendo eventuali backtick markdown.
        5. Fa il parse del JSON.
        6. Per ogni elemento dell'array chiama _build_delta.

    L'unico try/except di tutto il file è attorno a json.loads — se l'LLM restituisce JSON malformato il programma crasha con un errore chiaro.
    Tutto il resto fallisce rumorosamente senza essere intercettato.
"""
def parse_feedback(
    raw_text: str,
    state: ClusteringState,
    client: object,
) -> list[FeedbackDelta]:
    if not raw_text:
        return []

    cluster_summary = _build_cluster_summary(state)
    prompt = _PARSE_FEEDBACK_PROMPT.format(
        cluster_summary=cluster_summary,
        raw_text=raw_text,
    )

    # Chiama Claude Haiku per parsare il feedback
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    cleaned_text = response.content[0].text.strip()
    # Rimuove i backtick markdown se l'LLM li ha aggiunti (es. ```json ... ```)
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.split("```")[1]
        if cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:]
        cleaned_text = cleaned_text.strip()

    # Parsa il JSON — se l'LLM ha restituito testo non valido crasha qui
    raw_items = json.loads(cleaned_text)

    assert isinstance(raw_items, list), (
        f"parse_feedback: LLM returned non-list JSON: {type(raw_items)}"
    )

    valid_cluster_ids = {c.id for c in state.clusters}

    # Costruisce un FeedbackDelta per ogni elemento dell'array
    return [_build_delta(item, valid_cluster_ids) for item in raw_items]
