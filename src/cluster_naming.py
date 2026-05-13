"""
cluster_naming.py — Chiede all'LLM di dare un nome e una descrizione a ogni cluster.

Questo file risolve un problema semplice: dopo che HDBSCAN o KMeans hanno raggruppato le recensioni, i cluster hanno solo un numero identificativo.
Bisogna dare loro un nome leggibile e una descrizione.

Il sistema manda le prime 5 recensioni di ogni cluster all'LLM e chiede di rispondere con un JSON del tipo:
    {"name": "Knitting Supplies", "description": "Reviews about yarn and needles."}

Design: ClusterNamer è un'interfaccia. Qualsiasi oggetto che ha un metodo name_cluster() con la firma giusta va bene — non serve ereditare nulla.
Questo permette di usare Anthropic, Google o OpenAI intercambiabilmente senza cambiare nient'altro nel sistema.
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

"""
class ClusterNamer(Protocol):
    L'interfaccia che qualsiasi namer deve rispettare.

    Basta avere un metodo name_cluster(sample_texts, cluster_id) che restituisce un dizionario con "name" e "description".
    Non serve ereditare da questa classe.
"""
@runtime_checkable
class ClusterNamer(Protocol):
    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        """
        Given sample texts from a cluster, return a name and description.

        Args:
            sample_texts: up to 5 representative texts from the cluster.
            cluster_id: integer ID for this cluster (used in prompt context).

        Returns:
            dict with keys "name" (str, 2-5 words) and "description" (str, 1-2 sentences).

        Raises:
            AssertionError: if LLM response is missing "name" or "description" keys.
        """
        ...

"""
    Funzione base che chiama un client Anthropic per nominare un cluster.

    È la funzione condivisa usata da AnthropicClusterNamer. Gli altri namer Google, OpenAI) hanno la loro logica di chiamata perché i loro SDK hanno interfacce diverse.

    Cosa fa:
        1. Prende le prime max_samples recensioni (default 5).
        2. Costruisce un prompt che chiede all'LLM di rispondere SOLO con JSON.
        3. Chiama l'API.
        4. Rimuove eventuali backtick markdown dalla risposta.
        5. Parsa il JSON e verifica che contenga "name" e "description".

    Crasha subito se le chiavi mancano — una risposta malformata dell'LLM non deve propagarsi silenziosamente nel sistema.
"""
def name_cluster(
    client: object,
    sample_texts: list[str],
    cluster_id: int,
    max_samples: int = 5,
) -> dict[str, str]:
    samples = sample_texts[:max_samples]
    assert len(samples) > 0, f"Cannot name cluster {cluster_id}: no sample texts provided"

    prompt = (
        f"You are analyzing a cluster of customer reviews from an arts and crafts store. "
        f"Here are {len(samples)} representative reviews from cluster {cluster_id}:\n\n"
        + "\n---\n".join(samples)
        + "\n\nRespond ONLY with a JSON object with exactly two keys:\n"
        "  'name': a 2-5 word label for what unifies these reviews\n"
        "  'description': 1-2 sentences describing what these reviews have in common\n"
        "No other text. No markdown. Just the JSON object."
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    # Rimuove i backtick markdown se l'LLM li ha aggiunti
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    result = json.loads(raw_text)

    # Verifica che la risposta abbia le due chiavi richieste e che non siano vuote
    assert "name" in result and "description" in result, (
        f"LLM returned bad schema for cluster {cluster_id}: {result}"
    )
    assert isinstance(result["name"], str) and len(result["name"]) > 0, (
        f"LLM 'name' is empty or not a string for cluster {cluster_id}: {result}"
    )
    assert isinstance(result["description"], str) and len(result["description"]) > 0, (
        f"LLM 'description' is empty or not a string for cluster {cluster_id}: {result}"
    )

    return {"name": result["name"], "description": result["description"]}

"""
class AnthropicClusterNamer:
    Namer che usa il client Anthropic (Claude Haiku).
    Delega interamente alla funzione name_cluster() qui sopra.
"""
class AnthropicClusterNamer:
    def __init__(self, client: object) -> None:
        self._client = client

    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        return name_cluster(self._client, sample_texts, cluster_id)

"""
    Namer che usa Google AI Studio (Gemini).

    Stessa logica di AnthropicClusterNamer ma con l'SDK di Google, che ha un'interfaccia diversa (client.models.generate_content invece di client.messages.create).

    Richiede: pip install google-generativeai
    Chiave API: variabile d'ambiente GOOGLE_API_KEY.
"""
class GoogleClusterNamer:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        from google import genai  # type: ignore[import]
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        from google import genai  # type: ignore[import]
        samples = sample_texts[:5]
        assert len(samples) > 0, f"Cannot name cluster {cluster_id}: no sample texts provided"

        prompt = (
            f"You are analyzing a cluster of customer reviews from an arts and crafts store. "
            f"Here are {len(samples)} representative reviews from cluster {cluster_id}:\n\n"
            + "\n---\n".join(samples)
            + "\n\nRespond ONLY with a JSON object with exactly two keys:\n"
            "  'name': a 2-5 word label for what unifies these reviews\n"
            "  'description': 1-2 sentences describing what these reviews have in common\n"
            "No other text. No markdown. Just the JSON object."
        )

        response = self._client.models.generate_content(model=self._model, contents=prompt)
        raw_text = response.text.strip()
        # Strip markdown code fences if Gemini wraps the JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        result = json.loads(raw_text.strip())

        assert "name" in result and "description" in result, (
            f"LLM returned bad schema for cluster {cluster_id}: {result}"
        )
        assert isinstance(result["name"], str) and len(result["name"]) > 0, (
            f"LLM 'name' is empty or not a string for cluster {cluster_id}: {result}"
        )
        assert isinstance(result["description"], str) and len(result["description"]) > 0, (
            f"LLM 'description' is empty or not a string for cluster {cluster_id}: {result}"
        )
        return {"name": result["name"], "description": result["description"]}

"""
class OpenAIClusterNamer:
    Namer che usa OpenAI (GPT-4o-mini).

    Stessa logica degli altri namer ma con l'SDK OpenAI, che usa client.chat.completions.create e response.choices[0].message.content.

    Richiede: pip install openai
    Chiave API: variabile d'ambiente OPENAI_API_KEY.
"""
class OpenAIClusterNamer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # type: ignore[import]
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        samples = sample_texts[:5]
        assert len(samples) > 0, f"Cannot name cluster {cluster_id}: no sample texts provided"

        prompt = (
            f"You are analyzing a cluster of customer reviews from an arts and crafts store. "
            f"Here are {len(samples)} representative reviews from cluster {cluster_id}:\n\n"
            + "\n---\n".join(samples)
            + "\n\nRespond ONLY with a JSON object with exactly two keys:\n"
            "  'name': a 2-5 word label for what unifies these reviews\n"
            "  'description': 1-2 sentences describing what these reviews have in common\n"
            "No other text. No markdown. Just the JSON object."
        )

        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        result = json.loads(raw_text)

        assert "name" in result and "description" in result, (
            f"LLM returned bad schema for cluster {cluster_id}: {result}"
        )
        assert isinstance(result["name"], str) and len(result["name"]) > 0, (
            f"LLM 'name' is empty or not a string for cluster {cluster_id}: {result}"
        )
        assert isinstance(result["description"], str) and len(result["description"]) > 0, (
            f"LLM 'description' is empty or not a string for cluster {cluster_id}: {result}"
        )
        return {"name": result["name"], "description": result["description"]}

"""
def name_all_clusters( )
    Nomina tutti i cluster in una volta sola.

    Viene chiamata una volta sola da build_initial_clustering_state per dare nome e descrizione a tutti i cluster iniziali.

    Restituisce un dizionario cluster_id -> {"name": ..., "description": ...}.
"""
def name_all_clusters(
    cluster_items: dict[int, list[int]],
    id_to_text: dict[int, str],
    namer: ClusterNamer,
) -> dict[int, dict[str, str]]:
    results = {}
    for cluster_id, item_ids in sorted(cluster_items.items()):
        sample_texts = [id_to_text[i] for i in item_ids[:5]]
        results[cluster_id] = namer.name_cluster(sample_texts, cluster_id)
    return results
