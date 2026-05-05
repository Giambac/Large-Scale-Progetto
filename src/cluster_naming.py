"""
cluster_naming.py — LLM-based cluster name and description generation (FOUND-02).

Design: ClusterNamer is a Protocol (structural subtyping) so any object
that implements name_cluster(sample_texts, cluster_id) -> dict works.
This enables swapping Anthropic/OpenAI clients without changing clustering.py.

The concrete AnthropicClusterNamer uses claude-haiku-4-5 for cost efficiency.
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class ClusterNamer(Protocol):
    """
    Protocol for LLM-based cluster naming.
    Any object implementing name_cluster() satisfies this interface.
    """
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


def name_cluster(
    client: object,
    sample_texts: list[str],
    cluster_id: int,
    max_samples: int = 5,
) -> dict[str, str]:
    """
    Call an Anthropic-compatible client to name a cluster.

    Args:
        client: Anthropic client instance (anthropic.Anthropic).
        sample_texts: review texts from this cluster.
        cluster_id: integer cluster ID (for prompt context only).
        max_samples: number of sample texts to include in prompt (default 5).

    Returns:
        {"name": str, "description": str}

    Raises:
        AssertionError: if LLM returns JSON without "name" or "description" keys.
            Message contains "bad schema" (tested by test_cluster_naming.py).
    """
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


class AnthropicClusterNamer:
    """
    Concrete ClusterNamer backed by an Anthropic client.
    Satisfies the ClusterNamer Protocol.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    def name_cluster(
        self,
        sample_texts: list[str],
        cluster_id: int,
    ) -> dict[str, str]:
        return name_cluster(self._client, sample_texts, cluster_id)


def name_all_clusters(
    cluster_items: dict[int, list[int]],
    id_to_text: dict[int, str],
    namer: ClusterNamer,
) -> dict[int, dict[str, str]]:
    """
    Name all clusters. Returns {cluster_id: {"name": str, "description": str}}.
    Called once during build_initial_clustering_state.
    """
    results = {}
    for cluster_id, item_ids in sorted(cluster_items.items()):
        sample_texts = [id_to_text[i] for i in item_ids[:5]]
        results[cluster_id] = namer.name_cluster(sample_texts, cluster_id)
    return results
