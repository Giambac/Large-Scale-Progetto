"""
mapping.py — Pluggable mapping function layer for post-session generalization (GEN-01).

After oracle acceptance, codify oracle preferences into a reusable mapping function
that assigns new items to clusters without oracle interaction.

Design decisions:
  D-02: MappingProtocol is structural (runtime_checkable Protocol), mirrors StrategyProtocol.
  D-03: OracleRuleSet is a Pydantic model; extracted from AuditLog FB-04 instructional entries.
  D-04: LLMMappingStrategy uses extracted rules + cluster descriptions as context.
  D-05: CentroidMappingStrategy uses cosine similarity; no LLM call at inference.

Registry: MAPPING_REGISTRY = {"llm": LLMMappingStrategy, "centroid": CentroidMappingStrategy}

References: GEN-01, D-02, D-03, D-05 (06-CONTEXT.md)
"""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import anthropic
import numpy as np
from pydantic import BaseModel

from src.embedding_store import EMBEDDING_MODEL, EmbeddingStore
from src.logging_setup import deviation
from src.serialization import load_audit_log
from src.state import ClusteringState


# ---------------------------------------------------------------------------
# OracleRuleSet — Pydantic model for codified oracle preferences (D-03)
# ---------------------------------------------------------------------------

class OracleRuleSet(BaseModel):
    """
    Codified oracle preferences extracted from AuditLog instructional feedback (FB-04).

    Fields:
        synonyms:      Groups of interchangeable terms that the oracle treats as equivalent.
                       e.g. [["error", "fail", "crash"], ["billing", "payment"]]
        focus_areas:   Topics or categories the oracle wants highlighted.
                       e.g. ["billing complaints", "feature requests"]
        exclusions:    Concepts or attributes the oracle wants ignored during assignment.
                       e.g. ["product names", "version numbers"]
        cluster_rules: Explicit assignment rules stated by the oracle.
                       e.g. ["technical issues go in cluster A", "complaints about billing go in cluster B"]
    """

    synonyms: list[list[str]]
    focus_areas: list[str]
    exclusions: list[str]
    cluster_rules: list[str]


# ---------------------------------------------------------------------------
# extract_oracle_rules — build OracleRuleSet from AuditLog (D-03)
# ---------------------------------------------------------------------------

_EXTRACT_RULES_PROMPT = """You are extracting oracle clustering preferences from a list of instructional statements.
Each statement is a soft instruction a human oracle gave during a clustering conversation.

Instructions:
{instructions}

Classify these instructions into the following categories and return ONLY valid JSON with no markdown:
{{
  "synonyms": [["term1", "term2"], ...],
  "focus_areas": ["area1", "area2", ...],
  "exclusions": ["exclusion1", "exclusion2", ...],
  "cluster_rules": ["rule1", "rule2", ...]
}}

Rules:
- synonyms: groups of interchangeable terms (words/phrases the oracle treats as equivalent)
- focus_areas: topics or categories the oracle wants emphasized
- exclusions: concepts or attributes the oracle said to ignore
- cluster_rules: explicit assignment rules (e.g. "X goes in cluster Y")
- If a category has no entries, use an empty list []
- Return ONLY the JSON object, nothing else
"""


def extract_oracle_rules(audit_log_path: str) -> OracleRuleSet:
    """
    Extract oracle preferences from AuditLog instructional feedback (FB-04, D-03).

    Reads audit_log.jsonl, scans each turn for deltas with type=="instructional",
    collects all instruction_text strings, and classifies them into OracleRuleSet
    fields via a single LLM call (claude-haiku-4-5).

    The audit_log.jsonl stores ClusteringState snapshots. When the JSONL was written
    by a run that also stored raw delta information (future sessions), those appear
    in a "deltas" key on each line. This function reads the raw JSONL lines directly
    to handle both formats gracefully.

    Args:
        audit_log_path: Path to the session's audit_log.jsonl file.

    Returns:
        OracleRuleSet with extracted preferences. All fields are empty lists when
        no FB-04 entries are found (deviation() is called in that case).

    Raises:
        AssertionError: If audit_log_path does not exist or is empty (via load_audit_log).
        json.JSONDecodeError: If JSONL is malformed (fail loudly).
        anthropic.APIError: Re-raised from LLM call boundary.
    """
    # load_audit_log validates existence and non-empty — asserts propagate loudly
    _states = load_audit_log(audit_log_path)  # validates file exists and non-empty

    # Read raw JSONL to find instructional deltas — ClusteringState does not store deltas,
    # but future audit_log formats or extended serialization may include a "deltas" key.
    instructional_texts: list[str] = []
    with open(audit_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)  # JSONDecodeError propagates (fail loudly)
            deltas = record.get("deltas", [])
            for delta in deltas:
                if isinstance(delta, dict) and delta.get("type") == "instructional":
                    text = delta.get("instruction_text", "")
                    if text:
                        instructional_texts.append(text)

    if not instructional_texts:
        deviation(
            "no_fb04_entries",
            audit_log_path=audit_log_path,
        )
        return OracleRuleSet(synonyms=[], focus_areas=[], exclusions=[], cluster_rules=[])

    instructions_block = "\n".join(f"- {t}" for t in instructional_texts)
    prompt = _EXTRACT_RULES_PROMPT.format(instructions=instructions_block)

    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError:
        raise  # re-raise — no swallowing

    raw = resp.content[0].text.strip()
    parsed = json.loads(raw)  # JSONDecodeError propagates (fail loudly)

    return OracleRuleSet(
        synonyms=parsed["synonyms"],
        focus_areas=parsed["focus_areas"],
        exclusions=parsed["exclusions"],
        cluster_rules=parsed["cluster_rules"],
    )


# ---------------------------------------------------------------------------
# MappingProtocol — structural interface (D-02, mirrors StrategyProtocol)
# ---------------------------------------------------------------------------

@runtime_checkable
class MappingProtocol(Protocol):
    """
    Interface for post-session item-to-cluster mapping strategies.

    Structural subtyping: implementations do NOT need to inherit from this class.
    Mirrors StrategyProtocol from src/strategy.py exactly.

    Method:
        assign(item_text, state, rule_set, embedding_store) -> cluster_id (str)
            Returns the cluster ID (as a string matching one of the cluster IDs
            in state.clusters) for the given new item.
    """

    def assign(
        self,
        item_text: str,
        state: ClusteringState,
        rule_set: OracleRuleSet,
        embedding_store: EmbeddingStore,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# LLMMappingStrategy — LLM-based assignment using extracted rules (D-04)
# ---------------------------------------------------------------------------

class LLMMappingStrategy:
    """
    Assigns new items to clusters using an LLM with cluster descriptions and
    OracleRuleSet rules as context (D-04).

    One anthropic.Anthropic() client call per assign() call.
    LLM response is validated against known cluster IDs before use (T-06-01-01 threat mitigation).
    """

    def __init__(self) -> None:
        pass  # no args — consistent with StrategyProtocol pattern

    def assign(
        self,
        item_text: str,
        state: ClusteringState,
        rule_set: OracleRuleSet,
        embedding_store: EmbeddingStore,
    ) -> str:
        """
        Assign item_text to a cluster using LLM reasoning.

        Builds a prompt with:
          - Cluster names + descriptions (all clusters in state.clusters)
          - OracleRuleSet fields formatted as bullet lists
          - item_text
          - Instruction to reply with ONLY the cluster ID integer

        Args:
            item_text:       The text of the new item to assign.
            state:           Current ClusteringState with cluster definitions.
            rule_set:        Extracted oracle preferences (OracleRuleSet).
            embedding_store: Not used by LLM strategy (present for protocol conformance).

        Returns:
            Cluster ID as a string (e.g. "0", "1", "2") matching one of state.clusters.

        Raises:
            ValueError: If LLM returns a cluster ID not in state.clusters.
            anthropic.APIError: Re-raised from LLM call boundary.
        """
        assert len(state.clusters) > 0, (
            "LLMMappingStrategy.assign: state has no clusters — cannot assign item"
        )

        valid_cluster_ids = {str(c.id) for c in state.clusters}

        # Build cluster summary section
        cluster_lines = []
        for c in state.clusters:
            cluster_lines.append(f"  Cluster {c.id}: {c.name}\n  Description: {c.description}")
        clusters_block = "\n".join(cluster_lines)

        # Build oracle rules sections
        rule_parts: list[str] = []
        if rule_set.synonyms:
            syn_lines = "\n".join(
                f"  - {' / '.join(group)}" for group in rule_set.synonyms
            )
            rule_parts.append(f"Synonym groups (treat as interchangeable):\n{syn_lines}")
        if rule_set.focus_areas:
            fa_lines = "\n".join(f"  - {a}" for a in rule_set.focus_areas)
            rule_parts.append(f"Focus areas:\n{fa_lines}")
        if rule_set.exclusions:
            ex_lines = "\n".join(f"  - {e}" for e in rule_set.exclusions)
            rule_parts.append(f"Exclusions (ignore these when assigning):\n{ex_lines}")
        if rule_set.cluster_rules:
            cr_lines = "\n".join(f"  - {r}" for r in rule_set.cluster_rules)
            rule_parts.append(f"Cluster assignment rules:\n{cr_lines}")

        rules_block = "\n\n".join(rule_parts) if rule_parts else "(no explicit rules)"

        prompt = (
            f"You are assigning a new item to one of the following clusters.\n\n"
            f"Clusters:\n{clusters_block}\n\n"
            f"Oracle preferences:\n{rules_block}\n\n"
            f"New item text:\n  {item_text}\n\n"
            f"Which cluster ID does this item belong to? "
            f"Reply with ONLY the cluster ID integer, nothing else."
        )

        client = anthropic.Anthropic()
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError:
            raise  # re-raise — no swallowing

        value = resp.content[0].text.strip()

        if value not in valid_cluster_ids:
            raise ValueError(
                f"LLMMappingStrategy: LLM returned unknown cluster id {value!r}; "
                f"valid ids: {sorted(valid_cluster_ids)}"
            )

        return value


# ---------------------------------------------------------------------------
# CentroidMappingStrategy — cosine similarity to cluster centroids (D-05)
# ---------------------------------------------------------------------------

class CentroidMappingStrategy:
    """
    Assigns new items to clusters by cosine similarity to cluster centroids (D-05).

    Cluster centroids are computed from the embeddings of items currently assigned
    to each cluster (hard-assignment mean). No LLM call at inference — faster and
    embedding-space grounded, but ignores explicit oracle semantic rules.

    SentenceTransformer model is loaded lazily in __init__ (same model as EmbeddingStore).
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def assign(
        self,
        item_text: str,
        state: ClusteringState,
        rule_set: OracleRuleSet,
        embedding_store: EmbeddingStore,
    ) -> str:
        """
        Assign item_text to the nearest cluster centroid by cosine similarity.

        Centroid = mean of embeddings of all hard-assigned items in each cluster.
        New item embedding = self._model.encode([item_text])[0].

        Args:
            item_text:       The text of the new item to assign.
            state:           Current ClusteringState with cluster definitions and item_ids.
            rule_set:        Not used by centroid strategy (present for protocol conformance).
            embedding_store: Provides embeddings for existing cluster items.

        Returns:
            Cluster ID as a string (e.g. "0", "1", "2") matching one of state.clusters.

        Raises:
            ValueError: If any cluster has no hard-assigned items (cluster.item_ids is empty).
            AssertionError: If state has no clusters.
        """
        assert len(state.clusters) > 0, (
            "CentroidMappingStrategy.assign: state has no clusters — cannot assign item"
        )

        # Compute centroids — fail loudly if any cluster has no items
        centroids: list[tuple[str, np.ndarray]] = []
        for cluster in state.clusters:
            if not cluster.item_ids:
                raise ValueError(
                    f"cluster {cluster.id} has no items — cannot compute centroid"
                )
            vecs = np.array([embedding_store.get(iid) for iid in cluster.item_ids])
            centroid = np.mean(vecs, axis=0)
            centroids.append((str(cluster.id), centroid))

        # Embed the new item
        item_vec: np.ndarray = self._model.encode(
            [item_text], convert_to_numpy=True
        )[0]

        # Assign to nearest centroid by cosine similarity
        best_cluster_id: str = centroids[0][0]
        best_sim: float = -2.0  # sentinel below any valid cosine value

        norm_item = np.linalg.norm(item_vec)  # computed once before the loop
        for cluster_id_str, centroid in centroids:
            norm_centroid = np.linalg.norm(centroid)
            cosine_sim = float(
                np.dot(item_vec, centroid) / (norm_item * norm_centroid + 1e-10)
            )
            if cosine_sim > best_sim:
                best_sim = cosine_sim
                best_cluster_id = cluster_id_str

        return best_cluster_id


# ---------------------------------------------------------------------------
# MAPPING_REGISTRY — mirrors STRATEGY_REGISTRY from src/strategy.py
# ---------------------------------------------------------------------------

MAPPING_REGISTRY: dict[str, type] = {
    "llm": LLMMappingStrategy,
    "centroid": CentroidMappingStrategy,
}
