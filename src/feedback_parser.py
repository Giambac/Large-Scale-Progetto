"""
feedback_parser.py — parse_feedback(): the only module in Phase 2 that calls the LLM for parsing (D-05).

Converts raw oracle utterance text into a list of structured FeedbackDelta objects.
Post-parse validation asserts every cluster_id in every delta against the current
ClusteringState — hallucinated cluster IDs crash immediately with AssertionError (T-02-02).
Unknown feedback types crash immediately with AssertionError (T-02-03).
json.JSONDecodeError propagates (T-02-04 — accepted: fail loudly).
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

VALID_FEEDBACK_TYPES = frozenset({"global", "split", "merge", "move_item", "instructional"})

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


def _build_cluster_summary(state: ClusteringState) -> str:
    """Return a compact string listing active cluster IDs and names.

    Example: "0: Alpha, 1: Beta, 2: Gamma"
    Pure function — no I/O.
    """
    return ", ".join(f"{c.id}: {c.name}" for c in state.clusters)


def _build_delta(item: dict, valid_cluster_ids: set[int]) -> FeedbackDelta:
    """Map one parsed JSON item to a FeedbackDelta.

    Asserts:
    - item["type"] is in VALID_FEEDBACK_TYPES
    - All referenced cluster_ids exist in valid_cluster_ids
    - merge cluster_a_id != cluster_b_id (self-merge guard)

    Raises:
        AssertionError: unknown type, hallucinated cluster_id, or self-merge
        KeyError: LLM omitted a required field — fail loudly, do not catch
    """
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

    # feedback_type == "instructional" (only remaining valid type)
    return InstructionalFeedback(instruction_text=item["instruction_text"])


def parse_feedback(
    raw_text: str,
    state: ClusteringState,
    client: object,
) -> list[FeedbackDelta]:
    """Convert raw oracle utterance into structured FeedbackDelta objects.

    Fast path: empty raw_text returns [] without calling the LLM.

    Args:
        raw_text: Raw oracle utterance string.
        state:    Current ClusteringState — used to validate cluster IDs.
        client:   Provider tuple (provider, sdk_client) — see src/llm_call.py::build_client.

    Returns:
        List of FeedbackDelta objects (may be empty).

    Raises:
        AssertionError: LLM returned unknown type or hallucinated cluster_id.
        json.JSONDecodeError: LLM returned non-JSON text (fail loudly — propagates).
        KeyError: LLM omitted a required field in a delta item (fail loudly).
    """
    if not raw_text:
        return []

    from src.llm_call import chat

    cluster_summary = _build_cluster_summary(state)
    prompt = _PARSE_FEEDBACK_PROMPT.format(
        cluster_summary=cluster_summary,
        raw_text=raw_text,
    )

    response_text = chat(
        client,
        system=None,
        user=prompt,
        max_tokens=512,
    )

    cleaned_text = response_text.strip()
    # Strip markdown code fences (same pattern as cluster_naming.py)
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.split("```")[1]
        if cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:]
        cleaned_text = cleaned_text.strip()

    raw_items = json.loads(cleaned_text)

    assert isinstance(raw_items, list), (
        f"parse_feedback: LLM returned non-list JSON: {type(raw_items)}"
    )

    valid_cluster_ids = {c.id for c in state.clusters}

    return [_build_delta(item, valid_cluster_ids) for item in raw_items]
