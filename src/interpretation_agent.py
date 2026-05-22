"""
src/interpretation_agent.py — Interpretation Agent (pre-parser normalizer).

Sits between the raw oracle utterance and feedback_parser.py.
Rewrites ambiguous user text into unambiguous text that the parser
can handle without hallucinating cluster IDs or crashing.

Single public function: interpret_feedback(raw_text, state, client) -> str

Design decisions:
- Returns raw_text unchanged on any exception (fail-open: better to pass
  ambiguous text to the parser than to silently drop the turn).
- Uses max_tokens=256 — rewrites are always shorter than the original.
- Keeps the prompt in a module-level constant so callers cannot accidentally
  mutate it and tests can inspect it directly.
"""
from __future__ import annotations

from src.state import ClusteringState

_INTERPRETATION_PROMPT = """\
You are a preprocessing agent. Your only job is to rewrite \
the user's message so it is unambiguous for a downstream parser.

Current clusters (use ONLY these IDs):
{cluster_list}
(Format: ID | Name | N items)

Recent turns:
{transcript}

Current message: "{raw_text}"

Examples:

Input:  "split the biggest cluster"
Output: "split cluster 12"

Input:  "merge cluster 1 and 5"
Output: "merge cluster 3 and cluster 7"

Input:  "looks good but merge the last two"
Output: "I am satisfied. However, merge cluster 7 and cluster 12."

Input:  "perfect"
Output: "I am satisfied with the current clustering."

Input:  "merge them"
Output: "merge cluster 3 and cluster 7"

Input:  "change the title of cluster 1 to Ciao"
Output: "rename cluster 1 to Ciao"

Input:  "rename the first cluster to Positive Reviews"
Output: "rename cluster 3 to Positive Reviews"

Input:  "the third cluster must be renamed in Ciao"
Output: "rename cluster 3 to Ciao"

Input:  "call the second cluster Food Issues"
Output: "rename cluster 7 to Food Issues"

Input:  "the first cluster should be called Billing Problems"
Output: "rename cluster 1 to Billing Problems"

Input:  "I want the last cluster to be named Service Complaints"
Output: "rename cluster 3 to Service Complaints"

Input:  "give the biggest cluster a better name: Quality Issues"
Output: "rename cluster 12 to Quality Issues"

Input:  "delete cluster 2"
Output: "delete cluster 7"

Input:  "remove the smallest cluster"
Output: "delete cluster 7"

Input:  "delete the last cluster and merge its items into the closest cluster"
Output: "delete cluster 4"

Rules:
- Replace ALL positional or vague references ("the biggest", "the last two", \
"cluster 1", "them") with exact IDs from the cluster list above.
- "the biggest" = cluster with most items. "the last two" = last two by list order.
- "cluster N" where N is a position (not a real ID) = map to the ID at that position, BUT ONLY if N <= number of clusters in the list. If N is larger than the number of clusters, leave it unchanged.
- If the message contains BOTH satisfaction ("looks good", "perfect", "great") \
AND a request, split into two sentences: satisfaction first, then the request.
- If the message is ONLY satisfaction, rewrite as: \
"I am satisfied with the current clustering."
- ANY request to rename a cluster ("rename", "call it", "should be called", "must be renamed", "give it a name", "I want it named", "change the title", "retitle", "let's call it") → rewrite as: "rename cluster <ID> to <new name>"
- Delete / remove cluster requests ("delete cluster X", "remove cluster X", "get rid of cluster X", "disband cluster X") → rewrite as: "delete cluster <ID>"
- If a reference cannot be resolved with confidence, leave that part unchanged.
- NEVER invent or modify item IDs (numbers after "item"). Item IDs are opaque — you do not know which ones exist. Leave them exactly as written.
- Never invent operations not present in the original message.

Output ONLY the rewritten text. No explanation. No JSON. No markdown.\
"""


def _build_cluster_list(state: ClusteringState) -> str:
    """Return a compact string listing active cluster IDs, names, and sizes.

    Example:
        3 | Billing Issues    | 142 items
        7 | Login Problems    |  89 items
       12 | Feature Requests  | 201 items
    """
    return "\n".join(
        f"{c.id:>3} | {c.name:<20} | {len(c.item_ids)} items"
        for c in state.clusters
    )


def _build_transcript(turn_history: list[str] | None) -> str:
    """Format recent turns for the prompt.

    turn_history is a list of strings like:
        ["Turn 3 - user: keep Billing separate", "Turn 4 - system: ok"]

    Returns "(none)" when history is empty or None.
    """
    if not turn_history:
        return "(none)"
    return "\n".join(turn_history)


def interpret_feedback(
    raw_text: str,
    state: ClusteringState,
    client: object,
    turn_history: list[str] | None = None,
) -> str:
    """Rewrite raw_text so cluster references use exact IDs.

    Args:
        raw_text:     Raw oracle utterance.
        state:        Current ClusteringState — used to build the cluster list.
        client:       Provider tuple (provider, sdk_client) — see src/llm_call.py.
        turn_history: Optional list of recent turn strings for context.

    Returns:
        Rewritten text (str). Falls back to raw_text on any exception so the
        caller always receives a usable string — never raises.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    from src.llm_call import chat

    prompt = _INTERPRETATION_PROMPT.format(
        cluster_list=_build_cluster_list(state),
        transcript=_build_transcript(turn_history),
        raw_text=raw_text,
    )

    try:
        result = chat(client, system=None, user=prompt, max_tokens=256).strip()
    except Exception as _e:
        print(f"[InterpretationAgent] ⚠️  LLM call failed ({_e!r}), using raw text")
        return raw_text

    # Sanity check: if the result is suspiciously long (>3x input) or empty,
    # fall back to raw_text to avoid passing garbage to the parser.
    if not result or len(result) > len(raw_text) * 3 + 100:
        print(f"[InterpretationAgent] ⚠️  output suspicious (len={len(result)}), using raw text")
        return raw_text

    print(f"[InterpretationAgent] ✅ rewrite OK")
    print(f"[InterpretationAgent]   raw_text        : {raw_text!r}")
    print(f"[InterpretationAgent]   interpreted_text: {result!r}")
    return result