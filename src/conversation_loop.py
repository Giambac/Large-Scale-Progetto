"""
conversation_loop.py — Plain Python while loop orchestrator (D-01, D-02).

The loop owns all I/O (D-04): JSONL AuditLog write + SocketIO emit.
f_* functions are pure and called from here; they do not write anything.

GlobalFeedback accumulator (FB-01):
    The loop maintains a `global_instructions: list[str]` that persists across all turns.
    This list is passed to f_next_state each turn. When f_next_state processes a GlobalFeedback
    delta, it appends instruction_text to this list in-place. The accumulated list is available
    for Phase 3 to pass to the ClusterNamer for prompt enrichment.

Threading note: run_conversation() is designed to run in a background thread
via socketio.start_background_task() (see web/app.py). When socketio is None,
emit calls are skipped — useful for unit tests without a running Flask server.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

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


def _format_message(action: object, state: ClusteringState) -> str:
    """
    Convert an Action to a human-readable message for the oracle.
    Phase 2: plain text summary. Phase 3 Oracle Agent will parse this.
    """
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
) -> ClusteringState:
    """
    Plain Python while loop (D-01). Runs until a stopping condition fires.

    Args:
        initial_state: Starting ClusteringState (from build_initial_clustering_state).
        oracle: OracleProtocol instance (MockOracle in Phase 2, real Oracle in Phase 3).
        store: EmbeddingStore (read-only; used by f_next_state split path).
        namer: ClusterNamer (for renaming affected clusters after split/merge/move).
        strategy: StrategyProtocol for action selection. None -> RandomStrategy(seed=0).
        log_path: Path for the AuditLog JSONL file.
        criteria: StoppingCriteria. None -> StoppingCriteria() defaults (turn_budget=15).
        socketio: SocketIO instance for emitting state_update events. None -> skip emits.
        id_to_text: dict[item_id, text] for cluster naming. None -> empty dict.
        llm_client: Anthropic client for parse_feedback. None -> no parsing (deltas=[]).

    Returns:
        Final ClusteringState when the loop terminates.
    """
    if criteria is None:
        criteria = StoppingCriteria()
    if strategy is None:
        strategy = RandomStrategy(seed=0)
    if id_to_text is None:
        id_to_text = {}

    hierarchy = HierarchyStore()
    # Register all initial clusters in the hierarchy
    for cluster in initial_state.clusters:
        hierarchy.register(cluster.id)

    # FB-01: GlobalFeedback accumulator — persists across all turns in this session
    global_instructions: list[str] = []

    state = initial_state
    recent_magnitudes: list[float] = []

    while True:
        # Step 1: Compute uncertainty
        uncertainty_report = f_uncertainty(state)

        # Step 2: Select action
        action = f_next_best_step(state, strategy, uncertainty_report)

        # Step 3: Format message and get oracle reply
        message = _format_message(action, state)
        reply = oracle.reply(state, message)

        # Step 4: Parse oracle reply into FeedbackDelta list
        # parse_feedback is the only permitted try/except boundary in Phase 2.
        if llm_client is not None and reply.raw_text.strip():
            deltas = parse_feedback(reply.raw_text, state, llm_client)
        else:
            deltas = []

        # Step 5: Apply feedback to state (pure function).
        # global_instructions is passed in; f_next_state appends GlobalFeedback text in-place (FB-01).
        new_state = f_next_state(state, deltas, store, namer, hierarchy, id_to_text, global_instructions)

        # Step 6: Write AuditLog (D-04: loop owns the JSONL write)
        append_to_audit_log(new_state, log_path)

        # Step 7: Emit state update via SocketIO (skip if socketio is None — unit test mode)
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

        # Step 8: Record feedback magnitude for diminishing-returns tracking (Phase 4)
        recent_magnitudes.append(float(len(deltas)))

        # Step 9: Check stopping conditions
        stop = check_stopping(
            turn_index=new_state.turn_index,
            oracle_satisfied=reply.satisfied,
            recent_magnitudes=recent_magnitudes,
            criteria=criteria,
        )

        # Step 10: Advance state or break
        state = new_state
        if stop is not None:
            if socketio is not None:
                socketio.emit("session_stopped", {"reason": stop.value})
            break

    return state
