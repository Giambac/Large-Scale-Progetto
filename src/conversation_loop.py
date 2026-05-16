"""
conversation_loop.py — Plain Python while loop orchestrator (D-01, D-02).

The loop owns all I/O (D-04): JSONL AuditLog write + SocketIO emit.
f_* functions are pure and called from here; they do not write anything.

GlobalFeedback accumulator (FB-01):
    The loop maintains a `global_instructions: list[str]` that persists across all turns.
    This list is passed to f_next_state each turn. When f_next_state processes a GlobalFeedback
    delta, it appends instruction_text to this list in-place. The accumulated list is available
    for Phase 3 to pass to the ClusterNamer for prompt enrichment.

Phase 3 additions:
    - f_cognitive_load computed before oracle.reply() each turn (ORC-03, D-06)
    - global_instructions passed to OracleAgent.reply() if oracle is an OracleAgent (FB-04, D-12)
    - oracle_init event written to events.jsonl sidecar at run start (ORC-02, D-05)
    - drift_event written to events.jsonl when reply.contradiction_detected is True (ORC-04, D-10)
    - events.jsonl is a SEPARATE sidecar file from audit_log.jsonl (Pitfall 5 — never write
      oracle_init/drift_event to audit_log.jsonl; load_audit_log() would crash on non-state records)

Phase 4 additions:
    - DB writes per turn: turns.create() + oracle_feedback.create() after JSONL write (D-03)
    - real weighted magnitude via compute_magnitude() replaces float(len(deltas)) (D-13)
    - isinstance(oracle, OracleAgent) branch removed; oracle.reply() called unconditionally (D-27)
    - state_update SocketIO event extended with pairwise_accuracy, convergence_signal,
      contradiction_count (cumulative running total) fields (D-28)
    - PairBag updated each turn for pairwise accuracy computation (D-18, D-20)
    - deviation() fires when a DB-tracked resumed session starts without PairBag (D-21, V-4-05)
    - Write order per turn: JSONL first -> DB turns.create + oracle_feedback.create -> socket emit (D-03)

Threading note: run_conversation() is designed to run in a background thread
via socketio.start_background_task() (see web/app.py). When socketio is None,
emit calls are skipped — useful for unit tests without a running Flask server.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional, Callable

from src.agent_functions import f_output, f_next_best_step, f_next_state
from src.feedback_parser import parse_feedback
from src.hierarchy import HierarchyStore
from src.logging_setup import deviation
from src.serialization import append_to_audit_log
from src.state import ClusteringState
from src.stopping import StoppingCriteria, check_stopping, compute_magnitude, FeedbackMagnitudeWeights
from src.strategy import RandomStrategy
from src.uncertainty import f_uncertainty

if TYPE_CHECKING:
    import sqlite3
    from src.embedding_store import EmbeddingStore
    from src.cluster_naming import ClusterNamer
    from src.oracle_protocol import OracleProtocol
    from src.strategy import StrategyProtocol
    from src.judge import PairBag as _PairBagT


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


def _write_event(record: dict, events_path: str) -> None:
    """
    Append one JSON event record to the events sidecar file.

    Used for oracle_init and drift_event records (Phase 3).
    MUST NOT write to audit_log.jsonl — that file stores only ClusteringState lines
    and load_audit_log() will crash on non-state records (Pitfall 5).

    No try/except — fail loudly per CLAUDE.md.

    Args:
        record:      Plain dict, JSON-serializable (no numpy types).
        events_path: Path to the events.jsonl sidecar file.
    """
    parent = os.path.dirname(events_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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
    db_conn: "sqlite3.Connection | None" = None,  # D-04: open connection per run; None = no DB (unit test mode)
    experiment_id: int | None = None,              # FK for DB turn/feedback rows; None if db_conn is None
    pair_bag: "_PairBagT | None" = None,           # D-21: caller may provide pre-built bag for resume
    magnitude_weights: "FeedbackMagnitudeWeights | None" = None,  # D-13: defaults if None
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
        post_turn_callback: Optional callable (new_state, deltas) -> None. Called after
            each turn's AuditLog write. Used for projection recompute on split/merge (D-22).
        events_path: Path for oracle_init + drift_event sidecar JSONL file (Phase 3).
            None -> derived from log_path (same directory, filename events.jsonl).
            MUST be a different file from log_path — audit_log.jsonl stores only state records.
        db_conn: Open sqlite3.Connection for DB writes per turn (D-04). None -> no DB (unit test mode).
            Must be opened with check_same_thread=False — this function runs in a background thread.
            Caller is responsible for opening, committing, and closing the connection.
        experiment_id: FK for turn/feedback rows (DB-02, DB-03). None if db_conn is None.
            Must be set when db_conn is not None — asserted at DB write time.
        pair_bag: Pre-built PairBag for resume sessions (D-21). None -> new empty PairBag.
            Only used when db_conn is not None. For resumed DB-tracked sessions, pass the
            reconstructed bag (Phase 5 scope). In Phase 4, passing None triggers deviation()
            when db_conn is set.
        magnitude_weights: Weights for compute_magnitude() (D-13). None -> FeedbackMagnitudeWeights()
            defaults (global=1.0, cluster=0.5, point=0.2, instructional=0.1).

    Write order per turn (D-03):
        1. append_to_audit_log (JSONL — always written)
        2. turns.create() + oracle_feedback.create() (DB — only when db_conn is not None)
        3. post_turn_callback (projection recompute hook)
        4. socketio.emit("state_update") (UI update)

    Returns:
        Final ClusteringState when the loop terminates.
    """
    # Phase 3: ORC-03 — import f_cognitive_load at function body level, OUTSIDE the while loop.
    # Do NOT import inside the loop body.
    from src.cognitive_load import f_cognitive_load  # ORC-03: computed before oracle.reply()

    if criteria is None:
        criteria = StoppingCriteria()
    if strategy is None:
        strategy = RandomStrategy(seed=0)
    if id_to_text is None:
        id_to_text = {}
    if magnitude_weights is None:
        magnitude_weights = FeedbackMagnitudeWeights()

    # Phase 3: Derive events sidecar path (Pitfall 5 — separate from audit_log.jsonl)
    if events_path is None:
        _events_dir = os.path.dirname(log_path) or "."
        events_path = os.path.join(_events_dir, "events.jsonl")

    hierarchy = HierarchyStore()
    # Register all initial clusters in the hierarchy
    for cluster in initial_state.clusters:
        hierarchy.register(cluster.id)

    # FB-01: GlobalFeedback accumulator — persists across all turns in this session
    global_instructions: list[str] = []

    # Phase 3: Write oracle_init event if oracle is an OracleAgent (ORC-02, D-05)
    # Local import to avoid circular imports at module level.
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

    # Phase 4: PairBag initialization — lazy import guards against judge.py not existing yet.
    # src/judge.py is created by plan 04-03 (parallel wave). Import is deferred to function
    # body so `from src.conversation_loop import run_conversation` works even before judge.py exists.
    # PairBag is only instantiated and used when db_conn is not None.
    _pair_bag_was_none = pair_bag is None

    if db_conn is not None:
        # Lazy imports inside the DB-tracking block — only resolves when judge.py exists
        from src.judge import PairBag as _PairBag, compute_pairwise_accuracy as _compute_pairwise_accuracy, assemble_turn_metrics as _assemble_turn_metrics, assemble_feedback_rows as _assemble_feedback_rows

        if pair_bag is None:
            pair_bag = _PairBag()

        # D-21: In Phase 4, resumed DB sessions start without a PairBag — pairwise_accuracy will be
        # 0.0 until new feedback arrives. Fire deviation() to make this visible (V-4-05).
        if _pair_bag_was_none:
            deviation(
                "resumed DB session started without PairBag — pairwise_accuracy will be 0.0 until new feedback arrives",
                session_id="unknown",
            )

    # Safe defaults before first turn — used by state_update emit and loop break condition
    _cumulative_contradictions: int = 0  # D-28: running total (NOT per-turn binary — V-4-03)
    _pair_acc: float = 0.0               # safe default before first turn
    _stop_for_metrics: object = None     # safe default before first turn (StopReason | None)

    state = initial_state
    recent_magnitudes: list[float] = []

    while True:
        # Step 1: Compute uncertainty
        uncertainty_report = f_uncertainty(state)

        # Step 2: Select action
        action = f_next_best_step(state, strategy, uncertainty_report)

        # Step 3a: Format message and compute per-turn cognitive load (Phase 3 — ORC-03, D-06)
        # f_cognitive_load is imported at function body level (before while), not inside the loop.
        message = _format_message(action, state)
        cognitive_load = f_cognitive_load(state, message)

        # Step 3b: Get oracle reply — call unconditionally with global_instructions + cognitive_load (D-27)
        # isinstance branch removed per D-27: all OracleProtocol implementations must accept
        # global_instructions and cognitive_load kwargs (OracleProtocol.reply() and MockOracle.reply()
        # were updated in Plan 03 / oracle_protocol.py to accept **kwargs).
        reply = oracle.reply(
            state, message,
            global_instructions=global_instructions,
            cognitive_load=cognitive_load,
        )

        # Step 4: Parse oracle reply into FeedbackDelta list
        # parse_feedback is the only permitted try/except boundary in Phase 2.
        if llm_client is not None and reply.raw_text.strip():
            deltas = parse_feedback(reply.raw_text, state, llm_client)
        else:
            deltas = []

        # Step 5: Apply feedback to state (pure function).
        # global_instructions is passed in; f_next_state appends GlobalFeedback text in-place (FB-01).
        new_state = f_next_state(state, deltas, store, namer, hierarchy, id_to_text, global_instructions)

        # Step 5b: Update OracleAgent delta window and detect contradictions (Phase 3 — ORC-04, D-09)
        # IMPORTANT: use new_state.turn_index (AFTER f_next_state), not state.turn_index.
        # Modification B must run AFTER f_next_state so the turn_index stored in the deque
        # and referenced in drift_event records reflects the turn that just completed.
        if isinstance(oracle, _OracleAgent) and deltas:
            contradiction_detected, contradicted_turn = oracle.update_delta_window(
                deltas, new_state.turn_index
            )
            reply.contradiction_detected = contradiction_detected
            reply.contradicted_turn = contradicted_turn

        # Step 5c: Update PairBag with this turn's deltas (D-18, D-20)
        # Only when db_conn is not None — pair_bag is only initialized in that branch.
        if db_conn is not None:
            pair_bag.update(deltas, new_state, is_contradiction=reply.contradiction_detected)

        # Step 5d: Track cumulative contradictions for DB-02 (NOT per-turn binary flag — V-4-03)
        if reply.contradiction_detected:
            _cumulative_contradictions += 1

        # Step 6: Write AuditLog (D-04: loop owns the JSONL write; D-03: JSONL BEFORE DB writes)
        append_to_audit_log(new_state, log_path)

        # Step 6c: Log drift event to events sidecar if contradiction detected (Phase 3 — ORC-04, D-10)
        if reply.contradiction_detected:
            _write_event({
                "event": "drift_event",
                "turn": new_state.turn_index,
                "contradicted_turn": reply.contradicted_turn,
                "timestamp": new_state.timestamp,
            }, events_path)

        # Step 6a: DB writes — AFTER JSONL write, BEFORE post_turn_callback and socket emit (D-03)
        # All judge.py imports are lazy and already resolved at function start for the db_conn branch.
        if db_conn is not None and experiment_id is not None:
            from src.db import turns as _turns_db
            from src.db import oracle_feedback as _fb_db

            # Compute pairwise accuracy before writing turn row
            _pair_acc = _compute_pairwise_accuracy(new_state, pair_bag, new_state.turn_index)

            # Check stop reason for convergence_signal field — computed ONCE per turn (V-4-03)
            _stop_for_metrics = check_stopping(
                turn_index=new_state.turn_index,
                oracle_satisfied=reply.satisfied,
                recent_magnitudes=recent_magnitudes,
                criteria=criteria,
            )

            _turn_create = _assemble_turn_metrics(
                state=new_state,
                reply=reply,
                pair_acc=_pair_acc,
                stop_reason=_stop_for_metrics,
                action_type=action.action_type,
                experiment_id=experiment_id,
                cumulative_contradiction_count=_cumulative_contradictions,  # running counter (V-4-03)
            )
            _turn_row = _turns_db.create(db_conn, _turn_create)

            # Insert one oracle_feedback row per delta (DB-03: compound messages = multiple rows)
            _fb_rows = _assemble_feedback_rows(_turn_row.id, deltas, reply)
            for _fb in _fb_rows:
                _fb_db.create(db_conn, _fb)
            # Note: create() already commits after each insert per src/db/turns.py and oracle_feedback.py

        # Step 6b: Call post_turn_callback if provided (D-22: projection recompute hook)
        if post_turn_callback is not None:
            post_turn_callback(new_state, deltas)

        # Step 8: Record feedback magnitude for diminishing-returns tracking (Phase 4)
        # Real weighted magnitude per D-13 — replaces the Phase 2 stub of float(len(deltas)).
        recent_magnitudes.append(compute_magnitude(deltas, magnitude_weights))

        # Step 7: Emit state update via SocketIO (skip if socketio is None — unit test mode)
        if socketio is not None:
            import dataclasses

            # Compute _pair_acc and _stop_for_metrics if db_conn is None (unit-test mode, Step 6a was skipped).
            # Do NOT call check_stopping() or compute_pairwise_accuracy() again if already computed above
            # (one call per turn per V-4-03).
            if db_conn is None:
                _stop_for_metrics = check_stopping(
                    turn_index=new_state.turn_index,
                    oracle_satisfied=reply.satisfied,
                    recent_magnitudes=recent_magnitudes,
                    criteria=criteria,
                )
                # _pair_acc remains 0.0 when db_conn is None — no PairBag available in no-DB mode

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
                "pairwise_accuracy": _pair_acc,                                                        # D-28
                "convergence_signal": _stop_for_metrics.value if _stop_for_metrics else None,          # D-28
                "contradiction_count": _cumulative_contradictions,                                     # D-28: cumulative running total (V-4-03)
            })

        # Step 9: Check stopping conditions — reuse _stop_for_metrics already computed above.
        # When db_conn is not None, _stop_for_metrics was computed in Step 6a.
        # When db_conn is None and socketio is not None, it was computed in Step 7.
        # When both are None (pure unit test, no DB, no UI), compute it here for the loop break.
        if db_conn is None and socketio is None:
            _stop_for_metrics = check_stopping(
                turn_index=new_state.turn_index,
                oracle_satisfied=reply.satisfied,
                recent_magnitudes=recent_magnitudes,
                criteria=criteria,
            )
        stop_reason = _stop_for_metrics

        # Step 10: Advance state or break
        state = new_state
        if stop_reason is not None:
            if socketio is not None:
                socketio.emit("session_stopped", {"reason": stop_reason.value})
            break

    return state
