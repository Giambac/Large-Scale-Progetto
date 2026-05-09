"""web/app.py — Flask + Flask-SocketIO debug UI server (D-13, D-14, D-15)."""
from __future__ import annotations

import argparse
import io
import json as _json
import os
import sys
import threading

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO


# ── UMAP projection helpers (VIZ-V2-01) ──────────────────────────────────────

def _compute_projection(embeddings: "np.ndarray") -> "np.ndarray":
    """
    Compute 2D UMAP projection of embeddings (D-21).

    Uses fixed random_state=42 for reproducibility across runs on the same dataset.
    n_neighbors=15, min_dist=0.1 are reasonable defaults for 768-dim text embeddings.

    Args:
        embeddings: shape (N, dim)

    Returns:
        coords: shape (N, 2) float32
    """
    import numpy as np
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(embeddings)
    assert coords.shape == (embeddings.shape[0], 2), (
        f"UMAP output shape {coords.shape} != ({embeddings.shape[0]}, 2)"
    )
    return coords.astype("float32")


# Palette of 20 visually distinct hex colors for cluster membership.
# Cycles if there are more than 20 clusters.
_CLUSTER_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def _build_projection_payload(
    coords: "np.ndarray",
    state: "ClusteringState",
) -> dict:
    """
    Build the projection_update SocketIO payload from UMAP coords and ClusteringState.

    Payload format:
        coords:        list of N [x, y] pairs (floats)
        cluster_ids:   list of N cluster_id ints (one per item, ordered by item_id)
        max_probs:     list of N floats — max soft_prob per item (for opacity)
        cluster_colors: dict str(cluster_id) -> "#rrggbb"
    """
    import numpy as np
    N = len(state.assignments)
    assert coords.shape[0] == N, f"coords has {coords.shape[0]} rows, state has {N} items"

    # Build cluster_id list ordered by item_id (0..N-1)
    cluster_ids = [state.assignments[i] for i in range(N)]

    # Build max_probs ordered by item_id
    max_probs = [float(max(state.soft_probs[i])) for i in range(N)]

    # Assign a color to each unique cluster_id (stable across calls)
    sorted_cluster_ids = sorted({c.id for c in state.clusters})
    cluster_colors = {
        str(cid): _CLUSTER_COLORS[idx % len(_CLUSTER_COLORS)]
        for idx, cid in enumerate(sorted_cluster_ids)
    }

    return {
        "coords": coords.tolist(),       # list of [x, y] pairs
        "cluster_ids": cluster_ids,
        "max_probs": max_probs,
        "cluster_colors": cluster_colors,
    }


def _should_recompute_projection(deltas: list) -> bool:
    """
    Return True iff any delta is a SplitFeedback or MergeFeedback (D-22).

    Projection is recomputed ONLY on cluster-count changes (split/merge).
    NOT recomputed on point moves (embedding positions unchanged).
    NOT recomputed every turn (too expensive and disorienting).
    """
    from src.feedback import SplitFeedback, MergeFeedback
    return any(isinstance(d, (SplitFeedback, MergeFeedback)) for d in deltas)


def compute_and_emit_projection(
    store: object,
    state: "ClusteringState",
    sio: object,
) -> None:
    """
    Compute UMAP projection and emit projection_update event via SocketIO (D-24).

    Runs server-side. Emits coordinates as JSON over the projection_update event.
    Called from _run_conversation_background (background thread).
    Uses sio.emit() (instance method) — safe in background threads.
    """
    coords = _compute_projection(store.get_all())
    payload = _build_projection_payload(coords, state)
    sio.emit("projection_update", payload)


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for web/app.py.
    Called once at module load time.
    D-18: --backend hdbscan|kmeans. Default hdbscan. Unknown value asserts loudly.
    """
    parser = argparse.ArgumentParser(description="Clustering Agent Debug UI")
    parser.add_argument(
        "--backend",
        choices=["hdbscan", "kmeans"],
        default="hdbscan",
        help="Clustering backend to use (default: hdbscan)",
    )
    # parse_known_args so Flask/SocketIO can pass their own args without conflict
    args, _ = parser.parse_known_args()
    assert args.backend in ("hdbscan", "kmeans"), (
        f"Unknown --backend value: {args.backend!r}. Must be 'hdbscan' or 'kmeans'."
    )
    return args


_args = _parse_args()
_backend_name: str = _args.backend

# ── Module-level session state (single session per server run, D-15) ─────────
# Cleared on each POST /upload so the old session is discarded.
_session: dict = {
    "state": None,    # Current ClusteringState or None
    "task": None,     # Background task handle or None
    "log_path": "audit_log.jsonl",
}

# Lock to serialise concurrent /upload requests — prevents two background tasks
# from racing on _session["state"], the embeddings file, and the audit log.
_session_lock = threading.Lock()

# ── App and SocketIO init ─────────────────────────────────────────────────────
# async_mode='threading': safe for numpy/sklearn; no eventlet/gevent monkey-patching (RESEARCH.md)
# cors_allowed_origins='*': developer-only debug UI; no auth required
app = Flask(__name__, template_folder="templates", static_folder="static")
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    if _session["state"] is None:
        return jsonify({"status": "idle", "turn_index": None})
    state = _session["state"]
    return jsonify({
        "status": "running",
        "turn_index": state.turn_index,
        "cluster_count": len(state.clusters),
    })


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Accepts multipart/form-data with 'file' field (CSV or JSONL).
    Clears existing session state and starts a new conversation loop.
    D-15: single session per server run. _session_lock prevents concurrent
    uploads from racing on shared session state and background task.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400
    f = request.files["file"]
    assert f.filename, "Upload received empty filename"

    content = f.read().decode("utf-8")
    records = _parse_upload(content, f.filename)
    assert len(records) >= 2, f"Dataset too small: {len(records)} records (need >= 2)"

    with _session_lock:
        # Reset session state (D-15): clear old state before starting fresh session
        _session["state"] = None
        _session["task"] = None

        # Start the background task — heavy lifting (embeddings, clustering, LLM) runs there.
        # The background task asserts ANTHROPIC_API_KEY internally (fail-loudly boundary).
        _session["task"] = socketio.start_background_task(
            _run_conversation_background,
            records,
            _session["log_path"],
        )
    return jsonify({"status": "session_started", "records": len(records)}), 200


# ── Background task ───────────────────────────────────────────────────────────

def _run_conversation_background(records: list[dict], log_path: str) -> None:
    """
    Runs the full conversation pipeline in a background thread.

    Steps:
    1. Compute embeddings (EmbeddingStore.compute_and_save)
    2. Build initial ClusteringState (build_initial_clustering_state)
    3. Construct oracle, namer, client
    4. Call run_conversation — emits state_update each turn

    CRITICAL: Uses socketio.emit() (instance method) — never use the context-bound
    module-level emit function from flask_socketio, which raises RuntimeError in background
    threads (RESEARCH.md Pitfall 1).
    """
    import numpy as np

    from src.cluster_naming import AnthropicClusterNamer
    from src.clustering import HDBSCANBackend, KMeansBackend, build_initial_clustering_state
    from src.conversation_loop import run_conversation
    from src.embedding_store import EmbeddingStore
    from src.oracle_protocol import MockOracle, OracleReply
    from src.stopping import StoppingCriteria
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY environment variable not set"
    client = anthropic.Anthropic(api_key=api_key)
    namer = AnthropicClusterNamer(client)

    texts = [r["text"] for r in records]
    os.makedirs("embeddings", exist_ok=True)
    store = EmbeddingStore.compute_and_save(texts, "embeddings/session_embeddings.npy")

    # D-18: instantiate the backend selected via --backend CLI flag
    if _backend_name == "kmeans":
        backend = KMeansBackend()
        # backend._k is set during build_initial_clustering_state via backend.fit()
    elif _backend_name == "hdbscan":
        backend = HDBSCANBackend()
    else:
        assert False, f"Unknown backend: {_backend_name!r}"

    initial_state = build_initial_clustering_state(
        store.get_all(), records, namer, backend=backend
    )
    _session["state"] = initial_state

    # D-17: log chosen K to audit_log.jsonl as backend_init event so runs are reproducible
    if _backend_name == "kmeans":
        _k_chosen = backend.k
        _backend_init_event = {
            "event": "backend_init",
            "backend": "kmeans",
            "k": _k_chosen,
            "turn": 0,
        }
        with open(log_path, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_backend_init_event) + "\n")
        print(f"[startup] KMeansBackend: K={_k_chosen} (selected via BIC on GMM)")
    else:
        _backend_init_event = {
            "event": "backend_init",
            "backend": "hdbscan",
            "k": len(initial_state.clusters),
            "turn": 0,
        }
        with open(log_path, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_backend_init_event) + "\n")
        print(f"[startup] HDBSCANBackend: K={len(initial_state.clusters)} clusters discovered")

    id_to_text = {i: r["text"] for i, r in enumerate(records)}

    # Emit initial projection (D-22: once after initial clustering)
    compute_and_emit_projection(store, initial_state, socketio)

    # Phase 2: use a neutral MockOracle (Phase 3 replaces with real Oracle Agent)
    neutral_reply = OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)
    oracle = MockOracle(script=[neutral_reply] * 30)

    def _projection_post_turn(new_state, deltas):
        # D-22: re-emit projection only on cluster-count changes (split/merge).
        # MockOracle never produces SplitFeedback or MergeFeedback in Phase 2,
        # so this fires zero times at runtime — but the wiring is correct and
        # will activate automatically when the real Oracle Agent ships in Phase 3.
        if _should_recompute_projection(deltas):
            compute_and_emit_projection(store, new_state, socketio)

    final_state = run_conversation(
        initial_state=initial_state,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=log_path,
        criteria=StoppingCriteria(turn_budget=30),
        socketio=socketio,  # INSTANCE METHOD — context-free; safe in background thread
        id_to_text=id_to_text,
        llm_client=client,
        post_turn_callback=_projection_post_turn,  # D-22: recompute on split/merge
    )
    _session["state"] = final_state


# ── Upload parser helper ──────────────────────────────────────────────────────

def _parse_upload(content: str, filename: str) -> list[dict]:
    """Parse CSV or JSONL upload into list of dicts with 'text' field."""
    import csv
    import json

    records: list[dict] = []
    if filename.endswith(".jsonl"):
        for line in content.splitlines():
            line = line.strip()
            if line:
                record = json.loads(line)
                assert "text" in record, f"JSONL record missing 'text' field: {record}"
                records.append({"item_id": len(records), "text": record["text"]})
    else:
        # Assume CSV with 'text' column
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            assert "text" in row, f"CSV row missing 'text' column: {row}"
            records.append({"item_id": len(records), "text": row["text"]})
    return records


# ── SocketIO connect handler ──────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    # Single session: no per-client state to initialize
    pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # NEVER enable Flask debug mode — it exposes the Werkzeug debugger (security risk)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
