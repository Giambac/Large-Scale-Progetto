"""web/app.py — Flask + Flask-SocketIO debug UI server (D-13, D-14, D-15)."""
from __future__ import annotations

import io
import os

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

# ── Module-level session state (single session per server run, D-15) ─────────
# Cleared on each POST /upload so the old session is discarded.
_session: dict = {
    "state": None,    # Current ClusteringState or None
    "task": None,     # Background task handle or None
    "log_path": "audit_log.jsonl",
}

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
    D-15: single session per server run.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400
    f = request.files["file"]
    assert f.filename, "Upload received empty filename"

    content = f.read().decode("utf-8")
    records = _parse_upload(content, f.filename)
    assert len(records) >= 2, f"Dataset too small: {len(records)} records (need >= 2)"

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
    from src.clustering import build_initial_clustering_state
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
    initial_state = build_initial_clustering_state(store.get_all(), records, namer)
    _session["state"] = initial_state

    id_to_text = {i: r["text"] for i, r in enumerate(records)}

    # Phase 2: use a neutral MockOracle (Phase 3 replaces with real Oracle Agent)
    neutral_reply = OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)
    oracle = MockOracle(script=[neutral_reply] * 30)

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
