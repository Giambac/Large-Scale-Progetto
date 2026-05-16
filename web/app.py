"""web/app.py — FastAPI + python-socketio (ASGI) debug UI server (D-13, D-14, D-15)."""
from __future__ import annotations

import argparse
import asyncio
import datetime
import io
import json as _json
import os
import shutil
import sys
import threading
import time

# Ensure the project root is on sys.path so `src` is importable regardless
# of which directory the server is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socketio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


# ── Session persistence helpers (UI-V2-01) ───────────────────────────────────

# Directory where named session directories are stored (D-26)
SESSIONS_DIR = "sessions"


def _make_session_timestamp() -> str:
    """
    Generate a filesystem-safe session timestamp string (D-26).
    Uses hyphens instead of colons: "2026-05-08T14-32-00"
    (colons are forbidden in Windows paths and inconvenient on POSIX too).
    """
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _write_session_state(state: "ClusteringState", session_dir: str) -> None:
    """
    Write the latest ClusteringState to sessions/<session_dir>/state.json (D-27).

    Called via post_turn_callback after every f_next_state turn — same timing as
    AuditLog JSONL write (D-04). Overwrites state.json on each call so it always
    holds the most recent state. Uses serialize_state() from serialization.py.
    """
    from src.serialization import serialize_state
    os.makedirs(session_dir, exist_ok=True)
    state_path = os.path.join(session_dir, "state.json")
    line = serialize_state(state)
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(line)


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
    import time
    import numpy as np
    import umap as umap_lib
    from sklearn.decomposition import PCA

    N, dim = embeddings.shape
    print(f"[timing] _compute_projection: input shape ({N}, {dim})")

    t0 = time.perf_counter()
    pca = PCA(n_components=50, random_state=42)
    reduced = pca.fit_transform(embeddings)
    print(f"[timing] PCA {dim}→50: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(reduced)
    print(f"[timing] UMAP 50→2 ({N} items): {time.perf_counter() - t0:.2f}s")

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
    emitter: object,
) -> None:
    """
    Compute UMAP projection and emit projection_update event (D-24).

    Runs server-side in a background thread. The emitter bridges the
    worker thread to the asyncio event loop via run_coroutine_threadsafe.
    """
    coords = _compute_projection(store.get_all())
    payload = _build_projection_payload(coords, state)
    emitter.emit("projection_update", payload)


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
    # parse_known_args so FastAPI/uvicorn can pass their own args without conflict
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
    "session_dir": None,   # path to sessions/<timestamp>/ for current session
    "progress": None,
}

# Lock to serialise concurrent /upload requests — prevents two background tasks
# from racing on _session["state"], the embeddings file, and the audit log.
_session_lock = threading.Lock()

# The running asyncio event loop — captured in lifespan; used by SocketIOEmitter.
_loop: asyncio.AbstractEventLoop | None = None


# ── SocketIOEmitter shim ──────────────────────────────────────────────────────

class SocketIOEmitter:
    """
    Bridges a sync worker thread to an asyncio.AsyncServer.

    The conversation loop calls `.emit(event, payload)` synchronously from
    a worker thread (CPU-bound clustering work runs there to avoid blocking
    the event loop). This wrapper schedules the coroutine on the main loop
    via run_coroutine_threadsafe — the standard, monkey-patch-free pattern.

    Fire-and-forget by design: matches the original Flask-SocketIO threading
    semantic. If the loop is closed during shutdown, the future errors out
    but the worker keeps running — same behavior as the previous stack.
    """
    def __init__(self, sio: socketio.AsyncServer, loop_getter) -> None:
        self._sio = sio
        self._loop_getter = loop_getter  # callable returning the asyncio loop (deferred so startup captures it)

    def emit(self, event: str, payload: dict) -> None:
        loop = self._loop_getter()
        assert loop is not None, "Event loop not captured yet — server not fully started"
        asyncio.run_coroutine_threadsafe(self._sio.emit(event, payload), loop)


# ── App and SocketIO init ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# async_mode='asgi': native ASGI, no monkey-patching required (RESEARCH.md)
# cors_allowed_origins='*': developer-only debug UI; no auth required
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# ASGI entrypoint — uvicorn runs this, not `app` directly.
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Emitter shim: resolves loop lazily so it can be module-level despite lifespan init.
emitter = SocketIOEmitter(sio, lambda: _loop)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/status")
async def status():
    if _session["state"] is None:
        return {"status": "idle", "turn_index": None}
    state = _session["state"]
    return {
        "status": "running",
        "turn_index": state.turn_index,
        "cluster_count": len(state.clusters),
    }


@app.get("/sessions")
async def list_sessions():
    """
    List all past sessions by scanning SESSIONS_DIR (D-28).
    Returns JSON list ordered newest-first.
    Each entry has: session_id, timestamp, cluster_count, turn_count.
    """
    from src.serialization import deserialize_state
    sessions = []
    if not os.path.isdir(SESSIONS_DIR):
        return []
    for entry in sorted(os.scandir(SESSIONS_DIR), key=lambda e: e.name, reverse=True):
        if not entry.is_dir():
            continue
        state_path = os.path.join(entry.path, "state.json")
        if not os.path.exists(state_path):
            continue
        line = open(state_path, encoding="utf-8").read().strip()
        if not line:
            continue
        state = deserialize_state(line)
        name_path = os.path.join(entry.path, "name.txt")
        session_name = open(name_path, encoding="utf-8").read().strip() if os.path.exists(name_path) else entry.name
        sessions.append({
            "session_id": entry.name,
            "timestamp": entry.name,
            "name": session_name,
            "cluster_count": len(state.clusters),
            "turn_count": state.turn_index,
        })
    return sessions


@app.post("/resume/{session_id}")
async def resume_session(session_id: str):
    """
    Load a past session by session_id (D-27: load state.json directly, no AuditLog replay).
    Sets _session["state"] to the loaded ClusteringState.
    Emits a state_update SocketIO event so the client UI updates immediately.
    """
    import dataclasses
    from src.serialization import deserialize_state

    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail=f"Session directory not found: {session_dir}")
    state_path = os.path.join(session_dir, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=404, detail=f"state.json not found in session {session_id}")

    line = open(state_path, encoding="utf-8").read().strip()
    assert line, f"state.json is empty for session {session_id}"
    state = deserialize_state(line)

    _session["state"] = state
    _session["session_dir"] = session_dir

    # Emit state_update so client UI reflects the resumed session (D-27)
    await sio.emit("state_update", {
        "turn_index": state.turn_index,
        "clusters": [dataclasses.asdict(c) for c in state.clusters],
        "soft_probs": {
            str(item_id): {
                str(c.id): prob
                for c, prob in zip(state.clusters, probs)
            }
            for item_id, probs in state.soft_probs.items()
        },
        "cognitive_load": 0.0,       # not available for resumed sessions
        "pairwise_accuracy": 0.0,    # D-28: not available for resumed sessions (PairBag reconstruction is Phase 5)
        "convergence_signal": None,  # D-28
        "contradiction_count": 0,    # D-28
    })

    return {
        "status": "resumed",
        "session_id": session_id,
        "turn_index": state.turn_index,
        "cluster_count": len(state.clusters),
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), backend: str = Form("hdbscan")):
    if backend not in ("hdbscan", "kmeans"):
        raise HTTPException(status_code=400, detail=f"Invalid backend: {backend}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload received empty filename")

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    records = _parse_upload(content, file.filename)
    if len(records) < 2:
        raise HTTPException(status_code=400, detail=f"Dataset too small: {len(records)} records (need >= 2)")

    with _session_lock:
        _session["state"] = None
        _session["task"] = None
        t = threading.Thread(
            target=_run_conversation_background,
            args=(records, _session["log_path"], backend),
            daemon=True,
        )
        t.start()
        _session["task"] = t

    return JSONResponse({"status": "session_started", "records": len(records)}, status_code=200)


# ── Background task ───────────────────────────────────────────────────────────
def _generate_session_name(namer: object, cluster_names: list[str]) -> str:
    """Generate a short human-readable session name from cluster names using the LLM."""
    import re, time
    try:
        clusters_str = ", ".join(cluster_names[:8])
        prompt = (
            f"Given these cluster names from a dataset: {clusters_str}\n\n"
            "Generate a very short session title (3-6 words) that captures the main theme. "
            "Respond with ONLY the title, no punctuation, no quotes."
        )
        if hasattr(namer, '_client') and hasattr(namer._client, 'messages'):
            # Anthropic
            response = namer._client.messages.create(
                model="claude-haiku-4-5", max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        elif hasattr(namer, '_client') and hasattr(namer._client, 'chat'):
            # OpenAI/Groq
            response = namer._client.chat.completions.create(
                model=namer._model, max_completion_tokens=32,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        else:
            # Google
            response = namer._client.models.generate_content(
                model=namer._model, contents=prompt
            )
            return response.text.strip()
    except Exception as e:
        print(f"[session_name] Failed: {e}")
        return ""

def _run_conversation_background(records: list[dict], log_path: str, backend_name: str = "hdbscan") -> None:
    """
    Runs the full conversation pipeline in a background thread.

    Steps:
    1. Compute embeddings (EmbeddingStore.compute_and_save)
    2. Build initial ClusteringState (build_initial_clustering_state)
    3. Construct oracle, namer, client
    4. Call run_conversation — emits state_update each turn via SocketIOEmitter

    The emitter bridges worker-thread → asyncio event loop via
    run_coroutine_threadsafe (see SocketIOEmitter). No monkey-patching involved.
    """
    import numpy as np

    from src.cluster_naming import AnthropicClusterNamer, GoogleClusterNamer, OpenAIClusterNamer
    from src.clustering import HDBSCANBackend, KMeansBackend, build_initial_clustering_state
    from src.conversation_loop import run_conversation
    from src.embedding_store import EmbeddingStore
    from src.llm_key import resolve_llm_key
    from src.oracle_protocol import MockOracle, OracleReply
    from src.stopping import StoppingCriteria

    provider, api_key = resolve_llm_key()
    if provider == "anthropic":
        import anthropic
        namer = AnthropicClusterNamer(anthropic.Anthropic(api_key=api_key))
    elif provider == "openai":
        namer = OpenAIClusterNamer(api_key=api_key)
    else:
        assert provider == "google"
        namer = GoogleClusterNamer(api_key=api_key)

    # Create session directory (D-26)
    session_ts = _make_session_timestamp()
    session_dir = os.path.join(SESSIONS_DIR, session_ts)
    os.makedirs(session_dir, exist_ok=True)
    _session["session_dir"] = session_dir

    # Use session-scoped audit log (replaces module-level "audit_log.jsonl")
    session_log_path = os.path.join(session_dir, "audit_log.jsonl")

    time.sleep(1.0)
    _session["progress"] = {"stage": "embeddings", "pct": 0, "msg": "Computing embeddings..."}
    emitter.emit("progress_update", _session["progress"])
    texts = [r["text"] for r in records]
    os.makedirs("embeddings", exist_ok=True)
    _cached_path = "embeddings/embeddings.npy"
    _session_emb_path = "embeddings/session_embeddings.npy"
    if os.path.exists(_cached_path):
        import numpy as _np
        _cached_shape = _np.load(_cached_path, mmap_mode="r").shape
        if _cached_shape[0] == len(texts):
            print(f"[startup] Reusing cached embeddings from {_cached_path} (shape: {_cached_shape})")
            shutil.copy2(_cached_path, _session_emb_path)
            store = EmbeddingStore.load(_session_emb_path)
        else:
            store = EmbeddingStore.compute_and_save(texts, _session_emb_path)
    else:
        store = EmbeddingStore.compute_and_save(texts, _session_emb_path)

    # Copy embeddings to session dir for future resume (D-26)
    session_embeddings_path = os.path.join(session_dir, "embeddings.npy")
    shutil.copy2(_session_emb_path, session_embeddings_path)
    _session["progress"] = {"stage": "embeddings", "pct": 100, "msg": "Embeddings ready"}
    emitter.emit("progress_update", _session["progress"])

    # D-18: instantiate the backend selected via --backend CLI flag
    if backend_name == "kmeans":
        backend = KMeansBackend()
    elif backend_name == "hdbscan":
        backend = HDBSCANBackend()
    else:
        assert False, f"Unknown backend: {backend_name!r}"

    _session["progress"] = {"stage": "clustering", "pct": 0, "msg": "Clustering in progress..."}
    emitter.emit("progress_update", _session["progress"])
    t0 = time.perf_counter()
    initial_state = build_initial_clustering_state(
        store.get_all(), records, namer, backend=backend
    )
    print(f"[timing] build_initial_clustering_state: {time.perf_counter() - t0:.2f}s")
    _session["progress"] = {"stage": "clustering", "pct": 100, "msg": "Clustering complete"}
    emitter.emit("progress_update", _session["progress"])
    _session["state"] = initial_state

    # Generate human-readable session name
    cluster_names = [c.name for c in initial_state.clusters]
    session_name = _generate_session_name(namer, cluster_names)
    if session_name:
        with open(os.path.join(session_dir, "name.txt"), "w", encoding="utf-8") as _f:
            _f.write(session_name)
        print(f"[session] Name: {session_name}")

    # D-17: log chosen K to session-scoped audit_log.jsonl as backend_init event so runs are reproducible
    if backend_name == "kmeans":
        _k_chosen = backend.k
        _backend_init_event = {
            "event": "backend_init",
            "backend": "kmeans",
            "k": _k_chosen,
            "turn": 0,
        }
        with open(session_log_path, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_backend_init_event) + "\n")
        print(f"[startup] KMeansBackend: K={_k_chosen} (selected via BIC on GMM)")
    else:
        _backend_init_event = {
            "event": "backend_init",
            "backend": "hdbscan",
            "k": len(initial_state.clusters),
            "turn": 0,
        }
        with open(session_log_path, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_backend_init_event) + "\n")
        print(f"[startup] HDBSCANBackend: K={len(initial_state.clusters)} clusters discovered")

    # Write initial state snapshot (D-27)
    _write_session_state(initial_state, session_dir)

    # Phase 4: Open DB connection for this run (D-04)
    # check_same_thread=False MANDATORY — this function runs in a background thread
    from src.db.connection import connect as _db_connect, init_schema as _db_init_schema
    from src.db.experiments import ExperimentCreate as _ExperimentCreate, create as _exp_create, update as _exp_update
    from src.db.experiments import ExperimentUpdate as _ExperimentUpdate
    from datetime import datetime as _dt, timezone as _tz

    _db_conn = _db_connect()  # opens experiments.db at repo root
    _db_init_schema(_db_conn)

    # Create experiment row at run start
    _exp_start_ts = _dt.now(_tz.utc).isoformat()
    _exp_create_model = _ExperimentCreate(
        name=f"interactive-{session_ts}",
        strategy_id="interactive",  # D-08: distinct from "random"/"uncertainty_driven"/"boundary_driven" strategy IDs
        persona_id="web_session",   # Phase 5 will replace with actual OracleSpec.persona_id
        seed=0,                     # Phase 5 will replace with actual experiment seed
        dataset=session_ts,         # session timestamp as dataset identifier until Phase 5 parameterizes
        start_timestamp=_exp_start_ts,
        details={},
    )
    _experiment = _exp_create(_db_conn, _exp_create_model)
    _experiment_id = _experiment.id

    id_to_text = {i: r["text"] for i, r in enumerate(records)}

    # Emit initial projection (D-22: once after initial clustering)
    _session["progress"] = {"stage": "umap", "pct": 0, "msg": "Computing UMAP projection..."}
    emitter.emit("progress_update", _session["progress"])
    compute_and_emit_projection(store, initial_state, emitter)
    _session["progress"] = {"stage": "umap", "pct": 100, "msg": "Projection ready"}
    emitter.emit("progress_update", _session["progress"])

    # Phase 2: use a neutral MockOracle (Phase 3 replaces with real Oracle Agent)
    neutral_reply = OracleReply(raw_text="", satisfied=False, turn_cognitive_load=0.0)
    oracle = MockOracle(script=[neutral_reply] * 15)

    def _per_turn_callback(new_state, deltas):
        # D-27: write state.json after every turn (same timing as JSONL write)
        _write_session_state(new_state, session_dir)
        # D-22: re-emit projection only on cluster-count changes (split/merge).
        if _should_recompute_projection(deltas):
            compute_and_emit_projection(store, new_state, emitter)

    final_state = run_conversation(
        initial_state=initial_state,
        oracle=oracle,
        store=store,
        namer=namer,
        strategy=None,
        log_path=session_log_path,
        criteria=StoppingCriteria(turn_budget=15),
        socketio=emitter,  # SocketIOEmitter satisfies the duck-typed interface
        id_to_text=id_to_text,
        llm_client=None,
        post_turn_callback=_per_turn_callback,
        db_conn=_db_conn,              # Phase 4: DB writes per turn (D-04)
        experiment_id=_experiment_id,  # Phase 4: FK for turn/feedback rows
        # pair_bag=None intentionally — new session starts with empty bag.
        # NOTE (D-21 / V-4-05): for RESUMED sessions, pair_bag must be reconstructed
        # from audit_log.jsonl + events.jsonl before passing here. Reconstruction is
        # Phase 5 scope. The deviation() call inside run_conversation() will flag resumed
        # DB sessions that start without a bag (pairwise_accuracy will be 0.0 until new
        # feedback arrives).
    )
    _session["state"] = final_state
    _write_session_state(final_state, session_dir)

    # Phase 4: Seal experiment row with end-of-run summary (D-04)
    _exp_update(_db_conn, _experiment_id, _ExperimentUpdate(
        total_turns=final_state.turn_index,
        end_timestamp=_dt.now(_tz.utc).isoformat(),
        details={"turns_to_convergence": final_state.turn_index},
    ))
    _db_conn.close()


# ── Upload parser helper ──────────────────────────────────────────────────────
def _detect_text_field(records: list[dict]) -> str:
    """
    Detect the text field by picking the string column with highest average length
    across a sample of records. Reliable for any text dataset regardless of column name.
    """
    TEXT_FIELDS_PRIORITY = ("text", "reviewText", "Description", "body", "content", "review", "comment")

    if not records:
        assert False, "No records to detect text field from"

    # First try known field names
    for key in TEXT_FIELDS_PRIORITY:
        if key in records[0]:
            return key

    # Sample up to 20 records and pick string field with highest average length
    sample = records[:20]
    string_fields = {
        key for key, val in sample[0].items()
        if isinstance(val, str)
    }

    best_key = None
    best_avg = 0.0
    for key in string_fields:
        avg_len = sum(len(r.get(key, "")) for r in sample) / len(sample)
        if avg_len > best_avg:
            best_avg = avg_len
            best_key = key

    assert best_key is not None, f"No string field found. Available fields: {list(records[0].keys())}"
    print(f"[parse] Auto-detected text field: '{best_key}' (avg length: {best_avg:.0f} chars)")
    return best_key

def _parse_upload(content: str, filename: str) -> list[dict]:
    """Parse CSV or JSONL upload into list of dicts with 'text' field.
    Auto-detects the text field by average string length if not a known field name.
    """
    import csv
    import json

    records_raw = []
    if filename.endswith(".jsonl"):
        for line in content.splitlines():
            line = line.strip()
            if line:
                records_raw.append(json.loads(line))
    else:
        reader = csv.DictReader(io.StringIO(content))
        records_raw = list(reader)

    assert records_raw, "Empty dataset"

    text_field = _detect_text_field(records_raw)

    return [
        {"item_id": i, "text": r[text_field]}
        for i, r in enumerate(records_raw)
        if r.get(text_field, "").strip()
    ]

# ── SocketIO connect handler ──────────────────────────────────────────────────

@sio.event
async def connect(sid, environ):
    # Re-emit current state to newly connected/reconnected client.
    import dataclasses
    state = _session.get("state")
    if state is not None:
        await sio.emit("state_update", {
            "turn_index": state.turn_index,
            "clusters": [dataclasses.asdict(c) for c in state.clusters],
            "soft_probs": {
                str(item_id): {
                    str(c.id): prob
                    for c, prob in zip(state.clusters, probs)
                }
                for item_id, probs in state.soft_probs.items()
            },
            "cognitive_load": 0.0,
            "pairwise_accuracy": 0.0,    # D-28: safe default for reconnect (no PairBag available here)
            "convergence_signal": None,  # D-28
            "contradiction_count": 0,    # D-28
        }, to=sid)

@sio.event
async def request_progress(sid):
    """Client asks for current progress on (re)connect."""
    progress = _session.get("progress")
    if progress and _session.get("state") is None:
        await sio.emit("progress_update", progress, to=sid)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:asgi_app", host="0.0.0.0", port=5001, reload=False)
