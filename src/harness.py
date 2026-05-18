"""
src/harness.py — N×M×K ablation harness (ALAB-02, EXP-V2-01 partial).

Orchestrates parallel execution of the {strategies × personas × seeds}
cross-product using ThreadPoolExecutor. Each combo runs as its own
experiment with oracle_type='llm' (D-11). Per-worker DB connections per
Phase 4 D-04 + Phase 5 D-08 (WAL mode, check_same_thread=False).

Fail-loudly contract (CLAUDE.md): NO try/except around fut.result().
as_completed() surfaces worker exceptions immediately.

Pydantic stays inside src/db/ (Phase 4 D-10) — config parsing here uses
plain dataclasses (OracleSpec, NoiseParams come from src/oracle_agent.py).

KEY DESIGN DECISIONS (per checker revision):
- B-02 fix: production path uses a REAL OracleAgent(spec, noise, client=
  <anthropic.Anthropic>) per combo. HARNESS_DRY_RUN=1 env flag swaps in
  MockOracle for CI/unit-test mode. Default is the REAL oracle.
- W-01 fix: run_baseline is invoked exactly once per (persona, seed) pair
  (deduplicated across strategies — baseline does not depend on strategy
  choice). Without this the D-14 'no_dialogue (baseline)' table row is
  empty.
- W-04 fix: the embedding store is computed ONCE in the orchestrator
  thread and reused across every combo. Without this, 27 workers each
  re-embed the same dataset, multiplying cost by ~27x.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.logging_setup import deviation
from src.strategy import (
    BoundaryDrivenStrategy,
    RandomStrategy,
    UncertaintyDrivenStrategy,
)

if TYPE_CHECKING:
    import sqlite3
    import numpy as np
    from src.db.experiments import ExperimentRead
    from src.embedding_store import EmbeddingStore


# Module-level strategy registry (CONTEXT.md specifics line 161, D-09).
STRATEGY_REGISTRY: dict[str, type] = {
    "random": RandomStrategy,
    "uncertainty_driven": UncertaintyDrivenStrategy,
    "boundary_driven": BoundaryDrivenStrategy,
}


@dataclass
class _Combo:
    strategy_id: str
    persona_id: str
    seed: int


def _is_dry_run() -> bool:
    """HARNESS_DRY_RUN=1 swaps in MockOracle for CI/test runs without API keys."""
    return os.environ.get("HARNESS_DRY_RUN", "") == "1"


def _build_llm_client():
    """
    Build a provider-agnostic LLM client tuple via src.llm_call.build_client.

    Returns None when HARNESS_DRY_RUN=1 — caller MUST then use MockOracle.
    Returns (provider, sdk_client) otherwise; OracleAgent + chat() consume the
    tuple directly. Supported providers are listed in src.llm_call.DEFAULT_MODELS.
    """
    if _is_dry_run():
        return None

    # Lazy import keeps `python -m examples.run_harness --help` fast.
    from src.llm_call import build_client
    from src.llm_key import resolve_llm_key

    return build_client(*resolve_llm_key())


def _build_oracle(spec, noise, llm_client, events_path=None):
    """
    Construct the oracle for one combo (B-02 fix).

    Production path: REAL OracleAgent(spec, noise, client=<provider tuple>).
    The client tuple shape is produced by src.llm_call.build_client.
    HARNESS_DRY_RUN=1 path: MockOracle(satisfied=True). The dry-run path
    produces 1-turn convergence for every combo (useful only for CI smoke
    tests; the resulting rows have no statistical power).
    """
    if llm_client is None:
        # Dry-run path — MockOracle satisfied=True on turn 0. Fires deviation()
        # so the row in experiments.db is visibly marked as a dry-run artifact.
        from src.oracle_protocol import MockOracle, OracleReply
        deviation(
            "harness running with MockOracle (HARNESS_DRY_RUN=1) — "
            "rows have no statistical power; real LLM runs require an LLM API key",
        )
        return MockOracle(script=[
            OracleReply(raw_text="", satisfied=True, turn_cognitive_load=0.0)
        ])

    # Production path — real LLM oracle.
    from src.oracle_agent import OracleAgent
    return OracleAgent(
        spec=spec,
        noise_params=noise,
        client=llm_client,
        events_path=events_path,
    )


def _parse_personas(config: dict) -> dict[str, tuple]:
    """
    Convert YAML `personas:` section into {name: (OracleSpec, NoiseParams)}.

    Lazy import of dataclasses to keep module import cheap; OracleAgent
    instantiation happens later in _build_oracle.
    """
    from src.oracle_agent import NoiseParams, OracleSpec

    personas_cfg = config.get("personas", {})
    assert personas_cfg, "harness config has no `personas:` section"
    out: dict[str, tuple] = {}
    for name, body in personas_cfg.items():
        spec = OracleSpec(
            preferred_k=int(body["preferred_k"]),
            semantic_axes=list(body["semantic_axes"]),
            persona_description=str(body["persona_description"]),
        )
        noise = NoiseParams(
            consistency_rate=float(body["noise"]["consistency_rate"]),
            drift_probability=float(body["noise"]["drift_probability"]),
            sycophancy_resistance=float(body["noise"]["sycophancy_resistance"]),
        )
        out[name] = (spec, noise)
    return out


def _build_combos(harness_cfg: dict) -> list[_Combo]:
    """
    Compute cross-product of strategies × personas × seeds, minus the exclude list.

    exclude entries are dicts: {strategy: str, persona: str, seed: int}.
    Missing exclude key or empty list = no exclusions.
    """
    strategies = harness_cfg["strategies"]
    personas = harness_cfg["personas"]
    seeds = harness_cfg["seeds"]
    exclude_raw = harness_cfg.get("exclude") or []
    exclude_set = {
        (e["strategy"], e["persona"], int(e["seed"])) for e in exclude_raw
    }

    combos: list[_Combo] = []
    for s in strategies:
        assert s in STRATEGY_REGISTRY, (
            f"Unknown strategy '{s}' — not in STRATEGY_REGISTRY ({list(STRATEGY_REGISTRY)})"
        )
        for p in personas:
            for seed in seeds:
                if (s, p, int(seed)) in exclude_set:
                    deviation(
                        "harness combo excluded by config",
                        strategy=s, persona=p, seed=int(seed),
                    )
                    continue
                combos.append(_Combo(strategy_id=s, persona_id=p, seed=int(seed)))
    return combos


def _load_dataset(dataset_path: str) -> list[dict]:
    """Same record-load pattern as examples/run_baseline.py lines 34-49."""
    import json
    records: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" not in rec:
                assert "reviewText" in rec or "body" in rec or "content" in rec, (
                    f"Record {i} has no 'text' field: {list(rec.keys())}"
                )
                text_key = next(k for k in ("reviewText", "body", "content") if k in rec)
                rec = {"item_id": i, "text": rec[text_key]}
            else:
                rec["item_id"] = i
            records.append(rec)
    assert records, f"Dataset {dataset_path} is empty"
    return records


def _build_embedding_store(records: list[dict]) -> "EmbeddingStore":
    """
    Compute the embedding store ONCE in the orchestrator thread (W-04 fix).

    Without this, 27 worker threads would each re-embed the same dataset.
    The returned EmbeddingStore is read-only (numpy array in memory) and
    safe to share across worker threads.

    The on-disk .npy file is intentionally NOT unlinked — the harness
    treats it as a session cache. Cleanup is the developer's concern.
    """
    from src.embedding_store import EmbeddingStore
    import tempfile

    texts = [r["text"] for r in records]
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        cache_path = tmp.name
    # NOTE: deliberately NOT unlinked — see docstring. Plan 05 analysis layer
    # does not read it; OS temp cleanup handles eviction.
    return EmbeddingStore.compute_and_save(texts, cache_path)


def _run_one(
    combo: _Combo,
    config: dict,
    personas: dict[str, tuple],
    records: list[dict],
    embedding_store: "EmbeddingStore",
    llm_client,
) -> "ExperimentRead":
    """
    Run ONE interactive combo: open a per-thread DB connection, build a REAL
    OracleAgent + strategy, write the experiments row tagged oracle_type='llm',
    invoke run_conversation with the real llm_client threaded through, seal
    the row, return ExperimentRead.

    Per Phase 4 D-04 + Phase 5 D-08: caller does NOT share connections
    across threads. This function owns the connection lifecycle inside
    its thread.

    W-04 fix: receives the pre-computed embedding_store from the orchestrator;
    does NOT recompute embeddings.
    B-02 fix: receives a real llm_client and builds a real OracleAgent (or
    MockOracle when llm_client is None — HARNESS_DRY_RUN=1).
    """
    # Lazy imports — heavy modules resolved per-thread (mirrors run_baseline pattern)
    from src.clustering import HDBSCANBackend, build_initial_clustering_state
    from src.conversation_loop import run_conversation
    from src.db import experiments as exp_db
    from src.db.connection import connect, init_schema
    from src.db.experiments import ExperimentCreate, ExperimentUpdate
    from src.stopping import StoppingCriteria

    spec, noise = personas[combo.persona_id]

    # Stopping criteria from config
    stopping_cfg = config.get("stopping", {})
    criteria = StoppingCriteria(
        magnitude_threshold_epsilon=stopping_cfg.get("epsilon", 0.05),
        magnitude_fallback_turns=stopping_cfg.get("n_fallback", 3),
    )

    dataset_path = config["harness"]["dataset"]
    log_dir = config["harness"].get("log_dir", "sessions")
    run_slug = f"{combo.strategy_id}-{combo.persona_id}-seed{combo.seed}"
    session_dir = os.path.join(
        log_dir,
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f") + "-" + run_slug,
    )
    os.makedirs(session_dir, exist_ok=True)
    log_path = os.path.join(session_dir, "audit_log.jsonl")
    events_path = os.path.join(session_dir, "events.jsonl")

    db = connect()
    try:
        init_schema(db)

        start_ts = datetime.now(timezone.utc).isoformat()
        exp_create = ExperimentCreate(
            name=run_slug,
            strategy_id=combo.strategy_id,
            persona_id=combo.persona_id,
            seed=combo.seed,
            dataset=dataset_path,
            oracle_type="llm",   # D-11
            start_timestamp=start_ts,
            details={
                "noise": {
                    "consistency_rate": noise.consistency_rate,
                    "drift_probability": noise.drift_probability,
                    "sycophancy_resistance": noise.sycophancy_resistance,
                },
                "preferred_k": spec.preferred_k,
                "semantic_axes": spec.semantic_axes,
                "dry_run": llm_client is None,
            },
        )
        experiment = exp_db.create(db, exp_create)

        # Build initial clustering using the SHARED embedding_store (W-04 fix).
        id_to_text = {r["item_id"]: r["text"] for r in records}

        # _NullNamer mirrors run_baseline (D-22, D-26 — no LLM cluster naming in harness runs)
        class _NullNamer:
            def name_cluster(self, items, cluster_id=None):
                return {"name": f"Cluster {cluster_id}", "description": ""}

            def describe_cluster(self, items, name, cluster_id=None):
                return ""

        backend = HDBSCANBackend()
        initial_state = build_initial_clustering_state(
            embedding_store.get_all(), records, _NullNamer(), backend=backend
        )

        # Build strategy from registry (seeded for determinism)
        strategy_cls = STRATEGY_REGISTRY[combo.strategy_id]
        strategy = strategy_cls(seed=combo.seed)

        # Build the oracle: REAL OracleAgent (production) or MockOracle (HARNESS_DRY_RUN=1).
        # B-02 fix: production path MUST use the real oracle.
        oracle = _build_oracle(spec, noise, llm_client, events_path=events_path)

        # Run the conversation loop. DB writes per turn are handled by run_conversation
        # via db_conn + experiment_id (Phase 4 D-03). Final sealing happens below.
        # llm_client is threaded through so parse_feedback runs against the real model.
        final_state = run_conversation(
            initial_state=initial_state,
            oracle=oracle,
            store=embedding_store,
            namer=_NullNamer(),
            strategy=strategy,
            log_path=log_path,
            criteria=criteria,
            socketio=None,
            id_to_text=id_to_text,
            llm_client=llm_client,
            db_conn=db,
            experiment_id=experiment.id,
        )

        end_ts = datetime.now(timezone.utc).isoformat()
        # Total turns = final_state.turn_index + 1 (turn_index is 0-based after build_initial)
        total_turns = final_state.turn_index + 1
        sealed = exp_db.update(db, experiment.id, ExperimentUpdate(
            total_turns=total_turns,
            end_timestamp=end_ts,
            details={
                "turns_to_convergence": total_turns,
                "session_dir": session_dir,
            },
        ))
        return sealed
    finally:
        db.close()


def _run_baseline_for_combo(
    persona_id: str,
    seed: int,
    persona_data: tuple,
    dataset_path: str,
    records: list[dict],
    criteria,
    llm_client,
) -> "ExperimentRead":
    """
    W-01 fix: invoke run_baseline once per (persona, seed) pair.

    Builds the same OracleAgent the interactive path uses (or MockOracle
    under HARNESS_DRY_RUN=1) so the baseline row reflects a real one-turn
    oracle reply, not just satisfied=True. Owns its own DB connection
    (per-thread, D-08).
    """
    from src.db.connection import connect, init_schema
    from src.judge import run_baseline

    spec, noise = persona_data
    oracle = _build_oracle(spec, noise, llm_client, events_path=None)

    db = connect()
    try:
        init_schema(db)
        return run_baseline(
            records=records,
            persona_id=persona_id,
            seed=seed,
            db=db,
            oracle=oracle,
            llm_client=llm_client,
            dataset_name=dataset_path,
            criteria=criteria,
        )
    finally:
        db.close()


def run_harness(
    config_path: str,
    db: "sqlite3.Connection | None" = None,
) -> list["ExperimentRead"]:
    """
    Run the full {strategies × personas × seeds} cross-product (ALAB-02)
    PLUS the per-(persona, seed) no-dialogue baseline rows (W-01 fix).

    Args:
        config_path: Path to harness.yaml (D-09 shape).
        db:          Optional orchestrator-level connection for tests that want to
                     pre-seed schema. Per-worker connections are ALWAYS opened
                     independently via connect() (D-08).

    Returns:
        list[ExperimentRead] — every interactive combo row PLUS every baseline
        row (one per persona × seed). The Plan 05 analysis layer groups by
        strategy_id (interactive: random/uncertainty_driven/boundary_driven;
        baseline: no_dialogue) when producing the D-14 table.

    Fail-loudly: NO try/except around fut.result(); worker exceptions propagate
    through as_completed() immediately.
    """
    import yaml

    from src.db.connection import connect, init_schema
    from src.stopping import StoppingCriteria

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "harness" in config, f"{config_path} missing top-level `harness:` section"

    # Ensure the schema exists at least once before workers race (each worker also
    # calls init_schema, which is idempotent — safe regardless).
    owned_orchestrator_db = False
    if db is None:
        db = connect()
        owned_orchestrator_db = True
    try:
        init_schema(db)
    finally:
        if owned_orchestrator_db:
            db.close()

    # Load dataset once in the orchestrator thread — workers share the list (read-only).
    dataset_path = config["harness"]["dataset"]
    records = _load_dataset(dataset_path)
    personas = _parse_personas(config)
    combos = _build_combos(config["harness"])
    assert combos, "harness cross-product produced 0 combos (check exclude list)"

    # W-04 fix: compute the embedding store ONCE; share across all workers.
    embedding_store = _build_embedding_store(records)

    # B-02 fix: build the real llm_client ONCE; share across all workers.
    # anthropic.Anthropic is thread-safe per its docs.
    llm_client = _build_llm_client()

    # Stopping criteria for baseline runs (matches _run_one).
    stopping_cfg = config.get("stopping", {})
    criteria = StoppingCriteria(
        magnitude_threshold_epsilon=stopping_cfg.get("epsilon", 0.05),
        magnitude_fallback_turns=stopping_cfg.get("n_fallback", 3),
    )

    # W-01 fix: deduplicate baseline jobs across strategies — one per (persona, seed).
    baseline_keys = sorted({(c.persona_id, c.seed) for c in combos})

    max_workers = int(config["harness"].get("max_workers", 4))

    results: list = []
    # Fail-loudly: NO try/except around fut.result(). as_completed() raises immediately.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        interactive_futures = {
            pool.submit(
                _run_one, combo, config, personas, records, embedding_store, llm_client,
            ): combo
            for combo in combos
        }
        baseline_futures = {
            pool.submit(
                _run_baseline_for_combo,
                persona_id, seed, personas[persona_id],
                dataset_path, records, criteria, llm_client,
            ): (persona_id, seed)
            for (persona_id, seed) in baseline_keys
        }
        all_futures = {**interactive_futures, **baseline_futures}
        for fut in as_completed(all_futures):
            results.append(fut.result())
    return results
