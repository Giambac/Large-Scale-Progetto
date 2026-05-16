"""Tests for src/harness.py (ALAB-02, D-07..D-11, plus B-02/W-01/W-04 fixes)."""
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from src.harness import (
    STRATEGY_REGISTRY,
    _Combo,
    _build_combos,
    _build_oracle,
    _is_dry_run,
    _parse_personas,
    run_harness,
)


def test_strategy_registry_keys():
    assert set(STRATEGY_REGISTRY.keys()) == {
        "random", "uncertainty_driven", "boundary_driven",
    }


def test_build_combos_full_cross_product():
    cfg = {
        "strategies": ["random", "uncertainty_driven", "boundary_driven"],
        "personas": ["curious", "skeptical", "drifty"],
        "seeds": [1, 2, 3],
        "exclude": [],
    }
    combos = _build_combos(cfg)
    assert len(combos) == 3 * 3 * 3   # D-07
    # Spot-check determinism: combos should follow strategies × personas × seeds order
    assert combos[0] == _Combo("random", "curious", 1)


def test_build_combos_exclude_drops_specific_combo():
    cfg = {
        "strategies": ["random", "uncertainty_driven"],
        "personas": ["curious"],
        "seeds": [1, 2],
        "exclude": [{"strategy": "random", "persona": "curious", "seed": 2}],
    }
    combos = _build_combos(cfg)
    assert len(combos) == 2 * 1 * 2 - 1
    assert _Combo("random", "curious", 2) not in combos
    assert _Combo("random", "curious", 1) in combos
    assert _Combo("uncertainty_driven", "curious", 2) in combos


def test_build_combos_rejects_unknown_strategy():
    cfg = {
        "strategies": ["random", "ghost_strategy"],
        "personas": ["curious"],
        "seeds": [1],
        "exclude": [],
    }
    with pytest.raises(AssertionError, match="ghost_strategy"):
        _build_combos(cfg)


def test_parse_personas_returns_dataclasses():
    cfg = {
        "personas": {
            "curious": {
                "preferred_k": 5,
                "semantic_axes": ["topic"],
                "persona_description": "test",
                "noise": {
                    "consistency_rate": 0.9,
                    "drift_probability": 0.05,
                    "sycophancy_resistance": 0.7,
                },
            },
        },
    }
    from src.oracle_agent import NoiseParams, OracleSpec

    out = _parse_personas(cfg)
    assert "curious" in out
    spec, noise = out["curious"]
    assert isinstance(spec, OracleSpec)
    assert isinstance(noise, NoiseParams)
    assert spec.preferred_k == 5
    assert noise.consistency_rate == 0.9


def test_threadpool_propagates_worker_exception():
    """
    Fail-loudly contract (CLAUDE.md, D-08): a worker exception must surface
    immediately via fut.result(). No try/except around as_completed loop.
    """
    def _boom(_):
        raise RuntimeError("simulated worker failure")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_boom, i) for i in range(4)]
        with pytest.raises(RuntimeError, match="simulated worker failure"):
            for fut in as_completed(futures):
                fut.result()


# ── B-02: real oracle by default; MockOracle only under HARNESS_DRY_RUN=1 ──

def test_dry_run_flag_swaps_in_mock_oracle(monkeypatch):
    monkeypatch.setenv("HARNESS_DRY_RUN", "1")
    assert _is_dry_run() is True

    from src.oracle_agent import NoiseParams, OracleSpec
    from src.oracle_protocol import MockOracle

    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="t")
    noise = NoiseParams(consistency_rate=0.9, drift_probability=0.05, sycophancy_resistance=0.7)
    oracle = _build_oracle(spec, noise, llm_client=None)
    assert isinstance(oracle, MockOracle)


def test_real_oracle_constructed_when_client_provided():
    """When llm_client is provided (production path), _build_oracle returns a real OracleAgent."""
    from src.oracle_agent import NoiseParams, OracleAgent, OracleSpec

    spec = OracleSpec(preferred_k=3, semantic_axes=["topic"], persona_description="t")
    noise = NoiseParams(consistency_rate=0.9, drift_probability=0.05, sycophancy_resistance=0.7)
    fake_client = object()  # OracleAgent.__init__ doesn't call the client; type-irrelevant here.
    oracle = _build_oracle(spec, noise, llm_client=fake_client)
    assert isinstance(oracle, OracleAgent)


# ── W-01: run_baseline dedup — one row per (persona, seed) pair ────────────

def test_run_harness_invokes_baseline_once_per_persona_seed(tmp_path, monkeypatch):
    """
    W-01: 3 strategies × 2 personas × 2 seeds = 12 interactive combos but
    only 2 × 2 = 4 baseline rows (one per persona+seed). The harness MUST
    deduplicate baseline jobs across strategies.
    """
    monkeypatch.setenv("HARNESS_DRY_RUN", "1")

    cfg_path = tmp_path / "harness.yaml"
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text(
        '{"text": "a sample sentence"}\n{"text": "another sentence"}\n',
        encoding="utf-8",
    )
    cfg_path.write_text(
        f"""harness:
  strategies: [random, uncertainty_driven, boundary_driven]
  personas: [curious, skeptical]
  seeds: [1, 2]
  exclude: []
  max_workers: 1
  dataset: {dataset_path.as_posix()}
  log_dir: {(tmp_path / 'sessions').as_posix()}
personas:
  curious:
    preferred_k: 3
    semantic_axes: [topic]
    persona_description: "test"
    noise:
      consistency_rate: 0.9
      drift_probability: 0.05
      sycophancy_resistance: 0.7
  skeptical:
    preferred_k: 5
    semantic_axes: [topic]
    persona_description: "test"
    noise:
      consistency_rate: 0.75
      drift_probability: 0.05
      sycophancy_resistance: 0.9
stopping:
  epsilon: 0.05
  n_fallback: 3
""",
        encoding="utf-8",
    )

    from src.db.experiments import ExperimentRead

    def _fake_run_one(combo, config, personas, records, embedding_store, llm_client):
        return _row("random-or-similar", combo.persona_id, combo.seed, "llm",
                    f"{combo.strategy_id}-{combo.persona_id}-seed{combo.seed}")

    baseline_call_log: list[tuple] = []
    def _fake_baseline(persona_id, seed, persona_data, dataset_path, records, criteria, llm_client):
        baseline_call_log.append((persona_id, seed))
        return _row("no_dialogue", persona_id, seed, "llm",
                    f"no_dialogue-{persona_id}-seed{seed}")

    def _row(strategy_id, persona_id, seed, oracle_type, slug):
        return ExperimentRead(
            id=len(baseline_call_log) + 100, name=slug, slug=slug,
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
            deleted_at=None,
            strategy_id=strategy_id, persona_id=persona_id, seed=seed,
            dataset=dataset_path.as_posix(),
            oracle_type=oracle_type, total_turns=1, convergence_reason=None,
            start_timestamp="2026-05-17T00:00:00+00:00",
            end_timestamp="2026-05-17T00:00:01+00:00",
            details={},
        )

    # Patch _build_embedding_store (avoid running real embedding model in tests).
    class _FakeStore:
        def get_all(self):
            import numpy as np
            return np.zeros((2, 4), dtype=np.float64)

    mem = sqlite3.connect(":memory:", check_same_thread=False)
    mem.row_factory = sqlite3.Row

    with patch("src.harness._run_one", side_effect=_fake_run_one), \
         patch("src.harness._run_baseline_for_combo", side_effect=_fake_baseline), \
         patch("src.harness._build_embedding_store", return_value=_FakeStore()):
        results = run_harness(str(cfg_path), db=mem)

    # 3 strategies × 2 personas × 2 seeds = 12 interactive combos
    # Plus 2 personas × 2 seeds = 4 baseline rows (NOT 12 — dedup)
    assert len(results) == 12 + 4
    assert len(baseline_call_log) == 4
    assert sorted(baseline_call_log) == sorted([
        ("curious", 1), ("curious", 2), ("skeptical", 1), ("skeptical", 2),
    ])
    mem.close()


# ── W-04: embedding store is computed ONCE in the orchestrator ─────────────

def test_embedding_store_built_once_in_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_DRY_RUN", "1")

    cfg_path = tmp_path / "harness.yaml"
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")
    cfg_path.write_text(
        f"""harness:
  strategies: [random]
  personas: [curious]
  seeds: [1, 2, 3]
  exclude: []
  max_workers: 1
  dataset: {dataset_path.as_posix()}
  log_dir: {(tmp_path / 'sessions').as_posix()}
personas:
  curious:
    preferred_k: 3
    semantic_axes: [topic]
    persona_description: "test"
    noise: {{consistency_rate: 0.9, drift_probability: 0.05, sycophancy_resistance: 0.7}}
stopping: {{epsilon: 0.05, n_fallback: 3}}
""",
        encoding="utf-8",
    )

    from src.db.experiments import ExperimentRead

    class _FakeStore:
        def get_all(self):
            import numpy as np
            return np.zeros((2, 4), dtype=np.float64)

    embed_call_count = {"n": 0}
    def _fake_build_store(records):
        embed_call_count["n"] += 1
        return _FakeStore()

    def _fake_run_one(combo, config, personas, records, embedding_store, llm_client):
        return ExperimentRead(
            id=combo.seed, name="x", slug="x",
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
            deleted_at=None,
            strategy_id=combo.strategy_id, persona_id=combo.persona_id, seed=combo.seed,
            dataset="d", oracle_type="llm", total_turns=1, convergence_reason=None,
            start_timestamp="2026-05-17T00:00:00+00:00",
            end_timestamp="2026-05-17T00:00:01+00:00",
            details={},
        )

    def _fake_baseline(persona_id, seed, *a, **kw):
        return ExperimentRead(
            id=seed + 1000, name="b", slug="b",
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
            deleted_at=None,
            strategy_id="no_dialogue", persona_id=persona_id, seed=seed,
            dataset="d", oracle_type="llm", total_turns=1, convergence_reason=None,
            start_timestamp="2026-05-17T00:00:00+00:00",
            end_timestamp="2026-05-17T00:00:01+00:00",
            details={},
        )

    mem = sqlite3.connect(":memory:", check_same_thread=False)
    mem.row_factory = sqlite3.Row

    with patch("src.harness._build_embedding_store", side_effect=_fake_build_store), \
         patch("src.harness._run_one", side_effect=_fake_run_one), \
         patch("src.harness._run_baseline_for_combo", side_effect=_fake_baseline):
        results = run_harness(str(cfg_path), db=mem)

    # Embedding store should be built EXACTLY ONCE (W-04), not 3x for 3 combos.
    assert embed_call_count["n"] == 1, (
        f"W-04 violation: _build_embedding_store called {embed_call_count['n']} times, expected 1"
    )
    mem.close()


def test_run_harness_writes_oracle_type_llm(tmp_path, monkeypatch):
    """
    Mock _run_one + _run_baseline_for_combo to avoid touching the embedding
    pipeline, but exercise the full orchestrator path: yaml load,
    cross-product expansion, ThreadPoolExecutor scheduling, ExperimentRead
    aggregation. The DB write contract for oracle_type='llm' is enforced
    in _run_one itself (verified by the fake row's oracle_type='llm' value).
    """
    monkeypatch.setenv("HARNESS_DRY_RUN", "1")

    cfg_path = tmp_path / "harness.yaml"
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text(
        '{"text": "a sample sentence"}\n{"text": "another sentence"}\n',
        encoding="utf-8",
    )
    cfg_path.write_text(
        f"""harness:
  strategies: [random]
  personas: [curious]
  seeds: [1]
  exclude: []
  max_workers: 1
  dataset: {dataset_path.as_posix()}
  log_dir: {(tmp_path / 'sessions').as_posix()}
personas:
  curious:
    preferred_k: 3
    semantic_axes: [topic]
    persona_description: "test"
    noise: {{consistency_rate: 0.9, drift_probability: 0.05, sycophancy_resistance: 0.7}}
stopping: {{epsilon: 0.05, n_fallback: 3}}
""",
        encoding="utf-8",
    )

    from src.db.experiments import ExperimentRead

    class _FakeStore:
        def get_all(self):
            import numpy as np
            return np.zeros((2, 4), dtype=np.float64)

    def _fake_run_one(combo, config, personas, records, embedding_store, llm_client):
        return ExperimentRead(
            id=1, name="random-curious-seed1", slug="random-curious-seed1",
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
            deleted_at=None,
            strategy_id=combo.strategy_id, persona_id=combo.persona_id, seed=combo.seed,
            dataset=config["harness"]["dataset"],
            oracle_type="llm",
            total_turns=1,
            convergence_reason=None,
            start_timestamp="2026-05-17T00:00:00+00:00",
            end_timestamp="2026-05-17T00:00:01+00:00",
            details={},
        )

    def _fake_baseline(persona_id, seed, *a, **kw):
        return ExperimentRead(
            id=999, name="no_dialogue-curious-seed1", slug="no_dialogue-curious-seed1",
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
            deleted_at=None,
            strategy_id="no_dialogue", persona_id=persona_id, seed=seed,
            dataset="d", oracle_type="llm", total_turns=1, convergence_reason=None,
            start_timestamp="2026-05-17T00:00:00+00:00",
            end_timestamp="2026-05-17T00:00:01+00:00",
            details={},
        )

    mem = sqlite3.connect(":memory:", check_same_thread=False)
    mem.row_factory = sqlite3.Row

    with patch("src.harness._build_embedding_store", return_value=_FakeStore()), \
         patch("src.harness._run_one", side_effect=_fake_run_one), \
         patch("src.harness._run_baseline_for_combo", side_effect=_fake_baseline):
        results = run_harness(str(cfg_path), db=mem)

    # 1 interactive + 1 baseline = 2 rows
    assert len(results) == 2
    for r in results:
        assert r.oracle_type == "llm"
    strategies = sorted(r.strategy_id for r in results)
    assert strategies == ["no_dialogue", "random"]
    mem.close()
