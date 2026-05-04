---
phase: 1
slug: pre-code-obligations-and-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pytest.ini or pyproject.toml — Wave 0 installs |
| **Quick run command** | `pytest tests/phase1/ -x -q` |
| **Full suite command** | `pytest tests/phase1/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/phase1/ -x -q`
- **After every plan wave:** Run `pytest tests/phase1/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | PRE-01 | — | Hash mismatch raises AssertionError | unit | `pytest tests/phase1/test_dataset.py::test_hash_mismatch -x -q` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | PRE-01 | — | Split is 80/20 and reproducible | unit | `pytest tests/phase1/test_dataset.py::test_split_ratio -x -q` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | PRE-02 | — | Stopping criteria spec constants are present and typed | unit | `pytest tests/phase1/test_stopping_criteria.py -x -q` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 2 | FOUND-01 | — | EmbeddingStore loads from dataset, shape is (N, 768) | unit | `pytest tests/phase1/test_embedding_store.py::test_shape -x -q` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 2 | FOUND-01 | — | EmbeddingStore is read-only (mutation raises) | unit | `pytest tests/phase1/test_embedding_store.py::test_readonly -x -q` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 2 | FOUND-02 | — | Initial clustering assigns all items to clusters | unit | `pytest tests/phase1/test_clustering.py::test_all_items_assigned -x -q` | ❌ W0 | ⬜ pending |
| 1-03-02 | 03 | 2 | FOUND-03 | — | soft_probs shape is (N, K), sums to 1.0 per row | unit | `pytest tests/phase1/test_clustering.py::test_soft_probs_shape -x -q` | ❌ W0 | ⬜ pending |
| 1-04-01 | 04 | 3 | FOUND-04 | — | ClusteringState serializes and deserializes with no data loss | unit | `pytest tests/phase1/test_serialization.py::test_roundtrip -x -q` | ❌ W0 | ⬜ pending |
| 1-04-02 | 04 | 3 | FOUND-04 | — | Integer dict keys are preserved after deserialization | unit | `pytest tests/phase1/test_serialization.py::test_int_keys -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/phase1/__init__.py` — test package init
- [ ] `tests/phase1/test_dataset.py` — stubs for PRE-01 (hash lock, split ratio)
- [ ] `tests/phase1/test_stopping_criteria.py` — stubs for PRE-02 (criteria spec)
- [ ] `tests/phase1/test_embedding_store.py` — stubs for FOUND-01 (shape, read-only)
- [ ] `tests/phase1/test_clustering.py` — stubs for FOUND-02/FOUND-03 (assignments, soft_probs)
- [ ] `tests/phase1/test_serialization.py` — stubs for FOUND-04 (round-trip, int keys)
- [ ] `tests/conftest.py` — shared fixtures (sample dataset slice, tiny ClusteringState)
- [ ] `pytest` install — `pip install pytest`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LLM-generated cluster names are coherent natural language | FOUND-02 | LLM output is non-deterministic; quality is subjective | Run pipeline on 100-item slice, inspect printed cluster names |
| `all_points_membership_vectors()` returns valid data with `prediction_data=True` | FOUND-03 | Requires live HDBSCAN fit on real embeddings | Run `python -m src.clustering.initial` and assert no AttributeError |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
