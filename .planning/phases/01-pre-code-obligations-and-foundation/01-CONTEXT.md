# Phase 1: Pre-Code Obligations and Foundation - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Lock all irreversible data and evaluation decisions (dataset split + stopping criteria specs), then build the foundational pipeline: text ingestion → sentence embeddings → HDBSCAN clustering → ClusteringState → JSONL AuditLog. By the end of this phase the system can produce a well-formed, serializable ClusteringState from raw text, and the held-out split is sealed with a hash file that the rest of the project must never touch.

</domain>

<decisions>
## Implementation Decisions

### Dataset & Split

- **D-01:** Use **Amazon Reviews 2023 — Arts, Crafts & Sewing** subset (~15K reviews). Download only this category; do not download the full dataset.
- **D-02:** Text field to embed and cluster: **`review_text` only**. Do not concatenate title or other fields.
- **D-03:** Held-out split ratio: **80% train / 20% held-out**. Split is stratified by nothing (random), seeded for reproducibility.
- **D-04:** At split time, write a **SHA-256 hash file** of the held-out file. System must verify this hash at startup and crash (assertion error) if it doesn't match. Hash file and held-out file are committed to the repo and never modified.

### HDBSCAN Library

- **D-05:** Use the **standalone `hdbscan` package** (`pip install hdbscan`), not `sklearn.cluster.HDBSCAN`. Reason: `soft_clusters_` gives a true per-point probability vector over all K clusters — required for `f_uncertainty` and soft-assignment calibration.
- **D-06:** Windows install issues (OpenMP / compiled extensions) are handled as they come — no sklearn fallback is planned. If the package fails to install, investigate and fix (conda, pre-built wheel, or WSL).
- **D-07:** The soft assignment for each item is the full `soft_clusters_` row (shape `[K]`, sums to 1.0). This is the canonical soft assignment stored in `ClusteringState.soft_probs`.

### Stopping Criteria (PRE-02 — code-ready specs)

All three conditions are OR-combined: any one firing stops the loop.

- **D-08:** **Oracle satisfaction** — primary: `OracleReply.satisfied = True` explicit token emitted by the oracle. Secondary (behavioral fallback): weighted feedback magnitude drops below threshold `ε` for `N_fallback` consecutive turns (values for ε and N_fallback are implementation decisions for Phase 4; Phase 1 only specifies the structure).
- **D-09:** **Turn budget** — hard cap of **15 turns**. At turn 15 the loop stops unconditionally, regardless of convergence state. Turn index is 0-based; the budget fires when `turn_index >= 15`.
- **D-10:** **Diminishing returns** — feedback magnitude is computed as a **weighted sum by feedback type**: global feedback > cluster-level > point-level > instructional (exact weights are Phase 4 decisions; Phase 1 specifies this is the weighting scheme, not the weights themselves). Magnitude near zero for N consecutive turns triggers this condition.

### ClusteringState Schema

- **D-11:** Item IDs are **sequential integers** assigned once at dataset load (0 to N−1). Cluster IDs are sequential integers assigned at cluster creation and **never reused** after a cluster is deleted.
- **D-12:** **Embeddings live in a separate read-only `EmbeddingStore`**, not inside `ClusteringState`. `ClusteringState` references items by ID only. The `EmbeddingStore` is computed once at session start and never mutated.
- **D-13:** `ClusteringState` fields (minimal + soft assignments):
  ```
  ClusteringState:
    turn_index: int
    timestamp: str (ISO 8601)
    clusters: list of Cluster
      Cluster:
        id: int
        name: str
        description: str
        item_ids: list[int]
    assignments: dict[int, int]       # item_id → cluster_id
    soft_probs: dict[int, list[float]] # item_id → probability vector [float × K]
  ```
- **D-14:** `ClusteringState` is serialized to **JSONL** every turn (one JSON object per line, one line per turn). Deserialization must reproduce the exact same object — no data loss. This is tested explicitly in Phase 1.

### Claude's Discretion

- Exact HDBSCAN hyperparameters (`min_cluster_size`, `min_samples`) — researcher/planner decides based on dataset size (~15K points).
- Embedding batch size for `all-mpnet-base-v2` inference — choose for laptop memory constraints.
- Exact values for `ε` (magnitude threshold) and `N_fallback` (consecutive turns) in stopping criteria — deferred to Phase 4.
- JSONL file location and rotation policy — planner decides.

### Folded Todos (from STATE.md)

- **Select primary dataset and lock held-out split with hash** — folded: D-01 through D-04 above.
- **Write the three stopping criteria as code-ready specifications** — folded: D-08 through D-10 above.
- **Decide sklearn HDBSCAN vs. standalone `hdbscan` package at Phase 1 gate** — folded: D-05 through D-07 above.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning Artifacts
- `.planning/REQUIREMENTS.md` — PRE-01, PRE-02, FOUND-01 through FOUND-04 define the 6 Phase 1 requirements in full
- `.planning/ROADMAP.md` §Phase 1 — success criteria (4 items) that define done
- `.planning/PROJECT.md` — dataset-agnostic constraint, fail-loudly coding philosophy, architecture constraints

### Reference Scripts (review before implementing)
- `Conversational Clustering Script.txt` — prototype/reference; review for existing design patterns before writing ClusteringState
- `Multi Agent Personalities Script.txt` — not relevant to Phase 1; defer to Phase 3

### No external specs
No ADRs or external specs exist yet. Requirements are fully captured in decisions above and REQUIREMENTS.md.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — no code exists yet. This is the first phase.

### Established Patterns
- **Fail loudly:** No `try/except` outside CLI entry point and API calls. Use `assert` freely. Let unexpected state crash immediately. This applies to hash verification, state deserialization, and embedding loading.
- **Dataset-agnostic:** All code must accept any text corpus via a configurable field name — `review_text` is the v1 default, not a hardcoded string.

### Integration Points
- `EmbeddingStore` is the boundary between Phase 1 (foundation) and Phase 2 (Clustering Agent). Phase 2 reads from it; Phase 1 writes it once.
- `ClusteringState` schema (D-13) is the contract every subsequent phase depends on — any field change after Phase 1 is a breaking change.
- JSONL AuditLog written here is consumed by Phase 4 (Judge Agent metrics) and Phase 5 (bootstrap CIs).

</code_context>

<specifics>
## Specific Ideas

- Dataset: Amazon Reviews 2023 Arts, Crafts & Sewing — not a generic "some reviews dataset". Planner should find the exact Hugging Face dataset ID or download URL.
- Turn budget of 15 is intentionally tight — chosen to force efficient interaction strategies in ablation experiments (Phase 5).
- Weighted feedback magnitude scheme (D-10) should be documented as a named formula in code, not an inline magic number.

</specifics>

<deferred>
## Deferred Ideas

- **LangGraph vs. plain Python orchestrator** — flagged in STATE.md as unresolved before Phase 2. Not a Phase 1 decision; planner for Phase 2 must resolve this.
- **Oracle cognitive-load weight parameters** — Phase 3 decision.
- **Exact ε and N_fallback values** for stopping criteria — Phase 4 decision (Judge Agent).
- **Human validation study protocol** — must be written before Phase 6, but not in Phase 1 scope.

</deferred>

---

*Phase: 1-Pre-Code Obligations and Foundation*
*Context gathered: 2026-05-04*
