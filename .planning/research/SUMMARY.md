# Project Research Summary

**Project:** Conversational Clustering
**Milestone:** v2.0 — Experimentation Flexibility & Scale
**Domain:** Human-in-the-loop conversational clustering research system (subsequent-milestone additions)
**Researched:** 2026-05-21
**Confidence:** HIGH (stack + architecture code-grounded; features MEDIUM-HIGH; pitfalls HIGH for code-grounded items)

> Scope: this milestone ADDS capabilities to a shipped v1 system (3-agent loop, `feedback_parser`, `OracleAgent`, FastAPI+uvicorn+socketio UI, single-fit clustering). Research is scoped to the NEW v2 features only. The settled v1 design (pure `f_*` functions, single while-loop, read-only `EmbeddingStore`, audit-log-as-replay-source) is treated as ground truth and not re-derived. (FEATURES.md Part B is preserved v1 history and is out of scope for this synthesis.)

## Executive Summary

v2.0 turns a working single-fit conversational clusterer into a configurable, scalable research instrument. The shape is well understood because three technical researchers (stack, architecture, pitfalls) independently read the actual codebase and converged on the same conclusions: the milestone is mostly a sequence of careful **extensions to existing seams**, gated by **two foundational refactors** that everything else depends on. There is no greenfield system here — the work succeeds or fails on how cleanly each new feature grafts onto `run_conversation`, `EmbeddingStore`, `parse_feedback`, the `ClusteringBackend` Protocol, the `post_turn_callback` projection hook, and `ClusteringState` (the single source of truth).

The recommended approach is dependency-locked and non-negotiable in its ordering: **(1)** replace the hardcoded `EMBEDDING_DIM = 384` constant with a dynamic, manifest-/backend-carried dimension before any second embedding backend can exist; **(2)** define an artifact-provenance manifest schema (model, dim, normalized flag, library versions, input hash, n_items) that both Colab artifacts and the OpenAI/SentenceTransformer backends validate against on load; **(3)** make the UMAP projection recolor-not-refit (cache coords once, since coords are a fixed function of immutable embeddings); **(4)** reshape the loop to be oracle-initiated; **(5)** add per-query KMeans re-fit on fixed embeddings plus the query filter; **(6)** build the coordination agent last. Stack choices are deliberately conservative and reuse what is already installed — `huggingface_hub` for Colab-to-local artifact handoff, the already-imported `openai` SDK for embeddings, `pydantic` for config validation, and `ruamel.yaml` (the only genuinely new dependency) for round-trip YAML oracle configs.

The single linchpin risk that the whole back half of the milestone hinges on is **cluster-ID stability across per-query KMeans re-fits**. KMeans label assignment is arbitrary between fits; a naive re-fit churns every cluster ID, silently breaking merge/split history, cluster names, and the UMAP recolor (points jump colors for no reason). The fix is a Hungarian-style alignment step (`scipy.optimize.linear_sum_assignment` on item-set Jaccard / centroid cosine) that re-maps new clusters to prior IDs, combined with hard K-gating so re-fit NEVER re-optimizes K (the project's locked "K changes only via oracle intent" anti-feature). The recolor-not-refit UMAP phase hard-depends on this alignment landing first. The other concentrated risk is the coordination agent's cross-session state merge, which breaks every single-state/single-writer/single-log cardinality assumption and is explicitly flagged for its own research spike.

## Key Findings

### Recommended Stack

See [STACK.md](STACK.md). The strategy is "reuse what's installed, add almost nothing." Versions are pinned to what was probed live on the machine (2026-05-21), so recommendations are known-good. The one genuinely new dependency is `ruamel.yaml` (round-trip comment/order preservation for human-edited, version-controlled oracle configs); `PyYAML` is the fallback if the team rejects a new dep. Critically, the local env already runs the modern `huggingface_hub 1.7.1` + `sentence-transformers 5.4.1` pairing, so **no down-pinning is required** — but the stale `sentence-transformers>=2.7` floor in `requirements.txt` is misleading and must be bumped to `>=5.4`. Local torch is **CPU-only** (`2.11.0+cpu`) — this is the concrete motivation for offloading embedding compute to Colab GPU.

**Core technologies (new for v2.0):**
- `huggingface_hub` (>=1.7,<2): Colab-to-local artifact handoff via a private dataset repo — already installed transitively, gives versioned/authed/resumable transfer of `embeddings.npy` + `initial_state.json` + `manifest.json`.
- `openai` (>=2.26,<3): second embedding backend (`text-embedding-3-small`, 1536-dim) — **reuses the same `build_client("openai", key)` already in `src/llm_call.py`**; no new auth surface.
- `ruamel.yaml` (>=0.18,<0.19): versioned oracle config YAML — keeps git diffs clean; the only new install.
- `pydantic` (>=2.12, already present): validate parsed YAML into a typed `OracleConfig` at the boundary (fail-loudly).

**Explicitly forbidden:** `eventlet`/`gevent` (hard project rule — monkey-patches stdlib, corrupts numpy/sklearn/UMAP locking; the new embedding work touches numpy heavily); running FastAPI/socketio on Colab (Colab is compute-ONLY); re-hardcoding a new dim literal (e.g. `OPENAI_DIM = 1536`); storing oracle configs in `experiments.db` (locked: YAML in git).

### Expected Features

See [FEATURES.md](FEATURES.md) Part A. Five new features, each tagged EXTEND (modifies existing code) or NEW (new subsystem). The EXTEND/NEW split is the most actionable finding for sequencing: most of the milestone reuses existing code.

**EXTEND existing code (lower risk):**
- **Query filter** — EXTENDS `feedback_parser` + `_contradicts`; only genuinely new code is a `normalize(deltas)` pass (dedupe, latest-intent-wins within batch, compound-split). Do NOT build a parallel parser.
- **Versioned YAML oracle config** — EXTENDS `OracleAgent`, externalizes the currently-inline `_build_system_prompt`.
- **Interactive UMAP recolor** — EXTENDS the existing server-side projection + `state_update` emit; generalizes the coord cache already present in study/watch workers.
- **Real-time chat view** — EXTENDS `/study` + `/watch` routes; presentation layer over the existing event stream (never a parser-bypass side channel).

**Genuinely NEW subsystems (higher risk):**
- **Oracle-initiated onboarding + initial query** — NEW orchestration around the existing loop (dataset intro -> first query -> first clustering -> converge); keep "autonomous-first" as an ablation condition.
- **Coordination agent** — NEW orchestrator-worker layer (pairwise decompose -> N `run_conversation` sessions -> contradiction-validated recombine); HIGH complexity, deferred to last.

**Must have (table stakes for the milestone):** dataset introduction/summary; oracle initial query driving the first KMeans re-fit; query validated/normalized before clustering; live recolor view; real-time chat transcript; latest-intent-wins honored at query time.

**Anti-features to keep OUT:** re-embed-per-query (re-FIT only, embeddings fixed — locked); query filter that semantically "fixes"/second-guesses oracle intent (normalize structure only — the oracle IS ground truth); automatic K optimization triggered by query (K from oracle intent only); always-on parallel coordination (gate to genuinely decomposable ops); free-form chat that bypasses the parser/filter; eventlet/gevent.

### Architecture Approach

See [ARCHITECTURE.md](ARCHITECTURE.md). Every v2 feature hooks into one of five named existing seams: the `backend` parameter of `build_initial_clustering_state`; the `post_turn_callback(new_state, deltas)` hook; the `EMBEDDING_DIM` constant; the `parse_feedback(...) -> deltas` step; and `ClusteringState` as single source of truth. The two architectural pillars are a **clean artifact contract** (Colab is a pure producer, the local app a pure consumer — the boundary is a directory of files validated by a manifest, NOT shared code; the local runtime imports zero Colab-only deps) and **preserving the single-state / single-writer / single-replay-log invariants** through every change (the coordination agent is the only feature that stresses these, and must keep ONE authoritative state with sub-sessions running in the existing no-I/O `run_conversation` mode).

**Major components (new/modified):**
1. `src/embedding_backend.py` (NEW) — `EmbeddingBackend` Protocol mirroring `ClusteringBackend`, with `SentenceTransformerBackend` (384) and `OpenAIEmbeddingBackend` (1536) impls; dim is a property of the instance, not a constant.
2. `src/artifacts.py` + `manifest.json` (NEW) — Colab artifact loader/validator; the manifest provenance schema (model, dim, normalized, versions, input hash, n_items) is the shared dependency of both Colab artifacts and the embedding backends.
3. `rebuild_state_from_refit` helper + Step 5.5 in `run_conversation` (MODIFIED) — per-query KMeans re-fit on `store.get_all()` (read-only), with the cluster-ID alignment step.
4. `src/query_filter.py` (NEW) — pre-parse NL normalization stage between `oracle.reply()` and `parse_feedback`.
5. Pre-loop bootstrap block (MODIFIED `build_initial_clustering_state` with `defer=True`) — oracle-initiated entry.
6. Coordination agent + pure `merge(authoritative, [sub_results])` (NEW) — orchestrator-worker layer above `run_conversation`; built last.
7. Projection helpers (MODIFIED) — `_should_recompute_projection` -> `_should_recolor`; compute coords once, recolor on any cluster change.

### Critical Pitfalls

See [PITFALLS.md](PITFALLS.md). Top items, with prevention:

1. **Cluster-ID churn on every re-fit (THE linchpin risk)** — KMeans labels are arbitrary between fits, so naive re-fit breaks merge/split history, cluster names, and the UMAP recolor. Avoid with a Hungarian alignment step (`scipy.optimize.linear_sum_assignment` on item-set overlap / centroid cosine) that preserves old IDs/names for matched clusters; mint new IDs only when K genuinely increased via oracle intent. The UMAP recolor phase hard-depends on this.
2. **Unintentional K drift during re-fit** — re-instantiating the backend per query can re-run BIC K-selection, violating "K changes ONLY via oracle intent." Persist K in `ClusteringState`/session, pass it in explicitly on re-fit, assert `new_k == prev_k` unless feedback was split/merge; reach BIC only at the first clustering.
3. **Embedding dim mismatch (384 vs 1536) + normalization mismatch** — do not "remove the assert"; thread real dim + model + `normalized` flag through artifact metadata and assert on load. Normalize at every backend boundary (assert unit norm); never compare distances/BIC across normalization regimes. The mapping layer must re-embed with the SAME model that produced the stored vectors.
4. **OpenAI cost / rate-limit / no-cache** — batch <=2048 inputs (<=8192 tokens each), retry-with-backoff at the API boundary only, content-hash cache keyed by `sha256(text+model_id)`; never embed inside the turn loop. Colab is where bulk embedding should run.
5. **Colab artifact version skew / truncation** — ship a provenance sidecar and assert dtype `float32`, shape `(N, dim)`, dim/model/lib-version match, and input hash on load; write artifacts atomically (temp -> fsync -> rename); assert `len(records) == n_items`.
6. **Coordination agent state-merge / partial failure** — define ONE authoritative `ClusteringState`; sub-sessions produce proposals applied sequentially through the single-writer pipeline; on partial failure abort the WHOLE op (no partial merge), recover from the last good audit-log state; never reach for eventlet/gevent.

## Implications for Roadmap

The three technical researchers converged on a single dependency-locked build order. Phases are numbered 7-12 (continuing from v1's phases 1-6).

### Phase 7: Pluggable embedding backends + dynamic dimension [FOUNDATIONAL — do FIRST]
**Rationale:** `EMBEDDING_DIM = 384` is a module constant asserted in 4+ sites in `embedding_store.py` and 3 cache-shape sites in `web/app.py`; a second model (OpenAI 1536) cannot coexist with it, and a stale 1536-dim cache could be silently reused as 384. Everything embedding-related is blocked until this lands.
**Delivers:** `EmbeddingBackend` Protocol + ST/OpenAI impls; `store.dim` learned from data; cache files keyed by model name; normalize-at-boundary contract (assert unit norm); content-hash embedding cache; provenance metadata schema.
**Uses:** `openai` SDK (reuse `build_client`), `pydantic`, numpy `.npy`.
**Avoids:** Pitfalls 4 (dim mismatch), 5 (normalization mismatch), 6 (OpenAI cost/cache).

### Phase 8: Colab compute-only artifact pipeline
**Rationale:** The Colab manifest must record `embedding_dim`; building it before Phase 7 would bake in 384 again. Local torch is CPU-only, so GPU offload is the motivation.
**Delivers:** `src/artifacts.py` loader, `manifest.json` contract (shares Phase 7's provenance schema), HF Hub handoff, web-worker `colab_import` branch; atomic writes + `n_items` assert.
**Uses:** `huggingface_hub`, `requirements-colab.txt` mirroring local versions.
**Avoids:** Pitfalls 7 (version skew), 8 (truncation/GPU variance), 2 (sklearn `n_init`/seed pinning Colab==local).

### Phase 9: Interactive UMAP cache-once / recolor [prereq for re-fit]
**Rationale:** Per-query re-fit changes every cluster ID, so the projection must recolor not refit; coords are a fixed function of immutable embeddings. This capability must exist before re-fit lands.
**Delivers:** `_should_recompute_projection` -> `_should_recolor`; compute coords once (ideally in Colab artifact), recolor payload on any cluster change.
**Implements:** generalizes the coord cache already in study/watch workers.
**Avoids:** Pitfall 11 (stale UMAP geometry / layout jitter). NOTE: hard-depends on the ID-stability alignment from Phase 11 to be correct — coordinate sequencing carefully (see Ordering Rationale).

### Phase 10: Oracle-initiated flow + drop HDBSCAN (KMeans-only)
**Rationale:** The milestone's central reframe (cluster on demand, not autonomously); the first clustering becomes the first oracle-driven re-fit, so this naturally precedes/pairs with re-fit. Re-fit semantics are KMeans-specific (HDBSCAN has no fixed-K re-fit), so KMeans-only must settle here.
**Delivers:** pre-loop bootstrap block (`defer=True` placeholder state -> intro -> first query -> first fit); intro written to `events.jsonl` sidecar (not an audit state line); KMeans-only interactive default; "autonomous-first" retained as ablation.
**Addresses:** dataset introduction, oracle initial query (FEATURES table stakes).
**Avoids:** Pitfall 9 (empty/degenerate first query — needs a documented default + assert-before-first-clustering).

### Phase 11: Re-fit KMeans per query + query filter [THE engineering core]
**Rationale:** Depends on Phase 10 (oracle-initiated first fit) and Phase 9 (recolor). This is where the linchpin cluster-ID alignment layer is built.
**Delivers:** Step 5.5 re-fit on `store.get_all()` (embeddings FIXED); Hungarian ID-alignment layer; K-gating (BIC only at first clustering, assert K unchanged unless split/merge); `src/query_filter.py` (pre-parse NL normalization, EXTENDS `feedback_parser`/`_contradicts`); determinism test.
**Addresses:** query filter, oracle query -> first re-fit (FEATURES table stakes/differentiators).
**Avoids:** Pitfalls 1 (ID churn), 2 (nondeterminism), 3 (K drift), 10 (over-filtering / contradiction double-handling — filter normalizes, feedback layer arbitrates).

### Phase 12: Coordination agent (N parallel sessions) [LAST — needs own research spike]
**Rationale:** Highest risk; breaks single-state/single-writer cardinality; locked as last. Build only after F1-F5 + recolor stabilize.
**Delivers:** orchestrator-worker layer above `run_conversation`; pure `merge()` reduce; sub-sessions run in existing no-I/O mode (`db_conn=None, socketio=None`); ONE authoritative state + ONE audit line per merged turn.
**Avoids:** Pitfall 12 (state-merge conflicts / partial-failure -> abort whole op; no eventlet/gevent; `threading.Thread` per sub-session, mind BLAS oversubscription).

### Phase Ordering Rationale

- **F2 (dynamic dim) before everything embedding-related** — the `EMBEDDING_DIM` constant gates pluggable backends AND silently gates cache reuse; a second model cannot coexist with it. Named foundational prerequisite #1.
- **The provenance/manifest schema (foundational prerequisite #2)** is defined in Phase 7 and consumed by Phase 8 (Colab) — both the artifact pipeline and the embedding backends validate against it (model, dim, normalized, versions, input hash).
- **F1 (Colab) after F2** — the manifest must record `embedding_dim`; building Colab first re-bakes 384.
- **Recolor before re-fit** — re-fit changes all cluster IDs, so the projection must recolor-not-refit; that capability (Phase 9) is a prerequisite for Phase 11.
- **Oracle-initiated + KMeans-only before re-fit-per-query** — the first clustering becomes the first oracle-driven re-fit; re-fit is KMeans-specific.
- **Coordination agent last** — locked decision; highest risk; needs the rest stable.
- **Note on a cross-phase dependency:** Phase 9 (recolor) is correct ONLY once the Phase 11 ID-alignment layer exists (recolor with churned IDs shows wrong colors). The two researchers ordered these slightly differently (architecture put recolor at 9; pitfalls put ID-stability at 9). The roadmapper should either (a) build the recolor mechanism in Phase 9 but defer its correctness-validation until the alignment lands in Phase 11, or (b) pull the ID-alignment layer forward to sit with re-fit and validate recolor immediately after. Flag this coupling explicitly when planning.

### Research Flags

Phases likely needing a deeper `/gsd-research-phase` spike during planning:
- **Phase 12 (Coordination agent):** explicitly flagged by all three technical researchers for its own deep-research spike — cross-session state-merge semantics, contradiction recombination across N sessions, partial-failure rollback, and BLAS/thread-pool contention under N concurrent KMeans fits are unresolved design problems, not mechanics.
- **Phase 11 (Re-fit + ID alignment):** the cluster-ID reconciliation policy (preserve-stable-IDs vs. renumber) and the hierarchy-lineage gap (a wholesale re-fit has no clean split/merge lineage) are genuine design decisions worth a focused planning pass.

Phases with standard/well-documented patterns (likely skip research-phase):
- **Phase 7 (backends):** Protocol mirrors the existing `ClusteringBackend`; OpenAI/HF APIs documented and version-verified.
- **Phase 9 (UMAP recolor):** the cache-once/recolor pattern is standard and the cache half-exists already.
- **Phase 10 (oracle-initiated):** orchestration around the existing loop; HITL onboarding patterns are well-attested.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All key versions probed against the live local interpreter + verified against current PyPI/official docs (2026-05-21). |
| Features | MEDIUM-HIGH | Orchestrator/onboarding/intent patterns verified against multiple sources; UMAP-recolor and chat-view detail are LOWER (UMAP docs + implementation common-sense). |
| Architecture | HIGH | Grounded in direct reading of the existing codebase; integration seams verified against actual source (`conversation_loop.py`, `clustering.py`, `embedding_store.py`, `web/app.py`, etc.). |
| Pitfalls | HIGH (code-grounded) / MEDIUM (external API) | Code pitfalls verified by reading `src/`; OpenAI/sentence-transformers behavior verified against docs. |

**Overall confidence:** HIGH

### Gaps to Address

- **Cluster-ID reconciliation policy** — preserve-stable-IDs (Jaccard/centroid match) vs. renumber is a design decision to settle in Phase 11 planning; affects merge/split history, names, and recolor correctness.
- **Hierarchy lineage on wholesale re-fit** — `record_split/merge` assume incremental edits; a re-fit has no clean lineage. Decide: reset hierarchy on re-fit, or skip hierarchy recording for re-fit turns (Phase 11).
- **Intro-turn audit semantics** — is the oracle-initiated intro turn 0, turn -1, or an `events.jsonl` sidecar record? Suggested: sidecar, not an audit state line (Phase 10).
- **Coordination merge semantics** — the whole sub-state -> authoritative-state merge contract is unresolved and needs the Phase 12 spike (the only HIGH-risk gap).
- **Recolor/alignment cross-phase coupling** — recolor (Phase 9) is correct only with ID alignment (Phase 11); resolve the sequencing during roadmap planning (see Ordering Rationale).
- **`requirements.txt` drift** — bump stale floors (`sentence-transformers>=2.7` -> `>=5.4`, `openai>=1.30` -> `>=2.26,<3`, pin `huggingface_hub>=1.7,<2`); pre-existing `EMBEDDING_DIM`/docstring drift (docstrings say 768, constant is 384) — clean up during Phase 7.

## Sources

### Primary (HIGH confidence)
- Direct source reading: `src/conversation_loop.py`, `src/clustering.py`, `src/embedding_store.py`, `src/agent_functions.py`, `src/feedback_parser.py`, `src/feedback.py`, `src/state.py`, `src/mapping.py`, `web/app.py` — loop seams, `EMBEDDING_DIM` sites, KMeans determinism, ID remap, contradiction policy.
- Live local interpreter probe (2026-05-21) — exact installed versions incl. torch 2.11.0+cpu.
- OpenAI API docs — `text-embedding-3-small` 1536-dim, `dimensions` param, <=2048 inputs / 8192 tokens, unit-normalized output.
- huggingface_hub PyPI + v1.0 blog — upload/download API, httpx migration, `cached_download` removal.
- ruamel.yaml PyPI — round-trip comment/order preservation.
- OpenAI Agents SDK — planner/worker decomposition.
- UMAP docs — recolor-by-label, precompute+transform vs refit.
- `.planning/PROJECT.md` + `CLAUDE.md` — locked decisions, K-only-via-oracle anti-feature, no-eventlet/gevent, audit-log-as-replay-source, fail-loudly.

### Secondary (MEDIUM confidence)
- IPBC (interactive projection-based HITL clustering) — orientation + projection stability.
- AstronomicAL — onboarding/orient-then-query workflow.
- Multi-agent orchestration pattern surveys — orchestrator-worker (~70% of prod deployments), fan-out/fan-in.
- Text-to-SQL / intent-normalization references — NL -> structured-command translation.
- sentence-transformers/all-MiniLM-L6-v2 model card — 384-dim, L2-normalized via Normalize module.

### Tertiary (LOW confidence)
- UMAP recolor and chat-view implementation detail — partly implementation common-sense beyond what the cited docs cover; validate during Phase 9 planning.

---
*Research completed: 2026-05-21*
*Ready for roadmap: yes*
