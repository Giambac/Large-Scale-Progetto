# Architecture Research

**Domain:** Human-in-the-loop conversational clustering — v2.0 milestone integration
**Researched:** 2026-05-21
**Confidence:** HIGH (grounded in direct reading of the existing codebase; integration points verified against actual source)

> Scope note: this is a **subsequent-milestone** architecture study. It does NOT re-derive the
> base 3-agent / pure-`f_*` / single-loop design — that is settled and treated as ground truth.
> Every section below answers: *how does each v2 feature graft onto the existing seams, what is new
> vs. modified, what data-flow changes, and in what order to build it.*

---

## Standard Architecture (existing, as-built — the integration surface)

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  WEB / I/O LAYER  (web/app.py — FastAPI + python-socketio ASGI)            │
│   - routes: /upload, /resume, /study, /watch                               │
│   - background worker threads (threading.Thread, daemon)                   │
│   - SocketIOEmitter: worker-thread → loop via run_coroutine_threadsafe     │
│   - UMAP helpers: _compute_projection / _build_projection_payload /        │
│                   _should_recompute_projection / compute_and_emit_*        │
└──────────────┬─────────────────────────────────────────────────────────────┘
               │ calls (in worker thread)
┌──────────────▼─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION  (src/conversation_loop.py :: run_conversation)             │
│   - the ONLY place that does I/O: JSONL audit write, DB write, emit,       │
│     post_turn_callback (projection recompute)                              │
│   - while True: f_uncertainty → f_next_best_step → format → oracle.reply   │
│       → parse_feedback → f_next_state → write/emit → check_stopping        │
└──────────────┬─────────────────────────────────────────────────────────────┘
               │ calls pure functions (no I/O inside)
┌──────────────▼─────────────────────────────────────────────────────────────┐
│  PURE AGENT FUNCTIONS  (src/agent_functions.py, uncertainty.py, etc.)      │
│   f_output · f_uncertainty · f_next_best_step · f_next_state               │
└──────────────┬─────────────────────────────────────────────────────────────┘
               │ reads
┌──────────────▼───────────────────┬────────────────────┬────────────────────┐
│  EmbeddingStore (read-only .npy)  │ ClusteringBackend  │  ClusteringState   │
│  EMBEDDING_DIM = 384 (hardcoded)  │ Protocol (.fit)    │  (single source    │
│  computed ONCE, never re-embedded │ run ONCE at start  │   of truth)        │
└───────────────────────────────────┴────────────────────┴────────────────────┘
               │ persisted
┌──────────────▼─────────────────────────────────────────────────────────────┐
│  PERSISTENCE: audit_log.jsonl (replay source of truth) · events.jsonl       │
│              sidecar · experiments.db (SQLite query index)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities (as they exist today)

| Component | Responsibility | Where |
|-----------|----------------|-------|
| `run_conversation` | Owns the while-loop + ALL I/O (D-01..D-04). Pure functions called from here. | `src/conversation_loop.py` |
| `f_next_state` | Apply feedback deltas → new `ClusteringState` (pure). Split/merge/move. | `src/agent_functions.py` |
| `EmbeddingStore` | Read-only `(N, EMBEDDING_DIM)` cache, computed once, `flags.writeable=False`. | `src/embedding_store.py` |
| `ClusteringBackend` Protocol | `fit(embeddings) -> (labels, soft_probs)`. HDBSCAN + KMeans implement it. Run ONCE at start. | `src/clustering.py` |
| `build_initial_clustering_state` | embeddings → backend.fit → group → LLM-name → assemble `ClusteringState(turn=0)`. | `src/clustering.py` |
| Projection helpers | Server-side UMAP, refit on split/merge only via `_should_recompute_projection(deltas)`. | `web/app.py` |
| `parse_feedback` | LLM call: raw oracle text → `list[FeedbackDelta]`. The untrusted boundary. | `src/feedback_parser.py` |
| Web workers | `_run_conversation_background` / `_run_study_background` / `_run_watch_background` — 3 near-duplicate orchestration copies. | `web/app.py` |

**Key existing seams the v2 work hooks into (memorize these — every feature touches one):**
1. The `backend` parameter of `build_initial_clustering_state` (already a Protocol — KMeans re-fit reuses it).
2. The `post_turn_callback(new_state, deltas)` hook (already used for projection recompute).
3. `EMBEDDING_DIM` (the one global constant that blocks pluggable embeddings).
4. The parse step `parse_feedback(...) -> deltas` (query filter inserts right after).
5. `ClusteringState` as single source of truth (coordination agent must not break this).

---

## Feature-by-Feature Integration

### F1 — Colab compute-only (artifact contract)

**New vs modified:** New artifact loader + classmethod; **no change** to the loop or socketio runtime.

**Clean separation — the artifact contract.** Colab is a pure producer; the local app is a pure
consumer. The boundary is a directory of plain files with a manifest, NOT shared code:

```
artifacts/<dataset_id>/
├── manifest.json        # {dataset_id, embedding_model, embedding_dim, n_items,
│                        #  backend, k, schema_version, sha256 of each file, created_utc}
├── embeddings.npy       # (N, dim) float32  → feeds EmbeddingStore.load()
├── records.jsonl        # [{item_id, text}, ...] aligned to embedding rows (item_id == row)
└── initial_state.json   # serialized ClusteringState(turn_index=0)  → serialization.py format
```

**How it loads without leaking into the socketio runtime:** `EmbeddingStore.load()` is already
file-based — Colab just produces the `.npy`. The new piece is a
`load_initial_state(path) -> ClusteringState` (reuse `deserialize_state` from `serialization.py`)
and a thin `load_artifacts(dir) -> (store, records, initial_state)` function in a **new module
`src/artifacts.py`**. The web worker gains an `if mode == "colab_import"` branch that calls
`load_artifacts` instead of `compute_and_save` + `build_initial_clustering_state`. Colab dependencies
(`sentence_transformers`, `umap`, heavy GPU libs) never get imported in the local process — the
contract is files on disk, validated by `manifest.json` (assert `embedding_dim`, `n_items`,
`schema_version`, and recompute sha256). **This is the anti-leak invariant: the local runtime imports
zero Colab-only code; the only shared artifact is the serialization format already in `serialization.py`.**

**Critical alignment invariant:** `records.jsonl` row order MUST equal embedding row order MUST equal
`item_id`. The existing code already assumes `item_id == row index` (D-11); the manifest must assert
`len(records) == embeddings.shape[0]` on load (mirror the existing assert in
`build_initial_clustering_state`).

**Risk:** LOW. The seam (`EmbeddingStore.load` + serialized state) already exists; this is mostly a
manifest + validator + a worker branch.

---

### F2 — Pluggable embedding backend + dynamic dimension  ⚠ FOUNDATIONAL PREREQUISITE

**New vs modified:** New `EmbeddingBackend` Protocol (new). **Modified:** `embedding_store.py`
(remove the constant), every `EMBEDDING_DIM` reference site (3 in `web/app.py`, ~6 in
`embedding_store.py`).

**The blocker — `EMBEDDING_DIM = 384` is a module-level constant asserted in 4+ places.** Verified
reference sites:
- `src/embedding_store.py:21` (definition), `:45-46` (`__init__` assert), `:80-81`
  (`compute_and_save` assert), `:97/:104` (docstrings).
- `web/app.py:462, 496` (`_run_conversation_background` cache-shape check),
  `:933, 960` (`_run_study_background`), `:1149, 1194` (`_run_watch_background`).

**Required refactor (do this BEFORE any second backend lands):**
1. **`EmbeddingStore` learns its dim from the data, not a constant.** Drop the
   `embeddings.shape[1] == EMBEDDING_DIM` assert; replace with `self.dim = embeddings.shape[1]` set in
   `__init__`, and expose `store.dim`. The `ndim == 2` assert stays (still fail-loudly on garbage).
2. **The cache-shape checks in `web/app.py` (3 copies) must compare against the *expected* dim of the
   *selected* embedding backend**, not a global. Get it from the backend (`backend.dim`) or from the
   manifest (Colab path). This is the cascading impact: the cache-reuse logic
   (`_cached_shape == (len(texts), EMBEDDING_DIM)`) currently silently assumes 384 — with two models
   producing 384 vs 1536 dims, a stale cache from the *other* model must NOT be reused. The shape
   check already guards row count; extend it to also guard dim, and ideally key the cache filename by
   model name (`embeddings__<model>.npy`).
3. **New `EmbeddingBackend` Protocol** (mirror `ClusteringBackend`), new module
   `src/embedding_backend.py`:
   ```python
   @runtime_checkable
   class EmbeddingBackend(Protocol):
       @property
       def dim(self) -> int: ...                 # declared statically per model
       def encode(self, texts: list[str]) -> np.ndarray: ...   # (N, dim) float32
   ```
   Implementations: `SentenceTransformerBackend(model="all-MiniLM-L6-v2", dim=384)`,
   `OpenAIEmbeddingBackend(model="text-embedding-3-small", dim=1536)`. `compute_and_save` takes a
   backend instead of hardcoding `SentenceTransformer(EMBEDDING_MODEL)`.

**Cascading impacts to flag for roadmapper:**
- Anything that hardcodes `(768,)` / `(384,)` in docstrings or test fixtures (the docstrings in
  `embedding_store.py` even say "768" while the constant is 384 — pre-existing drift; fix while here).
- `manifest.json` (F1) must carry `embedding_dim` so the Colab path validates against the backend.
- KMeans soft-prob math (`_compute_soft_probs`) is dim-agnostic already (broadcasts over `dim`) — no
  change. UMAP's PCA-to-50 step caps at `min(50, dim)`; with dim=1536 it's fine, but verify
  `n_components <= min(N, dim)`.

**Risk:** MEDIUM. Mechanical but wide-reaching; the silent-cache-reuse trap is the real danger.
**This must ship before any second embedding model is added** — otherwise the assert fires or, worse,
a 1536-dim cache silently reused as 384.

---

### F3 — Re-fit KMeans per oracle query (embeddings FIXED)

**New vs modified:** **Modified** `run_conversation` (one new step) + a new pure helper. **Drop**
HDBSCAN from the interactive path (KMeans-only).

**Where it hooks into the single while-loop.** Today the loop applies oracle feedback purely via
`f_next_state` (split/merge/move on the existing partition). The new behavior: after parsing the
oracle's query into instructions/deltas, **re-run `KMeansBackend.fit(store.get_all())` on the
fixed embeddings** to get a fresh partition, then reconcile it into a new `ClusteringState`. The hook
is a new **Step 5.5** between `f_next_state` (Step 5) and the audit write (Step 6):

```
Step 5:   new_state = f_next_state(state, deltas, ...)        # existing delta application
Step 5.5: if refit_requested(deltas):                          # NEW
              labels, soft_probs = kmeans_backend.fit(store.get_all())   # embeddings FIXED
              new_state = rebuild_state_from_refit(new_state, labels, soft_probs, namer, k)
Step 6:   append_to_audit_log(new_state, ...)                  # unchanged
```

`store.get_all()` returns the read-only array — **the CORE INVARIANT (never re-embed) is preserved**;
only the cluster fit is recomputed. `K` is held in `KMeansBackend._k` and changes ONLY through oracle
intent (the loop sets `kmeans_backend._k = new_k` when the oracle asks for a different K — never via
silhouette/BIC re-optimization, per the locked anti-feature).

**Dependent state that MUST refresh after a re-fit** (this is the integration hazard list):
| State | Refresh rule |
|-------|--------------|
| `soft_probs` | Recomputed by `KMeansBackend.fit` (returned directly). Replaces, not merges. |
| `assignments` | New `labels` → remap to cluster ids. |
| `clusters` (names) | Re-fit produces new partitions → must re-name affected clusters via `namer`. Reuse stable cluster ids where item-set overlap is high to avoid disorienting renumbering (a reconciliation policy decision). |
| Projection cache (F7) | Cluster membership changed → **recolor** (do NOT refit UMAP — coords are embedding-derived and embeddings are fixed). This is exactly why F7 (recolor-not-refit) is a prerequisite. |
| `PairBag` (judge) | Pairwise-accuracy bag references item pairs, not cluster ids — survives a re-fit; `compute_pairwise_accuracy` reads `new_state.assignments`, so it auto-reflects the re-fit. Verify accuracy semantics still hold across a wholesale repartition. |
| `hierarchy` | `record_split/merge` assume incremental edits. A wholesale re-fit has no clean split/merge lineage — either reset hierarchy on re-fit or skip hierarchy recording for re-fit turns (decision needed). |

**Build-order dependency:** F3 depends on (a) KMeans-only being the interactive default (today the
web default is `hdbscan`), (b) the projection cache being recolor-not-refit (F7), and (c) ideally the
oracle-initiated flow (F4) so the first KMeans fit is itself query-driven. **Build F4 + F7 before F3.**

**Risk:** MEDIUM-HIGH. The cluster-id reconciliation policy (stable ids vs. renumber) and the
hierarchy-lineage gap are genuine design decisions, not mechanics.

---

### F4 — Oracle-initiated flow (defer first clustering)

**New vs modified:** **Modified** `build_initial_clustering_state` (allow a "pre-clustering" state)
and the loop entry; new "introduction" turn before turn 0.

**Reshape.** Today: worker computes embeddings → `build_initial_clustering_state` (runs backend
immediately) → loop starts at an already-clustered `turn_index=0`. New flow:
`dataset introduction → oracle's first query → first clustering → conversational session`.

Two clean options; **recommended: option B**:
- **Option A (sentinel state):** introduce a `ClusteringState` with a single "all items" cluster
  (K=1) as the pre-clustering state, then the oracle's first query triggers the first real
  `KMeansBackend.fit` via the F3 re-fit hook. Reuses F3 machinery — first fit is just the first
  re-fit. Keeps the loop uniform.
- **Option B (deferred-init parameter):** `build_initial_clustering_state` gains
  `defer: bool = False`. When `defer=True`, it returns a minimal placeholder state (K=1 "unclustered")
  and does NOT call `backend.fit`. The loop's first iteration shows the dataset introduction, takes the
  oracle's opening query, then performs the first fit on turn 1.

**Loop entry change:** the `while True` currently assumes `initial_state` is fully clustered. With
F4, the first iteration must (a) present an introduction message instead of `_format_message(action,...)`,
(b) read the oracle's opening query, (c) run the first fit. Cleanest: a small **pre-loop bootstrap
block** before `while True` that does intro→query→first-fit, producing the real turn-0 state, after
which the existing loop runs unchanged. This isolates the new flow and keeps the loop body intact.

**Build-order dependency:** F4 naturally unifies with F3 (first clustering = first re-fit). Build
**F4 and F3 together or F4 immediately before F3.** Both depend on KMeans-only being settled.

**Risk:** MEDIUM. Touches the loop's entry contract; the audit-log/turn-index semantics for the
"intro turn" (is it turn 0? a turn -1 event?) need a decision (suggest: write intro as an `events.jsonl`
sidecar record, NOT an `audit_log.jsonl` state line — same rule as `oracle_init`, per Pitfall 5).

---

### F5 — Query filter (stage between oracle reply and f_next_state)

**New vs modified:** **New** module `src/query_filter.py`; **modified** loop to insert one step;
extends `feedback_parser` conceptually (runs adjacent to it).

**Placement.** The loop today does: `oracle.reply()` → `parse_feedback(raw_text)` → `f_next_state(deltas)`.
The query filter sits **between the oracle reply and f_next_state**, translating natural-language
oracle queries into "simple, contradiction-free clusterer instructions" before they become deltas (or
after parse, before apply). Two valid insertion points:
- **Pre-parse (recommended):** `raw_text → query_filter → cleaned_instruction → parse_feedback`. The
  filter normalizes/deconflicts NL ("split A, but actually merge A and B" → resolve to latest intent
  per the locked "latest intent wins" rule) before the structured-extraction LLM call. Keeps
  `parse_feedback`'s JSON schema contract intact.
- **Post-parse:** `deltas → query_filter → contradiction-free deltas`. Operates on structured deltas
  (drop a `split` that a later `merge` in the same batch contradicts). This overlaps with the existing
  per-delta `deviation()` validation in `f_next_state` — risk of double-handling.

Recommend **pre-parse filtering** for NL normalization + keep the structured-delta validation where it
is. New step:
```
reply = oracle.reply(...)
filtered = query_filter(reply.raw_text, state, llm_client)   # NEW — contradiction-free NL
deltas = parse_feedback(filtered, state, llm_client)         # unchanged contract
new_state = f_next_state(state, deltas, ...)                 # unchanged
```

**Risk:** LOW-MEDIUM. Self-contained stage; the only subtlety is not duplicating the contradiction
logic already in `f_next_state`.

---

### F6 — Coordination agent (deferred, LAST) ⚠ RISKIEST INTEGRATION

**New vs modified:** **New** orchestration layer above `run_conversation`; the hardest constraint is
preserving "ClusteringState is the single source of truth" across **N parallel clusterer sessions**.

**The decomposition.** A complex oracle operation is decomposed into pairwise sub-operations fanned
out to N clusterer sessions, each running its own `run_conversation` with its own `ClusteringState`.
This directly tensions the single-source-of-truth invariant: there are now N states, not one.

**State-merge strategy (the core design problem).** The invariant is satisfiable only if you define a
**single authoritative state** and treat the N sub-session states as *derived, ephemeral working
copies* that are deterministically merged back. Recommended pattern — **scatter/gather with a
canonical merge**:

```
                    ┌─ Coordination Agent (NEW) ──────────────────┐
                    │  owns the ONE authoritative ClusteringState  │
                    └───────────┬──────────────────────────────────┘
       decompose into pairwise sub-ops │ scatter (fan-out)
        ┌──────────────┬──────────────┼──────────────┐
        ▼              ▼              ▼              ▼
   session 1       session 2      session 3      session N
 (sub-state 1)   (sub-state 2)  (sub-state 3)  (sub-state N)
        └──────────────┴──────────────┴──────────────┘
                    │ gather → deterministic merge
                    ▼
        new authoritative ClusteringState (turn += 1)
```

Merge rules to preserve the invariant:
1. **Disjoint partitioning:** each sub-session operates on a disjoint subset of clusters/items
   (pairwise sub-ops are by construction local — e.g., "is cluster A vs cluster B a split?"). Disjoint
   scope means sub-results compose without conflict.
2. **Single writer:** ONLY the coordination agent writes the authoritative state, audit log, and DB.
   Sub-sessions run with `db_conn=None`, `socketio=None`, `log_path=<scratch>` (no I/O) — they are pure
   compute, exactly like the unit-test mode `run_conversation` already supports. **This reuses the
   existing "no-I/O" path of `run_conversation` (verified: `db_conn=None`/`socketio=None` are already
   first-class).**
3. **Deterministic conflict resolution:** if two sub-ops touch overlapping items (should be rare given
   disjoint scope), apply the locked "latest intent wins" rule with a fixed sub-op ordering so merges
   are reproducible (required for the replay-from-`audit_log.jsonl` guarantee).
4. **One audit line per merged turn:** the coordination agent writes ONE `ClusteringState` per merged
   coordination turn to the authoritative `audit_log.jsonl` — sub-session scratch logs are never the
   replay source of truth.

**Why this is the riskiest:** the existing system has exactly one state, one writer, one log. N
sessions break every cardinality assumption: parallel `threading.Thread` workers all calling sklearn
(the CLAUDE.md eventlet/gevent ban exists precisely because of numpy/sklearn locking — N concurrent
KMeans fits must each be in their own thread with no monkey-patching, and the GIL/BLAS thread-pool
contention needs care). The merge must be a *pure function* `merge(authoritative, [sub_results]) ->
new_authoritative` so it's testable and replayable. **Flag for roadmapper: this phase needs its own
deep-research spike before implementation.**

**Risk:** HIGH. Build last, after F1–F5 + F7 stabilize, exactly as locked.

---

### F7 — Interactive UMAP: cache coords once, recolor on cluster change

**New vs modified:** **Modified** projection helpers + worker; cache `coords` once.

**Today:** `compute_and_emit_projection` calls `_compute_projection` (PCA→UMAP refit, expensive) on
initial clustering AND again on every split/merge (`_should_recompute_projection`). Since embeddings
are FIXED (CORE INVARIANT) and UMAP is purely a function of embeddings, **the 2D coords never change** —
only cluster colors do. So the refit-on-split/merge is wasted compute and causes disorienting layout
shifts.

**Change:** compute `coords = _compute_projection(store.get_all())` **once** (at first clustering or
artifact import), cache it (`sess["coords"]` already does this in study/watch workers — generalize to
the main worker), and on any cluster change emit only a lightweight **recolor** payload
(`cluster_ids` + `cluster_colors` + `max_probs`, reusing `_build_projection_payload` but with the
cached `coords`). `_should_recompute_projection` is replaced by `_should_recolor` (true on ANY cluster
change: split, merge, move, AND re-fit). `_compute_projection` is invoked exactly once per dataset.

**This is a prerequisite for F3** (re-fit per query): a wholesale re-fit changes every cluster_id, so
the projection must recolor — and it must NOT refit (coords are embedding-derived, embeddings fixed).

**Risk:** LOW. The cache already half-exists in the study/watch workers; this consolidates it and
flips the recompute trigger from "refit" to "recolor."

---

## Recommended Build Order (dependency-respecting)

```
Phase  7  ── F2  Dynamic-dim refactor + EmbeddingBackend Protocol   [FOUNDATIONAL — do FIRST]
              (remove EMBEDDING_DIM constant; fix 3 cache-shape sites; store.dim)
                                   │  (unblocks pluggable embeddings + Colab manifest)
Phase  8  ── F1  Colab compute-only artifact contract (manifest + loader)
              (depends on F2 so manifest carries embedding_dim)
Phase  9  ── F7  Interactive UMAP cache-once / recolor                [prereq for F3]
Phase 10  ── F4  Oracle-initiated flow (defer first clustering)       [pairs with F3]
              + drop HDBSCAN from interactive default (KMeans-only)
Phase 11  ── F3  Re-fit KMeans per oracle query (depends on F4 + F7)
              + F5 Query filter (independent, can fold in here or earlier)
Phase 12  ── F6  Coordination agent (N parallel sessions) [LAST — needs own spike]
```

**Why this order (the hard constraints):**
- **F2 before everything embedding-related:** the `EMBEDDING_DIM` constant is asserted in 4+ places
  and silently gates cache reuse. A second model (OpenAI 1536-dim) cannot coexist with it. This is the
  named foundational prerequisite.
- **F1 after F2:** the Colab manifest must record `embedding_dim`; the loader validates against the
  backend's declared dim. Building F1 first would bake in 384 again.
- **F7 before F3:** re-fit changes all cluster ids → projection must recolor-not-refit; that capability
  is F7.
- **F4 before/with F3:** the first clustering becomes the first oracle-driven re-fit; building the
  oracle-initiated entry first lets F3 reuse one fit path.
- **KMeans-only + oracle-initiated before re-fit-per-query:** re-fit semantics are KMeans-specific
  (HDBSCAN has no fixed-K re-fit); the interactive default must be KMeans first.
- **F6 last:** highest risk, breaks the single-state cardinality, needs the rest stable. (Locked.)

---

## Data Flow

### Per-turn flow, today
```
f_uncertainty → f_next_best_step → _format_message → oracle.reply
   → parse_feedback → f_next_state → [audit JSONL → DB → post_turn_callback → emit] → check_stopping
```

### Per-turn flow, after v2 (interactive KMeans-refit path)
```
f_uncertainty → f_next_best_step → _format_message → oracle.reply
   → query_filter (F5)                          ← NEW stage
   → parse_feedback → f_next_state
   → if refit_requested: KMeansBackend.fit(store.get_all()) → rebuild_state   ← NEW (F3, embeddings FIXED)
   → [audit JSONL → DB → recolor-emit (F7, cached coords) → emit] → check_stopping
```

### Session bootstrap, today vs. after F4
```
today:  embeddings → build_initial_clustering_state (fit NOW) → loop @ turn 0
after:  embeddings → defer-init placeholder → intro turn → oracle first query → FIRST fit → loop
        (Colab path: load_artifacts → store + records + initial_state, no local fit)
```

---

## Anti-Patterns (specific to this milestone)

### AP1: Re-embedding on re-fit
**What people do:** call `EmbeddingStore.compute_and_save` again inside the re-fit step.
**Why wrong:** violates the read-only-embeddings CORE INVARIANT; embeddings must be computed once.
**Do this instead:** re-fit reads `store.get_all()` (read-only array) and re-runs only `KMeansBackend.fit`.

### AP2: Refitting UMAP on cluster change
**What people do:** keep calling `_compute_projection` on every split/merge/re-fit.
**Why wrong:** embeddings are fixed → coords are constant; refitting wastes seconds and reshuffles the
layout, disorienting the oracle.
**Do this instead:** compute coords once, cache, recolor only (F7).

### AP3: N sub-sessions each writing audit/DB
**What people do:** let each coordination sub-session write its own audit_log.jsonl / DB rows.
**Why wrong:** destroys the single-source-of-truth + single-replay-log guarantee; N partial states
cannot be replayed deterministically.
**Do this instead:** sub-sessions run with `db_conn=None, socketio=None` (pure compute); only the
coordination agent writes ONE merged authoritative state per turn.

### AP4: Reintroducing a hardcoded dim after the F2 refactor
**What people do:** add `OPENAI_DIM = 1536` as a second module constant and branch on model name.
**Why wrong:** recreates the exact coupling F2 removed; the cache-shape and validation logic
re-fragments.
**Do this instead:** dim is a property of the `EmbeddingBackend` instance (`backend.dim`) and of the
data (`store.dim`); cache files keyed by model name.

### AP5: Monkey-patching for parallel sub-sessions
**What people do:** reach for eventlet/gevent to manage N concurrent sessions.
**Why wrong:** explicitly banned (CLAUDE.md) — corrupts numpy/sklearn/UMAP locking.
**Do this instead:** `threading.Thread` per sub-session (CPU-bound), gather results, merge on the main
thread; mind BLAS thread-pool oversubscription with N concurrent KMeans fits.

---

## Integration Points

### Summary table for roadmapper

| Feature | New components | Modified components | Seam used | Risk |
|---------|----------------|---------------------|-----------|------|
| F1 Colab | `src/artifacts.py`, `manifest.json` contract | web worker (import branch) | `EmbeddingStore.load` + `deserialize_state` | LOW |
| F2 Dynamic dim | `src/embedding_backend.py` (Protocol + 2 impls) | `embedding_store.py` (drop constant), `web/app.py` ×3 cache checks | `EMBEDDING_DIM` removal | MED (wide) |
| F3 Re-fit KMeans | `rebuild_state_from_refit` helper | `run_conversation` (Step 5.5) | `ClusteringBackend.fit` + loop body | MED-HIGH |
| F4 Oracle-initiated | pre-loop bootstrap block | `build_initial_clustering_state` (defer), loop entry | loop entry contract | MED |
| F5 Query filter | `src/query_filter.py` | `run_conversation` (one step) | between `oracle.reply` and `parse_feedback` | LOW-MED |
| F6 Coordination | coordination agent + pure `merge()` | new layer above `run_conversation` | no-I/O `run_conversation` path | **HIGH** |
| F7 Interactive UMAP | — | projection helpers, all 3 workers | `post_turn_callback` / cached coords | LOW |

### Internal Boundaries (preserved invariants)

| Boundary | Rule v2 must keep |
|----------|-------------------|
| Loop ↔ pure functions | Loop owns ALL I/O; `f_*` stay pure. Re-fit (F3) is called FROM the loop, not inside `f_next_state`. |
| EmbeddingStore | Read-only, computed once. Re-fit reads, never recomputes. |
| audit_log.jsonl | Source of truth for replay; ONE state line per turn; coordination agent writes ONE merged line. |
| events.jsonl | Sidecar only (intro turn, oracle_init, drift). Never a state line (Pitfall 5). |
| ClusteringState | Single source of truth — F6 keeps ONE authoritative state, sub-sessions are ephemeral. |
| `src/db/` | Still the ONLY SQL layer. Sub-sessions don't touch DB. |

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Colab (HuggingFace) | Compute-only producer; file artifacts on disk | Local app imports zero Colab-only deps; manifest validates contract |
| OpenAI `text-embedding-3-small` | New `EmbeddingBackend` impl (1536-dim) | API-call error handling only at the boundary (per CLAUDE.md) |

## Sources

- Direct source reading (HIGH confidence — verified, not inferred):
  - `src/conversation_loop.py` — loop steps, I/O ownership, no-I/O test mode (`db_conn`/`socketio` None)
  - `src/clustering.py` — `ClusteringBackend` Protocol, `KMeansBackend.fit`, `build_initial_clustering_state`
  - `src/embedding_store.py` — `EMBEDDING_DIM` constant + all assert sites
  - `src/agent_functions.py` — `f_next_state`, split/merge/move soft-prob mechanics
  - `src/feedback_parser.py` — `parse_feedback` contract (insertion point for F5)
  - `web/app.py` — projection helpers, `_should_recompute_projection`, 3 worker copies, `EMBEDDING_DIM` cache-check sites
  - `src/state.py` — `ClusteringState` (frozen schema, single source of truth)
  - `.planning/PROJECT.md` — v2.0 milestone goals + locked decisions
- `grep EMBEDDING_DIM` across `*.py` — confirmed 4 assert/check sites in `embedding_store.py` + 3 cache-shape sites in `web/app.py`

---
*Architecture research for: conversational-clustering v2.0 feature integration*
*Researched: 2026-05-21*
