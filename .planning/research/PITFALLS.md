# Pitfalls Research

**Domain:** Adding compute-offload (Colab), pluggable embedding/clustering backends, per-query re-fit, oracle-initiated flow, query filter, and a coordination agent to an existing conversational-clustering research system (milestone v2.0).
**Researched:** 2026-05-21
**Confidence:** HIGH for code-grounded pitfalls (read `src/clustering.py`, `src/embedding_store.py`, `src/state.py`, `src/mapping.py`, `src/feedback.py`); MEDIUM for external-API behavior (verified against OpenAI + sentence-transformers docs).

> **Scope note:** These pitfalls are specific to *adding the v2.0 features to this codebase*, not generic ML mistakes. Every prevention respects the two hard project rules: **K changes ONLY via oracle intent** (no auto K-selection) and **NEVER eventlet/gevent** (they monkey-patch stdlib and corrupt numpy/sklearn/UMAP locking). Phases referenced are the v2.0 phases 7–12.

---

## Phase map used in this document

These are *suggested* phase slots for the roadmapper. Ordering follows the locked decisions (Colab compute-only early; coordination agent last).

| Phase | Working title |
|-------|---------------|
| 7 | Pluggable embedding backends + dynamic dim (drop hardcoded 384) |
| 8 | Colab compute-only artifact pipeline (embeddings + initial clustering) |
| 9 | KMeans-only + stable per-query re-fit (cluster-ID stability layer) |
| 10 | Oracle-initiated flow + query filter |
| 11 | Interactive UMAP with cached coords (recolor-not-refit) |
| 12 | Coordination agent (N parallel sessions) — highest risk, last |

---

## Critical Pitfalls

### Pitfall 1: Cluster IDs churn on every per-query re-fit, silently breaking merge/split history and the UMAP recolor

**What goes wrong:**
KMeans assigns cluster *labels* arbitrarily — the cluster that was "id 2" last turn becomes "id 0" this turn even if the geometry barely changed. Worse, `build_initial_clustering_state` re-derives IDs with `enumerate(sorted(set(labels)), start=1)`, so IDs are positional, not semantic. After a per-query re-fit, every downstream structure that keys on cluster id breaks: `MergeFeedback(cluster_a_id, cluster_b_id)`, `SplitFeedback(cluster_id)`, `MoveItemFeedback(target_cluster_id)`, the merge/split lineage, the cluster name carried in `Cluster.name`, and the UMAP recolor (points jump colors for no visible reason). The `Cluster` docstring already promises "id... NEVER reused after deletion (D-11)" — naive re-fit violates this immediately.

**Why it happens:**
Developers treat re-fit as "just call `KMeansBackend.fit()` again." KMeans/Lloyd has no notion of label continuity between fits; centroid order depends on init. The existing code never had to solve this because v1 fit once and then applied *deltas* to a stable state.

**How to avoid:**
Introduce a **cluster-identity alignment step** that runs after every re-fit and maps new clusters back to previous IDs before constructing the new `ClusteringState`. Use the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on a cost matrix of overlap (Jaccard of item sets) or centroid cosine distance between old and new clusters. Preserve old IDs, names, and descriptions for matched clusters; only mint a new ID when K genuinely increased via oracle intent. Persist a `previous_state` reference so alignment has something to align against. Keep the existing "id never reused" invariant as an assertion.

**Warning signs:**
Cluster names that no longer match their contents after a query; merge/split throwing `KeyError`/`ValueError` on a cluster_id that "existed last turn"; UMAP colors reshuffling wholesale when only a small region changed; oracle saying "you renamed everything."

**Phase to address:** Phase 9 (this is the central engineering risk of per-query re-fit — build the alignment layer here, before UMAP in Phase 11 depends on it).

---

### Pitfall 2: KMeans nondeterminism makes "the same query" produce different clusterings

**What goes wrong:**
Even with identical embeddings and identical K, two re-fits can produce different assignments because of `n_init` random restarts and `random_state`. The current `KMeansBackend.fit()` hardcodes `random_state=0, n_init=10`, which is good — but a per-query re-fit path that rebuilds the backend, or a Colab-side fit using different sklearn defaults, can drift. Nondeterminism destroys reproducibility (a locked project requirement: "headline claims need CIs", "held-out split locked") and makes the alignment step in Pitfall 1 chase phantom changes.

**Why it happens:**
A new `KMeansBackend()` is instantiated per query (forgetting to pin seed), or sklearn version differs between Colab and local (default `n_init` changed from 10 to `"auto"` in scikit-learn 1.4), or someone "improves" convergence by raising `n_init` without pinning the seed.

**How to avoid:**
Pin `random_state` everywhere KMeans/GMM is constructed (already done — keep it). Pin the sklearn version in `requirements.txt` and assert it matches on Colab. Add a determinism test: fit twice on the same embeddings + same K, assert identical labels. Log the seed and sklearn version into the AuditLog every turn (the JSONL is the replay source of truth per CLAUDE.md).

**Warning signs:**
A determinism unit test that flakes; re-running an experiment produces different turn counts; CIs that won't reproduce; alignment step reporting changes when the oracle did nothing.

**Phase to address:** Phase 9 (re-fit), with a version-pinning check shared with Phase 8 (Colab).

---

### Pitfall 3: K drifts unintentionally during re-fit — violating "K changes ONLY via oracle intent"

**What goes wrong:**
The current `KMeansBackend` selects K once via BIC in `_select_k_via_bic` then fixes it. A per-query re-fit can accidentally re-run BIC selection (because `self._k` is reset when the backend is recreated per query), silently re-optimizing K every turn. This is the project's explicit **anti-feature** ("Automatic K optimization — anti-feature; K changes only through oracle intent"). It also interacts catastrophically with Pitfall 1: K changing means the alignment cost matrix is non-square and IDs churn legitimately.

**Why it happens:**
`self._k` is instance state; re-instantiating the backend per query loses it and triggers `_select_k_via_bic` again. Or the query filter / global feedback path triggers a "fresh" clustering that forgets the current K.

**How to avoid:**
Store the current K in the durable `ClusteringState` / session, not only in the backend instance. On re-fit, **pass K in explicitly** (`backend._k = current_k`) and assert K is unchanged unless the triggering feedback was a `SplitFeedback` (K+1) or `MergeFeedback` (K−1). Make BIC selection reachable ONLY on the very first clustering (turn 0 / first oracle query), never on subsequent re-fits. Add an assertion: `assert new_k == prev_k or feedback_changed_k`.

**Warning signs:**
K oscillating turn-to-turn in the AuditLog; BIC code executing after turn 0; number of clusters changing without a corresponding merge/split feedback delta.

**Phase to address:** Phase 9. Cross-check in Phase 10 (query filter must not silently emit "re-cluster fresh" that re-selects K).

---

### Pitfall 4: Embedding dimension mismatch — 384 hardcoded everywhere collides with 1536 (OpenAI)

**What goes wrong:**
`EMBEDDING_DIM = 384` is hardcoded in `embedding_store.py` and asserted in `EmbeddingStore.__init__` (`embeddings.shape[1] == EMBEDDING_DIM`). Swapping in OpenAI `text-embedding-3-small` (1536-dim) crashes that assert — which is correct fail-loud behavior — but the *real* danger is the silent path: loading a `.npy` produced by a different backend than the one configured, so K-means fits on 1536-dim vectors while the centroid-mapping generalization layer (`CentroidMappingStrategy`) re-embeds new items with the *local* MiniLM model (384-dim), producing a `(1536,) · (384,)` shape error or, worse, a quiet broadcast bug. Note `mapping.py` imports `EMBEDDING_MODEL` and re-encodes at inference time — that model MUST match whatever produced the stored vectors.

**Why it happens:**
"Dynamic dim" is implemented as "remove the assert" instead of "thread the actual dim through". The dim and the model name are two separate facts that must be stored *together with the artifact* and validated on load.

**How to avoid:**
Replace the module constant with **dim and model recorded in the embeddings artifact's sidecar metadata** (e.g. `embeddings.meta.json` carrying `{model, dim, normalized, lib_version, n_items}`). `EmbeddingStore.load()` reads dim from metadata and asserts `embeddings.shape[1] == meta.dim`. Store the embedding model identity in `ClusteringState`/session and assert the mapping layer re-embeds with the same model. Make `CentroidMappingStrategy` take the backend/model from session config, not the module-level `EMBEDDING_MODEL`.

**Warning signs:**
`AssertionError: Expected embedding dim 384, got 1536`; cosine/dot-product shape mismatch in `CentroidMappingStrategy.assign`; generalization accuracy collapsing because new items are embedded in a different space than the corpus.

**Phase to address:** Phase 7 (this is the core of "pluggable backends + dynamic dim"). Must land before Phase 8 stores Colab-produced artifacts.

---

### Pitfall 5: Silent normalization mismatch between OpenAI and sentence-transformers under L2 KMeans

**What goes wrong:**
OpenAI embeddings are returned as **unit-normalized** vectors (length 1). `all-MiniLM-L6-v2` *also* outputs L2-normalized vectors via its Normalize module — but only when encoded through the full SentenceTransformer pipeline; raw transformer output or other models (e.g. `all-mpnet`, which the stale docstrings still reference) are NOT necessarily unit length. The existing `KMeansBackend._compute_soft_probs` uses **raw L2 distance**, and `_select_k_via_bic` uses a Gaussian mixture — both are scale-sensitive. Mixing a normalized backend and an unnormalized one (or comparing runs across backends) makes distances, softmax temperatures, and BIC scores non-comparable, quietly skewing cluster shapes and soft-prob calibration without any crash.

**Why it happens:**
Cosine-vs-L2 assumptions are invisible. Developers assume "embeddings are embeddings." OpenAI's API doesn't expose a `normalize` flag (always normalized); sentence-transformers' `encode()` does NOT normalize unless `normalize_embeddings=True` is passed OR the model has a Normalize layer — easy to get wrong when adding a new model.

**How to avoid:**
**Normalize at the backend boundary, explicitly, for every backend.** Each embedding backend's contract must return unit-norm vectors (assert `np.allclose(norms, 1.0)` after encode). Record `normalized: true` in artifact metadata (Pitfall 4). Document that KMeans here is effectively spherical/cosine once inputs are unit-norm. Do NOT compare distances or BIC across runs with different normalization regimes.

**Warning signs:**
Soft-probs nearly uniform or nearly one-hot after a backend swap (temperature now mis-scaled); BIC picking wildly different K for the "same" data across backends; cluster quality dropping only for one backend.

**Phase to address:** Phase 7 (define the normalize-at-boundary contract when introducing the backend Protocol).

---

### Pitfall 6: OpenAI embeddings cost / rate-limit / batch-limit footguns, and no cache → paying repeatedly

**What goes wrong:**
Naively embedding a 12K–15K corpus by looping one text per request: 12K HTTPS round-trips, easy to hit RPM/TPM rate limits (429s), and slow. Also: a single request array must be **≤ 2048 inputs** and each input **≤ 8192 tokens**; oversized batches 400-error. Most expensive failure: re-embedding the whole corpus on every session/experiment run because there's no cache — directly violating the core invariant ("embeddings computed once; never re-embed per turn") and burning real money on every ablation.

**Why it happens:**
The v1 path computed embeddings locally for free, so cost discipline was never needed. The OpenAI path makes re-embedding a billable event, and the existing `compute_and_save` guard (`assert not os.path.exists(embed_path)`) is the *only* thing preventing re-computation — fragile across machines/Colab.

**How to avoid:**
Batch encode (chunks of ≤ 2048, respect token budget), with retry-with-exponential-backoff ONLY at the API-call boundary (allowed per fail-loudly rule). **Content-hash cache**: key embeddings by `sha256(text + model_id)` so identical corpora never re-pay; persist the cache as the `.npy` + sidecar artifact and treat it as immutable. Make the OpenAI path go through the *same* compute-once-then-load pathway as local; never embed inside the turn loop. Log token counts and estimated cost.

**Warning signs:**
429 errors / `RateLimitError`; a session that's slow on *every* start (cache miss); OpenAI dashboard cost growing per experiment run; `BadRequestError` about array length > 2048 or token limit.

**Phase to address:** Phase 7 (backend + cache contract); reinforced in Phase 8 (Colab is where bulk embedding should run, so the cache must be portable).

---

### Pitfall 7: Colab artifact version skew — vectors that look fine but are from a different model/library

**What goes wrong:**
Colab computes embeddings with one `sentence-transformers`/`transformers`/`torch` version; the local machine loads them and clusters/maps with another. Even the *same model name* can yield numerically different vectors across library versions or CPU-vs-GPU/float precision. The `.npy` loads cleanly (right shape, right dtype) so nothing crashes — but the local `CentroidMappingStrategy` re-embeds new items in a *slightly different* space, and reproducibility silently breaks. Also `.npy` dtype/shape mismatches: Colab may save float64 or a transposed array; the code assumes float32 `(N, dim)`.

**Why it happens:**
Colab pins nothing by default and upgrades packages frequently; "it ran in Colab" feels authoritative. Artifacts carry no provenance.

**How to avoid:**
Ship a **provenance sidecar** with every artifact: `{model_id, lib_versions, device, dtype, dim, n_items, seed, sha256_of_input}`. On local load, assert dtype `float32`, shape `(N, dim)`, dim matches, and **model_id + major lib versions match the local mapping model**; crash loudly on mismatch (this is exactly the "fail loudly" philosophy). Pin versions in a `requirements-colab.txt` that mirrors local. Cast to float32 explicitly on save. Prefer producing the *initial clustering* in Colab too (the milestone allows this) so the local side only consumes, reducing cross-environment fit divergence.

**Warning signs:**
Generalization accuracy that differs between "Colab vectors" and "local vectors" runs; `np.load` returning float64; clustering that looks subtly different after regenerating embeddings; silent dtype upcasting slowing KMeans.

**Phase to address:** Phase 8 (Colab pipeline) — but the metadata schema is shared with Phase 7's artifact format.

---

### Pitfall 8: Colab session timeout / GPU variance corrupts or half-writes the artifact

**What goes wrong:**
Colab disconnects (idle/max-runtime/GPU eviction) mid-embedding, leaving a truncated `.npy` or a partial upload to Drive/storage. Loading it later: either a shape that's `(K, dim)` with `K < N` (caught by the `len(records) == len(embeddings)` assert — good) or, if rows-per-text mapping drifts, vectors silently misaligned to the wrong texts (NOT caught — catastrophic). GPU availability varies (sometimes no GPU, sometimes T4 vs A100), changing throughput and occasionally numeric output.

**Why it happens:**
Long single-shot encode jobs on free Colab; no checkpointing; saving directly to the final path instead of write-temp-then-rename.

**How to avoid:**
Write artifacts atomically: encode to a temp file, `fsync`, then rename to final path only on success. Save `n_items` in metadata and assert `len == n_items` on load (extends the existing record-count assert to cross-environment). Checkpoint long encodes in chunks so a disconnect resumes rather than restarts. Keep embedding *order* identical to input order and store the `sha256_of_input` so misalignment is detectable. Don't depend on a specific GPU; pin seed and accept that GPU is a throughput convenience, not a correctness dependency.

**Warning signs:**
`len(records) != len(embeddings)` assert firing after a Colab run; a `.npy` smaller than expected; Drive showing an in-progress/partial file; embeddings that cluster "wrong" because rows are off-by-some.

**Phase to address:** Phase 8.

---

### Pitfall 9: Oracle-initiated flow with an empty/degenerate first query — "cluster on what?" before any clustering exists

**What goes wrong:**
The new flow is: dataset intro → oracle's first query → first clustering. If the oracle's first query is empty, vague ("just cluster it"), or contradictory, there's no prior state to fall back on (unlike v1, which always started from a default clustering). Naive handling either crashes on `state=None` in code paths that assume a `ClusteringState` exists (e.g. the query filter expects clusters to reference), or fabricates a degenerate K=1/K=N clustering. The mapping/feedback layers assume `len(state.clusters) > 0` (asserts in `mapping.py`).

**Why it happens:**
Inverting the loop (oracle-first instead of system-first) removes the guaranteed turn-0 state that every downstream component implicitly relied on.

**How to avoid:**
Define an explicit **bootstrap contract**: the first oracle query is interpreted by the query filter into an initial clustering *instruction*, and only then is the first KMeans fit run (with BIC K-selection allowed exactly here — see Pitfall 3). If the first query is empty/degenerate, fall back to a documented default ("cluster by overall topic") rather than crashing or producing K=1 — and surface that to the oracle. Keep the "always a complete assignment" invariant: once the first fit runs, `f_output` returns full state. Add an assert that no feedback/filter code runs before the first clustering exists.

**Warning signs:**
`NoneType has no attribute clusters`; K=1 or K=N initial clustering; query filter invoked with no clusters to reference; oracle's first message producing an empty cluster set.

**Phase to address:** Phase 10 (oracle-initiated flow + query filter are co-designed).

---

### Pitfall 10: Query filter over-filters, dropping oracle intent — or re-introduces contradictions the feedback layer already guards

**What goes wrong:**
The query filter translates NL oracle queries into "simple, contradiction-free clusterer instructions." Two failure modes: (a) **over-filtering** — it strips nuance and the oracle's actual intent is lost ("group by sentiment AND topic" → "group by topic"), so the system optimizes the wrong objective and the oracle gets frustrated; (b) **re-introducing contradictions** — it emits instructions that conflict with the existing "latest intent wins on contradictions" layer, producing double-handling or oscillation. The existing `feedback.py` already has type-priority ordering (global→split→merge→move→instructional, D-07) and a contradiction policy; a second filter that *also* resolves contradictions creates two competing arbiters.

**Why it happens:**
The filter is built as a standalone LLM prompt without a defined contract about *what it is allowed to drop* and *who owns contradiction resolution*. "Make it simple" is interpreted as "make it lossy."

**How to avoid:**
Define the filter's job narrowly: **normalize/translate, do not arbitrate.** Contradiction resolution stays in the existing feedback layer (single owner — respects single-source-of-truth). The filter should preserve all distinct intents as separate `FeedbackDelta` objects rather than collapsing them. Log the original query alongside the filtered output in the AuditLog so over-filtering is auditable/measurable. Add a regression test: known multi-intent queries must yield multiple deltas, not one. Treat "latest intent wins" as the *only* contradiction rule.

**Warning signs:**
Oracle repeatedly restating the same intent (signal it was dropped); two layers both rewriting/reordering feedback; oscillating clusterings turn-to-turn; AuditLog showing filtered instruction much shorter/simpler than the query in ways that lose constraints.

**Phase to address:** Phase 10.

---

### Pitfall 11: Stale UMAP geometry shown after re-clustering — recolor-vs-refit confusion

**What goes wrong:**
The decision is "cache 2D coords, **recolor** on cluster change, don't **refit** the projection." Done wrong, the system either (a) re-runs UMAP on every cluster change — slow, and worse, the layout *moves* so the oracle thinks the data changed when only labels did (and UMAP is itself nondeterministic without a fixed seed, compounding Pitfall 2's reproducibility issue); or (b) caches coords but recolors using *churned* cluster IDs (Pitfall 1), so colors are wrong even though positions are right; or (c) shows positions from the *original* embedding space while clusters now reflect a re-fit, so the geometry and the coloring disagree and points appear mis-colored relative to their neighbors.

**Why it happens:**
UMAP is expensive, so caching is obviously right — but the relationship "coords are a fixed function of the (immutable) embeddings; colors are a function of the (changing) assignments" isn't made explicit. Re-fit changes colors, not positions — but only if cluster IDs are stable.

**How to avoid:**
Compute UMAP **once** from the immutable embeddings (with a fixed `random_state`) at artifact-build time (ideally in Colab alongside embeddings — milestone-aligned), store coords in the artifact. On every turn, only re-map `assignment[item] → color`; never recompute coords. This **hard-depends on Pitfall 1's ID-stability layer** — recolor is only meaningful if IDs are aligned across re-fits. Document the invariant: "positions = f(embeddings) fixed; colors = f(assignments) per turn." Since embeddings are read-only and never re-embedded, coords are legitimately permanent.

**Warning signs:**
Points visibly relocating after a re-cluster; oracle confusion ("why did everything move?"); colors not matching spatial clusters; UMAP recompute appearing in turn-loop timing logs.

**Phase to address:** Phase 11 (depends on Phase 9 ID-stability and Phase 8 artifact coords).

---

### Pitfall 12: Coordination agent — state-merge conflicts and partial failures across N sessions break single-source-of-truth

**What goes wrong:**
The coordination agent decomposes complex operations into pairwise sub-operations across N clusterer sessions. Risks: (a) **two sessions mutate overlapping cluster state** and a naive merge double-applies or loses a feedback delta; (b) **partial failure** — 3 of 5 sub-sessions succeed, 2 crash (correct, fail-loudly!), leaving a half-merged global state with no clear rollback, violating "f_output always returns a complete, consistent assignment"; (c) **single-source-of-truth violation** — each session keeps its own `ClusteringState` and there's no defined authoritative merged state; the AuditLog (replay source of truth) can't represent N concurrent writers cleanly. With ASGI/asyncio (no eventlet/gevent), CPU-bound fits already run in `threading.Thread` and emit cross to the loop via `run_coroutine_threadsafe` — N sessions multiply the thread-to-loop coordination surface.

**Why it happens:**
Parallelism is added for scale without first defining the merge semantics and the authoritative state owner. "Pairwise sub-operations" implies a reduce step that's easy to get non-associative or non-idempotent.

**How to avoid:**
Defer to last (already decided — good). Define **one authoritative merged `ClusteringState`** and make sub-sessions produce *proposals* that the coordinator applies sequentially through the existing single-writer feedback pipeline (idempotent, ordered, latest-intent-wins). Make sub-operations idempotent and replayable. On partial failure, **fail loudly and abort the whole coordinated op** (don't commit a partial merge); the last good `ClusteringState` in the AuditLog is the recovery point. Keep emits crossing threads via `run_coroutine_threadsafe`; do NOT reach for eventlet/gevent to "simplify concurrency" — they corrupt numpy/sklearn/UMAP. Serialize the merged state to the single AuditLog only after a full successful reduce.

**Warning signs:**
Cluster item counts not summing correctly after a merge; a feedback delta applied twice; partial state committed after a sub-session crash; any import of eventlet/gevent; race conditions in the socket emits under N sessions.

**Phase to address:** Phase 12 (last, by design).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Keep `EMBEDDING_DIM = 384` constant, just relax the assert for OpenAI | Fast backend swap | Dim/model facts drift apart; mapping layer re-embeds in wrong space (Pitfall 4) | Never — thread dim through the artifact metadata in Phase 7 |
| Re-fit KMeans without an ID-alignment step | Ships per-query re-fit quickly | Merge/split history and UMAP recolor silently break (Pitfalls 1, 11) | Never — alignment is the point of Phase 9 |
| Embed via OpenAI inside the turn loop / per session start | No cache plumbing | Re-pays per run, violates compute-once invariant, hits rate limits (Pitfall 6) | Never — always compute-once + content-hash cache |
| Save Colab `.npy` with no provenance sidecar | One less file | Silent version/space skew, undetectable misalignment (Pitfalls 7, 8) | Never for shared artifacts; tolerable only for throwaway local experiments |
| Recompute UMAP per turn for "freshness" | No caching code | Layout jitters, nondeterministic, slow, confuses oracle (Pitfall 11) | Never — coords are f(immutable embeddings), compute once |
| Let the query filter resolve contradictions too | One LLM call does everything | Two contradiction arbiters; oscillation; lost intent (Pitfall 10) | Never — filter normalizes, feedback layer arbitrates |
| Commit partial coordinated merge on sub-session failure | "Some progress" | Inconsistent global state, broken replay (Pitfall 12) | Never — abort whole op, fail loudly |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenAI embeddings API | One request per text; ignoring 2048-input / 8192-token limits; no backoff | Batch ≤ 2048 inputs respecting token budget; retry-with-backoff at the API boundary only; content-hash cache |
| OpenAI vs sentence-transformers vectors | Assuming both are unit-norm and L2-comparable | Normalize at backend boundary, assert unit norm, record `normalized` in metadata; never compare distances/BIC across normalization regimes |
| Colab → local artifact handoff | Trusting a clean `.npy` load as correctness | Provenance sidecar (model, lib versions, dtype, dim, n_items, input hash); assert all on load; atomic temp-then-rename writes |
| sklearn KMeans across environments | Relying on default `n_init`/`random_state` | Pin `random_state` and explicit `n_init`; pin sklearn version Colab==local; determinism test |
| python-socketio under N sessions | Reaching for eventlet/gevent for concurrency | Keep asyncio + `threading.Thread` + `run_coroutine_threadsafe`; never monkey-patch (corrupts numpy/sklearn/UMAP) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-query KMeans re-fit on full corpus | Each oracle turn takes seconds–minutes | Re-fit is acceptable per the milestone (embeddings fixed); pin K, warm-start centroids from prev fit where valid; profile | Noticeable at ~10K+ items per turn |
| UMAP recomputed per turn | Multi-second stalls every cluster change | Compute coords once from immutable embeddings, recolor only | Any corpus where UMAP > ~1s |
| OpenAI re-embedding per run | Slow session start every time + cost | Content-hash cache, compute-once | Every experiment run / ablation sweep |
| N coordinated sessions each holding full embeddings in RAM | Memory blow-up | Share one read-only `EmbeddingStore`; sessions hold only state/assignments | At large N × large corpus |

## Security / Integrity Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| OpenAI API key in code or committed artifacts/notebooks | Leaked key, billing abuse | Keys in `.env` only (existing pattern); never in Colab notebook cells or sidecar metadata |
| Treating Colab artifacts as trusted without verification | Wrong-space vectors silently used in headline experiments → invalid CIs | Verify provenance + input hash before any experiment; held-out split stays locked |
| Mutating embeddings artifact in place | Breaks read-only invariant and reproducibility | Artifacts immutable; new model = new artifact + new hash |

## UX Pitfalls (oracle-facing)

| Pitfall | Oracle Impact | Better Approach |
|---------|---------------|-----------------|
| Cluster colors/IDs reshuffling after a query | Oracle thinks system "redid everything"; cognitive-load spike | ID-stability alignment + recolor-only UMAP (Pitfalls 1, 11) |
| Over-filtered first query producing a degenerate clustering | Oracle starts from nonsense, loses trust | Bootstrap contract + documented default + surface the interpretation (Pitfall 9) |
| Silent dropping of multi-intent queries | Oracle repeats themselves, frustration | Filter preserves distinct intents as separate deltas; log original query (Pitfall 10) |
| UMAP points moving between turns | "Why did the map change?" — misreads label change as data change | Fixed coords; positions never move (Pitfall 11) |

## "Looks Done But Isn't" Checklist

- [ ] **Pluggable backends:** Often missing the *metadata sidecar* — verify dim AND model AND normalized flag travel with the `.npy` and are asserted on load.
- [ ] **Per-query re-fit:** Often missing the *ID-alignment step* — verify cluster IDs/names survive a re-fit when the oracle didn't change K (run a no-op query, assert IDs unchanged).
- [ ] **Re-fit:** Often missing *K pinning* — verify BIC runs only at turn 0; assert K unchanged on subsequent re-fits unless split/merge feedback present.
- [ ] **OpenAI embeddings:** Often missing the *cache* — verify second session start is instant (cache hit) and no API call fires.
- [ ] **Colab artifacts:** Often missing *atomic writes + n_items assert* — verify a truncated artifact is rejected loudly, not loaded misaligned.
- [ ] **UMAP:** Often missing the *recolor-not-refit* guarantee — verify coords are byte-identical across turns; only colors change.
- [ ] **Query filter:** Often missing *contradiction-ownership separation* — verify the filter doesn't re-order/resolve; the feedback layer remains the only arbiter.
- [ ] **Oracle-initiated flow:** Often missing the *empty-first-query* path — verify a blank/vague first query yields a documented default, not a crash or K=1.
- [ ] **Coordination agent:** Often missing *partial-failure rollback* — verify a sub-session crash aborts the whole op and leaves the last good state intact.
- [ ] **All concurrency:** Verify zero imports of eventlet/gevent across new code.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Cluster-ID churn (1) | MEDIUM | Replay AuditLog with the alignment layer inserted; re-derive stable IDs from item-set overlap across the recorded sequence |
| KMeans nondeterminism (2) | LOW | Pin seed + sklearn version, add determinism test, re-run affected experiments |
| Unintentional K drift (3) | LOW | Persist K in session, gate BIC to turn 0, re-run |
| Dim mismatch (4) | LOW (fails loudly) | Add metadata sidecar; regenerate artifact with correct backend |
| Normalization mismatch (5) | MEDIUM | Re-normalize at boundary; re-fit; discard cross-regime comparisons |
| OpenAI cost/limits (6) | LOW | Add batching + backoff + content-hash cache; bulk-embed in Colab |
| Colab version skew (7) | MEDIUM | Pin `requirements-colab.txt`; regenerate artifacts; verify input hash |
| Colab truncation (8) | LOW (assert catches len) | Atomic writes + checkpointed encode; regenerate |
| Empty first query (9) | LOW | Add bootstrap default + assert-before-first-clustering |
| Over-filtering (10) | MEDIUM | Re-scope filter to normalize-only; add multi-intent regression test |
| Stale UMAP (11) | LOW | Compute coords once from embeddings; recolor only |
| Coordination merge (12) | HIGH | Abort to last good state in AuditLog; redesign merge as ordered single-writer reduce |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1 — Cluster-ID churn | 9 | No-op query leaves IDs/names unchanged; merge/split still resolve |
| 2 — KMeans nondeterminism | 9 (+8 versions) | Determinism test: two fits → identical labels |
| 3 — K drift | 9 (+10 filter check) | BIC executes only turn 0; K unchanged assert holds |
| 4 — Dim mismatch | 7 | Load asserts dim==meta.dim; mapping re-embeds in same space |
| 5 — Normalization mismatch | 7 | Post-encode unit-norm assert per backend |
| 6 — OpenAI cost/limits/cache | 7 (+8 bulk) | Second session start = cache hit, no API call |
| 7 — Colab version skew | 8 (schema w/ 7) | Provenance asserts; input-hash match |
| 8 — Colab truncation/GPU | 8 | Truncated artifact rejected; n_items assert |
| 9 — Empty first query | 10 | Blank first query → documented default, no crash |
| 10 — Query filter | 10 | Multi-intent query → multiple deltas; filter doesn't arbitrate |
| 11 — Stale UMAP | 11 (needs 9, 8) | Coords byte-identical across turns; colors change only |
| 12 — Coordination merge | 12 | Partial failure aborts whole op; single authoritative state; no eventlet/gevent |

## Sources

- Existing code (HIGH): `src/embedding_store.py` (hardcoded `EMBEDDING_DIM=384`, read-only store, compute-once guard), `src/clustering.py` (`KMeansBackend`, `random_state=0/n_init=10`, `_select_k_via_bic`, raw-L2 soft probs, positional ID remap), `src/state.py` (`Cluster.id` "never reused" invariant), `src/mapping.py` (`CentroidMappingStrategy` re-embeds with module-level `EMBEDDING_MODEL`), `src/feedback.py` (type-priority D-07, latest-intent-wins).
- `.planning/PROJECT.md` and `CLAUDE.md` (HIGH): locked decisions (re-cluster not re-embed; Colab compute-only; YAML prompts; coordination last), K-only-via-oracle anti-feature rule, no-eventlet/gevent rule, AuditLog as replay source of truth, fail-loudly philosophy.
- [Vector embeddings | OpenAI API](https://developers.openai.com/api/docs/guides/embeddings) (MEDIUM): embeddings returned unit-normalized; ≤2048 inputs/array; 8192 max input tokens.
- [text-embedding-3-small Model | OpenAI API](https://platform.openai.com/docs/models/text-embedding-3-small) (MEDIUM): 1536 dimensions.
- [sentence-transformers/all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (MEDIUM): 384-dim, L2-normalized output via Normalize module; `normalize_embeddings` controls explicit normalization.
- scikit-learn KMeans `n_init="auto"` default change (≥1.4) (MEDIUM, training-data + changelog knowledge): motivates version pinning Colab==local.

---
*Pitfalls research for: adding compute-offload + pluggable backends + per-query re-fit + oracle-initiated flow + query filter + coordination agent to a conversational-clustering research system*
*Researched: 2026-05-21*
