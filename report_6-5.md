# Report 6-5 — OpenAI fallback + LLM-oracle web viewer

Session 2026-05-18. Two related changes implemented together so the existing
OpenAI key in `.env` can drive both the CLI harness and a new browser-watch UI
for the LLM oracle.

## Context

- `.env` had only `OPENAI_API_KEY`. Two hard `assert provider == "anthropic"` gates were crashing the CLI harness (`src/harness.py`) and the human study route (`web/app.py`).
- All nine LLM call sites used Anthropic's call shape (`client.messages.create(...)` + `response.content[0].text`). OpenAI uses a different shape (`chat.completions.create(...)` + `response.choices[0].message.content`).
- No way to *watch* the LLM oracle dialogue in the browser — `/` uses `MockOracle` (no real dialogue) and `/study/` waits on human input. Wanted a third path for observation.

## Approach

One thin provider-agnostic helper, no class hierarchy. Every existing 3-line Anthropic block becomes one `chat(...)` call. Watch viewer reuses the study cluster/UMAP rendering but replaces the human-input loop with an `OracleAgent` driver.

## Files changed / created

### New files

| Path | Purpose |
|---|---|
| `src/llm_call.py` | `build_client(provider, api_key) -> (provider, sdk_client)` + `chat(client_tuple, *, system, user, max_tokens, model=None) -> str`. `DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5", "openai": "gpt-4.1-nano"}`. Lazy SDK imports, fail-loudly. |
| `web/templates/watch.html` | Three-column layout — cluster cards / UMAP grid / chat log. `data-mode="watch"` on body. No human input. |
| `experiments/configs/smoke.yaml` | 1-strategy × 1-persona × 1-seed harness config (`random` × `curious` × `1`). |

### Fix A — OpenAI fallback

| File | Change |
|---|---|
| `src/llm_key.py` | `_PRIORITY` reduced to `[("openai", "OPENAI_API_KEY")]`. Anthropic + Google commented out. Docstring + error message updated. |
| `src/oracle_agent.py` | Dropped `isinstance(self._client, anthropic.Anthropic)` branch and the dual try/except. One `chat(...)` call. `model` default changed to `None` (resolves from `DEFAULT_MODELS[provider]` inside `chat`). |
| `src/feedback_parser.py` | `client.messages.create(...)` → `chat(client, system=None, user=prompt, max_tokens=512)`. Argument doc updated to "provider tuple". |
| `src/mapping.py` | Removed `import anthropic`. Both `anthropic.Anthropic() + messages.create + try/except anthropic.APIError: raise` blocks → `chat(build_client(*resolve_llm_key()), ...)`. The re-raise was a no-op; deleted per fail-loudly. |
| `src/cluster_naming.py` | Added `make_namer(provider, api_key, model=None)` factory routing to existing `AnthropicClusterNamer` / `OpenAIClusterNamer` / `GoogleClusterNamer`. |
| `src/harness.py` | `_build_llm_client()` simplified to `return build_client(*resolve_llm_key())`. Anthropic-only assertion removed. `_build_oracle` docstring updated. |
| `examples/evaluate_mapping.py` | `anthropic.Anthropic()` + `client.messages.create(...)` → `build_client(*resolve_llm_key())` + `chat(...)`. Dropped `import anthropic`. |
| `tests/phase2/test_feedback_parser.py` | `_make_mock_client` now returns `("anthropic", mock_client)` tuple. Live `test_parse_feedback_real_llm` uses `resolve_llm_key()` instead of hard-coded Anthropic check. |

### Fix B — `/watch` LLM-oracle viewer

| File | Change |
|---|---|
| `web/app.py` (study path) | Removed Anthropic-only assertion at `_run_study_background:823`. Now uses `build_client()` + `make_namer()`. `_detect_satisfaction()` simplified to use `chat()`. |
| `web/app.py` (new routes) | `POST /watch/sessions` — body `{dataset_path, backend, persona, strategy, seed}`. Validates `persona ∈ {curious, skeptical, drifty}` and `strategy ∈ {random, uncertainty_driven, boundary_driven}`. Creates DB row with `strategy_id="oracle_watch"`, `oracle_type="llm_agent"`. Spawns `_run_watch_background`. `GET /watch/{id}` serves `watch.html`. `GET /watch/{id}/replay` returns cached `{projection, state, chat, ended}`. |
| `web/app.py` (worker) | `_run_watch_background(session_id)` mirrors `_run_study_background` up through clustering + projection, then replaces the `feedback_event.wait()` blocking with: `f_next_best_step` → `_format_message` → emit `watch_agent_message` → `oracle.reply` → emit `watch_oracle_turn` → `parse_feedback` → `f_next_state` → emit `study_state`. Stops on `reply.satisfied` or `STUDY_MAX_TURNS=30`. Loads personas from `experiments/configs/harness.yaml` via `_parse_personas`. |
| `web/app.py` (watch tolerance) | Wraps `parse_feedback`+`f_next_state` in try/except for `AssertionError`/`KeyError`/`ValueError`. On failure emits an `[apply-feedback error — skipping turn]` bubble and continues with unchanged state. The CLI/research path (`run_conversation`) still fails loudly per CLAUDE.md. |
| `web/app.py` (replay cache) | Worker now writes `sess["watch_projection"]`, `sess["watch_state"]`, `sess["watch_chat"]` so a late-loading or reloaded browser can hydrate via `/replay`. |
| `web/templates/watch.html` | New template, reuses the study three-column layout. Right panel is a `#chat-log` of `.bubble-agent` (left, blue) and `.bubble-oracle` (right, yellow) bubbles; `[SATISFIED]` bubbles get a green border. `body data-mode="watch"`. |
| `web/static/study.js` | Detects `body.dataset.mode === "watch"` and gates handlers. New `appendBubble(role, turn, text, satisfied)`. New SocketIO handlers `watch_agent_message`, `watch_oracle_turn`. New `hydrateWatchFromReplay()` called on `connect` — `fetch('/watch/<id>/replay')` then replays projection + state + bubbles. `study_ended` and `setStatus()` work in both modes. |
| `web/templates/index.html` | Added second form `#watch-form` with `<input>` dataset path + persona/strategy/backend `<select>`s. Inline JS `POST`s `/watch/sessions` and redirects to `watch_url`. |

### Bonus fixes (discovered while testing)

| File | Change | Reason |
|---|---|---|
| `web/app.py` (3 places) | Embedding cache validator changed from `if _cached_shape[0] == len(texts):` to `if _cached_shape == (len(texts), EMBEDDING_DIM):`. Added `EMBEDDING_DIM` to the three `from src.embedding_store import EmbeddingStore` imports. | Stale cache file `embeddings/embeddings.npy` was `(12000, 768)` from a previous `all-mpnet-base-v2` run; current code expects `(N, 384)` from `all-MiniLM-L6-v2`. Old validator passed on row count alone, then crashed inside `EmbeddingStore.__init__`. |
| `src/llm_call.py` and `src/cluster_naming.py` | Default OpenAI model `gpt-4o-mini` → `gpt-4.1-nano` (verified the latter exists in the active OpenAI account via `client.models.list()`). The `OpenAIClusterNamer.__init__` default also went from `gpt-5.4-nano` → `gpt-4.1-nano`. | User requested. |

## Naming choices (no `.planning/` collision)

- Route prefix: `/watch` (vs `/study`)
- DB `strategy_id`: `"oracle_watch"` (distinct from `human_study`, `interactive`, `random`, `uncertainty_driven`, `boundary_driven`, `no_dialogue`)
- DB `oracle_type`: `"llm_agent"`
- Session dir prefix: `watch-<timestamp>` (vs `study-<timestamp>`)
- Template: `web/templates/watch.html`
- SocketIO events: `watch_agent_message`, `watch_oracle_turn` (other events `study_state`, `study_projection`, `study_ended` reused)
- New helper module: `src/llm_call.py`

None of these collide with `.planning/` naming conventions (`NN-PLAN.md`, `NN-SPEC.md`, `NN-UAT.md`, `NN-CONTEXT.md`, etc.).

## Verification

| Check | Result |
|---|---|
| `from src.llm_call import build_client, chat, DEFAULT_MODELS` etc. all import | OK |
| `resolve_llm_key()` returns `('openai', 'sk-proj-…')` | OK |
| `chat(client, system='You are terse.', user='Say hi in 3 words.', max_tokens=20)` round-trip | `'Hello, how are you?'` |
| `pytest tests/phase2/test_feedback_parser.py` | 9/9 pass with new tuple-shape mock |
| Watch session id=6 (curious + random + seed=1) | `total_turns=2`, ended `oracle_satisfied`, two real `gpt-4.1-nano`-generated bubbles in DB |
| Watch session id=8 (skeptical + uncertainty_driven + seed=2) | `total_turns=30` (hit `turn_budget`), 61 bubbles in replay cache, 9 final clusters |
| `gpt-4.1-nano` existence check via `client.models.list()` | OK |

## Open issues / not fixed

1. **UMAP mini-plots not rendering on either `/` (debug) or `/watch/{id}` pages.** Cluster cards appear; `mini-plots-container` stays at "Projection loading...". `/replay` confirms the `projection` payload is cached and returned — the gap is on the client side. Suspected: `renderMiniPlots()` in `study.js` may have a guard, the canvases never get appended, or the projection event arrives before the DOM is ready. Not investigated in this session. The user flagged this twice in testing.
2. **HDBSCAN on AG-News 2000 rows produces only 2 broad clusters.** Initial state shows `Cluster 1 ('Diverse News Topics', 1494 items)` and `Cluster 2 ('Diverse News and Events', 506 items)`. May require backend tuning (`min_cluster_size`, `min_samples`) or trying `kmeans` backend for finer granularity. Not a Fix A/B regression — it's the data + default HDBSCAN params.
3. **`parse_feedback` over-validates against initial state.** When the LLM returns multiple deltas in one turn and the first removes a cluster the second references, `_apply_merge` / `_apply_split` crash with `cluster_X_id N not found in state`. The watch loop now catches and skips the turn; the CLI/research path still fails loudly. A proper fix would re-validate each delta against the *current* (after-prior-deltas) state inside `f_next_state`. Out of scope for this session.
4. **`gpt-4.1-nano` `[SATISFIED]` discipline.** Curious persona converged in 2 turns; skeptical ran the full budget. Single sample only; if convergence behaviour deviates from Claude in research runs, may need a stronger system-prompt directive (e.g. `End with exactly the token [SATISFIED] when satisfied.`).
5. **Watch route is unauthenticated** and burns LLM tokens. Hard cap is `STUDY_MAX_TURNS=30`. Fine for solo testing; would need rate-limiting before sharing.

## How to test locally

```
# CLI (Fix A) — uses experiments/configs/smoke.yaml (1 combo)
python -m examples.run_harness --config experiments/configs/smoke.yaml

# Web (Fix B)
uvicorn web.app:asgi_app --host 127.0.0.1 --port 5000
# browser → http://127.0.0.1:5000 → "Watch LLM Oracle" form
#   dataset = dataset/ag_news_working.csv
#   persona = skeptical (or curious / drifty)
#   strategy = uncertainty_driven (or random / boundary_driven)
# → click Start, redirected to /watch/<session_id>
# → page is hydrated via /watch/<session_id>/replay on connect
```

Cross-check via DB:

```python
from src.db.connection import connect
for r in connect().execute("SELECT id, strategy_id, oracle_type, total_turns "
                            "FROM experiments WHERE strategy_id='oracle_watch' "
                            "ORDER BY id DESC LIMIT 5"):
    print(dict(r))
```
