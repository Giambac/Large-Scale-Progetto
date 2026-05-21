# Stack Research — v2.0 "Experimentation Flexibility & Scale"

**Domain:** Python conversational-clustering research system — v2.0 milestone additions (Colab compute offload, pluggable embedding backends, versioned YAML configs)
**Researched:** 2026-05-21
**Confidence:** HIGH (all key versions verified against the live local environment + current PyPI/official docs)

> **Supersession note:** This file previously held the v1 (milestone-1) stack research, which described an *aspirational* stack (LangGraph, MLflow, Typer/Rich, sklearn-native HDBSCAN) that does **not** match what was actually built. The shipped v1 system uses a plain `ClusteringBackend` `typing.Protocol`, `src/llm_call.py`/`src/llm_key.py` for LLM access, FastAPI + uvicorn + python-socketio for the UI, and standalone `hdbscan` + sklearn KMeans. The prior content remains in git history. This document is scoped to **only the NEW v2.0 capabilities**; the validated v1 web/DB/clustering/LLM plumbing is not re-researched.

## Environment baseline (verified on this machine, 2026-05-21)

Probed the live interpreter — these are the versions actually installed, so recommendations are pinned to what is known-good here:

| Package | Installed | Relevance |
|---------|-----------|-----------|
| `sentence-transformers` | **5.4.1** | local embedding backend |
| `huggingface_hub` | **1.7.1** | already present (transitive via sentence-transformers); reuse for artifact handoff |
| `openai` | **2.26.0** | OpenAI embedding backend client (same SDK as `llm_call.py`) |
| `pydantic` | **2.12.5** | config schema validation (already a dep) |
| `pyyaml` | **6.0.3** | already a dep |
| `numpy` | **2.4.3** | `.npy` artifact I/O |
| `scikit-learn` | **1.8.0** | KMeans backend |
| `torch` | **2.11.0+cpu** | **CPU-only locally** — this is the concrete motivation for Colab GPU offload |
| `ruamel.yaml` | not installed | candidate for round-trip YAML (see decision below) |

**Critical compatibility finding (resolved):** `huggingface_hub` 1.0 removed `cached_download`, which breaks *old* `sentence-transformers` (<3.x). The local env already runs the modern pairing `huggingface_hub 1.7.1` + `sentence-transformers 5.4.1`, which work together cleanly. So **no down-pinning is required** — but the stale `sentence-transformers>=2.7` line in `requirements.txt` is misleading and must be bumped to reflect reality (see Version Compatibility).

## Recommended Stack

### Core Technologies (new for v2.0)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `huggingface_hub` | **>=1.7,<2** | Colab→local artifact handoff: upload `embeddings.npy` + `initial_state.json` from Colab, download locally | Already installed transitively (zero new heavy dep). `HfApi.upload_file` / `hf_hub_download` give versioned, resumable, auth'd transfer of arbitrary files via a (private) dataset repo. Cleaner than Google Drive scraping or manual copy; the v1.0 `cached_download` removal is a non-issue because modern sentence-transformers is in use. |
| `openai` | **>=2.26,<3** | Second embedding backend via `client.embeddings.create(model="text-embedding-3-small", dimensions=...)` | **Same SDK already imported in `src/llm_call.py`** and same key path in `src/llm_key.py` — reuses the `OpenAI` client from `build_client("openai", key)`. No new auth surface. `text-embedding-3-small` defaults to 1536-dim; supports the `dimensions` param to shrink (Matryoshka) if you ever want dim parity with 384. |
| `ruamel.yaml` | **>=0.18,<0.19** | Read/write versioned oracle config YAML with comment + key-order preservation | Round-trips comments and ordering (PyYAML strips both on write). For *human-edited, version-controlled* config files this keeps git diffs clean and lets configs self-document. PyYAML is the fallback if the team prefers no new dep (see alternatives). |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pydantic` | **>=2.12,<3** (already present) | Validate the parsed oracle-config dict into a typed `OracleConfig` model (model name, prompt templates, persona, noise params) | Use on every config load — fail-loudly on malformed YAML at the boundary, then pass a typed object inward. Already a dep; no new install. |
| `pyyaml` | **>=6.0** (already present) | Fallback YAML parser if ruamel is rejected | Only if the team decides write-side comment preservation isn't worth a new dep. |
| `numpy` | **>=2.4** (already present) | `np.save`/`np.load` for the `.npy` embedding artifact on both Colab and local sides | The artifact format is already `.npy` (see `embedding_store.py`); keep it. Pin the same numpy major (2.x) on Colab to avoid format edge cases. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Google Colab (notebook) | Compute-only GPU runtime: load dataset → embed (sentence-transformers on GPU) → initial KMeans → upload artifacts to HF Hub | NOT a deploy target. No FastAPI/socketio ever runs on Colab. The notebook is a thin script: read JSONL, encode, fit KMeans, `np.save`, `HfApi.upload_file`. |
| HF private dataset repo | Transport bucket for `embeddings.npy` + `initial_state.json` + a `manifest.json` (model name, dim, dataset hash) | `repo_type="dataset"`, `private=True`. Token via `HF_TOKEN` env var, resolved the same no-overwrite `.env` way as `src/llm_key.py`. |

## Installation

```bash
# New for v2.0 (run in the LOCAL venv)
pip install "ruamel.yaml>=0.18,<0.19"

# Already installed locally — bump the requirements.txt floors to match reality:
#   huggingface_hub>=1.7,<2     (currently transitive/unpinned)
#   openai>=2.26,<3             (requirements.txt currently says >=1.30)
#   sentence-transformers>=5.4  (requirements.txt currently says >=2.7 — STALE)
#   pydantic>=2.12 ; pyyaml>=6.0 ; numpy>=2.4 ; scikit-learn>=1.8   (already present)

# On the Colab side (compute-only notebook), the runtime ships torch+CUDA; add:
pip install -q "sentence-transformers>=5.4" "huggingface_hub>=1.7,<2"
```

## Integration Points with Existing Stack

**1. Colab → local artifact handoff (HF Hub)**
- Colab notebook produces a bundle:
  - `embeddings.npy` — shape `(N, dim)` float32 (dim = 384 or 1536 depending on the backend used in Colab)
  - `initial_state.json` — the KMeans initial clustering (labels + soft probs) so local startup skips compute entirely
  - `manifest.json` — `{embedding_backend, embedding_model, embedding_dim, dataset_sha256, n_items, created_utc}` so the local side can assert the artifact matches the expected dataset/backend (fail-loudly).
- Upload via `HfApi().upload_file(path_or_fileobj=..., path_in_repo=..., repo_id=..., repo_type="dataset")`.
- Local side downloads via `hf_hub_download(repo_id, filename, repo_type="dataset")`, then `EmbeddingStore.load(path)` consumes the `.npy` unchanged.
- `HF_TOKEN` resolved through the existing `.env` mechanism (extend the `src/llm_key.py` pattern or add a tiny `hf_key.py` mirror). Boundary-only error handling per fail-loudly rule.

**2. Pluggable embedding backend (dynamic dim)**
- Mirror the existing `ClusteringBackend` `typing.Protocol` with an `EmbeddingBackend` Protocol: `encode(texts: list[str]) -> np.ndarray` plus a `dim: int` property.
  - `SentenceTransformerBackend` — wraps current code; `dim` from `model.get_sentence_embedding_dimension()`.
  - `OpenAIEmbeddingBackend` — calls `client.embeddings.create(model="text-embedding-3-small", input=batch, dimensions=...)`; reuse `build_client("openai", key)` from `llm_call.py`. **Batch ≤ 2048 inputs per request and ≤ 8192 tokens per input** (OpenAI hard limits); chunk accordingly and concatenate. Returns `np.array([d.embedding for d in resp.data], dtype=np.float32)`.
- **Dynamic dimension migration (required):** `EMBEDDING_DIM = 384` is currently a hardcoded constant asserted across `embedding_store.py`. Replace with a value carried on the store (set from `embeddings.shape[1]` at load, or from the backend's `dim`), and rewrite the `assert embeddings.shape[1] == EMBEDDING_DIM` checks to assert against the *expected dim recorded in the manifest/config*, not a literal. **Keep the assert** (fail-loudly) — only make the expected value dynamic. Also note the file's stale comments referencing `all-mpnet-base-v2`/768 while the constant is 384 — clean those up during this migration.

**3. Versioned oracle configs (YAML)**
- Store under e.g. `configs/oracle/<name>.yaml`, version-controlled in git (NOT in `experiments.db` — locked decision).
- Load: `ruamel.yaml.YAML(typ="rt")` → dict → `OracleConfig.model_validate(dict)` (pydantic). Validation failure = crash at the boundary.
- Persist the resolved config (or its sha) into the existing audit log / `experiments` row so each run records exactly which config version produced it.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| HF Hub for Colab→local transfer | Google Drive mount + manual download | If offline from HF or wanting zero external accounts; loses versioning/auth and is fiddly to script. |
| HF Hub | `gdown` / direct Drive API | Only if artifacts must stay inside Google's ecosystem; HF gives cleaner private versioned repos. |
| `ruamel.yaml` | `PyYAML` (already installed) | If configs are never hand-edited or comment/order preservation on write doesn't matter — then skip the new dep and use `yaml.safe_load`. |
| `text-embedding-3-small` (1536) | `text-embedding-3-large` (3072) | Higher quality at ~6.5x cost; not needed for 3–10 cluster research scale. |
| OpenAI `dimensions` param to shrink to 384 | Keep native 1536 | Only shrink for apples-to-apples dim parity with MiniLM in an ablation; otherwise keep native dim and let `EMBEDDING_DIM` be dynamic. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `eventlet` / `gevent` | HARD project rule — they monkey-patch stdlib (threading/socket/time) and corrupt numpy/sklearn/UMAP internals. The new embedding/Colab work touches numpy heavily. | uvicorn native asyncio (unchanged); CPU work in `threading.Thread`. |
| Replacing/abandoning `uvicorn` | The interactive UI stays local and unchanged; Colab is compute-ONLY and runs no server. | Keep FastAPI + uvicorn + python-socketio exactly as is. |
| Running FastAPI/socketio on Colab | Colab is for heavy compute only; tunneling a socket server from Colab adds fragility for zero benefit. | Produce artifacts on Colab, serve UI locally. |
| Pinning `huggingface_hub<1.0` "for safety" | Unnecessary — local already runs hf_hub 1.7.1 with sentence-transformers 5.4.1 successfully. Down-pinning would break the working install. | `huggingface_hub>=1.7,<2`. |
| A second/different LLM SDK for OpenAI embeddings | The `openai` SDK is already imported in `src/llm_call.py`; embeddings live on the same client. | Reuse `build_client("openai", key)` → `client.embeddings.create(...)`. |
| Hardcoding a new `EMBEDDING_DIM` literal (e.g. 1536) | Repeats the original bug; backends now have different dims. | Carry dim dynamically from the manifest/backend and assert against it. |
| Storing oracle configs in `experiments.db` | Locked milestone decision: configs are versioned YAML files in git. | YAML files under `configs/`. |
| Hand-rolled `aiohttp`/`requests` HF transfer | hf_hub already wraps resumable, auth'd, versioned transfer (now on httpx). | `huggingface_hub` API. |

## Stack Patterns by Variant

**If embedding on Colab GPU (the motivating case — local torch is CPU-only):**
- Run sentence-transformers (or OpenAI API) in the notebook, `np.save` + `HfApi.upload_file`.
- Local side never loads `torch` for embedding when an artifact exists — just `EmbeddingStore.load`.

**If using the OpenAI embedding backend:**
- No GPU needed anywhere; embedding is an API call. Colab offload becomes optional (cost/latency tradeoff). Batch ≤2048 inputs, ≤8192 tokens each; dim = 1536 (or set `dimensions=384`).

**If staying fully local + sentence-transformers:**
- Current path works but is CPU-bound (torch 2.11.0+cpu; the existing docstring estimates ~10–30 min for 12K texts) — this is exactly the latency Colab offload removes.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `huggingface_hub@1.7.1` | `sentence-transformers@5.4.1` | Verified working in the live local env. The `cached_download` removal in hf_hub 1.0 only breaks sentence-transformers <3.x. |
| `huggingface_hub@1.x` | hf_xet + httpx backend | 1.0 migrated transport to httpx (already a dep: `httpx>=0.27`) and uses hf_xet for transfers — no action needed. |
| `requirements.txt` `sentence-transformers>=2.7` | **MISMATCH** | Stale floor; installed is 5.4.1. Bump to `>=5.4` so a future resolver can't pull a 2.x incompatible with hf_hub 1.x. |
| `requirements.txt` `openai>=1.30` | installed `2.26.0` | Embeddings API stable across 1.x/2.x; bump floor to `>=2.26,<3` to match local and cap the major. |
| `numpy@2.4` on Colab vs local | `.npy` format | Keep both on numpy 2.x; `.npy` is stable within a major but pin to be safe. |
| `ruamel.yaml@0.18.x` | pydantic 2.x | Independent — ruamel parses to dict/CommentedMap, pydantic validates the dict. No interaction risk. |

## Sources

- [huggingface-hub · PyPI](https://pypi.org/project/huggingface-hub/) — current 1.x line, upload/download API — HIGH
- [huggingface_hub v1.0 blog](https://huggingface.co/blog/huggingface-hub-v1) — httpx migration, hf_xet, `cached_download` removal — HIGH
- [sentence-transformers issue #1602](https://github.com/huggingface/sentence-transformers/issues/1602) — cached_download ImportError on old sentence-transformers — HIGH
- [text-embedding-3-small | OpenAI API](https://developers.openai.com/api/docs/models/text-embedding-3-small) — default 1536-dim, `dimensions` param — HIGH
- [Vector embeddings | OpenAI API](https://developers.openai.com/api/docs/guides/embeddings) — 8192-token input limit, ≤2048 array limit — HIGH
- [ruamel.yaml · PyPI](https://pypi.org/project/ruamel.yaml/) — round-trip comment/order preservation vs PyYAML — HIGH
- Live local interpreter probe (2026-05-21) — exact installed versions incl. torch 2.11.0+cpu — HIGH

---
*Stack research for: conversational-clustering v2.0 (Colab offload, pluggable embeddings, versioned YAML configs)*
*Researched: 2026-05-21*
