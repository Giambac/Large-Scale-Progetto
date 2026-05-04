# Phase 1: Pre-Code Obligations and Foundation - Research

**Researched:** 2026-05-04
**Domain:** Dataset ingestion, sentence embeddings, HDBSCAN clustering, JSONL serialization, data splitting
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use Amazon Reviews 2023 — Arts, Crafts & Sewing subset (~15K reviews). Download only this category; do not download the full dataset.
- **D-02:** Text field to embed and cluster: `review_text` only. Do not concatenate title or other fields.
- **D-03:** Held-out split ratio: 80% train / 20% held-out. Split is stratified by nothing (random), seeded for reproducibility.
- **D-04:** At split time, write a SHA-256 hash file of the held-out file. System must verify this hash at startup and crash (assertion error) if it doesn't match. Hash file and held-out file are committed to the repo and never modified.
- **D-05:** Use the standalone `hdbscan` package (`pip install hdbscan`), not `sklearn.cluster.HDBSCAN`. Reason: `soft_clusters_` gives a true per-point probability vector over all K clusters — required for `f_uncertainty` and soft-assignment calibration.
- **D-06:** Windows install issues (OpenMP / compiled extensions) are handled as they come — no sklearn fallback is planned.
- **D-07:** The soft assignment for each item is the full `soft_clusters_` row (shape `[K]`, sums to 1.0). This is the canonical soft assignment stored in `ClusteringState.soft_probs`.
- **D-08:** Oracle satisfaction — primary: `OracleReply.satisfied = True` explicit token. Secondary (behavioral fallback): weighted feedback magnitude drops below threshold ε for N_fallback consecutive turns.
- **D-09:** Turn budget — hard cap of 15 turns. Loop stops unconditionally at `turn_index >= 15`.
- **D-10:** Diminishing returns — feedback magnitude is a weighted sum by feedback type: global > cluster-level > point-level > instructional. Magnitude near zero for N consecutive turns triggers this condition.
- **D-11:** Item IDs are sequential integers 0 to N−1. Cluster IDs are sequential integers never reused after deletion.
- **D-12:** Embeddings live in a separate read-only `EmbeddingStore`, not inside `ClusteringState`.
- **D-13:** ClusteringState fields: turn_index (int), timestamp (str ISO 8601), clusters (list[Cluster] with id/name/description/item_ids), assignments (dict[int,int]), soft_probs (dict[int, list[float]]).
- **D-14:** ClusteringState serialized to JSONL every turn, one JSON object per line, one line per turn. Deserialization must reproduce the exact same object.

### Claude's Discretion

- Exact HDBSCAN hyperparameters (`min_cluster_size`, `min_samples`) — researcher/planner decides based on dataset size (~15K points).
- Embedding batch size for `all-mpnet-base-v2` inference — choose for laptop memory constraints.
- Exact values for ε and N_fallback in stopping criteria — deferred to Phase 4.
- JSONL file location and rotation policy — planner decides.

### Deferred Ideas (OUT OF SCOPE)

- LangGraph vs. plain Python orchestrator — Phase 2 decision.
- Oracle cognitive-load weight parameters — Phase 3 decision.
- Exact ε and N_fallback values for stopping criteria — Phase 4 decision.
- Human validation study protocol — Phase 6 scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRE-01 | A text dataset is selected and a frozen held-out split is locked before any experiment runs | Dataset ID verified; SHA-256 hash pattern documented |
| PRE-02 | The three stopping conditions (oracle satisfaction, diminishing returns, turn budget) are operationalized as code-ready specifications before the Judge Agent is implemented | Stopping condition structure documented as typed Python specifications |
| FOUND-01 | System ingests any text dataset (CSV or JSONL) and computes sentence embeddings (all-mpnet-base-v2) stored in a read-only EmbeddingStore | `SentenceTransformer.encode()` API documented; numpy save/load pattern confirmed |
| FOUND-02 | System produces an initial HDBSCAN clustering with cluster names and natural-language descriptions | HDBSCAN API documented; LLM naming pattern documented |
| FOUND-03 | System produces soft assignments — per-point probability distribution over K clusters (not hard labels only) | `hdbscan.all_points_membership_vectors()` API documented; shape confirmed |
| FOUND-04 | System serializes full conversation + clustering state to JSONL AuditLog each turn (required for CI computation and reproducibility) | JSONL serialization pattern with custom encoder documented |
</phase_requirements>

---

## Summary

Phase 1 builds the irreversible data foundation and the operational clustering pipeline. All decisions were locked in CONTEXT.md; research focuses on implementation details for those decisions.

**Critical API correction:** D-07 references `soft_clusters_` as an attribute name, but no such attribute exists on the `hdbscan.HDBSCAN` object. The correct API is the module-level function `hdbscan.all_points_membership_vectors(clusterer)`, which returns `ndarray(n_samples, n_clusters)`. The planner must use this name internally — `soft_probs` in ClusteringState stores the converted rows.

**Critical field name correction:** The Amazon Reviews 2023 dataset field is `text`, not `review_text`. The system must be dataset-agnostic with configurable field names; `text` is the v1 default. CONTEXT.md D-02 says "review_text" — this appears to be a naming convention chosen by the project, not the actual HuggingFace field name. The data loader must map `text` → the internal field.

**Good news on Windows:** hdbscan 0.8.42 ships a pre-built `cp313-cp313-win_amd64.whl` for Python 3.13. No compilation required, no conda workaround needed. `pip install hdbscan` works on this machine.

**Primary recommendation:** Load dataset with streaming + `take(15000)`, embed with batch_size=32 on CPU, persist embeddings as `.npy`, run HDBSCAN with `prediction_data=True`, call `all_points_membership_vectors()` for soft probs, serialize ClusteringState to JSONL with a custom `json.JSONEncoder` that converts numpy types to Python native.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dataset download + sampling | CLI / script | — | One-time operation; output is a static file committed to repo |
| Train/held-out split + SHA-256 hash | CLI / script | — | Irreversible; runs once and freezes output |
| Sentence embedding computation | Pipeline module (EmbeddingStore) | — | Batch CPU inference; result persisted as .npy; never recomputed |
| HDBSCAN clustering | Pipeline module (ClusteringAgent) | EmbeddingStore | Reads embeddings, produces labels + soft probs |
| LLM cluster naming | Pipeline module | LLM API | Stateless LLM call per cluster |
| ClusteringState construction | Data layer | — | Pure Python dataclass assembly |
| JSONL AuditLog serialization | Data layer (serializer) | — | Write-once-per-turn; append-only file |
| Hash verification at startup | CLI entry point | — | Assert and crash immediately on mismatch |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| hdbscan | 0.8.42 | Density-based clustering + soft assignments | Only package providing `all_points_membership_vectors()` for true per-point cluster probability distribution |
| sentence-transformers | 5.4.1 | Sentence embedding via all-mpnet-base-v2 | Official wrapper for the locked model; handles tokenization, batching, truncation |
| datasets | 4.8.5 | Streaming download of Amazon Reviews 2023 from HuggingFace | Official HuggingFace library; supports streaming (no full 9M row download) |
| numpy | 2.4.4 | Embedding array storage and HDBSCAN input | Already installed; `.npy` is the canonical single-file array format |
| anthropic or openai | latest | LLM API calls for cluster naming/descriptions | Project's AI API for structured generation |

[VERIFIED: pip index versions — all versions confirmed against PyPI registry 2026-05-04]

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | 1.17.1 | HDBSCAN dependency (already present) | Transitively required; do not manage separately |
| scikit-learn | 1.8.0 | HDBSCAN dependency (already present) | Transitively required; do not manage separately |
| hashlib | stdlib | SHA-256 hash computation | Use `hashlib.sha256()` from stdlib; no extra install |
| dataclasses | stdlib | ClusteringState, Cluster typed containers | Use Python 3.10+ `@dataclass` |
| json | stdlib | JSONL serialization | Core serializer with custom encoder for numpy types |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| datasets streaming | Direct JSONL download from Amazon Reviews site | datasets library handles auth, caching, and streaming cleanly; direct download requires manual JSONL parsing |
| numpy .npy | HDF5 (h5py) | .npy is simpler for a single 2D float32 array; HDF5 adds value only for multiple heterogeneous arrays |
| Custom JSON encoder | orjson | orjson handles numpy natively and is faster; however, stdlib json is zero-dependency and sufficient for turn-level serialization frequency |

**Installation (pip — confirmed working on Python 3.13 Windows):**

```bash
pip install hdbscan==0.8.42 sentence-transformers==5.4.1 datasets==4.8.5
```

**Version verification:** Confirmed against PyPI 2026-05-04. hdbscan 0.8.42 ships `cp313-cp313-win_amd64.whl` — no compilation on Windows.

## Architecture Patterns

### System Architecture Diagram

```
                    [One-Time Setup]
                         |
    HuggingFace ──► load_dataset(streaming=True)
                         │
                    take(15_000)
                         │
                  filter empty text
                         │
              save dataset/arts_crafts_15k.jsonl
                         │
                   80/20 random split
                         │
              ┌──────────┴──────────┐
              │                     │
         train.jsonl           held_out.jsonl
                                    │
                              sha256(held_out)
                                    │
                            held_out.sha256   ← committed to git, never modified

                    [Session Start]
                         │
              verify sha256(held_out) == held_out.sha256
              (assert crash on mismatch)
                         │
              load train.jsonl → list[str] texts
                         │
              SentenceTransformer.encode(batch_size=32)
              ──────────────────────────────────────►  embeddings.npy  (EmbeddingStore)
                         │
              HDBSCAN(prediction_data=True).fit(embeddings)
                         │
              ┌──────────┴──────────┐
              │                     │
         labels_              all_points_membership_vectors()
         (-1 = noise)         ndarray(N, K)
              │                     │
              └──────────┬──────────┘
                         │
              assign noise points to nearest cluster
                         │
              LLM API call per cluster → name + description
                         │
              assemble ClusteringState (turn_index=0)
                         │
              append to audit_log.jsonl (FOUND-04)
```

### Recommended Project Structure

```
.
├── dataset/
│   ├── arts_crafts_15k.jsonl        # raw sampled reviews (15K)
│   ├── train.jsonl                  # 80% split
│   ├── held_out.jsonl               # 20% split — committed, never modified
│   └── held_out.sha256              # SHA-256 hash — committed, never modified
├── embeddings/
│   └── embeddings.npy               # EmbeddingStore — computed once, read-only
├── src/
│   ├── data_loader.py               # dataset download, split, hash verify
│   ├── embedding_store.py           # EmbeddingStore class (load/save .npy)
│   ├── clustering.py                # HDBSCAN wrapper, soft assignment computation
│   ├── cluster_naming.py            # LLM calls for cluster names/descriptions
│   ├── state.py                     # ClusteringState, Cluster dataclasses
│   ├── serialization.py             # JSONL serialize/deserialize, custom encoder
│   └── stopping.py                  # StoppingCriteria spec (code, not prose)
├── audit_log.jsonl                  # append-only, one line per turn
└── scripts/
    └── setup_phase1.py              # CLI: download → split → embed → initial cluster
```

### Pattern 1: Dataset Download with Streaming

**What:** Load only the Arts, Crafts & Sewing category using HuggingFace `datasets` streaming. Take 15,000 records, filter out empty `text` fields, save to JSONL.

**When to use:** One-time setup. Never re-download.

```python
# Source: https://amazon-reviews-2023.github.io/data_loading/huggingface.html
from datasets import load_dataset

ds = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Arts_Crafts_and_Sewing",
    split="full",
    streaming=True,
    trust_remote_code=True,
)

TEXT_FIELD = "text"  # HuggingFace field name — NOT 'review_text'

records = []
for example in ds:
    text = example.get(TEXT_FIELD, "").strip()
    if text:
        records.append({"item_id": len(records), "text": text})
    if len(records) >= 15_000:
        break

assert len(records) == 15_000, f"Expected 15000 records, got {len(records)}"
```

**Field name note:** The HuggingFace field is `text`, not `review_text`. The internal system uses `text` as the canonical name. CONTEXT.md D-02 uses "review_text" as a conceptual label — the data loader maps HuggingFace's `text` field to item records.

### Pattern 2: SHA-256 Hash Split

**What:** Random 80/20 split, write held-out split, compute SHA-256, write hash file. At startup, verify hash matches — assert crash on mismatch.

**When to use:** Once at setup time. Hash verification runs at every session start.

```python
import hashlib, json, random

# Source: stdlib hashlib documentation
def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_held_out_hash(held_out_path: str, hash_path: str) -> None:
    with open(hash_path) as f:
        expected = f.read().strip()
    actual = compute_sha256(held_out_path)
    assert actual == expected, (
        f"Held-out split hash mismatch!\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {actual}\n"
        f"  File may have been modified: {held_out_path}"
    )
```

### Pattern 3: EmbeddingStore (Compute Once, Read-Only)

**What:** Encode all training texts with `all-mpnet-base-v2`, save as `.npy`. Load at session start. Never mutate.

**When to use:** One-time computation at setup. Every subsequent session reads the .npy file.

```python
# Source: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingStore:
    def __init__(self, embeddings: np.ndarray):
        assert embeddings.ndim == 2, "Embeddings must be 2D: (n_items, dim)"
        self._embeddings = embeddings  # shape: (N, 768) float32

    @classmethod
    def compute_and_save(cls, texts: list[str], save_path: str) -> "EmbeddingStore":
        model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = model.encode(
            texts,
            batch_size=32,          # safe for laptop CPU; increase if GPU available
            show_progress_bar=True,
            convert_to_numpy=True,  # returns float32 ndarray
        )
        np.save(save_path, embeddings)
        return cls(embeddings)

    @classmethod
    def load(cls, save_path: str) -> "EmbeddingStore":
        embeddings = np.load(save_path)
        return cls(embeddings)

    def get(self, item_id: int) -> np.ndarray:
        return self._embeddings[item_id]

    def get_all(self) -> np.ndarray:
        return self._embeddings
```

**Memory estimate:** 15,000 × 768 × 4 bytes (float32) ≈ 46 MB. Well within laptop RAM. [ASSUMED — no official benchmark for this exact size; math is straightforward.]

### Pattern 4: HDBSCAN with Soft Assignments

**What:** Fit HDBSCAN with `prediction_data=True`, then call `hdbscan.all_points_membership_vectors()` to get per-point soft assignment vectors.

**Critical:** There is NO `soft_clusters_` attribute. CONTEXT.md D-07 uses this as a conceptual label — the actual API is `all_points_membership_vectors()`.

**When to use:** After embedding, to produce the initial ClusteringState.

```python
# Source: https://hdbscan.readthedocs.io/en/latest/soft_clustering.html
import hdbscan
import numpy as np

def run_hdbscan(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        labels: ndarray(N,) with int cluster labels (-1 = noise)
        soft_probs: ndarray(N, K) where K = number of discovered clusters
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,      # ~0.3% of 15K; starting point, Claude's discretion
        min_samples=10,           # controls noise sensitivity
        prediction_data=True,     # REQUIRED for all_points_membership_vectors()
        metric="euclidean",       # [ASSUMED] euclidean on normalized embeddings
    )
    clusterer.fit(embeddings)

    # soft_probs shape: (N, K) — K is number of NON-NOISE clusters
    soft_probs = hdbscan.all_points_membership_vectors(clusterer)

    assert soft_probs.shape[0] == len(embeddings), "Row count mismatch"
    assert soft_probs.shape[1] > 0, "Zero clusters detected — check hyperparameters"

    return clusterer.labels_, soft_probs

def assign_noise_to_nearest(labels: np.ndarray, soft_probs: np.ndarray) -> dict[int, int]:
    """Hard assignment: noise points (-1) assigned to argmax of their soft_probs row."""
    assignments = {}
    for i, label in enumerate(labels):
        if label == -1:
            assignments[i] = int(np.argmax(soft_probs[i]))
        else:
            assignments[i] = int(label)
    return assignments
```

**Hyperparameter guidance for ~15K points (Claude's discretion):**
- `min_cluster_size=50`: 0.3% of 15K — produces 3–15 clusters typical for text at this scale [ASSUMED, based on 1-2% rule adjusted for embedding space]
- `min_samples=10`: reasonable noise sensitivity for review data
- These are starting values; the planner should specify them and document in code as named constants, not magic numbers.

### Pattern 5: LLM Cluster Naming

**What:** For each cluster, send representative sample texts to an LLM and get back a name and description.

**When to use:** After HDBSCAN, to populate `Cluster.name` and `Cluster.description`.

```python
# Source: [ASSUMED] — standard Anthropic API pattern for structured output
import anthropic, json

def name_cluster(
    client: anthropic.Anthropic,
    sample_texts: list[str],
    cluster_id: int,
    max_samples: int = 5,
) -> dict[str, str]:
    """Returns {'name': str, 'description': str}"""
    samples = sample_texts[:max_samples]
    prompt = (
        f"You are analyzing a cluster of customer reviews. "
        f"Here are {len(samples)} representative reviews from cluster {cluster_id}:\n\n"
        + "\n---\n".join(samples)
        + "\n\nRespond ONLY with a JSON object with keys 'name' (2-5 words) "
        "and 'description' (1-2 sentences describing what unifies these reviews)."
    )
    response = client.messages.create(
        model="claude-haiku-4-5",   # cheapest capable model for naming tasks
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(response.content[0].text)
    assert "name" in result and "description" in result, f"LLM returned bad schema: {result}"
    return result
```

### Pattern 6: ClusteringState Serialization (JSONL Round-Trip)

**What:** Serialize ClusteringState to a single JSON line. Deserialize back to the exact same object.

**Key risk:** `soft_probs` values come from numpy float32 — `json.dumps` will raise `TypeError: Object of type float32 is not JSON serializable` unless converted.

```python
import json, dataclasses
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class Cluster:
    id: int
    name: str
    description: str
    item_ids: list[int]

@dataclass
class ClusteringState:
    turn_index: int
    timestamp: str          # ISO 8601
    clusters: list[Cluster]
    assignments: dict[int, int]           # item_id -> cluster_id
    soft_probs: dict[int, list[float]]    # item_id -> [float × K]

class _StateEncoder(json.JSONEncoder):
    """Converts dataclasses and numpy types to JSON-native types."""
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def serialize_state(state: ClusteringState) -> str:
    """Returns a single JSON line (no trailing newline)."""
    return json.dumps(dataclasses.asdict(state), cls=_StateEncoder)

def deserialize_state(line: str) -> ClusteringState:
    """Reconstructs ClusteringState from a JSONL line."""
    d = json.loads(line)
    clusters = [Cluster(**c) for c in d["clusters"]]
    # dict keys in JSON are always strings; convert back to int
    assignments = {int(k): v for k, v in d["assignments"].items()}
    soft_probs = {int(k): v for k, v in d["soft_probs"].items()}
    return ClusteringState(
        turn_index=d["turn_index"],
        timestamp=d["timestamp"],
        clusters=clusters,
        assignments=assignments,
        soft_probs=soft_probs,
    )

def append_to_audit_log(state: ClusteringState, log_path: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(serialize_state(state) + "\n")
```

**Round-trip gotcha:** JSON object keys are always strings. `assignments` and `soft_probs` have `int` keys in Python but `str` keys after JSON parse. `deserialize_state` must `int(k)` all keys on load.

### Pattern 7: Stopping Criteria (Code-Ready Spec)

**What:** Three OR-combined conditions. Phase 1 writes these as typed Python — NOT prose.

```python
# Source: CONTEXT.md D-08, D-09, D-10
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class StopReason(Enum):
    ORACLE_SATISFIED = "oracle_satisfied"
    TURN_BUDGET = "turn_budget"
    DIMINISHING_RETURNS = "diminishing_returns"

@dataclass(frozen=True)
class StoppingCriteria:
    turn_budget: int = 15               # D-09: hard cap; fires when turn_index >= 15

    # D-08 behavioral fallback — exact values deferred to Phase 4
    magnitude_threshold_epsilon: float = float("nan")  # placeholder: set in Phase 4
    magnitude_fallback_turns: int = -1                  # placeholder: set in Phase 4

def check_stopping(
    turn_index: int,
    oracle_satisfied: bool,
    recent_magnitudes: list[float],
    criteria: StoppingCriteria,
) -> Optional[StopReason]:
    """
    Returns the first triggered stop reason, or None if loop should continue.
    Conditions are OR-combined; order: oracle_satisfied > turn_budget > diminishing_returns.
    """
    # D-08: primary satisfaction token
    if oracle_satisfied:
        return StopReason.ORACLE_SATISFIED

    # D-09: hard turn budget
    if turn_index >= criteria.turn_budget:
        return StopReason.TURN_BUDGET

    # D-10: diminishing returns — placeholder logic; values injected in Phase 4
    # This condition is intentionally left as a stub; Phase 4 sets epsilon and N.
    # Do not implement the threshold check here; defer to Phase 4.

    return None

@dataclass
class FeedbackMagnitudeWeights:
    """D-10: weighting scheme. Exact values are Phase 4 decisions."""
    global_feedback: float = float("nan")       # placeholder
    cluster_level: float = float("nan")         # placeholder
    point_level: float = float("nan")           # placeholder
    instructional: float = float("nan")         # placeholder
```

### Anti-Patterns to Avoid

- **Hardcoding field names:** Never write `example["review_text"]`. Use a configurable `text_field: str` parameter defaulting to `"text"`.
- **Swallowing HDBSCAN exceptions:** If HDBSCAN produces 0 clusters (all noise), assert and crash — do not silently fall back to a single cluster.
- **Storing embeddings in ClusteringState:** D-12 explicitly forbids this. ClusteringState references items by ID only.
- **Calling `model.encode()` on every turn:** Embeddings are computed once and persisted. Never re-encode.
- **Using `np.float32` in JSON without conversion:** Raises `TypeError`. Always call `.tolist()` or cast to `float()` before `json.dumps`.
- **Forgetting `int(k)` on dict key deserialization:** JSON always deserializes object keys as strings. Failing to cast back to int will cause KeyError mismatches silently.
- **Using sklearn HDBSCAN:** `sklearn.cluster.HDBSCAN` does not expose `all_points_membership_vectors()`. The standalone `hdbscan` package is the locked choice (D-05).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sentence embeddings | Custom transformer inference loop | `SentenceTransformer.encode()` | Handles padding, truncation (384 token limit), batching, device selection |
| Density-based clustering | Custom density estimator | `hdbscan.HDBSCAN` | Cluster hierarchy, noise handling, soft assignments — all extremely complex to implement correctly |
| Soft cluster probabilities | Custom distance-to-centroid softmax | `hdbscan.all_points_membership_vectors()` | Library uses correct probabilistic membership model over the condensed tree |
| HuggingFace dataset streaming | Manual HTTP download + JSONL parsing | `datasets.load_dataset(streaming=True)` | Handles auth, caching, schema validation, streaming with `.take()` |
| Numpy type JSON serialization | Custom float32 → float conversion | Custom `JSONEncoder` subclass (stdlib) | Centralizes conversion; prevents TypeError surprises as schema evolves |
| SHA-256 hashing | Manual byte accumulation | `hashlib.sha256()` (stdlib) | Already handles chunked file reading correctly |

**Key insight:** In scientific Python, the "obvious" custom solution for soft clustering (softmax over distances) is mathematically incorrect. HDBSCAN's soft membership is defined over the condensed cluster tree — a data structure that doesn't exist in custom implementations.

## Common Pitfalls

### Pitfall 1: `soft_clusters_` Does Not Exist

**What goes wrong:** Code raises `AttributeError: 'HDBSCAN' object has no attribute 'soft_clusters_'` at runtime.

**Why it happens:** CONTEXT.md D-07 uses the term "soft_clusters_" as a conceptual label. The actual API uses the module-level function `hdbscan.all_points_membership_vectors(clusterer)`.

**How to avoid:** Always call `hdbscan.all_points_membership_vectors(clusterer)` — never `.soft_clusters_`.

**Warning signs:** Any code that accesses `clusterer.soft_clusters_` — catch this in code review before the PR.

### Pitfall 2: `prediction_data=True` Omitted

**What goes wrong:** `hdbscan.all_points_membership_vectors(clusterer)` raises an error about missing prediction data.

**Why it happens:** The soft assignment computation requires extra data structures built during `fit()` only when `prediction_data=True`.

**How to avoid:** Always instantiate `HDBSCAN(prediction_data=True)`. Document this in the constructor call.

**Warning signs:** `ValueError: No prediction data was generated` or similar.

### Pitfall 3: JSON Key Type Loss on Round-Trip

**What goes wrong:** `state.assignments[0]` works before serialization; `deserialized.assignments[0]` raises `KeyError` — because the key is now the string `"0"`.

**Why it happens:** JSON object keys are always strings. Python's `json.loads` does not convert them back to int.

**How to avoid:** `deserialize_state` must do `{int(k): v for k, v in d["assignments"].items()}` for both `assignments` and `soft_probs`.

**Warning signs:** KeyError on integer item_id after loading from JSONL.

### Pitfall 4: All-Noise HDBSCAN Output

**What goes wrong:** All 15K points receive label `-1` (noise). The cluster K is 0. `all_points_membership_vectors()` returns shape `(N, 0)`.

**Why it happens:** `min_cluster_size` is too large, or embeddings are too uniformly distributed.

**How to avoid:** Assert after fit: `assert len(set(clusterer.labels_) - {-1}) > 0`. If this fails, adjust hyperparameters. Fail loudly — do not silently create a dummy cluster.

**Warning signs:** `clusterer.labels_` contains only `-1` values.

### Pitfall 5: Amazon Reviews 2023 Full Dataset Download

**What goes wrong:** `load_dataset(..., streaming=False)` downloads all 9M Arts, Crafts & Sewing reviews (~several GB) before any processing.

**Why it happens:** Default HuggingFace `load_dataset()` downloads and caches the full split.

**How to avoid:** Always use `streaming=True` + `.take(15_000)`. This is the only safe pattern on a laptop.

**Warning signs:** Disk usage climbing into GB range; HuggingFace progress bar showing millions of rows.

### Pitfall 6: Held-Out Hash File Modified

**What goes wrong:** Hash verification at session start crashes because the held-out file was re-generated (e.g., after rerunning the split script with a different seed).

**Why it happens:** Developer reruns the download/split script, regenerating `held_out.jsonl` with different contents.

**How to avoid:** The split script must detect that `held_out.jsonl` and `held_out.sha256` already exist and refuse to overwrite them (assert and crash). Only the initial run creates these files.

**Warning signs:** Running setup script twice produces an assertion error on hash mismatch.

## Runtime State Inventory

Not applicable — this is a greenfield phase with no rename/refactor/migration scope.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | Yes | 3.13.12 | — |
| conda | Package management | Yes | 26.1.1 | — |
| hdbscan | FOUND-02, FOUND-03 | Not yet | (0.8.42 on PyPI, cp313 wheel available) | None — pip install will work |
| sentence-transformers | FOUND-01 | Not yet | (5.4.1 on PyPI) | None — pip install |
| datasets | PRE-01 | Not yet | (4.8.5 on PyPI) | None — pip install |
| numpy | All | Yes | 2.4.4 | — (already installed) |
| scipy | hdbscan dep | Yes | 1.17.1 | — (already installed) |
| scikit-learn | hdbscan dep | Yes | 1.8.0 | — (already installed) |
| Internet access | Dataset download | Assumed | — | Cannot proceed without it |
| LLM API key | FOUND-02 (cluster naming) | Unknown | — | Required; set in env var |

[VERIFIED: pip dry-run confirmed hdbscan 0.8.42 cp313 win_amd64 wheel exists and all deps already present]
[VERIFIED: pip index versions — numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0 already installed]

**Missing dependencies with no fallback:**
- `hdbscan`, `sentence-transformers`, `datasets` — must be installed before any pipeline code runs
- LLM API key — required for cluster naming (FOUND-02); must be configured as environment variable

**Missing dependencies with fallback:**
- None — all missing items have a clear install path.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (to be installed if not present) |
| Config file | None — see Wave 0 |
| Quick run command | `pytest src/tests/ -x -q` |
| Full suite command | `pytest src/tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRE-01 | Held-out split exists and hash verifies | unit | `pytest src/tests/test_data_loader.py::test_split_and_hash -x` | Wave 0 |
| PRE-01 | Hash mismatch causes assertion crash | unit | `pytest src/tests/test_data_loader.py::test_hash_mismatch_crashes -x` | Wave 0 |
| PRE-02 | `check_stopping()` returns correct reason for each condition | unit | `pytest src/tests/test_stopping.py -x` | Wave 0 |
| FOUND-01 | EmbeddingStore.compute_and_save produces correct shape | unit | `pytest src/tests/test_embedding_store.py::test_shape -x` | Wave 0 |
| FOUND-01 | EmbeddingStore.load round-trips without data loss | unit | `pytest src/tests/test_embedding_store.py::test_load_roundtrip -x` | Wave 0 |
| FOUND-02 | HDBSCAN produces at least 2 clusters on synthetic data | unit | `pytest src/tests/test_clustering.py::test_min_clusters -x` | Wave 0 |
| FOUND-02 | Cluster names and descriptions are non-empty strings | unit | `pytest src/tests/test_cluster_naming.py -x` | Wave 0 |
| FOUND-03 | soft_probs rows sum to ~1.0 per point | unit | `pytest src/tests/test_clustering.py::test_soft_probs_sum -x` | Wave 0 |
| FOUND-03 | soft_probs shape matches (N, K) | unit | `pytest src/tests/test_clustering.py::test_soft_probs_shape -x` | Wave 0 |
| FOUND-04 | serialize_state → deserialize_state round-trips without data loss | unit | `pytest src/tests/test_serialization.py::test_roundtrip -x` | Wave 0 |
| FOUND-04 | append_to_audit_log creates one JSONL line per call | unit | `pytest src/tests/test_serialization.py::test_append -x` | Wave 0 |
| FOUND-04 | JSON keys are int after deserialization (not str) | unit | `pytest src/tests/test_serialization.py::test_key_types -x` | Wave 0 |

**Note:** FOUND-02 cluster naming test uses a mocked LLM client to avoid live API calls in unit tests.

### Sampling Rate

- **Per task commit:** `pytest src/tests/ -x -q`
- **Per wave merge:** `pytest src/tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `src/tests/__init__.py` — empty init
- [ ] `src/tests/test_data_loader.py` — covers PRE-01
- [ ] `src/tests/test_stopping.py` — covers PRE-02
- [ ] `src/tests/test_embedding_store.py` — covers FOUND-01
- [ ] `src/tests/test_clustering.py` — covers FOUND-02, FOUND-03
- [ ] `src/tests/test_cluster_naming.py` — covers FOUND-02 (mocked LLM)
- [ ] `src/tests/test_serialization.py` — covers FOUND-04
- [ ] `src/tests/conftest.py` — shared fixtures (tiny synthetic dataset, mock HDBSCAN output)
- [ ] Framework install: `pip install pytest` — if not yet installed

## Security Domain

This phase has no authentication, session management, user-facing input, or cryptographic operations beyond SHA-256 (for data integrity, not security). ASVS categories are not applicable.

| ASVS Category | Applies | Rationale |
|---------------|---------|-----------|
| V2 Authentication | No | No user authentication in this phase |
| V3 Session Management | No | No sessions — CLI batch process |
| V4 Access Control | No | No multi-user access control |
| V5 Input Validation | Partial | Dataset text field filtering (strip empty strings); assert on schema fields |
| V6 Cryptography | No | SHA-256 used for data integrity only, not security; stdlib `hashlib` is correct |

**Fail-loudly security note:** The `verify_held_out_hash()` function must use `assert` (not `if` + log) so that any held-out contamination crashes the process immediately.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ~15K records with non-empty `text` exist within the first few thousand iterations of the streaming dataset | Pattern 1 | If reviews are sparse, may need to stream more of the dataset; code should track count and warn |
| A2 | `min_cluster_size=50, min_samples=10` produces 3–15 usable clusters on 15K review embeddings | Pattern 4 | Wrong values → 0 clusters (all noise) or 100+ clusters; assert will catch this; hyperparameters need tuning |
| A3 | `all-mpnet-base-v2` CPU encoding of 15K texts completes in < 30 minutes on a laptop | Pattern 3 | May be slower; add `show_progress_bar=True` and document expected time; compute once and cache |
| A4 | CONTEXT.md D-02's "review_text" is a project-level naming convention, not the HuggingFace field name (which is "text") | Dataset section | If "review_text" is a v2 field or requires different loading config, data loader will fail — catch with assert |
| A5 | `metric="euclidean"` is appropriate for all-mpnet-base-v2 normalized embeddings in HDBSCAN | Pattern 4 | Cosine distance may produce better clusters; research suggests either works; euclidean is the HDBSCAN default |

## Open Questions (RESOLVED)

1. **LLM API provider selection**
   - What we know: Project requires an LLM for cluster naming; Anthropic Claude is available (this session uses it); OpenAI is an alternative.
   - What's unclear: Which API key the researcher has configured; whether the project commits to one provider or abstracts behind an interface.
   - Recommendation: Design `ClusterNamer` as a protocol/interface with a single concrete implementation for Phase 1; pick the API the researcher has a key for.
   - RESOLVED: ClusterNamer is a Protocol; AnthropicClusterNamer is the v1 implementation; key loaded from env ANTHROPIC_API_KEY

2. **Noise point assignment strategy**
   - What we know: HDBSCAN labels noise points as -1. D-13 requires complete assignments (all N items must appear in `assignments`). The current Pattern 4 assigns noise to `argmax(soft_probs[i])`.
   - What's unclear: Whether argmax is the intended strategy or whether noise items should form their own cluster.
   - Recommendation: Assign noise to `argmax(soft_probs)` — this preserves the anytime behavior requirement (CLUS-01) and avoids a spurious "noise" cluster. Document this decision explicitly in `clustering.py`.
   - RESOLVED: assign noise points via argmax(soft_probs[item_id]) — same as nearest cluster

3. **Actual achievable K for this dataset**
   - What we know: HDBSCAN discovers K automatically. The project targets 3–10 clusters.
   - What's unclear: Whether review data at 15K points + 768d embeddings will produce a sensible K with the starting hyperparameters.
   - Recommendation: Run a quick experiment in a notebook before fixing hyperparameters in code; document the chosen values and rationale.
   - RESOLVED: empirical; HDBSCAN with MIN_CLUSTER_SIZE=50 on 15K points expected to produce 10-40 clusters; verify in Phase 1 execution

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sklearn's KMeans for text clustering | HDBSCAN for density-based + noise-aware clustering | ~2017 | Eliminates need to specify K upfront; identifies outliers |
| Hard cluster labels only | Soft membership vectors via `all_points_membership_vectors()` | hdbscan >= 0.8 | Enables boundary detection, calibration testing |
| Direct HuggingFace download | Streaming + `.take(N)` | datasets >= 2.x | Avoids downloading full datasets (here: 9M rows) |
| JSON serialization of numpy manually | Custom JSONEncoder subclass | stdlib pattern | Centralizes all numpy type handling; prevents TypeError surprises |

**Deprecated/outdated:**
- `clusterer.soft_clusters_`: Never existed. Always use `hdbscan.all_points_membership_vectors(clusterer)`.
- `load_dataset(..., streaming=False)` for large categories: Do not use without `take()` — will attempt to cache all 9M rows.

## Sources

### Primary (HIGH confidence)
- [hdbscan readthedocs — Soft Clustering](https://hdbscan.readthedocs.io/en/latest/soft_clustering.html) — API for `all_points_membership_vectors()`, shape of output, `prediction_data=True` requirement
- [hdbscan readthedocs — API Reference](https://hdbscan.readthedocs.io/en/latest/api.html) — confirmed no `soft_clusters_` attribute; `probabilities_` and `labels_` are the instance attributes
- [HuggingFace — sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) — 768d output, 384 token max, 0.1B params, `model.encode()` API
- [Amazon Reviews'23 data loading](https://amazon-reviews-2023.github.io/data_loading/huggingface.html) — exact HuggingFace config name `raw_review_Arts_Crafts_and_Sewing`, field name `text`
- [HuggingFace — McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) — confirmed field names: `rating`, `title`, `text`, `asin`, `user_id`, `timestamp`, `verified_purchase`; category size: 9M ratings
- PyPI registry (pip dry-run) — hdbscan 0.8.42 cp313-cp313-win_amd64.whl confirmed installable; no compilation required

### Secondary (MEDIUM confidence)
- [hdbscan readthedocs — Parameter Selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) — `min_cluster_size` guidance; default `min_samples` equals `min_cluster_size`
- [hdbscan GitHub Issues #629, #318](https://github.com/scikit-learn-contrib/hdbscan/issues/629) — historical Windows install problems; resolved in 0.8.42 with pre-built wheels
- [Anthropic structured outputs guide](https://towardsdatascience.com/hands-on-with-anthropics-new-structured-output-capabilities/) — tool-based structured output pattern for Claude

### Tertiary (LOW confidence)
- Community guidance on `min_cluster_size` 1-2% rule for text clustering — found in multiple blog posts; not in official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified against PyPI; Windows wheel confirmed via pip dry-run
- Architecture: HIGH — based on locked decisions from CONTEXT.md + verified API docs
- hdbscan soft assignment API: HIGH — verified against official readthedocs
- Dataset field names: HIGH — verified against official HuggingFace card and data loading docs
- JSONL serialization: HIGH — stdlib pattern; numpy type issue well-documented
- Hyperparameter values: LOW — starting points only; require empirical validation

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 (stable stack; packages unlikely to break API in 30 days)
