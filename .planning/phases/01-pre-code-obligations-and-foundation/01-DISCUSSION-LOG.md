# Phase 1 — Discussion Log

**Date:** 2026-05-04
**Areas discussed:** Dataset & split, HDBSCAN library, Stopping criteria, ClusteringState schema

---

## Area 1: Dataset & Split

| Question | Options presented | Selected |
|----------|------------------|----------|
| Which dataset? | Amazon Reviews 2023 / IMDB / Support tickets | Amazon Reviews 2023 |
| Which category? | Musical Instruments / Software / Arts Crafts & Sewing / Specify | Arts, Crafts & Sewing |
| Split ratio? | 80/20 / 90/10 / 70/30 | 80/20 |
| Text field? | review_text only / title+review_text / title only | review_text only |

**Notes:** User asked for dataset dimension/characteristics before deciding. Chose Arts Crafts & Sewing after reviewing dataset sizes and embedding storage requirements for a laptop setup.

---

## Area 2: HDBSCAN Library

| Question | Options presented | Selected |
|----------|------------------|----------|
| sklearn vs standalone? | sklearn.HDBSCAN / standalone hdbscan / sklearn + manual | standalone hdbscan |
| Windows fallback? | Try standalone first, fallback sklearn / Standalone only | Standalone only |

**Notes:** User accepted Windows install risk in exchange for full multinomial soft_clusters_ output. No fallback plan.

---

## Area 3: Stopping Criteria

| Question | Options presented | Selected |
|----------|------------------|----------|
| Oracle satisfaction operationalization? | Explicit token / Behavioral inference / Both | Both (token + behavioral fallback) |
| Turn budget? | 50 / 30 / 100 | 15 (user-specified via Other) |
| Diminishing returns measurement? | Structural changes / Token count / Weighted feedback type | Weighted feedback type |

**Notes:** 15-turn budget is intentionally tight vs. the 50-turn option — user wants experiments under time pressure. Weighted feedback type chosen over simpler proxies.

---

## Area 4: ClusteringState Schema

| Question | Options presented | Selected |
|----------|------------------|----------|
| Item/cluster ID scheme? | Integer IDs at load time / Content-hash IDs | Integer IDs at load time |
| Embedding location? | Separate EmbeddingStore / Inside ClusteringState | Separate EmbeddingStore |
| State fields? | Minimal + soft assignments / + full oracle history / minimal only | Minimal + soft assignments |

**Notes:** All three schema decisions took the recommended option. Embeddings kept separate to keep JSONL serialization lightweight.

---

## Claude's Discretion

- HDBSCAN hyperparameters (min_cluster_size, min_samples)
- Embedding batch size for laptop memory
- Exact ε and N_fallback values (deferred to Phase 4)
- JSONL file location and rotation

## Deferred Ideas

- LangGraph vs. plain Python — Phase 2 decision
- Oracle cognitive-load weights — Phase 3 decision
- Human validation study protocol — needed before Phase 6
