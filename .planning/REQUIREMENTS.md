# Requirements: Conversational Clustering

**Defined:** 2026-04-29
**Core Value:** The interaction loop converges toward oracle-accepted clusterings efficiently, with every design decision measured against cognitive load and information gain.

## v1 Requirements

### Pre-Code Obligations

- [ ] **PRE-01**: A text dataset is selected and a frozen held-out split is locked before any experiment runs
- [ ] **PRE-02**: The three stopping conditions (oracle satisfaction, diminishing returns, turn budget) are operationalized as code-ready specifications before the Judge Agent is implemented

### Foundation

- [ ] **FOUND-01**: System ingests any text dataset (CSV or JSONL) and computes sentence embeddings (all-mpnet-base-v2) stored in a read-only EmbeddingStore — no dataset-specific configuration required
- [ ] **FOUND-02**: System produces an initial HDBSCAN clustering with cluster names and natural-language descriptions
- [ ] **FOUND-03**: System produces soft assignments — per-point probability distribution over K clusters (not hard labels only)
- [ ] **FOUND-04**: System serializes full conversation + clustering state to JSONL AuditLog each turn (required for CI computation and reproducibility)

### Clustering Agent

- [x] **CLUS-01**: `f_output` always returns a complete clustering assignment (anytime behavior — no partial states)
- [x] **CLUS-02**: `f_uncertainty` identifies boundary points, ambiguous assignments, and low-confidence clusters
- [x] **CLUS-03**: `f_next_best_step` selects the next action (show full clustering / show subset / ask targeted question / stop) via a pluggable Strategy interface
- [x] **CLUS-04**: `f_next_state` applies oracle feedback to update clustering state; latest oracle intent wins on contradictions

### Oracle Agent

- [ ] **ORC-01**: Oracle Agent is an LLM with an explicit preference specification and persona — not a bare LLM call
- [ ] **ORC-02**: Oracle Agent has explicit noise parameters: `consistency_rate`, `drift_probability`, `sycophancy_resistance`
- [ ] **ORC-03**: Oracle Agent receives a pre-computed cognitive-load score per turn and simulates fatigue/overload above threshold
- [ ] **ORC-04**: Oracle Agent tracks and surfaces preference drift — contradictions with prior intent are detected, logged, and optionally flagged to the Clustering Agent

### Judge Agent

- [x] **JUDG-01**: `f_eval` implements multi-signal convergence detection: oracle satisfaction signal + feedback magnitude decay + turn budget exhaustion
- [ ] **JUDG-02**: Per-turn metric bundle is recorded to AuditLog: turns-to-convergence counter, cognitive-load score, contradiction count, pairwise validation accuracy sample
- [ ] **JUDG-03**: No-dialogue baseline evaluation runs the same metric bundle on the un-conversed initial clustering, enabling isolation of dialogue contribution

### Feedback Types

- [x] **FB-01**: System accepts and acts on global oracle feedback ("too many clusters", "focus on billing complaints")
- [x] **FB-02**: System accepts and acts on cluster-level oracle feedback ("split this cluster", "merge A and B", "A is too large")
- [x] **FB-03**: System accepts and acts on point-level oracle feedback ("x belongs in B", "x and y should be together")
- [ ] **FB-04**: System accepts and acts on instructional oracle feedback ("treat 'error' and 'fail' as synonyms", "pay more attention to feature F") via LLM parsing into structured constraints

### Hierarchy

- [x] **HIER-01**: System maintains a cluster hierarchy that the oracle can drill into or zoom out of
- [x] **HIER-02**: Hierarchy is grown incrementally as the oracle refines (not computed in one shot upfront)

### Database

- [ ] **DB-01**: An `experiments` table stores one row per run with: strategy_id, persona_id, seed, dataset, total_turns, convergence_reason, headline metrics (turns-to-convergence, mean cognitive load, contradiction count), start/end timestamps
- [ ] **DB-02**: A `turns` table stores one row per turn per experiment (FK: experiment_id) with: turn_index, action_type (show_full/show_subset/ask_question/stop), system_message, cognitive_load_score, cumulative_contradiction_count, convergence_signal
- [ ] **DB-03**: An `oracle_feedback` table stores one row per parsed feedback item per turn (FK: turn_id), supporting multiple items per turn: feedback_type (global/cluster/point/instructional), raw_text, parsed_delta (JSON), target (cluster or item reference), is_contradiction

### Debug UI

- [x] **UI-01**: A web-based debug interface displays current clustering state (cluster names, assignments, soft probabilities), conversation history (turn-by-turn oracle feedback and system replies), and per-turn metrics (cognitive-load score, contradiction count, convergence signal) — intended for developer debugging, not end-user interaction
- [x] **UI-02**: The web UI allows uploading a dataset file (e.g. CSV or JSONL) to start a new session on a different dataset without restarting from the CLI

### Ablation & Evaluation

- [x] **ALAB-01**: `f_next_best_step` exposes a swappable Strategy interface with at least 3 registered strategies: Random, UncertaintyDriven, BoundaryDriven (and optionally InformationGain)
- [ ] **ALAB-02**: Multi-run experiment harness executes N strategies × M personas × K seeds automatically, logging all runs to AuditLog with `strategy_id` and `persona_id`
- [ ] **ALAB-03**: Bootstrap confidence intervals are computed on the headline quantitative claim (turns-to-convergence or information-gain-per-cognitive-load ratio)

### Generalization

- [ ] **GEN-01**: After oracle acceptance, system codifies oracle preferences into a reusable mapping function (prompt, classifier, or rule set) that assigns new items to clusters
- [ ] **GEN-02**: Generalization function is evaluated on the frozen held-out split; accuracy is validated by the oracle on a small sample

## v2 Requirements (Trio/Quartet Scope)

These requirements activate when the team grows to 3–4 people or v1 is complete and time permits.

### Headline Experiment

- **EXP-V2-01**: Run the full N oracles × M tasks experiment with both LLM oracles (for scale) and human oracles (N ≥ 10, within-subject, randomized task order), comparing the conversational interface against a sensible default-parameter baseline on three metrics: oracle satisfaction, turns-to-convergence, and generalization accuracy

### Evaluation Depth

- **EVAL-V2-01**: Soft-assignment calibration — reliability diagram showing whether system-flagged boundary points are genuinely ambiguous
- **EVAL-V2-02**: Human oracle validation study (N ≥ 10, within-subject, randomized order) — quantifies the simulated-vs-human gap as a reported finding

### Extended Clustering Backends

- **BACK-V2-01**: Multiple clustering backends (k-means, LLM-first) as swappable components alongside HDBSCAN
- **BACK-V2-02**: Representation choice exposed as a config parameter (raw features vs. sentence embeddings vs. fine-tuned embeddings)

### Visualization

- **VIZ-V2-01**: UMAP/t-SNE 2D projection of embedding space displayed in the web UI, with cluster membership color-coded and soft-assignment opacity encoding

### Persistence

- **UI-V2-01**: Persistent sessions — clustering state and conversation history are saved to disk and can be resumed across server restarts

### Synthetic Data

- **SYNTH-V2-01**: 4th agent for synthetic dataset generation — creates text corpora with known cluster structure for controlled experiments

## Milestone v2.0 Requirements (Experimentation Flexibility & Scale)

**Defined:** 2026-05-21
**Goal:** Make the system configurable and scalable for research — Colab compute offload, swappable embedding/clustering backends, oracle-initiated flow, query filter, and a coordination agent.
**Locked decisions:** re-cluster (re-fit KMeans), not re-embed, per query · Colab is compute-only · oracle configs in versioned YAML · coordination agent built last.

### Embedding & Compute

- [ ] **EMB-V2-01**: Embedding dimension is dynamic (carried from the backend/manifest), replacing the hardcoded `EMBEDDING_DIM = 384` constant; fail-loudly asserts retained but with a dynamic expected value
- [ ] **EMB-V2-02**: A pluggable `EmbeddingBackend` Protocol exposes selectable backends — SentenceTransformer (local, 384) and OpenAI `text-embedding-3-small` (1536) — chosen per run
- [ ] **EMB-V2-03**: Embedding artifacts carry a provenance manifest (model, dim, normalized flag, library versions, input hash, n_items) validated on load; a content-hash cache prevents re-embedding/re-paying

### Compute Offload (Colab)

- [ ] **COL-V2-01**: A Colab compute-only notebook computes embeddings + the initial clustering and exports artifacts (embeddings, initial state, manifest)
- [ ] **COL-V2-02**: The local app imports Colab artifacts via HuggingFace Hub, validated against the manifest; the local runtime imports zero Colab-only dependencies

### Clustering

- [ ] **CLUST-V2-01**: KMeans-only interactive clustering — HDBSCAN dropped from the interactive path
- [ ] **CLUST-V2-02**: KMeans re-fits on each oracle query with embeddings fixed; a cluster-ID alignment step preserves cluster IDs/names across re-fits; K never auto-reoptimizes (changes only via explicit oracle split/merge intent)

### Flow & Onboarding

- [ ] **FLOW-V2-01**: A dataset introduction/summary is presented to the user/oracle before any clustering runs
- [ ] **FLOW-V2-02**: The oracle issues an initial query that triggers the first clustering; the conversational session then begins. Autonomous-first clustering is retained as an ablation condition

### Query Filter

- [ ] **FILT-V2-01**: A query filter normalizes oracle natural language into simple, contradiction-free clusterer instructions, extending `feedback_parser` + the contradiction layer; it normalizes structure only and does not semantically second-guess oracle intent

### Oracle Configurability

- [ ] **OCFG-V2-01**: The oracle's LLM model is selectable per run (Anthropic or OpenAI)
- [ ] **OCFG-V2-02**: Oracle prompt/persona/noise configuration is stored in versioned YAML files and validated into a typed config object at load

### Visualization & Chat

- [ ] **VIZ-V2-02**: Interactive UMAP caches 2D projection coordinates once and recolors on cluster change, without per-turn refit
- [ ] **UX-V2-01**: Human chat works end-to-end in the study UI
- [ ] **UX-V2-02**: The LLM-oracle conversation is viewable in real time

### Coordination Agent (deferred to last phase)

- [ ] **COORD-V2-01**: A coordination agent decomposes a complex clustering operation into pairwise sub-operations
- [ ] **COORD-V2-02**: Pairwise sub-operations fan out to N clusterer sessions and recombine into a single authoritative `ClusteringState` (single-source-of-truth invariant preserved; whole-op abort on partial failure)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Automatic K optimization (silhouette, BIC) | Anti-feature: bypasses oracle as objective function; K changes only through oracle intent |
| Showing raw model internals to oracle | Increases cognitive load without proportional annotation quality gain (Prodigy principle) |
| Re-embedding the dataset per oracle query (v2.0) | Resolved as re-FIT (KMeans) on fixed embeddings; preserves the read-only EmbeddingStore invariant and avoids cost/geometry churn |
| Running the FastAPI/socketio UI on Colab (v2.0) | Colab is compute-only; sockets/UI run locally |
| Query filter that semantically rewrites oracle intent (v2.0) | The oracle IS the objective function; the filter normalizes structure only |
| eventlet / gevent (v2.0) | Monkey-patches stdlib and corrupts numpy/sklearn/UMAP locking |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRE-01 | Phase 1 | Pending |
| PRE-02 | Phase 1 | Pending |
| FOUND-01 | Phase 1 | Pending |
| FOUND-02 | Phase 1 | Pending |
| FOUND-03 | Phase 1 | Pending |
| FOUND-04 | Phase 1 | Pending |
| CLUS-01 | Phase 2 | Pending |
| CLUS-02 | Phase 2 | Pending |
| CLUS-03 | Phase 2 | Pending |
| CLUS-04 | Phase 2 | Pending |
| FB-01 | Phase 2 | Pending |
| FB-02 | Phase 2 | Pending |
| FB-03 | Phase 2 | Pending |
| HIER-01 | Phase 2 | Pending |
| HIER-02 | Phase 2 | Pending |
| UI-01 | Phase 2 | Pending |
| UI-02 | Phase 2 | Pending |
| ORC-01 | Phase 3 | Pending |
| ORC-02 | Phase 3 | Pending |
| ORC-03 | Phase 3 | Pending |
| ORC-04 | Phase 3 | Pending |
| FB-04 | Phase 3 | Pending |
| DB-01 | Phase 4 | Pending |
| DB-02 | Phase 4 | Pending |
| DB-03 | Phase 4 | Pending |
| JUDG-01 | Phase 4 | Complete |
| JUDG-02 | Phase 4 | Pending |
| JUDG-03 | Phase 4 | Pending |
| ALAB-01 | Phase 5 | Complete |
| ALAB-02 | Phase 5 | Pending |
| ALAB-03 | Phase 5 | Pending |
| GEN-01 | Phase 6 | Pending |
| GEN-02 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 33
- Unmapped: 0

---
*Requirements defined: 2026-04-29*
*Last updated: 2026-05-21 — milestone v2.0 requirements added (17 reqs across 7 categories); traceability filled by roadmapper*
