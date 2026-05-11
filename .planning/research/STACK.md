# Technology Stack

**Project:** Conversational Clustering — Multi-Agent Human-in-the-Loop System
**Researched:** 2026-04-29
**Overall confidence:** HIGH (core stack), MEDIUM (tooling periphery)

---

## Recommended Stack

### LLM Orchestration Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| LangGraph | 1.1.x (latest: 1.1.10) | Agent graph execution, state machine, human-in-the-loop interrupt | Only framework where stateful graph + HITL interrupts + durable execution are first-class citizens, not bolted on. Reached 1.0 stability in late 2025. |
| LangChain Core | 0.3.x | Message types, prompt templates, LLM abstraction | Underlies LangGraph; provides model-agnostic LLM calls without framework lock-in |

**Why LangGraph over alternatives:**

- **vs AutoGen 0.4**: AutoGen's AgentChat is conversational-first and good for free-form dialogue, but state persistence and explicit graph control are harder to achieve. The Clustering Agent's `f_next_best_step` decision logic needs deterministic branching — a graph node, not a chat turn.
- **vs CrewAI**: Role-based YAML-driven setup is optimized for pipelines with static roles and sequential execution. The oracle feedback loop is inherently cyclical and stateful; CrewAI requires workarounds for that.
- **vs Raw OpenAI API**: Raw API gives full transparency but forces manual implementation of state, checkpointing, conversation history management, and HITL pause/resume. LangGraph provides all of these; the cost is ~200 lines of boilerplate saved per agent.
- **vs PydanticAI**: Good for type-safe single-agent pipelines. Multi-agent orchestration and graph-level state machines are not its primary target.

**Confidence:** HIGH — LangGraph 1.1.x is the ecosystem default for stateful multi-agent Python systems as of 2026.

---

### LLM Provider

| Technology | Version/Model | Purpose | Why |
|------------|--------------|---------|-----|
| OpenAI API | gpt-4o-mini (oracle simulation, judge), gpt-4o (clustering agent when needed) | LLM backbone for all three agents | Best cost/capability ratio for high-volume oracle simulations; gpt-4o-mini at $0.15/M input tokens enables hundreds of ablation runs without budget pressure |
| openai (Python SDK) | 1.x | API client | Official, maintained, async-capable |

**Model assignment rationale:**
- **Clustering Agent**: gpt-4o (or gpt-4.1 if available). Proposal quality matters; pay for reasoning.
- **Oracle Agent**: gpt-4o-mini. Runs in a tight loop; cost matters more than ceiling capability. Persona + preference spec constrains output space.
- **Judge Agent**: gpt-4o-mini. Convergence detection is a structured classification task, not a reasoning task.

**Model-agnostic note:** Use LangGraph's LangChain model abstraction (`ChatOpenAI`, `ChatAnthropic`) so models can be swapped without changing agent logic. This is important for ablation experiments.

**Confidence:** HIGH for OpenAI; MEDIUM for specific model assignments (adjust after first cost profiling run).

---

### Embedding Library

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| sentence-transformers | 5.x (latest: 5.3.0) | Sentence embeddings for text clustering | Local inference, no API cost, reproducible, well-benchmarked on clustering tasks. Project constraint says one embedding representation — this is the standard choice. |

**Recommended model:** `all-mpnet-base-v2`

- Produces 768-dimensional vectors, 12 transformer layers, ~110M parameters
- Consistently ranks higher than `all-MiniLM-L6-v2` on MTEB clustering benchmarks (~87-88% STS-B vs ~84-85%)
- Speed is not the bottleneck here: embeddings are computed once at dataset load, not per conversation turn
- For datasets of 500-5000 texts (support tickets, reviews), inference takes seconds on CPU

**Why not OpenAI `text-embedding-3-small`:**
- $0.02/M tokens — negligible for one-time embedding, but adds an external API dependency
- Embeddings are not reproducible across API versions (model updates change vectors silently)
- Local `sentence-transformers` eliminates the API round-trip and is fully deterministic for research reproducibility

**Confidence:** HIGH — sentence-transformers is the ecosystem default for local text embedding; model choice is MEDIUM (could switch to `all-MiniLM-L6-v2` if CPU is severely constrained).

---

### Clustering Algorithm

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| scikit-learn (HDBSCAN) | 1.6+ (stable 1.8.0) | Initial clustering, soft assignment probabilities | HDBSCAN is now native in scikit-learn since 1.3; provides `probabilities_` attribute for soft assignments (required by PROJECT.md); automatically determines cluster count; handles variable-density clusters in embedding space |

**Why HDBSCAN over k-means:**
- k-means requires specifying K upfront and produces hard assignments. The project explicitly requires soft assignments and a navigable hierarchy, which HDBSCAN natively provides.
- HDBSCAN's dendrogram output enables hierarchical exploration (3-10 top-level clusters with sub-clusters).
- `probabilities_` per point directly feeds soft-assignment calibration metrics.

**Why scikit-learn's HDBSCAN over `hdbscan` package:**
- `sklearn.cluster.HDBSCAN` (added 1.3, stable in 1.6+) avoids an extra dependency.
- The standalone `hdbscan` package (`scikit-learn-contrib/hdbscan`) offers `all_points_membership_vectors()` for full probability distributions over all clusters. **Use the standalone package if full per-point multinomial soft assignments are needed** (not just membership strength scalar). Evaluate this at implementation time.

**Soft assignment implementation note:**
```python
from sklearn.cluster import HDBSCAN

clusterer = HDBSCAN(
    min_cluster_size=5,
    cluster_selection_method="eom",  # Excess-of-Mass: better for variable density
    metric="euclidean",              # Use after L2-normalizing embeddings
    store_centers="centroid",        # Required for f_output (show cluster center)
    prediction_data=True,            # Required for membership vectors
)
clusterer.fit(embeddings)
labels = clusterer.labels_          # Hard assignments (-1 = noise)
strengths = clusterer.probabilities_ # Soft membership scalar per point
```

**Confidence:** HIGH for algorithm choice; MEDIUM for whether sklearn's HDBSCAN or the standalone package is sufficient for full multinomial soft assignments (verify at Phase 1).

---

### State Management

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python dataclasses / TypedDict | stdlib | LangGraph state schema definition | LangGraph requires a TypedDict or dataclass for the graph state object. No extra dependency; type-safe; LangGraph uses it for checkpointing. |
| Pydantic v2 | 2.x | Validating oracle feedback messages, cluster proposals | Structured LLM output parsing via `.model_validate_json()`; integrates with LangGraph tool calling |

**State object design:**

The LangGraph graph state should contain:
- `turn_count: int`
- `current_clusters: list[ClusterState]` — each with label, description, member indices, centroid
- `soft_assignments: np.ndarray` — shape (N, K)
- `conversation_history: list[Message]`
- `oracle_preferences: OraclePreferenceSpec`
- `contradiction_log: list[Contradiction]`
- `convergence_signal: ConvergenceSignal | None`

**Confidence:** HIGH — this is standard LangGraph practice.

---

### Experiment Tracking

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| MLflow | 2.x (2.19+ for LLM tracing) | Logging ablation runs, metrics per turn, convergence curves | Runs entirely local with zero server setup; `mlruns/` directory is created automatically; `mlflow ui` for inspection. No SaaS account needed for a solo/pair academic project. |

**Why MLflow over alternatives:**
- **vs Weights & Biases**: W&B is developer-friendly with excellent UI, but requires account creation and SaaS dependency. Costs $50-200/user/month for teams; overkill for solo/pair scope.
- **vs simple JSON logging**: JSON is perfectly viable for early phases. Recommend starting with JSON and migrating to MLflow when ablation experiments begin (Phase 3+). MLflow's run comparison and parameter search are hard to replicate manually once you have 20+ runs.
- **vs Neptune**: Enterprise-scale, unnecessary here.

**Migration path:** Start with structured JSON logging (`experiments/run_{id}.json`) in Phases 1-2. Introduce MLflow in the ablation phase. This avoids early infrastructure overhead while ensuring the data format is migration-compatible.

**Confidence:** HIGH for MLflow as the right tool when ablations start; HIGH for JSON-first in early phases.

---

### CLI / Notebook Interface

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Typer | 0.12+ | CLI entry points, command definition | Built on Click with type-hint-based argument parsing; integrates natively with Rich for styled output; faster boilerplate than raw Click |
| Rich | 13.x | Terminal formatting: tables for cluster proposals, progress bars, dialogue panels | Makes the conversation loop readable in a terminal; critical for usability during human oracle sessions |
| Jupyter / IPython | latest | Notebook-based exploration and demo | PROJECT.md explicitly allows notebook scope; useful for interactive development and committee demos |

**CLI design pattern:**
```
python main.py run --dataset data/tickets.csv --strategy ask-first --turns 20
python main.py ablate --strategies all --oracle-persona "domain expert"
python main.py evaluate --run-id abc123
```

**Why Typer + Rich:**
- Typer generates `--help` automatically from type hints; no manual argparse boilerplate.
- Rich panels and tables are the right output primitive for "show cluster proposal, ask oracle" turns.
- Typer + Rich is the 2025 standard pairing for Python CLI research tooling.

**Confidence:** HIGH.

---

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.26+ / 2.x | Embedding arrays, soft assignment matrices | Always — core numeric substrate |
| pandas | 2.x | Dataset loading, result DataFrames | Dataset ingestion and output tabulation |
| scipy | 1.x | Cosine similarity, hierarchical linkage for cluster tree | Needed for L2-norm, cosine distance pre-processing before HDBSCAN |
| pytest | 8.x | Unit tests for `f_*` functions | Agent logic functions are pure functions; test them independently |
| python-dotenv | 1.x | API key management | Keep OpenAI keys out of source code |
| datasets (HuggingFace) | 3.x | Loading Amazon Reviews 2023, IMDB datasets | Standard interface for the candidate datasets mentioned in PROJECT.md |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Orchestration | LangGraph 1.1.x | AutoGen 0.4 | AutoGen is conversation-first; explicit graph state and HITL interrupt mechanics require more workaround |
| Orchestration | LangGraph 1.1.x | CrewAI | YAML role-based; cyclic oracle loop and dynamic state are awkward fits |
| Orchestration | LangGraph 1.1.x | Raw OpenAI API | Viable but requires manual re-implementation of state, checkpointing, HITL pause/resume |
| Embeddings | sentence-transformers (local) | OpenAI text-embedding-3-small | API dependency, non-reproducible across model versions, no advantage for fixed offline dataset |
| Embeddings | all-mpnet-base-v2 | all-MiniLM-L6-v2 | MiniLM is 5x faster but 3-4% weaker on clustering benchmarks; speed not a bottleneck here |
| Clustering | HDBSCAN (sklearn) | k-means | k-means requires K upfront, produces hard labels only; HDBSCAN is strictly more appropriate given soft-assignment and hierarchy requirements |
| Clustering | HDBSCAN (sklearn) | GMM (Gaussian Mixture) | GMM gives full probability distributions but assumes Gaussian structure in embedding space — not warranted |
| Tracking | MLflow (local) + JSON-first | Weights & Biases | W&B requires SaaS account; unnecessary external dependency for solo/pair scope |
| CLI | Typer + Rich | argparse | argparse is stdlib but verbose; Typer removes boilerplate without adding meaningful risk |

---

## Installation

```bash
# Core orchestration
pip install langgraph langchain-core langchain-openai

# Embeddings
pip install sentence-transformers

# Clustering
pip install scikit-learn>=1.6  # HDBSCAN included since 1.3
pip install hdbscan            # Optional: only if full multinomial soft assignments needed

# CLI + formatting
pip install typer rich

# Experiment tracking (introduce at ablation phase)
pip install mlflow

# Data + utilities
pip install numpy pandas scipy datasets python-dotenv

# Development
pip install pytest ipykernel jupyter
```

**Estimated install size:** ~2.5 GB (dominated by sentence-transformers + torch for CPU inference).

**Python version:** 3.11 or 3.12. LangGraph 1.1.x, sentence-transformers 5.x, and scikit-learn 1.6+ all support 3.12; use 3.11 for maximum compatibility with older transitive dependencies.

---

## Version Summary Table

| Library | Pinned Version | Confidence |
|---------|---------------|------------|
| langgraph | ~=1.1.0 | HIGH |
| langchain-core | ~=0.3.0 | HIGH |
| langchain-openai | ~=0.3.0 | HIGH |
| openai | ~=1.0 | HIGH |
| sentence-transformers | ~=5.3.0 | HIGH |
| scikit-learn | >=1.6,<2.0 | HIGH |
| hdbscan (standalone) | ~=0.8.1 | MEDIUM (use only if sklearn insufficient) |
| numpy | >=1.26 | HIGH |
| pandas | >=2.0 | HIGH |
| scipy | >=1.11 | HIGH |
| typer | >=0.12 | HIGH |
| rich | >=13.0 | HIGH |
| mlflow | >=2.19 | HIGH |
| datasets | >=3.0 | MEDIUM |
| pytest | >=8.0 | HIGH |
| python-dotenv | >=1.0 | HIGH |

---

## Critical Stack Decisions for Roadmap

1. **LangGraph is the single source of truth for agent state.** All three agents are LangGraph nodes or subgraphs. The shared TypedDict state flows through the graph. This eliminates ad-hoc global state and makes HITL pausing straightforward via `interrupt()`.

2. **Embeddings are computed once at startup, not per turn.** Pre-compute and cache the embedding matrix. Oracle feedback operates on cluster memberships, not raw text — the embedding layer is fixed after initialization.

3. **JSON-first logging, MLflow from Phase 3.** Early phases produce one or two runs; JSON is sufficient. When ablation experiments begin (3-5 strategies × N runs), migrate to MLflow. Design JSON schema to be MLflow-compatible from day one.

4. **HDBSCAN soft assignments via `probabilities_` from sklearn.** Verify whether scalar membership strength is sufficient for the soft-assignment requirement, or whether full multinomial probability vectors (requiring standalone `hdbscan` package's `all_points_membership_vectors`) are needed. This is a Phase 1 decision point.

5. **Model-agnostic via LangChain abstractions.** Use `ChatOpenAI` (or `ChatAnthropic`, `ChatGoogleGenerativeAI`) rather than the raw `openai` SDK in agent logic. Swapping the underlying model for ablations then requires changing one config value, not refactoring agent code.

---

## Sources

- LangGraph GitHub (version 1.1.10, April 2026): https://github.com/langchain-ai/langgraph
- LangGraph overview docs: https://docs.langchain.com/oss/python/langgraph/overview
- DataCamp: CrewAI vs LangGraph vs AutoGen: https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen
- OpenAgents multi-framework comparison (2026): https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared
- scikit-learn HDBSCAN docs (1.8.0): https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
- sentence-transformers releases (v5.3.0): https://github.com/huggingface/sentence-transformers/releases/tag/v5.3.0
- all-mpnet-base-v2 vs all-MiniLM-L6-v2 comparison: https://milvus.io/ai-quick-reference/what-are-some-popular-pretrained-sentence-transformer-models-and-how-do-they-differ-for-example-allminilml6v2-vs-allmpnetbasev2
- OpenAI API pricing (2026): https://openai.com/api/pricing/
- MLflow local tracking: https://mlflow.org/docs/latest/ml/tracking/quickstart/
- Typer + Rich CLI guide: https://dasroot.net/posts/2026/01/building-cli-tools-with-typer-and-rich/
- HDBSCAN soft clustering docs: https://hdbscan.readthedocs.io/en/latest/soft_clustering.html
