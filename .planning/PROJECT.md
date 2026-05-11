# Conversational Clustering

## What This Is

An AI research system that clusters any textual dataset by conversing with a human oracle — proposing groupings, explaining them, and refining them through dialogue. The system is dataset-agnostic: it accepts any text corpus (support tickets, reviews, articles, etc.) without modification. The oracle is the sole judge of quality, making this a study of how an AI can efficiently converge to a subjectively acceptable clustering under a tight cognitive-load budget, with no intrinsic ground truth.

## Core Value

The interaction loop converges toward oracle-accepted clusterings efficiently, with every design decision — what to show, what to ask, when to stop — measured against cognitive load and information gain.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Ingest text dataset and produce an initial clustering with cluster names and natural-language descriptions
- [ ] Accept oracle feedback at four levels: global, cluster-level, point-level, and instructional
- [ ] Produce soft assignments (per-point distribution over K clusters) and a navigable hierarchy
- [ ] Implement `f_next_best_step`: decide each turn whether to show, ask, or stop
- [ ] Maintain coherent state across turns; latest oracle intent wins on contradictions
- [ ] Implement a stopping signal (oracle satisfaction, diminishing returns, or turn budget)
- [ ] Run ablation experiments across 3–5 interaction strategies with LLM-simulated oracles
- [ ] Validate LLM-simulated oracle behavior against a small human study (N ≈ 5–10)
- [ ] Produce at least one defensible quantified claim with a confidence interval
- [ ] Enable generalization: codify oracle preferences into a function for new items

### Out of Scope (v1)

- Automatic K optimization (silhouette, BIC) — anti-feature; K changes only through oracle intent
- Showing raw model internals to oracle — increases cognitive load without proportional gain

### Trio/Quartet Scope (v2)

Features deferred until team grows to 3–4 or v1 is complete:

- **UMAP/t-SNE visualization** — 2D embedding projection in the web UI (VIZ-V2-01)
- **Multiple clustering backends** — k-means, LLM-first alongside HDBSCAN (BACK-V2-01)
- **Persistent sessions** — save and resume state across server restarts (UI-V2-01)
- **Full N×M headline experiment** — N oracles × M tasks, LLM oracles for scale + human oracles (N ≥ 10, within-subject), comparing conversational interface vs. default-parameter baseline on oracle satisfaction, turns-to-convergence, and generalization accuracy (EXP-V2-01)

## Context

This is an academic AI systems project (Large-Scale semester 2). The system models clustering as a human-in-the-loop optimization problem where the oracle *is* the objective function. The core research insight is that clustering is fundamentally under-defined without a human preference signal, and the interesting engineering question is how to extract that signal efficiently.

**Architecture:** A multi-agent system with three core agents:
- **Clustering Agent** — maintains state, proposes groupings, implements `f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state`
- **Oracle Agent** — LLM-simulated human oracle with a preference specification, persona, and cognitive-load constraint; stand-in for real humans at scale
- **Judge Agent** — evaluates convergence and quality; implements `f_eval`; produces the stopping signal and evaluation metrics

A 4th agent for synthetic data generation is acknowledged but deferred. Each agent starts as a naïve LLM prompt and is sharpened where bottlenecks emerge.

**Dataset:** The system is dataset-agnostic and must work on any text corpus without modification. Example domains: support tickets, product reviews, news articles, IMDB movies, Amazon Reviews 2023. Target: 3–10 top-level clusters, non-trivial conversation possible. For each dataset used in experiments, a frozen held-out split is locked before any experiment runs on that dataset.

**Evaluation approach (no ground truth):** Combine oracle satisfaction, turns to convergence, cognitive load per turn, contradiction/drift tracking, pairwise validation, soft-assignment calibration, and generalization performance. LLM-simulated oracles for scale; human study (N ≈ 5–10) for validation.

**Existing scripts in repo:**
- `Conversational Clustering Script.txt` — prototype or reference script
- `Multi Agent Personalities Script.txt` — multi-agent persona reference

## Constraints

- **Scope**: Solo/pair tier — CLI or notebook interface; dataset-agnostic (any text corpus via upload or CLI); one embedding representation
- **Research integrity**: All headline quantitative claims must include confidence intervals; human study requires consented, within-subject protocol
- **Fail loudly**: Code must crash on unexpected state — no silent failures, no defensive `if/else` chains, no swallowed exceptions. An assertion error or uncaught exception is always preferable to continuing with bad state. Error handling is only permitted at system boundaries (CLI entry point, API calls). Everywhere else: let it crash.
- **Cognitive load**: Must be budgeted from turn one, not discovered at study time
- **Evaluation validity**: Evaluation must measure something the dialogue actually influences, not just the underlying clustering's default behavior
- **LLM oracle validation**: Simulated oracle results must be validated against humans before any quantitative claim is published

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Soft assignments over hard labels | Enables boundary detection and calibration testing | — Pending |
| `f_*` function decomposition | Clean separation allows iterative sharpening of bottlenecks | — Pending |
| LLM-simulated oracles for scale | Human oracles are expensive; LLMs enable large-scale ablations | — Pending |
| Latest intent wins on contradictions | Treat preference evolution as signal, not error | — Pending |
| Sentence embeddings as representation | Standard, well-understood; oracle feedback acts on grouping, not embeddings | — Pending |
| 3-agent architecture | Clustering agent + Oracle agent + Judge agent — clean separation of concerns; oracle agent simulates human for scale | — Pending |
| Synthetic data agent deferred | Out of scope for v1; real or curated datasets used instead | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-29 after initialization*
