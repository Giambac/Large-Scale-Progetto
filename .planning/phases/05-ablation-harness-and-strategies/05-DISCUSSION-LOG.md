# Phase 5: Ablation Harness and Strategies - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-16
**Phase:** 5-Ablation-Harness-and-Strategies
**Areas discussed:** Strategy mechanics, Harness design, oracle_type DB field, Bootstrap CI script

---

## Strategy mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| UncertaintyDriven: Highest-uncertainty cluster | Target cluster with lowest mean soft-assignment confidence | ✓ |
| UncertaintyDriven: Highest-uncertainty point | Target single boundary point with lowest max-assignment probability | |
| UncertaintyDriven: You decide | Planner picks signal from f_uncertainty output | |

**User's choice:** Highest-uncertainty cluster

---

| Option | Description | Selected |
|--------|-------------|----------|
| BoundaryDriven: Points near cluster boundaries | Individual boundary points | |
| BoundaryDriven: Smallest cluster | Target cluster with fewest items | |
| BoundaryDriven: Cluster pair (mutual overlap) | Most-confused cluster pair | ✓ |
| BoundaryDriven: Group of points sharing boundary ambiguity | All items confused between the most-confused pair, shown as a coherent group | ✓ |

**User's choice:** Group of items confused between the most-confused cluster pair (A,B). Payload: `{cluster_a, cluster_b, item_ids}`. Evolved through clarification — user noted that single-point focus is useless conversationally unless used for generalization; a group sharing the same boundary ambiguity is actionable.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Degenerate fallback: Fall back to Random | Delegate to RandomStrategy when no valid target | |
| Degenerate fallback: Raise exception | Crash loudly on degenerate state | |
| Degenerate fallback: ask_question (oracle-directed) | Hand initiative to oracle with generic question | ✓ |

**User's choice:** ask_question with generic payload — hand initiative to oracle. User answered "Do nothing, ask a question to the oracle for clarification" when presented with the fallback options.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Oracle initiative: Full freedom within response | Oracle responds however it wants to any agent action | ✓ |
| Oracle initiative: Oracle-initiated turns | Human can push message at any point unprompted | |
| Oracle initiative: Deferred to Phase 6 | Lock for human study design | |

**User's choice:** Full freedom within response (already works). User initially asked "I need in general the user to be able to express what he wants and not be driven by the questions of the clustering agent" — clarified to mean oracle can volunteer any feedback type in response to any action, not oracle-initiated turns.

---

## Harness design

| Option | Description | Selected |
|--------|-------------|----------|
| 3 personas × 5 seeds = 45 runs | More data, wider CIs | |
| 3 personas × 3 seeds = 27 runs | Lighter, tunable | ✓ |
| You decide | Planner picks M and K | |

**User's choice:** 3×3×3 = 27 runs default, tunable later.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Sequential | Simple for-loop, one run at a time | |
| Parallel with threading | ThreadPoolExecutor, concurrent runs | ✓ |

**User's choice:** Parallel with threading.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Cross-product YAML | List strategies/personas/seeds; harness generates all combinations | |
| Explicit run list | Each run listed individually | |
| Cross-product with exclusions | Cross-product + optional exclude list | ✓ |

**User's choice:** Cross-product with exclusions.

---

## oracle_type DB field

| Option | Description | Selected |
|--------|-------------|----------|
| New indexed column | oracle_type TEXT NOT NULL DEFAULT 'llm' | ✓ |
| Inside details JSON | No migration, not directly filterable | |

**User's choice:** New indexed column.

**Notes:** User asked "Is this for the sessions I already have in memory?" — confirmed: only concern was preserving existing Phase 4 test data. User is fine restarting fresh, so no ALTER TABLE migration needed. Column added directly to CREATE TABLE DDL.

---

## Bootstrap CI script

| Option | Description | Selected |
|--------|-------------|----------|
| examples/compute_ci.py only | CLI script | |
| Jupyter notebook only | Interactive exploration | |
| Both — CLI + notebook | src/analysis.py shared logic + CLI + notebook | ✓ |

**User's choice:** Both. User asked "What is Bootstrap CI?" — explained bootstrapping approach. Then asked "Is the notebook useful to revise older test runs or not?" — confirmed: notebook queries DB directly so useful for any past/future runs, not just current results.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Printed table | Human-readable comparison table | |
| JSON to stdout | Machine-readable | |
| Both — table + --json flag | Default table, --json flag for JSON | ✓ |

**User's choice:** Both — table by default, `--json` flag.

---

## Claude's Discretion

- Exact threshold for "items confused between cluster pair" in BoundaryDriven (e.g., `P(A) > 0.3 AND P(B) > 0.3` vs. top-N by proximity)
- Action-type selection logic within each strategy (when show_subset vs. ask_question)
- `max_workers` default value
- Exact OracleSpec + NoiseParams values for curious/skeptical/drifty personas
- Whether show_full message is also enriched with sample item texts
- SQL index on oracle_type column

## Deferred Ideas

- **Oracle-initiated turns** — human pushes message unprompted. Deferred to Phase 6 human study UI.
- **Mutable noise params (fatigue simulation)** — consistency_rate degrading turn-by-turn. Still deferred from Phase 3.
- **show_full enrichment** with actual item texts — left to planner.
