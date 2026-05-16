# Phase 4: Judge Agent - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-16
**Phase:** 4-Judge-Agent
**Areas discussed:** Database choice & lifecycle, Schema shape & relationships, DB code layer organization, Diminishing-returns values (JUDG-01), Pairwise validation (JUDG-02), No-dialogue baseline (JUDG-03), Judge structure + WR-04, STRICT_MODE + CI, Documentation discipline
**Recipe followed:** wazzup "how to build simple applications" (user-provided this session)

---

## 1. Database choice & lifecycle

### Q1.1 — DB scope

| Option | Description | Selected |
|--------|-------------|----------|
| Single shared experiments.db at repo root | One file across all runs; cross-run analysis is a single SELECT; tests use `:memory:` per fixture | ✓ |
| Per-session DB inside sessions/<timestamp>/ | Self-contained sessions; cross-run queries become UNIONs across files | |
| Hybrid (per-session + nightly merge) | Sessions stay self-contained; analysis script merges | |

**Notes:** User asked "what about the tests?" before answering; clarified that tests use `sqlite3.connect(":memory:", check_same_thread=False)` per `conftest.py` fixture, so the production DB is never touched in tests. Then selected the recommended option.

### Q1.2 — DB vs JSONL coexistence

| Option | Description | Selected |
|--------|-------------|----------|
| Both — JSONL stays per-session, DB stores summary + cross-run rows | JSONL is source of truth for replay; DB is queryable index | ✓ |
| DB only — stop writing audit_log.jsonl | All per-turn data moves to DB-02 turns table; breaks Phase 1–3 load_audit_log readers | |
| JSONL only with a SQLite VIEW on top | DB as a query layer over JSONL; doesn't satisfy DB-01/02/03 | |

### Q1.3 — Write order per turn

| Option | Description | Selected |
|--------|-------------|----------|
| JSONL first, then DB, in the same loop step | Preserves Phase 2 D-04 "loop owns JSONL write"; DB derivable from JSONL on crash | ✓ |
| DB first, then JSONL | Reverses ownership; doesn't match Phase 2 D-04 | |
| Single transaction wrapping both | Pure but couples the JSONL writer to SQLite transaction lifecycle | |

### Q1.4 — Connection lifecycle

| Option | Description | Selected |
|--------|-------------|----------|
| One connection per run; commit per turn; WAL mode | Caller owns the unit of work (recipe §6); WAL allows concurrent read for the UI; check_same_thread=False mandatory | ✓ |
| One connection per run, single commit at end | Atomicity but loses partial progress on crash | |
| One connection per turn | Wastes overhead; recipe §6 explicitly advises against | |

---

## 2. Schema shape & relationships

### Q2.1 — Entity shape per table

| Option | Description | Selected |
|--------|-------------|----------|
| experiments=full recipe shape; turns + oracle_feedback=minimal | Recipe §2 explicitly endorses dropping name/slug for `message`-like entities | ✓ |
| All three follow full shape | Uniform but invents meaningless name/slug for turns | |
| All three minimal — just DB-01/02/03 columns | Loses details JSON forward-compat lever | |

### Q2.2 — Rels shape

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated FK columns on each child table | Real FK integrity via PRAGMA foreign_keys=ON; direct joins; recipe §3 endorses for "frequent rels" | ✓ |
| Single polymorphic rels table | Recipe §3 purist form; no FK integrity at DB level | |
| Hybrid (direct FK + rels for ad-hoc future) | Speculative | |

### Q2.3 — Cascade rule on experiment delete

| Option | Description | Selected |
|--------|-------------|----------|
| Cascade to turns + oracle_feedback | Soft-delete propagates via deletion.py orchestrator; hard-delete via ON DELETE CASCADE; rule in docs/MODEL.md | ✓ |
| Restrict — refuse to delete if turns exist | Safer for accidents; no real use-case in v1 | |
| No deletion at all in Phase 4 | Immutable research records; simplest | |

### Q2.4 — details JSON content on experiments

| Option | Description | Selected |
|--------|-------------|----------|
| Indexed: strategy_id, persona_id, seed, dataset, convergence_reason, total_turns. Details: headline metrics + run metadata | "Promote to a column only when you actually query by it" per recipe §2 | ✓ |
| Maximalist — every metric is its own column | Bigger schema, more migrations | |
| Minimalist — only REQUIREMENTS.md columns | More conservative reading of "headline metrics" | |

---

## 3. DB code layer organization

### Q3.1 — Layer layout

| Option | Description | Selected |
|--------|-------------|----------|
| New src/db/ subpackage with per-entity modules | Mirrors recipe wazzup/api/ layout 1:1; the only layer that touches SQL | ✓ |
| Single src/db.py module | Simpler today; bloats fast; loses deletion.py orchestrator pattern | |
| Keep SQL in src/judge.py | Recipe §6 violation; blocks Phase 5 harness reuse | |

### Q3.2 — Model type

| Option | Description | Selected |
|--------|-------------|----------|
| Dataclasses, matching existing convention | Continues Phase 1–3 dataclass-only pattern; fewer deps | |
| Pydantic triples per entity, matching recipe | Free validation; model_dump_json round-trip for details JSON; new entry point for Pydantic in codebase | ✓ |
| Mixed — dataclasses internally, Pydantic only at HTTP boundary | Defers the decision | |

**Notes:** User chose recipe-aligned Pydantic explicitly. CONTEXT.md flags this as the entry point for Pydantic in the codebase; Pydantic stays INSIDE src/db/ only.

### Q3.3 — Error conventions

| Option | Description | Selected |
|--------|-------------|----------|
| Recipe §6 conventions: NotFound + IntegrityError + None/[] returns | Consistent across entity modules; future HTTP layer gets free 404/409 translation | ✓ |
| Custom DB-layer exceptions only | Fail-loudly purist; diverges from recipe + stdlib sqlite3 | |
| Assert-only | Too coarse for HTTP translation; loses recipe §6 win | |

### Q3.4 — Slug generation

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-derive from strategy+persona+seed with collision suffix | Reproducible, greppable; uses make_slug + partial unique index per recipe §6 | ✓ |
| Random UUID slug | Opaque; loses human-readable URL win | |
| Caller-provided slug, no auto-derivation | Forces friction at every caller; Phase 5 has to invent a convention anyway | |

---

## 4. Diminishing-returns values (JUDG-01)

### Q4.1 — Magnitude weights

| Option | Description | Selected |
|--------|-------------|----------|
| 4 / 3 / 2 / 1 (integer ladder) | Simple, monotonic, easy to read in logs | |
| 1.0 / 0.5 / 0.2 / 0.1 (geometric falloff) | Stronger emphasis on global feedback; 10x spread between global and instructional | ✓ |
| Defer — placeholder values, tune in Phase 5 | All-equal stub; defer tuning | |

**Notes:** User explicitly required the parameters to remain tunable per-experiment. Already supported by Phase 1's `FeedbackMagnitudeWeights` dataclass; D-16 routes overrides through the YAML config.

### Q4.2 — Epsilon

| Option | Description | Selected |
|--------|-------------|----------|
| ε = 0.15 | Just above a single instructional | |
| ε = 0.3 | Above a single point-level; more aggressive | |
| ε = 0.0 (strict zero) | Only an empty-feedback turn counts | |
| ε = 0.05 (user free-text) | Effectively a "true-silence" threshold; below the smallest weighted magnitude (0.1) | ✓ |

**Notes:** User picked 0.05 (lower than any presented option) and re-emphasized tunability.

### Q4.3 — N_fallback

| Option | Description | Selected |
|--------|-------------|----------|
| N = 3 | Three silent turns; immune to single quiet turn; tunable | ✓ |
| N = 2 | Faster; risks stopping after one "pause for thought" | |
| N = 5 | Conservative; effectively rarely fires | |

**Notes:** User confirmed N=3 and re-emphasized tunability.

### Q4.4 — Tuning surface

| Option | Description | Selected |
|--------|-------------|----------|
| CLI flags + recorded on experiments row | No config file; CLI sets each knob | |
| YAML config file per experiment, CLI accepts --config path | experiments/configs/<name>.yaml; PyYAML dep; config snapshot in experiments.details | ✓ |
| Python-only — StoppingCriteria construction; no CLI surface | Phase 5 harness owns it; awkward for ad-hoc CLI tuning | |

---

## 5. Pairwise validation accuracy (JUDG-02)

### Q5.1 — Truth source

| Option | Description | Selected |
|--------|-------------|----------|
| Accumulated oracle feedback | Pairs derived from prior MoveItem/Split/Merge feedback; no new oracle calls per turn; honest with "no intrinsic ground truth" framing | ✓ |
| Held-out labels from locked dataset | Amazon Reviews has no cluster labels; would have to synthesize, contaminating PRE-01 | |
| Fresh oracle calls each turn | Cleanest but spikes cognitive load + token cost | |

### Q5.2 — Pair extraction

| Option | Description | Selected |
|--------|-------------|----------|
| Same-cluster + different-cluster pairs from all feedback | MoveItem → same-cluster pairs; Split → diff-cluster pair; Merge → cross-product same-cluster pairs | ✓ |
| Only MoveItemFeedback pairs | Simpler; loses signal from splits/merges (common in early turns) | |
| Defer extraction logic to planner | Leaves most under-specified metric still under-specified | |

### Q5.3 — Sample size + recording

| Option | Description | Selected |
|--------|-------------|----------|
| N = min(50, bag_size); turns.details JSON; seeded by turn_index | Hard cap prevents big merges from dominating; deterministic replay | ✓ |
| All-pairs (no cap) | O(N²) in feedback count; problematic in Phase 5 ablations | |
| Fixed N=20 | Noisier; needs more runs for bootstrap CI | |
| N = min(100, bag_size) | Higher cap; less variance | |

**Notes:** User asked for explanation of what a "pair" is and what the 50 cap practically does before answering. Selected option 1 with explicit "make this tunable" requirement — D-19 routes through the same YAML config as D-16.

### Q5.4 — Contradiction handling

| Option | Description | Selected |
|--------|-------------|----------|
| Overwrite: remove superseded pairs, add new ones | Honors CLUS-04 "latest intent wins"; bag reflects current oracle preferences | ✓ |
| Keep old pairs (additive bag) | Captures drift; metric becomes hard to interpret late in conversation | |
| Tag superseded pairs, exclude from sampling | Both; over-engineered for v1 | |

### Q5.5 — Bag location + resume

| Option | Description | Selected |
|--------|-------------|----------|
| In-memory in Judge module; reconstructed from audit_log.jsonl + events.jsonl on resume | Derived state; no new persistence surface | ✓ |
| Persist in dedicated DB table | 4th table for derived state; violates "JSONL is source of truth" | |
| Per-session pair_bag.json file | Yet another file; same "derived state" objection | |

---

## 6. No-dialogue baseline (JUDG-03)

### Q6.1 — Oracle role

| Option | Description | Selected |
|--------|-------------|----------|
| Oracle runs ONCE on initial clustering | One turn, one row, all metrics filled; directly answers "how good is default?" | ✓ |
| Oracle skipped entirely | pairwise_accuracy undefined; loses the comparison point | |
| Oracle runs N times but loop never applies feedback | Muddies comparison; not really "no dialogue" | |

### Q6.2 — Distinguishing baseline vs interactive

| Option | Description | Selected |
|--------|-------------|----------|
| strategy_id = "no_dialogue" | Baseline is just another strategy; single schema | ✓ |
| Separate kind column experiments.kind ENUM | Redundant with strategy_id; two ways to say the same thing | |
| Separate baselines table | 4th table, UNION queries, recipe §2 violation | |

### Q6.3 — Pairing

| Option | Description | Selected |
|--------|-------------|----------|
| By (dataset, persona_id, seed) tuple — implicit join | Natural composite key; no ordering constraint | ✓ |
| Explicit baseline_experiment_id FK on interactive | Forces baseline-first ordering | |
| Tag both runs with shared run_group slug | Extra column; redundant | |

### Q6.4 — Entry point

| Option | Description | Selected |
|--------|-------------|----------|
| examples/run_baseline.py + reusable function in src/judge.py | Recipe §10 pattern; Phase 5 harness imports the function | ✓ |
| examples/run_baseline.py only, logic inline | Doesn't match recipe §10 thin-wrapper pattern | |
| --baseline flag on existing entry point | Conflates two distinct flows | |

---

## 7. Judge structure + WR-04 cleanup

### Q7.1 — f_eval location

| Option | Description | Selected |
|--------|-------------|----------|
| Keep stopping.py as-is (fill numbers); new src/judge.py for everything else | check_stopping IS f_eval; pure function pattern; matches f_* decomposition | ✓ |
| Wrap everything in JudgeAgent class | Stateful agent for mostly pure compute; breaks f_* convention | |
| Inline in conversation_loop.py | Bloats loop with 3 concerns; SQL leakage risk | |

### Q7.2 — WR-04 fix

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — fix as part of Phase 4 wiring | Three small edits; loop is already being modified; required for Phase 5 | ✓ |
| Defer to a follow-up cleanup phase | Leaves known protocol/dispatch split | |
| Fix signature, leave isinstance branch | Half-fix | |

### Q7.3 — UI panels (Phase 2 D-14 deferred)

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — contradictions + convergence + pairwise sparkline | ~30 lines vanilla JS; reuses existing state_update event | ✓ |
| Add just contradictions + convergence (skip pairwise UI) | Lighter touch; loses visual signal | |
| Skip UI — DB-only | Pushes Phase 2 UI-01 gap further | |

---

## 8. STRICT_MODE + deviation() + CI

### Q8.1 — deviation() rollout

| Option | Description | Selected |
|--------|-------------|----------|
| New code only — Phase 4 uses it; Phase 1–3 left alone | Migrate opportunistically; existing asserts already serve fail-loudly | ✓ |
| Full backfill — audit Phase 1–3 | Large scope, easy to regress | |
| Skip deviation() entirely | Loses STRICT_MODE knob + "grep deviation" audit | |

### Q8.2 — CI setup

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — .github/workflows/ci.yml with STRICT_MODE matrix [0, 1] | Recipe §12; both must pass; ~35 lines YAML | |
| Yes, single-leg — just pytest, no matrix | Defer matrix until deviation() has enough coverage | |
| Defer CI to a later phase | Local-only; no automated enforcement | ✓ |

### Q8.3 — Local tooling (pyproject.toml)

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — ruff + pytest -W error + branch coverage | Each tool would have caught a Phase 3 REVIEW.md finding (IN-01 / CR-02 / WR-02) | |
| Ruff only — skip the other two | Surgical; avoids triaging existing numpy/sklearn warnings | |
| Skip tooling config entirely | Defer along with CI | ✓ |

**Notes:** User asked "Why do I want to set this?" first; the assistant explained the three tools' concrete payoffs grounded in Phase 3 REVIEW.md findings. User then chose to defer.

---

## 9. Documentation discipline

### Q9.1 — docs/MODEL.md

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — new docs/MODEL.md covering DB-01/02/03 schema + cascade rules | Recipe §15; living spec; ~40 lines; spec-first discipline | ✓ |
| Skip — keep schema spec inline in src/db/connection.py | Loses separation between spec and code | |
| Embed in CONTEXT.md | Phase-archived; nobody reads it after the phase closes | |

### Q9.2 — CLAUDE.md updates

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — additive rules under Key Constraints and Web Stack | New rules about src/db/ as only SQL layer, PRAGMAs, deviation(), JSONL contract | ✓ |
| Skip CLAUDE.md updates | Future AI sessions re-derive | |
| Defer to end of Phase 4 (post-execute) | Mistakes more likely during execute itself | |

**Notes:** User explicitly required "keep the Fail Loudly". D-32 makes updates additive only — the existing "Coding Philosophy — Fail Loudly" section stays intact. `deviation()` is documented as complementary to assertions, not a replacement.

### Q9.3 — Spec-first rule

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — docs/MODEL.md is the spec; SQL must match | Edit MODEL.md FIRST; drift is a workflow violation | ✓ |
| MODEL.md exists but isn't gated | Reconcile periodically; drift compounds | |
| Skip the spec-first rule | MODEL.md becomes stale snapshot | |

### Q9.4 — Other docs (DEMO.md + README.md)

| Option | Description | Selected |
|--------|-------------|----------|
| README.md updates only — skip DEMO.md for now | DEMO.md paired with seed.py which doesn't exist; add when Phase 5 introduces it | ✓ |
| Both — README + minimal DEMO.md | DEMO.md without paired script rots fast | |
| Skip both | Loses new-contributor quickstart for baseline runner | |

---

## Claude's Discretion

The following areas are left to the planner/executor without further user input:

- Exact Pydantic field validators and types for the three Create/Read/Update triples (D-10)
- Exact YAML config schema shape (field names, nesting) (D-16)
- Additional SQL indexes beyond the FKs and partial slug-uniqueness index — driven by Phase 5 query patterns
- `src/logging_setup.py` log format (JSON-line vs key=value) (D-29)
- `RotatingFileHandler` config when `LOG_FILE_PATH` is set (recipe §11)
- Exact SocketIO event shape additions for D-28 (extend `state_update` vs new event)
- Sparkline rendering details for D-28 (canvas vs SVG vs inline divs)
- HDBSCAN `min_cluster_size` / KMeans BIC search range for `run_baseline()` initial clustering — reuse Phase 1/2 defaults

---

## Deferred Ideas

Captured in CONTEXT.md `<deferred>` section. Brief recap:

- CI with STRICT_MODE matrix (recipe §12) — deferred per D-30
- Local lint/test tooling config — deferred per D-30
- Backfill of `deviation()` to Phase 1–3 code — explicitly out of scope per D-29
- Mass-rewriting Phase 1–3 dataclasses to Pydantic — explicitly out of scope per D-10
- Polymorphic rels table — deferred per D-06
- `docs/DEMO.md` + `examples/seed.py` — deferred to Phase 5
- Full mutation API on experiments (only end-of-run summary update ships) — D-09
- Health/readiness/migrations/real-auth — out of scope for v1 per PROJECT.md tier
- Async DB driver / Postgres — out of scope; SQLite is in-process and sync per D-01
- Pre-existing project-level todos (install umap-learn/hdbscan; human study protocol) — not Phase 4 scope
