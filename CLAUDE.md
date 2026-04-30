# Conversational Clustering — Project Guide

## Project Overview

A 3-agent AI research system that clusters text data through dialogue with a human oracle.
See `.planning/PROJECT.md` for full context.

## GSD Workflow

This project uses GSD (Get Shit Done) for structured execution.

**Check current state:** `/gsd-progress`
**Resume after break:** `/gsd-resume-work`
**Next action:** `/gsd-plan-phase 1`

## Agent Architecture

Three core agents:
- **Clustering Agent** — maintains state, implements `f_output`, `f_uncertainty`, `f_next_best_step`, `f_next_state`
- **Oracle Agent** — LLM-simulated human oracle with preference spec, persona, noise params (`consistency_rate`, `drift_probability`, `sycophancy_resistance`)
- **Judge Agent** — `f_eval`, convergence detection, per-turn metric logging, no-dialogue baseline

## Key Constraints

- `f_output` must ALWAYS return a complete assignment (anytime behavior — never partial state)
- K changes ONLY through oracle intent — no automatic K optimization (silhouette, BIC, etc.)
- AuditLog serialized to JSONL every turn from Phase 1 — never skip this
- Held-out split locked before ANY experiment runs — never modify it
- All headline quantitative claims must include bootstrap 95% CIs

## Coding Philosophy — Fail Loudly

**Let the code crash. Do not make it robust.**

- No silent failures — if something is wrong, raise an exception or let it propagate
- No defensive `if/else` chains to paper over unexpected state
- No swallowed exceptions (`except: pass`, `except Exception as e: log and continue`)
- Use `assert` statements freely to document and enforce invariants
- Error handling is only permitted at two places: the CLI entry point and external API calls (LLM, embeddings)
- Everywhere else: unexpected state = crash immediately with a clear traceback

A crash during development is a feature. It tells you exactly what broke and where. Silent wrong answers are the actual failure mode to avoid.

## Planning Artifacts

```
.planning/
├── PROJECT.md        # Vision, constraints, key decisions
├── REQUIREMENTS.md   # 28 v1 requirements with REQ-IDs
├── ROADMAP.md        # 6-phase breakdown
├── STATE.md          # Current project state
└── research/         # Domain research (stack, features, architecture, pitfalls)
```
