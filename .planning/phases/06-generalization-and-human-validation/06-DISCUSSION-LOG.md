# Phase 6: Generalization and Human Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-17
**Phase:** 6-Generalization-and-Human-Validation
**Areas discussed:** Mapping function, Human study UI, Oracle satisfaction for humans, GEN-02 validation mechanic

---

## Mapping Function

| Option | Description | Selected |
|--------|-------------|----------|
| LLM-prompt classifier | LLM called once per item at classification time; cluster descriptions + FB-04 constraints in prompt | ✓ (combined) |
| Embedding nearest-centroid | Cosine similarity to cluster centroids; no LLM at inference; ignores explicit oracle rules | ✓ (combined) |
| Hybrid: extract-once, classify-cheap | LLM extracts rules once; inference uses rules + cosine similarity, no per-item LLM call | — |

**User's choice:** Implement both LLM and centroid strategies and compare. For LLM classifier, also extract FB-04 rules from AuditLog and use them as context (not just cluster descriptions). Architecture should support adding an ensemble or combined strategy in future without changing evaluation code.

**Notes:**
- User asked whether the LLM in the hybrid option was the same model as option 1 — clarified: same client, different call timing (once at session end vs. once per item)
- User asked whether they could implement both and compare — yes, this is cleaner as a research finding
- User confirmed OracleRuleSet should be a Pydantic object
- Pluggable MappingProtocol mirrors StrategyProtocol pattern from Phase 2/5

---

## Human Study UI

| Option | Description | Selected |
|--------|-------------|----------|
| Existing debug UI as-is | No new frontend work; participants see all developer panels | |
| Study mode flag | ?study=true hides UMAP, soft-prob bars, per-turn metrics | |
| Minimal dedicated page | New /study route with stripped-down view | ✓ |

**User's choice:** Minimal dedicated page (/study route). Full cluster detail (names, descriptions, expandable item list). No conversation history visible — current state only. Free text input.

**Additional features specified by user:** Faceted UMAP representation — K per-cluster mini-plots sharing global coordinate space. Hover/click item in list highlights its position in corresponding mini-plot.

**Session start:** Researcher pre-loads session; participant joins via URL/session ID. Participant dataset upload deferred to future.

---

## Oracle Satisfaction for Humans

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit 'Done' button | Participant clicks when satisfied; triggers stop signal | |
| LLM parses free-text for satisfaction | Per-turn LLM call detects satisfaction intent in participant message | ✓ |
| Turn budget only | No satisfaction detection; session runs to turn limit | |

**User's choice:** LLM parses free-text for satisfaction, with safety turn limit of 30 (configurable). When satisfaction detected, confirm with participant before stopping ("It looks like you're satisfied. End session?").

**Notes:** User specified the 30-turn cap is tunable, not hardcoded.

---

## GEN-02 Validation Mechanic

| Option | Description | Selected |
|--------|-------------|----------|
| Oracle labels items — ground truth | Oracle assigns 20–30 held-out items to clusters; mapping function predictions compared against oracle labels | ✓ |
| Oracle reviews assignments — binary agree/disagree | Mapping function assigns first; oracle reviews each | |
| Oracle runs mini-conversation on held-out subset | Short session on held-out items; blurs train/test boundary | |

**User's choice:** Oracle labels items as ground truth. Both mapping strategies evaluated on the same 20–30 item sample.

**Who labels:** LLM oracle only. User asked whether human participants were obligated to do GEN-02 labeling when running study sessions — answer: no, these are independent. GEN-02 uses LLM oracle; participant dataset upload for GEN-02 is deferred.

---

## Claude's Discretion

- Exact `OracleRuleSet` Pydantic field schema — design based on observed FB-04 constraint types
- Whether `extract_oracle_rules()` uses structured prompt or JSON mode
- Cosine similarity implementation (scipy vs. numpy)
- Number of held-out items for GEN-02 sample (default 25, configurable)
- `/study` page frontend layout
- How cluster centroids are computed (hard vs. soft-weighted mean)

## Deferred Ideas

- Participant-side dataset upload on study page — user noted future interest
- Oracle-initiated turns — deferred from Phase 5
- Ensemble MappingStrategy — protocol designed to support it; deferred until Phase 6 results show value
- Human GEN-02 labeling — participants label held-out items for human-oracle accuracy comparison
- Mutable noise params / fatigue simulation
- Soft-assignment calibration (EVAL-V2-01)
