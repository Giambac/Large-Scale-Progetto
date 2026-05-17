---
phase: "06"
plan: "01"
subsystem: "mapping"
tags: [generalization, mapping, oracle-rules, llm-strategy, centroid-strategy, pydantic, protocol]
dependency_graph:
  requires:
    - src/strategy.py          # StrategyProtocol pattern mirrored exactly
    - src/embedding_store.py   # EmbeddingStore.get(), EMBEDDING_MODEL
    - src/state.py             # ClusteringState, Cluster
    - src/serialization.py     # load_audit_log for audit log validation
    - src/logging_setup.py     # deviation() for unexpected branches
    - src/feedback.py          # InstructionalFeedback shape (FB-04)
  provides:
    - src/mapping.py           # OracleRuleSet, extract_oracle_rules, MappingProtocol, LLMMappingStrategy, CentroidMappingStrategy, MAPPING_REGISTRY
  affects:
    - 06-02-PLAN.md            # GEN-02 evaluation imports MAPPING_REGISTRY and both strategies
tech_stack:
  added:
    - pydantic.BaseModel (already installed from Phase 4; OracleRuleSet model)
    - anthropic.Anthropic (already installed from Phase 2; LLM calls in both strategies)
    - sentence_transformers.SentenceTransformer (already installed; lazy load in CentroidMappingStrategy.__init__)
  patterns:
    - Protocol + @runtime_checkable structural subtyping (mirrors StrategyProtocol exactly)
    - REGISTRY dict mapping string keys to strategy classes (mirrors STRATEGY_REGISTRY)
    - Pydantic BaseModel for data contracts (consistent with Phase 4 discipline)
    - deviation() for unexpected-but-possible branches (no FB-04 entries, per CLAUDE.md)
    - try/except anthropic.APIError only at LLM boundary; all other exceptions propagate
key_files:
  created:
    - src/mapping.py
  modified: []
decisions:
  - "OracleRuleSet fields: synonyms (list[list[str]]), focus_areas (list[str]), exclusions (list[str]), cluster_rules (list[str]) — derived from FB-04 InstructionalFeedback parsed deltas in AuditLog"
  - "extract_oracle_rules reads raw JSONL lines (not deserialized ClusteringState) to handle both current and future delta-storing formats; calls deviation('no_fb04_entries') when none found"
  - "LLMMappingStrategy validates LLM cluster ID response against valid cluster ID set; raises ValueError on mismatch (T-06-01-01 threat mitigation)"
  - "CentroidMappingStrategy uses hard-assignment mean (mean of item embeddings by cluster.item_ids); raises ValueError for empty-item-list clusters (fail loudly)"
  - "CentroidMappingStrategy uses cosine_sim = dot(a,b)/(norm(a)*norm(b)+1e-10) to avoid division by zero"
  - "MAPPING_REGISTRY = {'llm': LLMMappingStrategy, 'centroid': CentroidMappingStrategy} — same dict shape as STRATEGY_REGISTRY"
metrics:
  duration: "~15 minutes"
  completed: "2026-05-17T11:57:35Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 0
---

# Phase 6 Plan 01: Mapping Function Core — OracleRuleSet, MappingProtocol, LLM + Centroid Strategies

**One-liner:** Pluggable mapping layer with OracleRuleSet Pydantic model and two strategies (LLM-context-aware and cosine-centroid) registered in MAPPING_REGISTRY, mirroring StrategyProtocol/STRATEGY_REGISTRY exactly.

## What Was Built

`src/mapping.py` — the complete pluggable mapping function layer for GEN-01:

1. **OracleRuleSet** (Pydantic BaseModel) — codified oracle preferences with four fields: `synonyms`, `focus_areas`, `exclusions`, `cluster_rules`. Consistent with Phase 4 Pydantic-in-src/ discipline.

2. **extract_oracle_rules(audit_log_path) -> OracleRuleSet** — reads raw JSONL lines from audit_log.jsonl, finds delta entries with `type="instructional"`, collects `instruction_text` values, and classifies them into OracleRuleSet fields via a single `claude-haiku-4-5` call. Calls `deviation("no_fb04_entries")` when no instructional deltas are found (the typical case for existing sessions, since ClusteringState serialization does not embed delta lists).

3. **MappingProtocol** — `@runtime_checkable Protocol` with `assign(item_text, state, rule_set, embedding_store) -> str`. Mirrors `StrategyProtocol` from `src/strategy.py` structurally (no inheritance needed).

4. **LLMMappingStrategy** — builds a prompt with all cluster names+descriptions, OracleRuleSet fields as bullet lists, and the new item text. One `claude-haiku-4-5` call per `assign()`. Validates returned cluster ID against known cluster IDs; raises `ValueError` on mismatch (threat T-06-01-01 mitigation).

5. **CentroidMappingStrategy** — computes cluster centroids (mean of `embedding_store.get(iid)` for each `cluster.item_ids`). Embeds new item with `SentenceTransformer(EMBEDDING_MODEL)` (loaded once in `__init__`). Assigns to nearest centroid by cosine similarity. Raises `ValueError` if any cluster has empty `item_ids` (fail loudly).

6. **MAPPING_REGISTRY** — `{"llm": LLMMappingStrategy, "centroid": CentroidMappingStrategy}` — same shape as `STRATEGY_REGISTRY`.

## Verification Results

All 5 plan verification checks pass:

1. `MAPPING_REGISTRY` keys == `{'llm', 'centroid'}` — PASSED
2. Both strategies satisfy `MappingProtocol` (isinstance check) — PASSED
3. `OracleRuleSet` construction and field access — PASSED
4. No `sqlite3` import, no `src.db` reference — PASSED
5. `assign()` signature: `(item_text, state, rule_set, embedding_store) -> str` — PASSED

## Deviations from Plan

### Auto-fixed Issues

None.

### Architectural Observations

**load_audit_log API mismatch:** The plan's interface spec states `load_audit_log(path: str) -> list[dict]`, but the actual `src/serialization.py` implementation returns `list[ClusteringState]`. The `extract_oracle_rules` function uses `load_audit_log` only for its side-effect of validating that the audit log exists and is non-empty (it asserts both). The function then re-opens the file to read raw JSONL lines, scanning for `"deltas"` keys — a forward-compatible approach that handles both the current format (no deltas field) and any future format that embeds delta lists. In practice, existing audit logs will almost always hit the `no_fb04_entries` deviation path, which is correct behavior (returns empty OracleRuleSet).

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes beyond what the plan specified. The two new LLM call sites (`extract_oracle_rules` and `LLMMappingStrategy.assign`) follow the established `try/except anthropic.APIError: raise` pattern from `oracle_agent.py` and `feedback_parser.py`.

## Commits

| Task | Description | Commit |
|------|-------------|--------|
| T-06-01-01 + T-06-01-02 | Complete src/mapping.py with all 6 exports | 8e98138 |

## Self-Check

- [x] `src/mapping.py` exists
- [x] Commit `8e98138` exists in git log
- [x] All 5 verification checks pass
- [x] No stubs (no TODO/FIXME/placeholder text)
- [x] No sqlite3 or src.db references

## Self-Check: PASSED
