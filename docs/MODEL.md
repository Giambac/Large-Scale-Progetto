# DB Layer: Data Model

DB layer: SQLite at `experiments.db` (repo root, gitignored). Source of truth for cross-run
analysis. JSONL `audit_log.jsonl` remains source of truth for state replay.

---

## Table: `experiments`

Full recipe shape — indexed columns + details JSON. One row per conversation run.

| Column | Type | Constraints | Purpose |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | Auto-increment row identifier |
| `name` | TEXT | NOT NULL | Human-readable run name (e.g. `random-trump-seed42`) |
| `slug` | TEXT | NOT NULL, unique among live rows via partial index | URL-safe identifier; immutable once written |
| `created_at` | TEXT | NOT NULL | ISO-8601 UTC timestamp (row insert time) |
| `updated_at` | TEXT | NOT NULL | ISO-8601 UTC timestamp (last mutation time) |
| `deleted_at` | TEXT | NULL = live row | ISO-8601 UTC timestamp; NULL means the experiment is live |
| `strategy_id` | TEXT | NOT NULL | Clustering strategy identifier (e.g. `random`, `uncertainty_driven`) |
| `persona_id` | TEXT | NOT NULL | Oracle persona identifier (e.g. `trump`, `curie`) |
| `seed` | INTEGER | NOT NULL | RNG seed for reproducible runs |
| `dataset` | TEXT | NOT NULL | Dataset name used for this run |
| `total_turns` | INTEGER | NULL until run ends | Number of conversation turns completed |
| `convergence_reason` | TEXT | NULL until run ends | One of: `oracle_satisfied`, `turn_budget`, `diminishing_returns` |
| `start_timestamp` | TEXT | NOT NULL | ISO-8601 UTC; when the conversation started |
| `end_timestamp` | TEXT | NULL until run ends | ISO-8601 UTC; when the conversation ended |
| `details` | TEXT | JSON | Headline metrics and run metadata: `mean_cognitive_load`, `mean_pairwise_accuracy`, `contradiction_count`, `turns_to_convergence`, `config_snapshot`, `oracle_init_params`, `git_commit` |

---

## Table: `turns`

Minimal shape — FKs + indexed columns + details JSON. One row per conversation turn.

| Column | Type | Constraints | Purpose |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | Auto-increment row identifier |
| `experiment_id` | INTEGER | NOT NULL, REFERENCES experiments(id) ON DELETE CASCADE | FK to parent experiment |
| `created_at` | TEXT | NOT NULL | ISO-8601 UTC timestamp (row insert time) |
| `turn_index` | INTEGER | NOT NULL | Zero-based turn counter within the experiment |
| `action_type` | TEXT | NOT NULL | Type of action taken this turn |
| `cognitive_load_score` | REAL | NOT NULL | Per-turn cognitive load from `f_cognitive_load()` |
| `cumulative_contradiction_count` | INTEGER | NOT NULL | Running total of oracle contradictions up to this turn |
| `convergence_signal` | TEXT | NULL = still running | One of the `StopReason` values, or NULL if conversation continues |
| `details` | TEXT | JSON | Per-turn extras: `pairwise_accuracy`, `pairwise_sample_size`, `raw_deltas_count` |

### Deliberately omitted fields

- **`turns.system_message`**: Stored in `audit_log.jsonl` only; not promoted to a DB column because
  the JSONL file is the source of truth for state replay (JSONL-first contract). Promoting it to
  the turns table would create a secondary copy that could diverge from the JSONL record.
  REQUIREMENTS.md DB-02 (`turns.system_message`) is satisfied via JSONL in accordance with D-05.

---

## Table: `oracle_feedback`

Minimal shape — FKs + indexed columns + details JSON. One row per feedback delta per turn.
Compound oracle messages produce multiple rows per turn (DB-03).

| Column | Type | Constraints | Purpose |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | Auto-increment row identifier |
| `turn_id` | INTEGER | NOT NULL, REFERENCES turns(id) ON DELETE CASCADE | FK to parent turn |
| `created_at` | TEXT | NOT NULL | ISO-8601 UTC timestamp (row insert time) |
| `feedback_type` | TEXT | NOT NULL | One of: `GlobalFeedback`, `SplitFeedback`, `MergeFeedback`, `MoveItemFeedback`, `InstructionalFeedback` |
| `raw_text` | TEXT | NOT NULL | Raw oracle utterance text |
| `parsed_delta` | TEXT | NULL if unparseable | JSON representation of the `FeedbackDelta` |
| `is_contradiction` | INTEGER | NOT NULL, DEFAULT 0 | SQLite boolean (0 or 1); 1 if this feedback contradicts a prior statement |
| `details` | TEXT | JSON | Any extra data not promoted to an indexed column |

### Deliberately omitted fields

- **`oracle_feedback.target`**: Captured via the `parsed_delta` JSON column; the full
  `FeedbackDelta` repr already encodes the target cluster/item reference. A dedicated column
  would be redundant and would require updating every time `FeedbackDelta` adds new target
  variants. REQUIREMENTS.md DB-03 (`oracle_feedback.target`) is satisfied via `parsed_delta`
  in accordance with D-05.

---

## Cascade Rules

**Soft-delete cascade (via `src/db/deletion.py`):**

1. `deletion.cascade_delete(db, experiment_id)` sets `deleted_at` on the experiment row.
2. It then sets `deleted_at` on all turns WHERE `experiment_id = ?` AND `deleted_at IS NULL`.
3. `oracle_feedback` has no `deleted_at` column (minimal shape per D-05); child rows are
   excluded from live reads by filtering their parent turns (`WHERE deleted_at IS NULL`).

**Hard delete (via ON DELETE CASCADE FK):**

Removing an `experiments` row removes all child `turns` rows via the FK cascade, which in
turn removes all child `oracle_feedback` rows. Used only for test teardown or explicit data purge.

Live reads for turns and experiments always filter `WHERE deleted_at IS NULL`.

---

## Slug Uniqueness

Partial unique index:

```sql
CREATE UNIQUE INDEX experiment_slug_live ON experiments(slug) WHERE deleted_at IS NULL;
```

- Only live (non-deleted) rows must have unique slugs.
- Soft-deleted slugs are reusable (a new run with the same strategy/persona/seed can reuse the slug).
- Slug pattern: `{strategy_id}-{persona_id}-seed{seed}` (e.g. `random-trump-seed42`).
- Collision suffix: `-2`, `-3`, ... (appended by `src/db/slugs.make_slug()`).
- Slugs are **immutable** once written.

---

## Indexes

| Index Name | Table | Columns | Where Clause | Purpose |
|---|---|---|---|---|
| `experiment_slug_live` | experiments | `(slug)` | `WHERE deleted_at IS NULL` | Partial unique index for live slug uniqueness |
| `idx_experiments_strategy` | experiments | `(strategy_id, persona_id, dataset)` | — | Phase 5 cross-run GROUP BY queries |
| `idx_turns_experiment` | turns | `(experiment_id, turn_index)` | — | Turn pagination and ordered replay |
| `idx_oracle_feedback_turn` | oracle_feedback | `(turn_id)` | — | Feedback lookup by turn |

---

## Timestamp Convention

All timestamps use:

```python
datetime.now(timezone.utc).isoformat()
```

**Never** `datetime.utcnow()` (deprecated, returns a naive datetime with no timezone info).
Timestamps are stored as TEXT ISO-8601 UTC strings (e.g. `2026-01-01T00:00:00+00:00`).

---

## Change Discipline (D-31)

**Edit this file FIRST when the schema changes. SQL follows.**

Drift between `docs/MODEL.md` and the `CREATE TABLE` statements in `src/db/connection.py`
is a workflow violation, not a quiet inconsistency.
