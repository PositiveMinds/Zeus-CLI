---
name: database
description: Design schemas, SQL, and migrations; optimize queries and indexes; keep data migrations safe.
version: 1.0.0
tags: [database, sql, schema, migrations, optimization]
depends_on: []
---

# Database Engineering

Design and change databases deliberately — schema changes are the most
expensive mistakes to reverse.

## Schema design
- **Model intent first**: entities, relationships, cardinality (1:1 / 1:N /
  N:M) and the access patterns that will query them. Schema follows queries,
  not the other way round.
- **Keys**: choose a stable natural key when one exists; otherwise a UUID or
  surrogate key. Prefer `BIGINT`/identity in MySQL-family, `SERIAL`/
  `IDENTITY` in Postgres, `TEXT` or `INTEGER` PK in SQLite.
- **Constraints**: `NOT NULL` unless there's a real reason for nulls; add
  `CHECK` for invariants, `UNIQUE` for identity, `FOREIGN KEY` for integrity.
- **Normalize, then denormalize deliberately** for hot read paths — never
  denormalize by default.

## Migrations
- One migration per logical change; write **both** `up` and `down` (rollback)
  paths. Never edit an applied migration — add a new one.
- Backfill data in the migration itself when adding a `NOT NULL` column to a
  table with rows. Test the down path in CI.
- Timestamp ordering, not sequential numbers, avoids merge conflicts.

## Query & index optimization
- Fetch only columns you need. Watch for N+1 — `JOIN` or batch.
- Index the columns used in `WHERE`, `JOIN`, `ORDER BY`; prefer composite
  indexes that match the exact predicate order. Watch cardinality — an index
  on a low-cardinality column (e.g. `gender`) is rarely worth it.
- `EXPLAIN` on slow queries; aim for index scans over seq scans on hot tables.
- Use parameterized queries always — never string-concatenate user input.

## Concurrency & safety
- Wrap multi-statement changes in transactions.
- `SELECT ... FOR UPDATE`/row locks only where the business logic needs them.
- Mind lock duration on long-running writes; batch large DELETEs/UPDATEs.

## Verification
- Run migrations against a scratch DB and confirm schema.
- Add a smoke query per new table/index and confirm it uses the index
  (`EXPLAIN QUERY PLAN` in SQLite, `EXPLAIN (ANALYZE, BUFFERS)` in Postgres).