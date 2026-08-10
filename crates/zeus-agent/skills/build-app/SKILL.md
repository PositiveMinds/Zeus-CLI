---
name: build-app
description: Full-stack app builds by composing specialist skills — requirements, database, API, frontend, security, tests, docs.
version: 1.0.0
tags: [application, fullstack, orchestration]
depends_on: [project-orientation, database, api, frontend, security, qa-testing, documentation]
---

# Build An Application (composable)

This skill is the **composer**: a single user request ("build a hospital
dashboard", "add offline sync", "ship a billing portal") expands into a
pipeline of specialist skills. It exists so you never need a separate agent
per task — one request chains many skills.

## How it composes
`read_skill build-app recursive:true` loads, in order:
1. `project-orientation` — understand the repo/stack before writing.
2. `database` — schema/DDL & migrations.
3. `api` — the endpoints.
4. `frontend` — screens wired to those endpoints.
5. `security` — auth/authz, secrets, injection hardening.
6. `qa-testing` — tests for each behavior.
7. `documentation` — README/docs updates.

## Executing the build
1. **Requirements** (2 min, out loud): restate what the user asked, the
   entities, the screens, the flows. Get a nod or flag ambiguities.
2. **Plan**: walk the pipeline bottom-up, listing concrete work items and the
   files you'll touch. Explicitly say which skills you're activating.
3. **Build bottom-up, verify each stage**:
   - schema → migrations run → seed smoke query
   - routes → handler tests → live request
   - UI → lint/typecheck → tests → visual check (if feasible)
4. **Security gates**: auth on every mutating route; no secrets; parameterized
   SQL. Use `security` skill checklist.
5. **Closing**: docs updated, full suite green, summary of what exists and
   what the user should try.

## Cross-cutting rules
- Every stage produces something the next consumes (contracts).
- Verify with the project's own build/test commands at each stage boundary —
  don't stack four stages on top of a fire.
- When one stage surfaces a problem that changes earlier work, go back and fix
  the earlier stage; the plan is a scratchpad, not scripture.
- Keep composition visible: tell the user which skills you used and why.