---
name: software-engineering
description: Orchestrate full-stack feature builds as a pipeline of specialist skills instead of ad-hoc edits.
version: 1.0.0
tags: [orchestration, workflow, planning]
depends_on: []
---

# Software Engineering Pipeline

You are acting as the **technical lead** for a multi-skill build. When the user
asks for anything substantial (a feature, an API, a screen, a full app), do NOT
skip straight to writing files. Compose the relevant specialist skills below so
each concern gets its own expert treatment.

## When to use
- Any request that spans two or more layers (backend + frontend, database + API,
  security + anything).
- Any "build X" / "implement Y" request where X has a schema, endpoints, UI,
  and tests.

## Pipeline
1. **Understand the project first.** Read the skill `project-orientation`, then
   use `read`/`glob`/`grep`/`code_index` to map the repo: entry points, existing
   schema, framework, test runner, style conventions.
2. **Plan before writing.** List the work items, pick which specialist skills
   apply, and announce the plan to the user before mutating files.
3. **Run the pipeline bottom-up (dependencies first):**
   - `database` — schema/DDL first when a datastore is involved.
   - `api` — endpoints that the frontend will consume.
   - `frontend` — screens wired to those endpoints.
   - `security` — a pass on auth/authz, secrets, and injection before tests.
   - `qa-testing` — tests for every behavioral change.
   - `documentation` — update docs/README for anything user-visible.
4. **Verify at each stage**: run the relevant build/test command from the shell
   before moving to the next skill.

## Composing skills
Load dependencies with `read_skill { "name": "<skill>", "recursive": true }` so
the whole chain arrives in context in one call. Skills are a checklist — the
instructions shape HOW you work; they don't replace reading the actual code.

## When NOT to use
- Single-file typo/small-refactor requests: just use the core tools directly.
- Read-only questions: answer from context, no pipeline needed.

## Definition of done
- Build/typecheck green
- Tests added/updated for changed behavior
- No dead code, no secrets, no inventory of the project left stale